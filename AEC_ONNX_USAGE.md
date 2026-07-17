# ULUNAS-AEC ONNX 流式推理调用指南

## 模型文件

`ulunas_aec_stream_ep_0033.onnx` — 流式双路 AEC（Acoustic Echo Cancellation）

## 核心参数

| 参数 | 值 | 说明 |
|------|-----|------|
| 采样率 | 48000 Hz | — |
| 帧长 (hop) | 480 samples | 每帧输入/输出 10ms |
| 窗长 (win) | 960 samples | 分析窗（20ms） |
| FFT 点数 (n_fft) | 960 | → 481 频点 |
| 窗函数 | `hann(960).pow(0.5)` | sqrt-Hann |
| 输入通道 | 单声道 | mic 和 far 都是 mono float32 |

## ONNX 模型接口

### 输入（9 个）

| 名称 | 形状 | 类型 | 说明 |
|------|------|------|------|
| `mic_spec` | `(1, 2, 1, 481)` | float32 | 麦克风 STFT 帧 [batch, R/I, time=1, freq] |
| `far_spec` | `(1, 2, 1, 481)` | float32 | 远端参考 STFT 帧 |
| `mic_conv` | `(1, 7368)` | float32 | Mic 编码器卷积缓存 |
| `mic_tfa` | `(1, 368)` | float32 | Mic 编码器 TFA 缓存 |
| `far_conv` | `(1, 7368)` | float32 | Far 编码器卷积缓存 |
| `far_tfa` | `(1, 368)` | float32 | Far 编码器 TFA 缓存 |
| `dec_conv` | `(1, 2880)` | float32 | 解码器卷积缓存 |
| `dec_tfa` | `(1, 244)` | float32 | 解码器 TFA 缓存 |
| `inter` | `(1, 4608)` | float32 | DPGRNN 中间缓存 |

### 输出（8 个）

| 名称 | 形状 | 类型 | 说明 |
|------|------|------|------|
| `mask` | `(1, 2, 1, 481)` | float32 | 复数频域 mask [batch, R/I, time=1, freq] |
| `mic_conv_o` ~ `inter_o` | 同输入形状 | float32 | 更新后的 7 个缓存 |

## 完整调用流程

```python
import numpy as np
import torch
import onnxruntime as ort

# ========== 1. 初始化 ==========
so = ort.SessionOptions()
so.intra_op_num_threads = 1
so.inter_op_num_threads = 1
sess = ort.InferenceSession("ulunas_aec_stream_ep_0033.onnx", so,
                            providers=['CPUExecutionProvider'])

# STFT 参数
WIN_LEN = 960   # window size
HOP_LEN = 480   # chunk size
N_FFT   = 960   # FFT size → 481 bins
WINDOW  = torch.hann_window(WIN_LEN).pow(0.5)  # sqrt-Hann

# 7 个 cache，初始化为全零
CACHE_SIZES = [7368, 368, 7368, 368, 2880, 244, 4608]
caches = [np.zeros((1, s), dtype=np.float32) for s in CACHE_SIZES]

# 滑动窗口缓冲区（960 样本）
mic_buffer = torch.zeros(WIN_LEN)
far_buffer = torch.zeros(WIN_LEN)


# ========== 2. 逐帧处理（每帧 480 samples） ==========
def process_frame(mic_480: list, far_480: list) -> list:
    """
    mic_480: 麦克风 480 样本 float32
    far_480: 远端参考 480 样本 float32（扬声器 loopback）
    返回: 回声消除后的 480 样本
    """
    global mic_buffer, far_buffer, caches

    mic = torch.tensor(mic_480, dtype=torch.float32)
    far = torch.tensor(far_480, dtype=torch.float32)

    # --- Step A: 更新滑动窗口 ---
    mic_buffer = torch.cat([mic_buffer[HOP_LEN:], mic])  # (960,)
    far_buffer = torch.cat([far_buffer[HOP_LEN:], far])

    # --- Step B: STFT（单帧，hop=win 产生 1 帧） ---
    mic_spec = torch.stft(mic_buffer, n_fft=N_FFT,
                          hop_length=WIN_LEN, win_length=WIN_LEN,
                          window=WINDOW, center=False,
                          return_complex=True)  # (481, 1)
    far_spec = torch.stft(far_buffer, n_fft=N_FFT,
                          hop_length=WIN_LEN, win_length=WIN_LEN,
                          window=WINDOW, center=False,
                          return_complex=True)  # (481, 1)

    # --- Step C: 拆实部/虚部 → (1, 2, 481, 1) ---
    # 关键：ONNX 输入是 R/I 两个独立通道，不是 complex64
    mf = mic_spec[..., -1]  # (481,) complex
    ff = far_spec[..., -1]
    mic_frame = np.stack([mf.real.numpy(), mf.imag.numpy()], axis=0)      # (2, 481)
    far_frame = np.stack([ff.real.numpy(), ff.imag.numpy()], axis=0)
    mic_frame = mic_frame[np.newaxis, :, :, np.newaxis]                   # (1, 2, 481, 1)
    far_frame = far_frame[np.newaxis, :, :, np.newaxis]

    # --- Step D: ONNX 推理 ---
    inputs = {
        'mic_spec': mic_frame, 'far_spec': far_frame,
        'mic_conv': caches[0], 'mic_tfa': caches[1],
        'far_conv': caches[2], 'far_tfa': caches[3],
        'dec_conv': caches[4], 'dec_tfa': caches[5],
        'inter':     caches[6],
    }
    outputs = sess.run(None, inputs)
    mask = torch.from_numpy(outputs[0])   # (1, 2, 1, 481)
    caches = outputs[1:]                  # 更新 cache

    # --- Step E: 复数乘 — enhanced = mask × mic ---
    # mask: (1, 2, 1, 481) → permute → (1, 2, 481, 1)
    mask_ri = mask.permute(0, 1, 3, 2)       # (1, 2, 481, 1)
    mr = mask_ri[:, 0:1, :, :]                # 实部 (1, 1, 481, 1)
    mi = mask_ri[:, 1:2, :, :]                # 虚部

    mic_frame_c = mic_spec[..., -1]            # (481,) complex
    mic_r = mic_frame_c.real.reshape(1, 1, 481, 1)
    mic_i = mic_frame_c.imag.reshape(1, 1, 481, 1)

    # 复数乘法: (a+bi)(c+di) = (ac-bd) + (ad+bc)i
    enhanced_real = mr * mic_r - mi * mic_i    # (1, 1, 481, 1)
    enhanced_imag = mr * mic_i + mi * mic_r

    # --- Step F: iFFT 还原时域 ---
    # 拼接实/虚 → (1, 2, 481, 1) → permute → (1, 481, 1, 2) → complex (1, 481, 1)
    enhanced_spec = torch.cat([enhanced_real, enhanced_imag], dim=1)
    spec_c = torch.view_as_complex(
        enhanced_spec.permute(0, 2, 3, 1).contiguous()
    )  # (1, 481, 1)

    # 单帧无重叠 → 直接用 irfft，不用 istft（避免 OLA 归一化在窗边缘除零）
    time_signal = torch.fft.irfft(
        spec_c.squeeze(0).squeeze(-1), n=N_FFT
    )  # (960,)

    return time_signal[:HOP_LEN].tolist()  # 取前 480 样本


# ========== 3. 重置（切换设备/模式时调用） ==========
def reset():
    global mic_buffer, far_buffer, caches
    for i, s in enumerate(CACHE_SIZES):
        caches[i] = np.zeros((1, s), dtype=np.float32)
    mic_buffer.zero_()
    far_buffer.zero_()
```

## 关键注意事项

1. **实部/虚部分开**：ONNX 模型输入输出是 `(B, 2, 1, 481)` — dim=1 是 R/I 两个独立 float32 通道，**不是** complex64 类型。必须 `stack([real, imag], axis=0)` 手动拆分。

2. **STFT 参数**：`hop_length=WIN_LEN`（即 960），非典型配置。目的是每次 STFT 只产生 **1 帧**，与模型 streaming 接口匹配。

3. **单帧 iFFT**：用 `torch.fft.irfft` 而非 `torch.istft`。因为单帧无重叠，istft 的 OLA 归一化会在 sqrt-Hann 窗边缘（值近似 0）除零报错。

4. **Cache 连续**：7 个 cache 必须在帧间保持连续传递——每个 `sess.run()` 输出后 7 个张量就是下一帧的输入。

5. **滑动窗口**：`mic_buffer` / `far_buffer` 各保持 960 样本，每帧丢弃最旧 480、追加新 480。STFT 对 960 样本做单帧分析。
