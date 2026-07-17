# Lightweight AEC 48K

轻量级声学回声消除（AEC）推理模块，支持 48kHz 实时流式处理。

## 项目结构

```
lightweight-aec-48k/
├── aec7_ep0185.onnx        # AEC ONNX 模型文件
├── aec7.py                 # AEC 模型训练代码
├── aec_inference.cpp        # C++ 推理模块（pybind11）
├── aec_processor.py         # Python AEC 处理器封装
├── DNS5_build_data_aec.py   # DNS5 数据集构建脚本
├── AEC_ONNX_USAGE.md        # ONNX 推理详细文档
└── README.md
```

## 核心参数

| 参数 | 值 | 说明 |
|------|-----|------|
| 采样率 | 48000 Hz | |
| 帧长 (hop) | 480 samples | 每帧 10ms |
| 窗长 (win) | 960 samples | 分析窗 20ms |
| FFT 点数 | 960 | → 481 频点 |
| 窗函数 | sqrt-Hann | `hann(960).pow(0.5)` |
| 输入通道 | 单声道 | mic 和 far 均为 mono float32 |

## 快速开始

### Python 使用

```python
from aec_processor import AecProcessor

# 初始化 AEC 处理器
aec = AecProcessor("aec7_ep0185.onnx")

# 逐帧处理（每帧 480 样本 = 10ms）
output = aec.process(mic_chunk, far_chunk)

# 重置状态
aec.reset()
```

### C++ 推理（通过 pybind11）

```python
import aec_inference

# 初始化
aec = aec_inference.AecProcessor("aec7_ep0185.onnx")

# 处理一帧
output = aec.process_frame(mic_list, far_list)

# 重置
aec.reset()
```

## 完整处理流程

```python
from aec_processor import AecProcessor, SpeakerCapture

# 1. 初始化
aec = AecProcessor("aec7_ep0185.onnx")
speaker = SpeakerCapture()

# 2. 启动扬声器采集（AEC 远端参考）
speaker.start()

# 3. 实时处理循环
while running:
    # 获取麦克风输入（480 样本）
    mic_data = get_mic_input()
    
    # 获取远端参考（480 样本）
    far_data = speaker.read(480)
    
    # AEC 处理
    if far_data:
        output = aec.process(mic_data, far_data)
    else:
        output = mic_data  # 无远端时直通
    
    # 输出处理后的音频
    play_output(output)

# 4. 清理
speaker.stop()
aec.reset()
```

## 构建 C++ 模块

### 依赖

- pybind11
- onnxruntime (1.24.4+)
- pffft

### 编译命令（Windows）

```bash
# 使用 setuptools
pip install pybind11
python setup.py build_ext --inplace

# 或直接使用 MSVC
cl /EHsc /O2 /MD aec_inference.cpp /I<pybind11_include> /I<onnxruntime_include> /I<pffft_include> /link /OUT:aec_inference.pyd
```

## ONNX 模型接口

### 输入（14 个）

| 名称 | 形状 | 说明 |
|------|------|------|
| `mic_frame` | `(1, 2, 481)` | 麦克风 STFT [R/I] |
| `far_frame` | `(1, 2, 481)` | 远端 STFT [R/I] |
| `res_enc_conv` | `(1, 135680)` | 参考编码器卷积缓存 |
| `res_enc_tfa` | `(1, 248)` | 参考编码器 TFA 缓存 |
| `mic_enc_conv` | `(1, 135680)` | 麦克风编码器卷积缓存 |
| `mic_enc_tfa` | `(1, 248)` | 麦克风编码器 TFA 缓存 |
| `deep_enc_tfa` | `(1, 336)` | 深度编码器 TFA 缓存 |
| `dec_conv` | `(1, 13440)` | 解码器卷积缓存 |
| `dec_tfa` | `(1, 496)` | 解码器 TFA 缓存 |
| `inter` | `(1, 7680)` | DPGRNN 中间缓存 |
| `res_prev1/2` | `(1, 1, 1, 320)` | 参考导数缓存 |
| `mic_prev1/2` | `(1, 1, 1, 320)` | 麦克风导数缓存 |

### 输出（14 个）

| 名称 | 形状 | 说明 |
|------|------|------|
| `enhanced_frame` | `(1, 2, 481)` | 增强后的 STFT |
| 其余 13 个 | 同输入 | 更新后的缓存 |

## 详细文档

参见 [AEC_ONNX_USAGE.md](AEC_ONNX_USAGE.md)

## License

MIT
