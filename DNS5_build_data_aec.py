#!/usr/bin/env python3
"""
AEC数据集生成 - 严格遵循EchoFree论文 Section IV-A
信号链:
  近端干净语音 → 卷积RIR → near(近端含混响, 即学习目标)
  远端干净语音 → 非线性失真 → 卷积RIR → 延迟(10~512ms) → echo(回声)
  mic = near + echo (按SER混合)
  far = 远端干净语音 (参考信号, 告诉模型"回声从哪来")
输出: far(远端参考), mic(麦克风混合), near(近端目标)  采样率 48kHz

纯CPU: scipy fftconvolve + SoundFile seek读片段 + 文件列表缓存
"""

import random
import numpy as np
import soundfile as sf
from pathlib import Path
import os, pickle, json
import scipy.signal as signal
from concurrent.futures import ThreadPoolExecutor
from threading import Lock

# ================== 配置 ==================
TRAIN_COUNT = 100000
TEST_COUNT = 100
SER_BINS = np.linspace(-8, 15, 24)  # -8~+15dB, ~1dB 步长
NOISE_SNR_BINS = np.linspace(0, 30, 15)
NOISE_RATIO = 0.0
PROB_FAR_SINGLE = 0.1    # 10% 远端单讲 (near=0)
PROB_NEAR_ONLY = 0.10    # 10% 近端单讲 (echo=0)
RIR_T60_RANGE = (0.05, 0.2)   # 100% 小房间 T60 (桌面近场)
REVERB_WET_RATIO = (0.0, 0.25)  # 混响湿声比例: 桌面近场, 直达声为主
NL_GAIN = (0.3, 3.0)     # 扬声器增益变化: 有时大音量, 有时小音量
CLIP_RANGE = (0.3, 0.9)  # tanh 削波范围: 大音量时削波更严重
RMS_TARGET = (0.9, 0.99) # 近端/远端 RMS 归一化目标
DELAY_MIN = 20            # 最小延迟 ms
DELAY_MAX = 200           # 最大延迟 ms (桌面音响→麦)
SR = 48000
LEN_SEC = 10
LEN_SAMPLES = SR * LEN_SEC

CLEAN_DIR = Path("_DNS5/clean_fullband")
NOISE_DIR = Path("_DNS5/noisy_fullband")
OUTPUT_DIR = Path("_DNS5_AEC")
FILE_CACHE = Path("_DNS5_AEC/_wav_list_cache.pkl")


# ================== 工具函数 ==================
def _sample_rir_t60(base_t60=None, jitter=0.02):
    """RIR的T60采样: 桌面近场小房间 [0.05, 0.2]s.
    
    如果传入 base_t60, 则在 base_t60 ± jitter 范围内采样 (用于共享房间特征).
    """
    if base_t60 is not None:
        lo, hi = RIR_T60_RANGE
        return np.clip(base_t60 + random.uniform(-jitter, jitter), lo, hi)
    return random.uniform(*RIR_T60_RANGE)


def read_wav_segment(path, n_samples=LEN_SAMPLES):
    """SoundFile seek 直接读随机10s片段"""
    info = sf.info(path)
    sr = info.samplerate
    total = info.frames
    need = int(n_samples * sr / SR) if sr != SR else n_samples

    if total > need:
        start = random.randint(0, total - need)
    else:
        start = 0
        need = total

    with sf.SoundFile(path, 'r') as f:
        f.seek(start)
        data = f.read(need, dtype='float32')

    if data.ndim > 1:
        data = data[:, 0]
    if sr != SR:
        ratio = SR / sr
        new_len = int(len(data) * ratio)
        data = np.interp(np.linspace(0, len(data), new_len),
                         np.arange(len(data)), data)

    if len(data) >= n_samples:
        return data[:n_samples]
    return np.tile(data, int(np.ceil(n_samples / len(data))))[:n_samples]


def make_rir(sr, t60):
    rir_len = int(t60 * sr)
    t = np.linspace(0, t60, rir_len)
    decay = np.exp(-6.91 * t / t60)
    noise = np.random.randn(rir_len) * decay
    b, a = signal.butter(1, 8000, fs=sr, btype='low')
    rir = signal.lfilter(b, a, noise)
    delay = random.randint(0, int(0.02 * sr))
    full = np.zeros(rir_len + delay, dtype=np.float32)
    full[delay] = 1.0
    full[delay + 1:delay + 1 + len(rir)] += rir[:len(full) - (delay + 1)]
    return full


def conv(audio, rir):
    return signal.fftconvolve(audio, rir, mode='full')[:len(audio)]


# ================== 文件列表缓存 ==================
def _load_file_cache():
    if FILE_CACHE.exists():
        try:
            return pickle.load(open(FILE_CACHE, 'rb'))
        except Exception:
            pass
    return {}


def _save_file_cache(cache):
    pickle.dump(cache, open(FILE_CACHE, 'wb'))


def get_wav_files(dur_min=3.0):
    fc = _load_file_cache()
    if 'clean' in fc and len(fc['clean']) > 0:
        print(f"  缓存命中: {len(fc['clean'])} 个clean文件")
        return fc['clean']

    print(f"  扫描clean目录 (首次)...")
    wavs = []
    for root, _, files in os.walk(CLEAN_DIR):
        for f in files:
            if not f.endswith('.wav'):
                continue
            p = os.path.join(root, f)
            try:
                if sf.info(p).duration >= dur_min:
                    wavs.append(p)
            except:
                pass

    fc['clean'] = wavs
    _save_file_cache(fc)
    print(f"  缓存已写入: {len(wavs)} 个clean文件")
    return wavs


def scan_noise():
    fc = _load_file_cache()
    if 'noise' in fc:
        print(f"  缓存命中: {len(fc['noise'])} 个noise文件")
        return fc['noise']

    print(f"  扫描noise目录...")
    noise_files = [str(f) for f in sorted(NOISE_DIR.rglob("*.wav"))]

    fc['noise'] = noise_files
    _save_file_cache(fc)
    print(f"  缓存已写入: {len(noise_files)} 个noise文件")
    return noise_files


# ================== 生成单样本 ==================
def _rms_normalize(signal_data, target_range=RMS_TARGET):
    """将信号 RMS 归一化到 target_range 范围内."""
    rms = np.sqrt(np.mean(signal_data ** 2)) + 1e-9
    target_rms = random.uniform(*target_range)
    return signal_data * (target_rms / rms)


def gen_sample(wav_paths, noise_paths):
    i_near, i_far = random.sample(range(len(wav_paths)), 2)

    near_clean = read_wav_segment(wav_paths[i_near])
    far_clean = read_wav_segment(wav_paths[i_far])

    # RMS 归一化: 每个人的语音统一到目标 RMS (0.9~0.99)
    near_clean = _rms_normalize(near_clean)
    far_clean = _rms_normalize(far_clean)

    # 共享房间特征: 基础 T60 + 小幅抖动
    t60_base = random.uniform(*RIR_T60_RANGE)

    # near = 近端 × RIR (独立 wet_ratio)
    rir_near = make_rir(SR, _sample_rir_t60(t60_base))
    wet_near = conv(near_clean, rir_near)
    wet_ratio_near = random.uniform(*REVERB_WET_RATIO)
    near = (1 - wet_ratio_near) * near_clean + wet_ratio_near * wet_near

    # echo = 远端 → 非线性失真 → RIR → 延迟
    echo = far_clean.copy()
    gain = random.uniform(*NL_GAIN)
    clip = random.uniform(*CLIP_RANGE)
    echo = echo * gain
    echo = np.tanh(echo / clip) * clip
    if random.random() < 0.2:
        d = random.uniform(0.01, 0.04)
        echo = echo + d * (echo ** 2) * np.sign(echo)
    echo_distorted = echo.copy()
    rir_echo = make_rir(SR, _sample_rir_t60(t60_base))
    wet_echo = conv(echo_distorted, rir_echo)
    wet_ratio_echo = random.uniform(*REVERB_WET_RATIO)  # 独立采样
    echo = (1 - wet_ratio_echo) * echo_distorted + wet_ratio_echo * wet_echo
    delay_ms = random.randint(DELAY_MIN, DELAY_MAX)
    delay_n = int(delay_ms / 1000 * SR)
    tap_index = delay_ms // 10
    if 0 < delay_n < LEN_SAMPLES:
        echo = np.concatenate([np.zeros(delay_n, dtype=np.float32),
                               echo[:LEN_SAMPLES - delay_n]])

    # SER控制回声量
    ser = random.choice(SER_BINS)
    s_pow = np.sum(near ** 2) + 1e-9
    e_pow = np.sum(echo ** 2) + 1e-9
    echo *= np.sqrt(s_pow / e_pow) * (10.0 ** (-ser / 20.0))

    # 三种说话人场景控制: 80% 双讲 / 10% 远端单讲 / 10% 近端单讲
    r = random.random()
    if r < PROB_FAR_SINGLE:
        near = np.zeros(LEN_SAMPLES, dtype=np.float32)
    elif r < PROB_FAR_SINGLE + PROB_NEAR_ONLY:
        echo = np.zeros(LEN_SAMPLES, dtype=np.float32)

    # mic = near + echo
    mic = near + echo

    # 30% 叠加噪声
    if random.random() < NOISE_RATIO and noise_paths:
        noise = read_wav_segment(random.choice(noise_paths))
        snr = random.choice(NOISE_SNR_BINS)
        arms = np.sqrt(np.mean(mic ** 2)) + 1e-9
        nrms = np.sqrt(np.mean(noise ** 2)) + 1e-9
        scale = arms / (nrms * (10.0 ** (snr / 20.0)))
        mic = mic + noise * scale

    # 限幅
    peak = np.max(np.abs(mic))
    if peak > 0.999:
        sc = 0.999 / peak
        mic *= sc
        near *= sc

    info = {'delay_ms': delay_ms, 'delay_samples': delay_n, 'tap_index': tap_index}
    return (np.clip(far_clean, -1, 1),
            np.clip(mic, -1, 1),
            np.clip(near, -1, 1),
            info)


# ================== 写盘 ==================
def write_one(path, data, sr):
    sf.write(path, data, sr)


# ================== 主函数 ==================
def main():
    import time
    print(f"Clean dir: {CLEAN_DIR}")
    print(f"Output dir: {OUTPUT_DIR}")
    print(f"SR={SR}, len={LEN_SEC}s (纯CPU)")

    t0 = time.perf_counter()
    wav_paths = get_wav_files()
    print(f"找到 {len(wav_paths)} 个wav文件 ({time.perf_counter()-t0:.1f}s)")
    if len(wav_paths) < 2:
        print("Error: 至少需要2个wav文件！")
        return

    t0 = time.perf_counter()
    noise_paths = scan_noise()
    print(f"找到 {len(noise_paths)} 个噪声文件 ({time.perf_counter()-t0:.1f}s)")

    with ThreadPoolExecutor(max_workers=4) as pool:
        for subset, count, prefix in [("train", TRAIN_COUNT, "train"),
                                       ("test", TEST_COUNT, "test")]:
            out = OUTPUT_DIR / subset
            for d in ["far", "mic", "near"]:
                (out / d).mkdir(parents=True, exist_ok=True)

            print(f"\n生成 {subset} ({count} 样本, delay=10~500ms 整数)...")
            t_start = time.perf_counter()

            # 收集延迟信息
            delays = {}
            lock = Lock()
            completed = [0]  # 实际完成计数 (list 以便闭包修改)

            def write_sample(i):
                far, mic, near, info = gen_sample(wav_paths, noise_paths)
                key = f"{subset}/mic/{prefix}_{i:06d}_mic.wav"
                sf.write(str(out / "far" / f"{prefix}_{i:06d}_far.wav"), far, SR)
                sf.write(str(out / "mic" / f"{prefix}_{i:06d}_mic.wav"), mic, SR)
                sf.write(str(out / "near" / f"{prefix}_{i:06d}_near.wav"), near, SR)
                with lock:
                    delays[key] = info
                    completed[0] += 1
                    n = completed[0]
                if n % 1000 == 0:
                    elapsed = time.perf_counter() - t_start
                    speed = n / elapsed
                    eta = (count - n) / speed
                    print(f"  {n}/{count}  {speed:.0f} samples/s  ETA: {eta:.0f}s")

            futures = []
            for i in range(count):
                futures.append(pool.submit(write_sample, i))

            # 等待全部完成
            for f in futures:
                f.result()

            elapsed = time.perf_counter() - t_start
            print(f"  {subset} 完成: {elapsed:.1f}s ({count/elapsed:.0f} samples/s)")

            # 写出子集 delays.json
            delays_path = str(OUTPUT_DIR / subset / "delays.json")
            with open(delays_path, 'w') as f:
                json.dump(delays, f, indent=2, ensure_ascii=False)
            print(f"  延迟信息: {delays_path} ({len(delays)} entries)")

    # ── 合并全部延迟信息到根目录 delays.json ──
    print("\n合并延迟信息 → delays.json ...")
    all_delays = {}
    for subset in ["train", "test"]:
        p = str(OUTPUT_DIR / subset / "delays.json")
        if os.path.exists(p):
            with open(p, 'r') as f:
                sub = json.load(f)
            all_delays.update(sub)
            print(f"  + {subset}: {len(sub)} entries")
    if not all_delays:
        print("  (no delay files found)")
        return

    root_delays_path = str(OUTPUT_DIR / "delays.json")
    with open(root_delays_path, 'w') as f:
        json.dump(all_delays, f, indent=2, ensure_ascii=False)
    print(f"  → {root_delays_path} ({len(all_delays)} total)")

    # 统计延迟分布
    taps = [v['tap_index'] for v in all_delays.values()]
    print(f"  Tap distribution: min={min(taps)} max={max(taps)} mean={np.mean(taps):.1f}")

    print("\nDone!")


if __name__ == "__main__":
    main()
