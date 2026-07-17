# Lightweight AEC 48K

[中文](README.md) | English

Lightweight Acoustic Echo Cancellation (AEC) inference module, supporting 48kHz real-time streaming processing.

## Project Structure

```
lightweight-aec-48k/
├── aec7_ep0185.onnx        # AEC ONNX model file
├── aec7.py                 # AEC model training code
├── aec_inference.cpp        # C++ inference module (pybind11)
├── aec_processor.py         # Python AEC processor wrapper
├── DNS5_build_data_aec.py   # DNS5 dataset build script
├── AEC_ONNX_USAGE.md        # ONNX inference detailed documentation
├── checkpoint_epoch_185.tar # Training checkpoint
└── README.md
```

## Core Parameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| Sample Rate | 48000 Hz | |
| Frame Size (hop) | 480 samples | 10ms per frame |
| Window Size (win) | 960 samples | Analysis window 20ms |
| FFT Size | 960 | → 481 frequency bins |
| Window Function | sqrt-Hann | `hann(960).pow(0.5)` |
| Input Channel | Mono | mic and far are both mono float32 |

## Quick Start

### Python Usage

```python
from aec_processor import AecProcessor

# Initialize AEC processor
aec = AecProcessor("aec7_ep0185.onnx")

# Process frame by frame (480 samples = 10ms)
output = aec.process(mic_chunk, far_chunk)

# Reset state
aec.reset()
```

### C++ Inference (via pybind11)

```python
import aec_inference

# Initialize
aec = aec_inference.AecProcessor("aec7_ep0185.onnx")

# Process one frame
output = aec.process_frame(mic_list, far_list)

# Reset
aec.reset()
```

## Complete Processing Pipeline

```python
from aec_processor import AecProcessor, SpeakerCapture

# 1. Initialize
aec = AecProcessor("aec7_ep0185.onnx")
speaker = SpeakerCapture()

# 2. Start speaker capture (AEC far-end reference)
speaker.start()

# 3. Real-time processing loop
while running:
    # Get microphone input (480 samples)
    mic_data = get_mic_input()
    
    # Get far-end reference (480 samples)
    far_data = speaker.read(480)
    
    # AEC processing
    if far_data:
        output = aec.process(mic_data, far_data)
    else:
        output = mic_data  # Passthrough when no far-end
    
    # Output processed audio
    play_output(output)

# 4. Cleanup
speaker.stop()
aec.reset()
```

## Build C++ Module

### Dependencies

- pybind11
- onnxruntime (1.24.4+)
- pffft

### Build Commands (Windows)

```bash
# Using setuptools
pip install pybind11
python setup.py build_ext --inplace

# Or directly with MSVC
cl /EHsc /O2 /MD aec_inference.cpp /I<pybind11_include> /I<onnxruntime_include> /I<pffft_include> /link /OUT:aec_inference.pyd
```

## ONNX Model Interface

### Inputs (14)

| Name | Shape | Description |
|------|-------|-------------|
| `mic_frame` | `(1, 2, 481)` | Microphone STFT [R/I] |
| `far_frame` | `(1, 2, 481)` | Far-end STFT [R/I] |
| `res_enc_conv` | `(1, 135680)` | Reference encoder conv cache |
| `res_enc_tfa` | `(1, 248)` | Reference encoder TFA cache |
| `mic_enc_conv` | `(1, 135680)` | Microphone encoder conv cache |
| `mic_enc_tfa` | `(1, 248)` | Microphone encoder TFA cache |
| `deep_enc_tfa` | `(1, 336)` | Deep encoder TFA cache |
| `dec_conv` | `(1, 13440)` | Decoder conv cache |
| `dec_tfa` | `(1, 496)` | Decoder TFA cache |
| `inter` | `(1, 7680)` | DPGRNN intermediate cache |
| `res_prev1/2` | `(1, 1, 1, 320)` | Reference derivative cache |
| `mic_prev1/2` | `(1, 1, 1, 320)` | Microphone derivative cache |

### Outputs (14)

| Name | Shape | Description |
|------|-------|-------------|
| `enhanced_frame` | `(1, 2, 481)` | Enhanced STFT |
| Remaining 13 | Same as input | Updated caches |

## Documentation

See [AEC_ONNX_USAGE.md](AEC_ONNX_USAGE.md)

## License

MIT
