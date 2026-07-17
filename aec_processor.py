"""
AEC (Acoustic Echo Cancellation) processor module.

This module provides AEC functionality including:
- AEC processor wrapper for C++ aimic module
- Speaker loopback capture (WASAPI) for far-end audio
- Thread-safe ring buffer for audio data
"""

import ctypes
from ctypes import wintypes, POINTER, byref, cast, c_void_p
import os
import struct
import threading
from typing import List, Optional

try:
    import aec_inference
    CPP_AVAILABLE = True
except ImportError:
    CPP_AVAILABLE = False
    print("[AEC] C++ module aec_inference not available")

SAMPLE_RATE = 48000
HOP_LENGTH = 480


class AecProcessor:
    """Streaming AEC — delegates to C++ aec_inference.AecProcessor."""

    HOP_LEN = 480

    def __init__(self, onnx_path: str) -> None:
        if not os.path.exists(onnx_path):
            raise FileNotFoundError(f"AEC model not found: {onnx_path}")
        self._cpp = aec_inference.AecProcessor(onnx_path)

    def reset(self) -> None:
        self._cpp.reset()

    def process(self, mic_chunk: list, far_chunk: list) -> list:
        """Process 480-sample mic + far chunks through AEC."""
        return self._cpp.process_frame(mic_chunk, far_chunk)


class RingBuffer:
    """Thread-safe ring buffer that automatically discards old data when full."""

    def __init__(self, capacity_samples: int) -> None:
        self._capacity: int = capacity_samples
        self._buffer: List[float] = [0.0] * capacity_samples
        self._write_pos: int = 0
        self._read_pos: int = 0
        self._count: int = 0
        self._lock: threading.Lock = threading.Lock()

    def write(self, data: List[float]) -> None:
        with self._lock:
            data_len = len(data)
            if data_len >= self._capacity:
                start = data_len - self._capacity
                self._buffer[:] = data[start:]
                self._write_pos = 0
                self._read_pos = 0
                self._count = self._capacity
                return

            discard = max(0, self._count + data_len - self._capacity)
            if discard > 0:
                self._read_pos = (self._read_pos + discard) % self._capacity
                self._count -= discard

            first_part = min(data_len, self._capacity - self._write_pos)
            self._buffer[self._write_pos:self._write_pos + first_part] = data[:first_part]

            if first_part < data_len:
                self._buffer[:data_len - first_part] = data[first_part:]

            self._write_pos = (self._write_pos + data_len) % self._capacity
            self._count = min(self._count + data_len, self._capacity)

    def read(self, n_samples: int) -> Optional[List[float]]:
        with self._lock:
            if self._count < n_samples:
                return None

            first_part = min(n_samples, self._capacity - self._read_pos)
            result = self._buffer[self._read_pos:self._read_pos + first_part]

            if first_part < n_samples:
                result = result + self._buffer[:n_samples - first_part]

            self._read_pos = (self._read_pos + n_samples) % self._capacity
            self._count -= n_samples
            return result

    def available(self) -> int:
        with self._lock:
            return self._count


class SpeakerCapture:
    """WASAPI loopback capture for far-end audio (AEC reference)."""

    _CLSID_MMDeviceEnumerator = "{BCDE0395-E52F-467C-8E3D-C4579291692E}"
    _IID_IMMDeviceEnumerator = "{A95664D2-9614-4F35-A746-DE8DB63617E6}"
    _IID_IMMDevice = "{D666063F-1587-4E43-81F1-B948E807363F}"
    _IID_IAudioClient = "{1CB9AD4C-DBFA-4C32-B178-C2F568A703B2}"
    _IID_IAudioCaptureClient = "{C8ADBD64-E71E-48A0-A4DE-185C395CD317}"

    AUDCLNT_SHAREMODE_SHARED = 0
    AUDCLNT_STREAMFLAGS_LOOPBACK = 0x00020000
    CLSCTX_ALL = 0x17
    COINIT_MULTITHREADED = 0x0
    DEVICE_CHECK_INTERVAL = 2.0

    def __init__(self, rate: int = 48000, channels: int = 1):
        self._rate = rate
        self._channels = channels
        self._buffer = RingBuffer(rate * 2)
        self._active = False
        self._lock = threading.Lock()
        self._capture_thread: Optional[threading.Thread] = None
        self._device_check_thread: Optional[threading.Thread] = None
        self._audio_client: Optional[ctypes.c_void_p] = None
        self._capture_client: Optional[ctypes.c_void_p] = None
        self._dev_ch: int = 2
        self._current_device_name: Optional[str] = None

    @property
    def active(self) -> bool:
        return self._active

    def start(self) -> bool:
        try:
            ole32 = ctypes.windll.ole32

            def _make_guid(s):
                buf = (ctypes.c_ubyte * 16)()
                ole32.CLSIDFromString(s, buf)
                return buf

            CLSID_MMDE = _make_guid(self._CLSID_MMDeviceEnumerator)
            IID_IMMDE = _make_guid(self._IID_IMMDeviceEnumerator)
            IID_IAC = _make_guid(self._IID_IAudioClient)
            IID_IACC = _make_guid(self._IID_IAudioCaptureClient)

            ole32.CoInitializeEx(None, self.COINIT_MULTITHREADED)

            pEnum = c_void_p()
            hr = ole32.CoCreateInstance(
                byref(CLSID_MMDE), None, self.CLSCTX_ALL,
                byref(IID_IMMDE), byref(pEnum))
            if hr < 0 or not pEnum:
                return False

            vtbl_enum = cast(pEnum, POINTER(POINTER(c_void_p))).contents
            fn_GetDefault = cast(vtbl_enum[4], ctypes.WINFUNCTYPE(
                ctypes.c_long, c_void_p, wintypes.DWORD, wintypes.DWORD, POINTER(c_void_p)))
            pDevice = c_void_p()
            hr = fn_GetDefault(pEnum, 0, 0, byref(pDevice))
            if hr < 0 or not pDevice:
                return False

            vtbl_dev = cast(pDevice, POINTER(POINTER(c_void_p))).contents
            fn_Activate = cast(vtbl_dev[3], ctypes.WINFUNCTYPE(
                ctypes.c_long, c_void_p, ctypes.c_void_p, wintypes.DWORD,
                c_void_p, POINTER(c_void_p)))
            pAudioClient = c_void_p()
            hr = fn_Activate(pDevice, byref(IID_IAC), self.CLSCTX_ALL,
                             None, byref(pAudioClient))
            if hr < 0 or not pAudioClient:
                return False

            vtbl_ac = cast(pAudioClient, POINTER(POINTER(c_void_p))).contents
            fn_GetMixFormat = cast(vtbl_ac[8], ctypes.WINFUNCTYPE(
                ctypes.c_long, c_void_p, POINTER(ctypes.c_void_p)))

            class WAVEFORMATEX(ctypes.Structure):
                _fields_ = [
                    ("wFormatTag", wintypes.WORD),
                    ("nChannels", wintypes.WORD),
                    ("nSamplesPerSec", wintypes.DWORD),
                    ("nAvgBytesPerSec", wintypes.DWORD),
                    ("nBlockAlign", wintypes.WORD),
                    ("wBitsPerSample", wintypes.WORD),
                    ("cbSize", wintypes.WORD),
                ]

            pWfx = ctypes.c_void_p()
            hr = fn_GetMixFormat(pAudioClient, byref(pWfx))
            if hr < 0 or not pWfx:
                return False
            wfx = cast(pWfx, POINTER(WAVEFORMATEX)).contents
            self._dev_ch = int(wfx.nChannels)
            ole32.CoTaskMemFree(pWfx)

            REFERENCE_TIME = 10000000
            fn_Initialize = cast(vtbl_ac[3], ctypes.WINFUNCTYPE(
                ctypes.c_long, c_void_p, wintypes.DWORD, wintypes.DWORD,
                ctypes.c_longlong, ctypes.c_longlong, ctypes.c_void_p, ctypes.c_void_p))
            hr = fn_Initialize(
                pAudioClient,
                self.AUDCLNT_SHAREMODE_SHARED,
                self.AUDCLNT_STREAMFLAGS_LOOPBACK,
                REFERENCE_TIME, 0,
                pWfx, None)
            if hr < 0:
                return False

            fn_GetService = cast(vtbl_ac[14], ctypes.WINFUNCTYPE(
                ctypes.c_long, c_void_p, ctypes.c_void_p, POINTER(c_void_p)))
            pCaptureClient = c_void_p()
            hr = fn_GetService(pAudioClient, byref(IID_IACC), byref(pCaptureClient))
            if hr < 0 or not pCaptureClient:
                return False

            fn_Start = cast(vtbl_ac[10], ctypes.WINFUNCTYPE(ctypes.c_long, c_void_p))
            hr = fn_Start(pAudioClient)
            if hr < 0:
                return False

            self._audio_client = pAudioClient
            self._capture_client = pCaptureClient

            cast(vtbl_dev[2], ctypes.WINFUNCTYPE(ctypes.c_ulong, c_void_p))(pDevice)
            cast(vtbl_enum[2], ctypes.WINFUNCTYPE(ctypes.c_ulong, c_void_p))(pEnum)

            self._active = True
            self._capture_thread = threading.Thread(target=self._capture_loop, daemon=True)
            self._capture_thread.start()
            self._device_check_thread = threading.Thread(target=self._device_check_loop, daemon=True)
            self._device_check_thread.start()
            return True

        except Exception:
            self._active = False
            return False

    def _capture_loop(self) -> None:
        vtbl_cc = cast(self._capture_client, POINTER(POINTER(c_void_p))).contents
        fn_GetBuffer = cast(vtbl_cc[3], ctypes.WINFUNCTYPE(
            ctypes.c_long, c_void_p,
            POINTER(ctypes.POINTER(ctypes.c_ubyte)),
            POINTER(wintypes.DWORD),
            POINTER(wintypes.DWORD),
            POINTER(ctypes.c_uint64),
            POINTER(ctypes.c_uint64)))
        fn_ReleaseBuffer = cast(vtbl_cc[4], ctypes.WINFUNCTYPE(
            ctypes.c_long, c_void_p, wintypes.DWORD))

        while self._active:
            try:
                pData = ctypes.POINTER(ctypes.c_ubyte)()
                numFrames = wintypes.DWORD()
                flags = wintypes.DWORD()
                devPos = ctypes.c_uint64()
                qpcPos = ctypes.c_uint64()
                hr = fn_GetBuffer(self._capture_client, byref(pData),
                                  byref(numFrames), byref(flags),
                                  byref(devPos), byref(qpcPos))
                if hr < 0 or numFrames.value == 0:
                    import time
                    time.sleep(0.001)
                    continue

                frame_count = numFrames.value
                ch = self._dev_ch
                fmt = f'{frame_count * ch}f'
                raw = list(struct.unpack(fmt, bytes(pData[:frame_count * ch * 4])))
                fn_ReleaseBuffer(self._capture_client, numFrames)

                if ch > 1:
                    mono = [0.0] * frame_count
                    for i in range(frame_count):
                        s = 0.0
                        for c in range(ch):
                            s += raw[i * ch + c]
                        mono[i] = s / ch
                else:
                    mono = raw
                self._buffer.write(mono)

            except Exception:
                import time
                time.sleep(0.005)

    def stop(self) -> None:
        self._active = False
        if self._device_check_thread:
            self._device_check_thread.join(timeout=1.0)
        if self._capture_thread:
            self._capture_thread.join(timeout=1.0)
        if self._audio_client:
            try:
                vtbl_ac = ctypes.cast(self._audio_client, ctypes.POINTER(ctypes.POINTER(ctypes.c_void_p))).contents
                fn_Stop = ctypes.cast(vtbl_ac[11], ctypes.WINFUNCTYPE(ctypes.c_long, ctypes.c_void_p))
                fn_Stop(self._audio_client)
                fn_Rel = ctypes.cast(vtbl_ac[2], ctypes.WINFUNCTYPE(ctypes.c_ulong, ctypes.c_void_p))
                fn_Rel(self._audio_client)
            except Exception:
                pass
            self._audio_client = None
        self._capture_client = None

    def read(self, n_samples: int) -> Optional[list]:
        return self._buffer.read(n_samples)

    def flush(self) -> None:
        self._buffer = RingBuffer(self._rate * 2)
