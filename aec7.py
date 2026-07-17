# -*- coding: utf-8 -*-
# ==============================================================================
# AEC7 — 两阶段可训练声学回声消除 (48kHz, 单文件)
#
# 合并自: aec7_common / aec7_dataset / aec7_offline / aec7_stream / aec7_train
# 运行: python aec7.py          → 训练 (自动启动 TensorBoard)
#        python aec7.py --ckpt X → 加载 checkpoint 推理
#        python aec7.py --causal → 因果性检查
# ==============================================================================

import os, re, sys, argparse, subprocess, time, random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch._dynamo
import torch._inductor.config
from torch.utils.data import DataLoader, SubsetRandomSampler, Dataset
from torch.utils.tensorboard import SummaryWriter
import soundfile as sf
from tqdm import tqdm

torch.set_float32_matmul_precision('high')

FILE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJ_ROOT = os.path.dirname(FILE_DIR)
sys.path.insert(0, FILE_DIR)

DATA = os.path.join(PROJ_ROOT, '_DNS5_AEC')
OUT = os.path.join(PROJ_ROOT, 'output', 'output_aec7')
TEST_MIC = os.path.join(PROJ_ROOT, 'epoch_end_test_wav', 'aec', 'test_000001_mic.wav')
TEST_FAR = os.path.join(PROJ_ROOT, 'epoch_end_test_wav', 'aec', 'test_000001_far.wav')

CHUNK = 10.0; SPE = 1000; EPO = 1000; BS = 10; CGN = 30.0
LR0 = 1e-3; LR_MAX = 1e-3; LR_MIN = 1e-5
SEED = 42

# ─── torch.compile persistent cache ───
torch._inductor.config.fx_graph_cache = True
torch._inductor.config.coordinate_descent_tuning = True  # 5-15% faster kernels, slower first compile
torch._dynamo.config.recompile_limit = 32
torch._dynamo.config.suppress_errors = True
_IND_CACHE = os.path.join(OUT, 'inductor_cache')
os.makedirs(_IND_CACHE, exist_ok=True)
os.environ['TORCHINDUCTOR_CACHE_DIR'] = _IND_CACHE

G_ALL = '1_All'; G_TRAIN = '2_Train'; G_TRAIN_Epoch = '2b_TrainEpoch'
G_VAL = '3_Val'; G_TEST = '4_Test'

# ══════════════════════════════════════════════════════════════════════════════
# 常量 (aec7_common)
# ══════════════════════════════════════════════════════════════════════════════

FS = 48000; HOP = 480; WIN = 960; NFFT = 960
FRONT_ENC_C = [16, 24, 36, 48]
DEEP_ENC_C = [72, 96]
DEC_C = [96, 72, 48, 32]
ERB_LOW = 240; ERB_HIGH = 80; ERB_TOTAL = ERB_LOW + ERB_HIGH
DELAY_TAPS_FRAMES = tuple(range(0, 52))
DELAY_K = len(DELAY_TAPS_FRAMES)
DELAY_MAX_FRAMES = max(DELAY_TAPS_FRAMES)
DELAY_BUF_FRAMES = DELAY_K
N_BINS = NFFT // 2 + 1

# ══════════════════════════════════════════════════════════════════════════════
# 构建块 (aec7_common)
# ══════════════════════════════════════════════════════════════════════════════

class RMSNorm(nn.Module):
    def __init__(self, num_channels, eps=1e-8):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(num_channels))

    def forward(self, x):
        norm = x.pow(2).mean(dim=1, keepdim=True).add(self.eps).rsqrt()
        return x * norm * self.weight[None, :, None, None]


class ERB(nn.Module):
    def __init__(self, erb_subband_1=100, erb_subband_2=41, nfft=960, high_lim=24000, fs=48000):
        super().__init__()
        erb_filters = self.erb_filter_banks(erb_subband_1, erb_subband_2, nfft, high_lim, fs)
        nfreqs = nfft // 2 + 1
        self.erb_subband_1 = erb_subband_1
        self.erb_subband_2 = erb_subband_2
        self.erb_fc = nn.Linear(nfreqs - erb_subband_1, erb_subband_2, bias=False)
        self.ierb_fc = nn.Linear(erb_subband_2, nfreqs - erb_subband_1, bias=False)
        self.erb_fc.weight = nn.Parameter(erb_filters, requires_grad=False)
        self.ierb_fc.weight = nn.Parameter(erb_filters.T, requires_grad=False)

    def hz2erb(self, freq_hz):
        return 21.4 * np.log10(0.00437 * freq_hz + 1)

    def erb2hz(self, erb_f):
        return (10 ** (erb_f / 21.4) - 1) / 0.00437

    def erb_filter_banks(self, erb_subband_1, erb_subband_2, nfft=960, high_lim=24000, fs=48000):
        low_lim = erb_subband_1 / nfft * fs
        erb_low = self.hz2erb(low_lim)
        erb_high = self.hz2erb(high_lim)
        erb_points = np.linspace(erb_low, erb_high, erb_subband_2)
        bins = np.round(self.erb2hz(erb_points) / fs * nfft).astype(np.int32)
        erb_filters = np.zeros([erb_subband_2, nfft // 2 + 1], dtype=np.float32)
        erb_filters[0, bins[0]:bins[1]] = (bins[1] - np.arange(bins[0], bins[1]) + 1e-12) \
                                            / (bins[1] - bins[0] + 1e-12)
        for i in range(erb_subband_2 - 2):
            erb_filters[i + 1, bins[i]:bins[i+1]] = (np.arange(bins[i], bins[i+1]) - bins[i] + 1e-12) \
                                                      / (bins[i+1] - bins[i] + 1e-12)
            erb_filters[i + 1, bins[i+1]:bins[i+2]] = (bins[i+2] - np.arange(bins[i+1], bins[i+2]) + 1e-12) \
                                                      / (bins[i+2] - bins[i+1] + 1e-12)
        erb_filters[-1, bins[-2]:bins[-1]+1] = 1 - erb_filters[-2, bins[-2]:bins[-1]+1]
        erb_filters = erb_filters[:, erb_subband_1:]
        return torch.from_numpy(np.abs(erb_filters))

    def bm(self, x):
        x = x.permute(0, 1, 3, 2)
        x_low = x[..., :self.erb_subband_1]
        x_high = self.erb_fc(x[..., self.erb_subband_1:])
        return torch.cat([x_low, x_high], dim=-1).permute(0, 1, 3, 2)

    def bs(self, x_erb):
        x_erb_low = x_erb[..., :self.erb_subband_1]
        x_erb_high = self.ierb_fc(x_erb[..., self.erb_subband_1:])
        return torch.cat([x_erb_low, x_erb_high], dim=-1)


def compute_derivatives(x):
    d1 = torch.zeros_like(x)
    d1[:, :, 1:, :] = x[:, :, 1:, :] - x[:, :, :-1, :]
    d2 = torch.zeros_like(d1)
    d2[:, :, 1:, :] = d1[:, :, 1:, :] - d1[:, :, :-1, :]
    return torch.cat([x, d1, d2], dim=1)


class FA(nn.Module):
    def __init__(self, nfreq, freq_comp_ratio=5):
        super().__init__()
        self.nfreq = nfreq
        self.r = freq_comp_ratio
        self.H = (nfreq + self.r - 1) // self.r
        self.padded_nfreq = self.H * self.r
        self.need_pad = nfreq % freq_comp_ratio != 0
        self.gru = nn.GRU(self.r, self.r, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(2 * self.r, self.r)

    def forward(self, x):
        B, C, T, f_dim = x.shape
        x = x.pow(2).mean(dim=1)
        if self.need_pad:
            x = F.pad(x, (0, self.padded_nfreq - self.nfreq))
        x = x.reshape(B * T, self.H, self.r)
        x, _ = self.gru(x)
        x = self.fc(x)
        if self.need_pad:
            x = x.reshape(B, T, self.padded_nfreq)[..., :self.nfreq]
        else:
            x = x.reshape(B, T, self.nfreq)
        return x


class cTFA(nn.Module):
    def __init__(self, channels, width):
        super().__init__()
        self.channels = channels
        self.ta_gru = nn.GRU(channels, channels * 2, 1, batch_first=True)
        self.ta_fc = nn.Linear(channels * 2, channels)
        self.fa = FA(width)

    def forward(self, x):
        zt = x.pow(2).mean(dim=-1).transpose(1, 2)
        at = self.ta_gru(zt)[0]
        at = self.ta_fc(at).transpose(1, 2)
        at = torch.sigmoid(at)
        af = self.fa(x)
        af = torch.sigmoid(af)
        return at[..., None] * x * af[:, None]


class Shuffle(nn.Module):
    def forward(self, x):
        x1, x2 = torch.chunk(x, chunks=2, dim=1)
        x = torch.stack([x1, x2], dim=1)
        x = x.transpose(1, 2).contiguous()
        x = x.reshape(x.shape[0], -1, x.shape[3], x.shape[4])
        return x


class XDWSBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel=(2, 3), stride=1, groups=2,
                 in_width=141, out_width=141, pt_pad2d=0, pf=1, use_deconv=False, is_last=False,
                 output_padding=0):
        super().__init__()
        self.pconv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, groups=groups),
            nn.BatchNorm2d(out_channels),
            nn.SiLU(),
            Shuffle() if groups == 2 else nn.Identity()
        )
        if use_deconv:
            self.dconv = nn.Sequential(
                nn.ZeroPad2d([0, 0, pt_pad2d, 0]),
                nn.ConvTranspose2d(out_channels, out_channels, kernel,
                                   stride=(1, stride), padding=(0, pf),
                                   output_padding=(0, output_padding), groups=out_channels),
                nn.BatchNorm2d(out_channels),
                nn.SiLU() if not is_last else nn.Identity(),
            )
        else:
            self.dconv = nn.Sequential(
                nn.ZeroPad2d([0, 0, pt_pad2d, 0]),
                nn.Conv2d(out_channels, out_channels, kernel,
                          stride=(1, stride), padding=(0, pf),
                          groups=out_channels),
                nn.BatchNorm2d(out_channels),
                nn.SiLU() if not is_last else nn.Identity(),
            )
        self.tfa = cTFA(out_channels, out_width) if not is_last else nn.Identity()

    def forward(self, x):
        h = self.pconv(x)
        h = self.dconv(h)
        h = self.tfa(h)
        return h


class DPGRNN(nn.Module):
    def __init__(self, input_size, width, hidden_size):
        super().__init__()
        self.input_size = input_size
        self.width = width
        self.hidden_size = hidden_size
        self.intra_rnn = nn.GRU(input_size, hidden_size // 2, batch_first=True, bidirectional=True)
        self.intra_fc = nn.Linear(hidden_size, input_size)
        self.intra_ln = nn.LayerNorm((width, input_size), eps=1e-8)
        self.inter_rnn = nn.GRU(input_size, hidden_size, batch_first=True, bidirectional=False)
        self.inter_fc = nn.Linear(hidden_size, input_size)
        self.inter_ln = nn.LayerNorm((width, input_size), eps=1e-8)

    def forward(self, x):
        B, C, T, F = x.shape
        intra_x = x.permute(0, 2, 3, 1).reshape(B * T, F, C)
        intra_x, _ = self.intra_rnn(intra_x)
        intra_x = self.intra_fc(intra_x)
        intra_x = intra_x.reshape(B, T, F, C)
        intra_x = self.intra_ln(intra_x)
        intra_out = x.permute(0, 2, 3, 1) + intra_x
        inter_x = intra_out.permute(0, 2, 1, 3).reshape(B * F, T, C)
        inter_x, _ = self.inter_rnn(inter_x)
        inter_x = self.inter_fc(inter_x)
        inter_x = inter_x.reshape(B, F, T, C).permute(0, 2, 1, 3)
        inter_x = self.inter_ln(inter_x)
        inter_out = intra_out + inter_x
        return inter_out.permute(0, 3, 1, 2)


class FrontEncoder(nn.Module):
    def __init__(self, in_channels=1):
        super().__init__()
        enc0 = {'kernel': (11, 3), 'stride': 2, 'groups': 1,
                'in_width': 320, 'out_width': 160, 'pt_pad2d': 10, 'pf': 1}
        enc1 = {'kernel': (11, 3), 'stride': 2, 'groups': 2,
                'in_width': 160, 'out_width': 80, 'pt_pad2d': 10, 'pf': 1}
        enc2 = {'kernel': (11, 3), 'stride': 2, 'groups': 2,
                'in_width': 80, 'out_width': 40, 'pt_pad2d': 10, 'pf': 1}
        enc3 = {'kernel': (10, 3), 'stride': 1, 'groups': 2,
                'in_width': 40, 'out_width': 40, 'pt_pad2d': 9, 'pf': 1}
        configs = [enc0, enc1, enc2, enc3]
        self.layers = nn.ModuleList()
        for i in range(4):
            ch = in_channels if i == 0 else FRONT_ENC_C[i - 1]
            self.layers.append(XDWSBlock(ch, FRONT_ENC_C[i], **configs[i], use_deconv=False))

    def forward(self, x):
        en_outs = []
        for layer in self.layers:
            x = layer(x)
            en_outs.append(x)
        return x, en_outs


class DeepEncoder(nn.Module):
    def __init__(self, in_channels=48):
        super().__init__()
        enc0 = {'kernel': (1, 5), 'stride': 1, 'groups': 2,
                'in_width': 40, 'out_width': 40, 'pt_pad2d': 0, 'pf': 2}
        enc1 = {'kernel': (1, 5), 'stride': 1, 'groups': 2,
                'in_width': 40, 'out_width': 40, 'pt_pad2d': 0, 'pf': 2}
        configs = [enc0, enc1]
        self.layers = nn.ModuleList()
        for i in range(2):
            ch = in_channels if i == 0 else DEEP_ENC_C[i - 1]
            self.layers.append(XDWSBlock(ch, DEEP_ENC_C[i], **configs[i], use_deconv=False))

    def forward(self, x):
        en_outs = []
        for layer in self.layers:
            x = layer(x)
            en_outs.append(x)
        return x, en_outs


class MultiTapDelayLine(nn.Module):
    def __init__(self, delays_frames=DELAY_TAPS_FRAMES):
        super().__init__()
        self.delays = delays_frames
        self.K = len(delays_frames)
        self.max_delay = max(delays_frames)

    def forward(self, far_feat):
        taps = []
        for d in self.delays:
            if d == 0:
                taps.append(far_feat)
            else:
                tapped = F.pad(far_feat[:, :, :-d, :], (0, 0, d, 0))
                taps.append(tapped)
        return torch.stack(taps, dim=1)

    def forward_stream(self, far_frame, delay_buf):
        D = delay_buf.shape[2]
        new_buf = torch.cat([delay_buf[:, :, 1:, :], far_frame], dim=2)
        taps = []
        for d in self.delays:
            if d == 0:
                taps.append(far_frame)
            else:
                idx = D - d
                taps.append(delay_buf[:, :, idx:idx+1, :])
        return torch.stack(taps, dim=1), new_buf

    def forward_chunk(self, far_feat, delay_buf):
        D, N = delay_buf.shape[2], far_feat.shape[2]
        aug = torch.cat([delay_buf, far_feat], dim=2)
        taps = []
        for d in self.delays:
            if d == 0:
                taps.append(far_feat)
            else:
                start = D - d
                taps.append(aug[:, :, start:start+N, :])
        new_buf = aug[:, :, N:, :]
        return torch.stack(taps, dim=1), new_buf

    @staticmethod
    def init_delay_buffer(batch_size, C, F, max_delay, device=None):
        return torch.zeros(batch_size, C, max_delay + 1, F, device=device)


# ─── STFT / iSTFT ───
def _stft_window(win_len, device='cpu', dtype=torch.float32):
    return torch.hann_window(win_len, device=device, dtype=dtype).pow(0.5)


def compute_stft(wav, win_len=960, hop_len=480, n_fft=960, window=None):
    if wav.dtype == torch.bfloat16:
        wav = wav.float()
    spec = torch.stft(wav, n_fft=n_fft, hop_length=hop_len, win_length=win_len,
                      window=window, center=True, pad_mode='reflect',
                      normalized=False, onesided=True, return_complex=True)
    return torch.stack([spec.real, spec.imag], dim=1)


def compute_istft(spec, length=None, win_len=960, hop_len=480, n_fft=960, window=None):
    if spec.dtype == torch.bfloat16:
        spec = spec.float()
    spec_c = torch.view_as_complex(spec.permute(0, 2, 3, 1).contiguous())
    return torch.istft(spec_c, n_fft=n_fft, hop_length=hop_len, win_length=win_len,
                       window=window, center=True, normalized=False,
                       onesided=True, length=length)


# ══════════════════════════════════════════════════════════════════════════════
# 数据集 (aec7_dataset)
# ══════════════════════════════════════════════════════════════════════════════

class AECDataset(Dataset):
    def __init__(self, root_dir: str, train: bool = True):
        self.root_dir = root_dir
        split = 'train' if train else 'test'
        self.mic_dir = os.path.join(root_dir, split, 'mic')
        self.far_dir = os.path.join(root_dir, split, 'far')
        self.near_dir = os.path.join(root_dir, split, 'near')
        for d in [self.mic_dir, self.far_dir, self.near_dir]:
            if not os.path.exists(d):
                raise FileNotFoundError(f"数据集目录不存在: {d}")
        self.mic_files = sorted(os.listdir(self.mic_dir))
        self.far_files = sorted(os.listdir(self.far_dir))
        self.near_files = sorted(os.listdir(self.near_dir))
        if len(self.mic_files) != len(self.far_files) or len(self.mic_files) != len(self.near_files):
            raise RuntimeError(f"文件数量不一致: mic={len(self.mic_files)}, far={len(self.far_files)}, near={len(self.near_files)}")
        self.n_samples = len(self.mic_files)
        print(f"{'训练' if train else '测试'}集: {self.n_samples} 个样本")

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        mic, _ = sf.read(os.path.join(self.mic_dir, self.mic_files[idx]))
        far, _ = sf.read(os.path.join(self.far_dir, self.far_files[idx]))
        near, _ = sf.read(os.path.join(self.near_dir, self.near_files[idx]))
        mic = torch.tensor(mic, dtype=torch.float32)
        far = torch.tensor(far, dtype=torch.float32)
        near = torch.tensor(near, dtype=torch.float32)
        min_len = min(mic.shape[0], near.shape[0])
        mic = mic[:min_len]
        near = near[:min_len]
        return mic, far, near


class Aec7Dataset:
    def __init__(self, root_dir, train=True, chunk_sec=10.0):
        self.ds = AECDataset(root_dir=root_dir, train=train)
        self.chunk_samples = int(chunk_sec * 48000)
        self.train = train

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        mic, far, near = self.ds[idx]
        mic = self._fix_len(mic, self.chunk_samples)
        far = self._fix_len(far, self.chunk_samples)
        near = self._fix_len(near, self.chunk_samples)
        return mic, far, near

    def _fix_len(self, x, target_len):
        if len(x) < target_len:
            reps = target_len // len(x) + 1
            x = x.repeat(reps)
        return x[:target_len]


def collate_aec7(batch):
    mic = torch.stack([x[0] for x in batch])
    far = torch.stack([x[1] for x in batch])
    near = torch.stack([x[2] for x in batch])
    return mic, far, near


# ══════════════════════════════════════════════════════════════════════════════
# 模型 (aec7_offline)
# ══════════════════════════════════════════════════════════════════════════════

class FusionLayer(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels * 2, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.SiLU(inplace=True)
        )

    def forward(self, res_feat, mic_feat):
        return self.conv(torch.cat([res_feat, mic_feat], dim=1))


class Aec7Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.erb = ERB(ERB_LOW, ERB_HIGH, NFFT)
        self.register_buffer('stft_window', _stft_window(WIN))
        self.leak_factor = 0.0
        self.linear_echo_filter = nn.Parameter(torch.zeros(2, N_BINS))
        self.res_norm = RMSNorm(num_channels=3)
        self.mic_norm = RMSNorm(num_channels=3)
        self.res_encoder = FrontEncoder(in_channels=3)
        self.mic_encoder = FrontEncoder(in_channels=3)
        self.fusion = FusionLayer(FRONT_ENC_C[-1], FRONT_ENC_C[-1])
        self.fusion_norm = RMSNorm(num_channels=FRONT_ENC_C[-1])
        self.deep_encoder = DeepEncoder(in_channels=FRONT_ENC_C[-1])
        self.deep_norm = RMSNorm(num_channels=DEEP_ENC_C[-1])
        self.dpgrnn = nn.Sequential(
            DPGRNN(DEEP_ENC_C[-1], 40, DEEP_ENC_C[-1]),
            DPGRNN(DEEP_ENC_C[-1], 40, DEEP_ENC_C[-1])
        )
        dec0 = {'kernel': (1, 1), 'stride': 1, 'groups': 2,
                'in_width': 40, 'out_width': 40, 'pt_pad2d': 0, 'pf': 0}
        dec1 = {'kernel': (2, 3), 'stride': 2, 'groups': 2,
                'in_width': 40, 'out_width': 80, 'pt_pad2d': 1, 'pf': 1,
                'output_padding': 1}
        dec2 = {'kernel': (2, 3), 'stride': 2, 'groups': 2,
                'in_width': 80, 'out_width': 160, 'pt_pad2d': 1, 'pf': 1,
                'output_padding': 1}
        dec3 = {'kernel': (1, 4), 'stride': 2, 'groups': 1,
                'in_width': 160, 'out_width': 320, 'pt_pad2d': 0, 'pf': 1}
        dec_configs = [dec0, dec1, dec2, dec3]
        dec_out_ch = [96, 72, 48, 32]
        skip_ch    = [96, 48, 36, 24]
        self.decoder = nn.ModuleList()
        self.dec_skip_norms = nn.ModuleList()
        self.dec_skip_proj = nn.ModuleList()
        for i in range(4):
            in_ch = DEEP_ENC_C[-1] if i == 0 else dec_out_ch[i - 1]
            out_ch = dec_out_ch[i]
            self.decoder.append(XDWSBlock(in_ch, out_ch, **dec_configs[i],
                                           use_deconv=True, is_last=False))
            self.dec_skip_norms.append(RMSNorm(num_channels=in_ch))
            self.dec_skip_proj.append(nn.Conv2d(skip_ch[i], in_ch, 1, bias=False)
                                       if skip_ch[i] != in_ch else nn.Identity())
        self.mask_conv = nn.Conv2d(dec_out_ch[-1], 1, 1)

    def _linear_echo_cancel(self, mic_spec, far_spec):
        gain_r = self.linear_echo_filter[0].unsqueeze(0).unsqueeze(-1)
        gain_i = self.linear_echo_filter[1].unsqueeze(0).unsqueeze(-1)
        echo_r = far_spec[:, 0] * gain_r - far_spec[:, 1] * gain_i
        echo_i = far_spec[:, 0] * gain_i + far_spec[:, 1] * gain_r
        echo_spec = torch.stack([echo_r, echo_i], dim=1)
        return mic_spec - echo_spec

    def _to_erb_feat(self, spec):
        real = spec.permute(0, 1, 3, 2)
        mag = torch.log10(torch.norm(real, dim=1, keepdim=True).clamp(1e-12))
        erb = self.erb.bm(mag.permute(0, 1, 3, 2)).permute(0, 1, 3, 2)
        return compute_derivatives(erb)

    def forward(self, mic, far, return_mask=False):
        if mic.dim() == 1:
            mic = mic.unsqueeze(0); far = far.unsqueeze(0)
        B, ns = mic.shape
        mic_spec = compute_stft(mic, window=self.stft_window)
        far_spec = compute_stft(far, window=self.stft_window)
        residual_spec = self._linear_echo_cancel(mic_spec, far_spec)
        mic_feat = self._to_erb_feat(mic_spec)
        res_feat = self._to_erb_feat(residual_spec)
        res_feat_normed = self.res_norm(res_feat)
        mic_feat_normed = self.mic_norm(mic_feat)
        res_enc, res_en_outs = self.res_encoder(res_feat_normed)
        mic_enc, mic_en_outs = self.mic_encoder(mic_feat_normed)
        fused = self.fusion(res_enc, mic_enc)
        fused = self.fusion_norm(fused)
        deep_out, deep_en_outs = self.deep_encoder(fused)
        x = self.deep_norm(deep_out)
        for dpgrnn_layer in self.dpgrnn:
            x = dpgrnn_layer(x)
        all_skips = [
            deep_en_outs[1],
            res_en_outs[3] + mic_en_outs[3],
            res_en_outs[2] + mic_en_outs[2],
            res_en_outs[1] + mic_en_outs[1],
        ]
        for i, dec_layer in enumerate(self.decoder):
            skip = self.dec_skip_proj[i](all_skips[i])
            if skip.shape[-1] != x.shape[-1]:
                skip = F.interpolate(skip, size=(skip.shape[2], x.shape[-1]),
                                     mode='bilinear', align_corners=False)
            tT = min(x.shape[2], skip.shape[2])
            x = x[:, :, :tT, :]
            skip = skip[:, :, :tT, :]
            x = dec_layer(self.dec_skip_norms[i](x) + skip)
            if x.shape[2] > tT:
                x = x[:, :, :tT, :]
        x = self.mask_conv(x)
        mask_erb = torch.sigmoid(x)
        mask_full = self.erb.bs(mask_erb)
        mask_full = torch.clamp(mask_full, 0.0, 1.0)
        mask_full = mask_full.permute(0, 1, 3, 2)
        t_min = min(mask_full.shape[3], mic_spec.shape[3])
        mask_full = mask_full[:, :, :, :t_min]
        enhanced = mic_spec[:, :, :, :t_min] * mask_full
        output = compute_istft(enhanced, length=ns, window=self.stft_window)
        if self.leak_factor > 0:
            output = output * (1.0 - self.leak_factor) + mic * self.leak_factor
        if return_mask:
            return output, mask_full
        return output


def print_summary(m):
    t = sum(p.numel() for p in m.parameters())
    trainable = sum(p.numel() for p in m.parameters() if p.requires_grad)
    print(f"Aec7 | 48kHz | Dual-Stream ERB AEC")
    print(f"  Dual-Stream: mic_erb ∥ residual_erb (mic - H*far) → Fusion → DPGRNN → Decoder")
    print(f"    Front-end encoders: 2 × {FRONT_ENC_C}")
    print(f"    Deep encoder: {DEEP_ENC_C}")
    print(f"    Decoder: {DEC_C}")
    print(f"    ERB: {ERB_LOW}+{ERB_HIGH}={ERB_TOTAL} bins, STFT: {WIN}/{HOP}/{NFFT}")
    print(f"  Total params: {t:,} ({trainable:,} trainable)")


# ══════════════════════════════════════════════════════════════════════════════
# 流式推理 (aec7_stream)
# ══════════════════════════════════════════════════════════════════════════════

class Aec7Stream(nn.Module):
    FRONT_ENC_CONV_SHAPES = [(16,10,320),(24,10,160),(36,10,80),(48,9,40)]
    FRONT_ENC_TFA_HIDDEN = [32,48,72,96]
    DEEP_ENC_CONV_SHAPES = [(72,0,40),(96,0,40)]
    DEEP_ENC_TFA_HIDDEN = [144,192]
    DEC_CONV_SHAPES = [(96,0,40),(72,2,40),(48,2,80),(32,0,160)]
    DEC_TFA_HIDDEN = [192,144,96,64]
    INTER_CACHE_SHAPES = [(1,40,96),(1,40,96)]

    def __init__(self, model):
        super().__init__()
        self.erb = model.erb
        self.res_norm      = model.res_norm
        self.mic_norm     = model.mic_norm
        self.res_encoder  = model.res_encoder
        self.mic_encoder  = model.mic_encoder
        self.linear_echo_filter = model.linear_echo_filter
        self.fusion = model.fusion
        self.fusion_norm = model.fusion_norm
        self.deep_encoder = model.deep_encoder
        self.deep_norm = model.deep_norm
        self.dpgrnn = model.dpgrnn
        self.decoder = model.decoder
        self.dec_skip_norms = model.dec_skip_norms
        self.dec_skip_proj = model.dec_skip_proj
        self.mask_conv = model.mask_conv
        self.leak_factor = model.leak_factor
        self._prev1 = None
        self._prev2 = None

    @classmethod
    def init_caches(cls, batch_size=1, device=None):
        def _m(shapes):
            t = sum(int(np.prod(s)) for s in shapes)
            return torch.zeros(batch_size, t, device=device)
        fe_c = cls.FRONT_ENC_CONV_SHAPES; fe_h = cls.FRONT_ENC_TFA_HIDDEN
        return {
            'res_enc_conv': _m(fe_c), 'res_enc_tfa': _m(fe_h),
            'mic_enc_conv': _m(fe_c), 'mic_enc_tfa': _m(fe_h),
            'deep_enc_conv': _m(cls.DEEP_ENC_CONV_SHAPES),
            'deep_enc_tfa':  _m(cls.DEEP_ENC_TFA_HIDDEN),
            'dec_conv': _m(cls.DEC_CONV_SHAPES),
            'dec_tfa':  _m(cls.DEC_TFA_HIDDEN),
            'inter': _m(cls.INTER_CACHE_SHAPES),
            'res_prev1': torch.zeros(batch_size, 1, 1, 320, device=device),
            'res_prev2': torch.zeros(batch_size, 1, 1, 320, device=device),
            'mic_prev1': torch.zeros(batch_size, 1, 1, 320, device=device),
            'mic_prev2': torch.zeros(batch_size, 1, 1, 320, device=device),
        }

    @staticmethod
    def _unpack(flat, shapes):
        bsz = flat.shape[0]; out, off = [], 0
        for s in shapes:
            n = int(np.prod(s))
            if n == 0: out.append(None)
            else: out.append(flat[:, off:off+n].reshape(bsz, s[0], s[1], s[2]))
            off += n
        return out

    @staticmethod
    def _unpack_gru(flat, hs, nl=1, nd=1):
        bsz, nd_, off = flat.shape[0], nl*nd, 0
        out = []
        for h in hs:
            if h==0: out.append(None)
            else: out.append(flat[:, off:off+h].reshape(nd_, bsz, h//nd_))
            off += h
        return out

    def _unpack_inter(self, inter):
        bsz, out, off = inter.shape[0], [], 0
        for s in self.INTER_CACHE_SHAPES:
            n = int(np.prod(s)); out.append(inter[:, off:off+n].reshape(s[0],s[1],s[2])); off += n
        return out

    @staticmethod
    def _pack(caches, bsz, device):
        v = [c.reshape(bsz, -1) for c in caches if c is not None]
        return torch.cat(v, dim=1) if v else torch.zeros(bsz, 0, device=device)

    @staticmethod
    def _stream_conv(x, conv, cache, delayed=False):
        if cache is None:
            y = conv(x)
            if isinstance(conv, nn.ConvTranspose2d) and y.shape[2] > 1:
                y = y[:, :, 1:2, :]
            return y, cache
        if delayed:
            y = conv(cache)
            if isinstance(conv, nn.ConvTranspose2d) and y.shape[2] > 1:
                y = y[:, :, 1:2, :]
            return y, torch.cat([cache[:, :, 1:, :], x], dim=2)
        inp = torch.cat([cache, x], dim=2)
        y = conv(inp)
        if isinstance(conv, nn.ConvTranspose2d) and y.shape[2] > 1:
            y = y[:, :, 1:2, :]
        return y, inp[:, :, 1:, :]

    @staticmethod
    def _stream_ctfa(ctfa, x, hc):
        zt = x.pow(2).mean(dim=-1).transpose(1, 2)
        at, hc = ctfa.ta_gru(zt, hc)
        at = torch.sigmoid(ctfa.ta_fc(at).transpose(1, 2))
        af = torch.sigmoid(ctfa.fa(x))
        return at[..., None]*x*af[:, None], hc

    @staticmethod
    def _stream_dpgrnn(dpgrnn, x, ic):
        B, C, T, Fd = x.shape
        ix = x.permute(0,2,3,1).reshape(B*T, Fd, C)
        ix, _ = dpgrnn.intra_rnn(ix); ix = dpgrnn.intra_fc(ix)
        ix = ix.reshape(B, T, Fd, C); ix = dpgrnn.intra_ln(ix)
        io_ = x.permute(0,2,3,1)+ix
        ex = io_.permute(0,2,1,3).reshape(B*Fd, T, C)
        ex, ic = dpgrnn.inter_rnn(ex, ic)
        ex = dpgrnn.inter_fc(ex).reshape(B, Fd, T, C).permute(0,2,1,3)
        ex = dpgrnn.inter_ln(ex)
        return (io_+ex).permute(0,3,1,2), ic

    def _stream_block(self, blk, x, cv, tv, delayed=False):
        h = blk.pconv(x)
        cm = None
        for m in blk.dconv:
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)): cm = m; break
        if cm is not None: h, cv = self._stream_conv(h, cm, cv, delayed)
        for m in blk.dconv:
            if not isinstance(m, (nn.Conv2d, nn.ConvTranspose2d, nn.ZeroPad2d)):
                h = m(h)
        if not isinstance(blk.tfa, nn.Identity) and tv is not None:
            h, tv = self._stream_ctfa(blk.tfa, h, tv)
        return h, cv, tv

    def _run_enc(self, enc, x, cv, tv):
        eo = []
        for i, blk in enumerate(enc.layers):
            x, cv[i], tv[i] = self._stream_block(blk, x, cv[i], tv[i])
            eo.append(x)
        return x, eo, cv, tv

    def _spec_to_feat(self, spec_frame, prev1, prev2):
        sf_ = spec_frame.unsqueeze(-1)
        real = sf_.permute(0, 1, 3, 2)
        mag = torch.log10(torch.norm(real, dim=1, keepdim=True).clamp(1e-12))
        erb = self.erb.bm(mag.permute(0, 1, 3, 2)).permute(0, 1, 3, 2)
        d1 = erb - prev1
        d2 = d1 - (prev1 - prev2)
        new_prev2 = prev1
        new_prev1 = erb
        return torch.cat([erb, d1, d2], dim=1), new_prev1, new_prev2

    def reset_deriv_cache(self, batch_size, device):
        self._prev1 = torch.zeros(batch_size, 1, 1, 320, device=device)
        self._prev2 = torch.zeros(batch_size, 1, 1, 320, device=device)

    def _linear_echo_cancel_frame(self, mic_frame, far_frame):
        gain_r = self.linear_echo_filter[0]
        gain_i = self.linear_echo_filter[1]
        fr, fi = far_frame[:, 0], far_frame[:, 1]
        echo_r = fr * gain_r - fi * gain_i
        echo_i = fr * gain_i + fi * gain_r
        echo_frame = torch.stack([echo_r, echo_i], dim=1)
        return mic_frame - echo_frame

    def forward(self, mic_frame, far_frame, caches):
        B = mic_frame.shape[0]; dev = mic_frame.device
        fe_c = self.FRONT_ENC_CONV_SHAPES; fe_h = self.FRONT_ENC_TFA_HIDDEN
        dc_c = self.DEEP_ENC_CONV_SHAPES; dc_h = self.DEEP_ENC_TFA_HIDDEN
        dd_c = self.DEC_CONV_SHAPES; dd_h = self.DEC_TFA_HIDDEN
        residual_frame = self._linear_echo_cancel_frame(mic_frame, far_frame)
        res_p1 = caches['res_prev1']
        res_p2 = caches['res_prev2']
        mic_p1 = caches['mic_prev1']
        mic_p2 = caches['mic_prev2']
        res_feat, res_p1_new, res_p2_new = self._spec_to_feat(residual_frame, res_p1, res_p2)
        mic_feat, mic_p1_new, mic_p2_new = self._spec_to_feat(mic_frame, mic_p1, mic_p2)
        res_feat_normed = self.res_norm(res_feat)
        mic_feat_normed = self.mic_norm(mic_feat)
        rcv = self._unpack(caches['res_enc_conv'], fe_c)
        rtv = self._unpack_gru(caches['res_enc_tfa'], fe_h)
        mcv = self._unpack(caches['mic_enc_conv'], fe_c)
        mtv = self._unpack_gru(caches['mic_enc_tfa'], fe_h)
        dcv = self._unpack(caches['deep_enc_conv'], dc_c)
        dtv = self._unpack_gru(caches['deep_enc_tfa'], dc_h)
        decv = self._unpack(caches['dec_conv'], dd_c)
        detv = self._unpack_gru(caches['dec_tfa'], dd_h)
        ic = self._unpack_inter(caches['inter'])
        xr, reo, rcv, rtv = self._run_enc(self.res_encoder, res_feat_normed, rcv, rtv)
        xm, meo, mcv, mtv = self._run_enc(self.mic_encoder, mic_feat_normed, mcv, mtv)
        fused = self.fusion(xr, xm)
        fused = self.fusion_norm(fused)
        den = []; xx = fused
        for i, blk in enumerate(self.deep_encoder.layers):
            xx, dcv[i], dtv[i] = self._stream_block(blk, xx, dcv[i], dtv[i])
            den.append(xx)
        xx = self.deep_norm(xx)
        for i, dl in enumerate(self.dpgrnn):
            xx, ic[i] = self._stream_dpgrnn(dl, xx, ic[i])
        all_skips = [den[1], reo[3]+meo[3], reo[2]+meo[2], reo[1]+meo[1]]
        for i, dl in enumerate(self.decoder):
            sk = self.dec_skip_proj[i](all_skips[i])
            if sk.shape[-1] != xx.shape[-1]:
                sk = F.interpolate(sk, size=(sk.shape[2], xx.shape[-1]),
                                   mode='bilinear', align_corners=False)
            tT = min(xx.shape[2], sk.shape[2])
            xx, sk = xx[:,:,:tT,:], sk[:,:,:tT,:]
            is_delayed = (self.DEC_CONV_SHAPES[i][1] > 1)
            xx, decv[i], detv[i] = self._stream_block(
                self.decoder[i], self.dec_skip_norms[i](xx)+sk, decv[i], detv[i], delayed=is_delayed)
            if xx.shape[2] > tT: xx = xx[:,:,:tT,:]
        xx = self.mask_conv(xx)
        mask_erb = torch.sigmoid(xx)
        mask_full = self.erb.bs(mask_erb)
        mask_full = torch.clamp(mask_full, 0.0, 1.0)
        enhanced = mic_frame * mask_full.squeeze(1)
        if self.leak_factor > 0:
            enhanced = enhanced * (1.0 - self.leak_factor) + mic_frame * self.leak_factor
        new_caches = {
            'res_enc_conv': self._pack(rcv,B,dev), 'res_enc_tfa': self._pack(rtv,B,dev),
            'mic_enc_conv': self._pack(mcv,B,dev), 'mic_enc_tfa': self._pack(mtv,B,dev),
            'deep_enc_conv': self._pack(dcv,B,dev), 'deep_enc_tfa': self._pack(dtv,B,dev),
            'dec_conv': self._pack(decv,B,dev), 'dec_tfa': self._pack(detv,B,dev),
            'inter': self._pack(ic,B,dev),
            'res_prev1': res_p1_new, 'res_prev2': res_p2_new,
            'mic_prev1': mic_p1_new, 'mic_prev2': mic_p2_new,
        }
        return enhanced, new_caches


def stream_infer(model, mic_wav, far_wav, device='cpu'):
    if mic_wav.dim() == 1:
        mic_wav = mic_wav.unsqueeze(0); far_wav = far_wav.unsqueeze(0)
    B, ns = mic_wav.shape
    stream = Aec7Stream(model).to(device).eval()
    caches = Aec7Stream.init_caches(B, device)
    mic_spec = compute_stft(mic_wav.to(device), window=model.stft_window.to(device))
    far_spec = compute_stft(far_wav.to(device), window=model.stft_window.to(device))
    T = mic_spec.shape[3]
    enhanced_specs = []
    for t in range(T):
        mf = mic_spec[:, :, :, t]
        ff = far_spec[:, :, :, t]
        out, caches = stream(mf, ff, caches)
        enhanced_specs.append(out.unsqueeze(-1))
    enhanced = torch.cat(enhanced_specs, dim=-1)
    output = compute_istft(enhanced, length=ns, window=model.stft_window.to(device))
    return output


def lr_fn(step):
    cycle_steps = (SPE // BS) * 10
    pos = step % cycle_steps
    return LR_MAX - (LR_MAX - LR_MIN) * pos / (cycle_steps - 1)


class HybridLoss(nn.Module):
    def __init__(self, use_vad=False, vad_db_below_peak=24.0, min_speech_frames=5,
                 min_silence_frames=9, vad_weight=1000, compress_factor=0.5,
                 lamda_ri=10, lamda_mag=20, lamda_amp=10, lamda_sisnr=0.5,
                 asym_mag_weight=2.5,
                 n_fft=960, hop_len=480, win_len=960, eps=1e-12):
        super().__init__()
        self.use_vad = use_vad; self.vad_db_below_peak = vad_db_below_peak
        self.vad_weight = vad_weight; self.compress_factor = compress_factor
        self.lamda_ri = lamda_ri; self.lamda_mag = lamda_mag
        self.lamda_amp = lamda_amp; self.lamda_sisnr = lamda_sisnr
        self.asym_mag_weight = asym_mag_weight
        self.n_fft = n_fft; self.hop_len = hop_len; self.win_len = win_len; self.eps = eps
        self.attack = min_speech_frames if min_speech_frames % 2 == 1 else min_speech_frames + 1
        self.release = min_silence_frames if min_silence_frames % 2 == 1 else min_silence_frames + 1
        self.register_buffer('window', torch.hann_window(win_len).pow(0.5))

    def get_vad_mask(self, mag):
        fe = torch.mean(mag, dim=1, keepdim=True)
        edb = 10.0 * torch.log10(fe + self.eps)
        th = edb.max() - self.vad_db_below_peak
        rm = (edb > th).float()
        me = -F.max_pool1d(-rm, kernel_size=self.attack, stride=1, padding=self.attack // 2)
        return F.max_pool1d(me, kernel_size=self.release, stride=1, padding=self.release // 2)

    def forward(self, y_pred, y_true, mic=None, leak_factor=0.0):
        dev = y_pred.device
        if leak_factor > 0 and mic is not None:
            ml = min(y_pred.shape[-1], mic.shape[-1])
            y_pred = y_pred[..., :ml] * (1.0 - leak_factor) + mic[..., :ml] * leak_factor
            y_true = y_true[..., :ml]
        ps = torch.stft(y_pred, self.n_fft, self.hop_len, self.win_len, self.window.to(dev), return_complex=True)
        ts = torch.stft(y_true, self.n_fft, self.hop_len, self.win_len, self.window.to(dev), return_complex=True)
        pr, pi = ps.real, ps.imag; tr, ti = ts.real, ts.imag
        pm = torch.sqrt(pr**2 + pi**2 + self.eps); tm = torch.sqrt(tr**2 + ti**2 + self.eps)
        c = self.compress_factor
        prc = pr / (pm**(1 - c) + self.eps); pic = pi / (pm**(1 - c) + self.eps)
        trc = tr / (tm**(1 - c) + self.eps); tic = ti / (tm**(1 - c) + self.eps)
        rl = torch.mean((prc - trc)**2); il = torch.mean((pic - tic)**2)
        mag_diff = pm**c - tm**c
        asym_w = torch.where(mag_diff < 0, self.asym_mag_weight, 1.0)
        ml = torch.mean(asym_w * (mag_diff ** 2))
        dot = torch.sum(y_true * y_pred, dim=-1, keepdim=True)
        yt2 = torch.sum(y_true**2, dim=-1, keepdim=True) + self.eps
        a = dot / yt2; st = a * y_true; en = y_pred - st
        sn2 = torch.sum(st**2, dim=-1); en2 = torch.sum(en**2, dim=-1) + self.eps
        sisnr = -10.0 * torch.log10(sn2 / en2 + self.eps).mean()
        al = torch.mean(torch.abs(y_pred - y_true))
        tl = self.lamda_ri * (rl + il) + self.lamda_mag * ml + self.lamda_amp * al + self.lamda_sisnr * sisnr
        if self.use_vad:
            vm = self.get_vad_mask(tm).detach(); sz = 1 - vm
            vl = torch.mean(sz * (torch.abs(pr) + torch.abs(pi)))
            tl = tl + self.vad_weight * vl
        else:
            vl = torch.tensor(0.0, device=dev)
        return tl, rl + il, ml, sisnr, al, vl


criterion = HybridLoss(use_vad=False)


def find_ckpt(d):
    m = {}
    if not os.path.isdir(d): return None, 0
    for f in os.listdir(d):
        if f.endswith('.tar') and f.startswith('checkpoint_'):
            try: m[int(re.search(r'(\d+)', f).group(1))] = f
            except: pass
    if m: mx = max(m.keys()); return os.path.join(d, m[mx]), mx
    return None, 0


def load_wav(p):
    w, _ = sf.read(p, dtype='float32')
    if w.ndim > 1: w = w.mean(1)
    return torch.from_numpy(w)


def save_wav(p, w, sr=FS):
    sf.write(p, w.detach().cpu().numpy(), sr)


def export_aec7_onnx(model, onnx_path, device='cpu'):
    B = 1; F_ = 481
    stream = Aec7Stream(model).to(device).eval()
    caches = Aec7Stream.init_caches(B, device)
    cache_keys = ['res_enc_conv', 'res_enc_tfa', 'mic_enc_conv', 'mic_enc_tfa',
                  'deep_enc_conv', 'deep_enc_tfa', 'dec_conv', 'dec_tfa', 'inter',
                  'res_prev1', 'res_prev2', 'mic_prev1', 'mic_prev2']
    cache_vals = [caches[k] for k in cache_keys]
    dummy_mic = torch.randn(B, 2, F_, device=device)
    dummy_far = torch.randn(B, 2, F_, device=device)
    in_names = ['mic_frame', 'far_frame'] + cache_keys
    out_names = ['enhanced_frame'] + [f'{k}_o' for k in cache_keys]

    class Wrapper(nn.Module):
        def __init__(self, stream):
            super().__init__()
            self.stream = stream
        def forward(self, mic_frame, far_frame,
                    res_enc_conv, res_enc_tfa,
                    mic_enc_conv, mic_enc_tfa, deep_enc_conv, deep_enc_tfa,
                    dec_conv, dec_tfa, inter,
                    res_prev1, res_prev2, mic_prev1, mic_prev2):
            caches = {'res_enc_conv': res_enc_conv, 'res_enc_tfa': res_enc_tfa,
                      'mic_enc_conv': mic_enc_conv, 'mic_enc_tfa': mic_enc_tfa,
                      'deep_enc_conv': deep_enc_conv, 'deep_enc_tfa': deep_enc_tfa,
                      'dec_conv': dec_conv, 'dec_tfa': dec_tfa, 'inter': inter,
                      'res_prev1': res_prev1, 'res_prev2': res_prev2,
                      'mic_prev1': mic_prev1, 'mic_prev2': mic_prev2}
            out, nc = self.stream(mic_frame, far_frame, caches)
            return (out,
                    nc['res_enc_conv'], nc['res_enc_tfa'],
                    nc['mic_enc_conv'], nc['mic_enc_tfa'], nc['deep_enc_conv'], nc['deep_enc_tfa'],
                    nc['dec_conv'], nc['dec_tfa'], nc['inter'],
                    nc['res_prev1'], nc['res_prev2'], nc['mic_prev1'], nc['mic_prev2'])

    wrapper = Wrapper(stream).eval()
    with torch.no_grad():
        test = wrapper(dummy_mic, dummy_far, *cache_vals)
    print(f"  ONNX dry-run: {len(test)} outputs, enhanced={test[0].shape}")
    os.makedirs(os.path.dirname(onnx_path) if os.path.dirname(onnx_path) else '.', exist_ok=True)
    torch.onnx.export(wrapper, (dummy_mic, dummy_far, *cache_vals), onnx_path,
                      input_names=in_names, output_names=out_names,
                      opset_version=17, do_constant_folding=True,
                      export_params=True, verbose=False, dynamo=False)
    print(f"  ONNX exported: {onnx_path} ({os.path.getsize(onnx_path) / 1024:.0f} KB)")


def infer_aec7_onnx(onnx_path, mic_wav, far_wav, device='cpu'):
    import onnxruntime as ort
    so = ort.SessionOptions(); so.intra_op_num_threads = 1; so.inter_op_num_threads = 1
    if mic_wav.dim() == 1:
        mic_wav = mic_wav.unsqueeze(0); far_wav = far_wav.unsqueeze(0)
    B, ns = mic_wav.shape
    mic_wav = mic_wav.to(device); far_wav = far_wav.to(device)
    w = Aec7Model().stft_window.to(device)
    mic_spec = compute_stft(mic_wav, window=w)
    far_spec = compute_stft(far_wav, window=w)
    T = mic_spec.shape[3]
    sess = ort.InferenceSession(onnx_path, so, providers=['CPUExecutionProvider'])
    onnx_inputs = {i.name for i in sess.get_inputs()}
    onnx_outputs = [o.name for o in sess.get_outputs()]
    cache_keys = ['res_enc_conv', 'res_enc_tfa', 'mic_enc_conv', 'mic_enc_tfa',
                  'deep_enc_conv', 'deep_enc_tfa', 'dec_conv', 'dec_tfa', 'inter',
                  'res_prev1', 'res_prev2', 'mic_prev1', 'mic_prev2']
    all_caches = Aec7Stream.init_caches(1, device='cpu')
    cache_vals = {k: all_caches[k].numpy() for k in cache_keys if k in onnx_inputs}
    active_keys = [k for k in cache_keys if k in onnx_inputs]
    enhanced_frames = []
    for t in range(T):
        mf = mic_spec[:, :, :, t].cpu().numpy()
        ff = far_spec[:, :, :, t].cpu().numpy()
        inputs = {'mic_frame': mf, 'far_frame': ff}
        for k in active_keys:
            inputs[k] = cache_vals[k]
        outs = sess.run(None, inputs)
        enhanced_frames.append(outs[0])
        out_map = dict(zip(onnx_outputs[1:], outs[1:]))
        for k in active_keys:
            ok = f'{k}_o'
            if ok in out_map:
                cache_vals[k] = out_map[ok]
    enhanced = np.stack(enhanced_frames, axis=-1)
    enhanced_t = torch.from_numpy(enhanced).to(device)
    output = compute_istft(enhanced_t, length=ns, window=w)
    return output


# ─── TensorBoard 子进程 ───
_tb_process = None

def start_tensorboard(log_dir):
    global _tb_process
    cmd = [sys.executable, '-m', 'tensorboard', f'--logdir={log_dir}', '--port=6006']
    print(f"启动 TensorBoard: {' '.join(cmd)}")
    _tb_process = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    print(f"  TensorBoard PID: {_tb_process.pid}, http://localhost:6006")


def stop_tensorboard():
    global _tb_process
    if _tb_process is not None:
        _tb_process.terminate()
        try: _tb_process.wait(timeout=5)
        except: _tb_process.kill()
        print("TensorBoard 已停止")
        _tb_process = None


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--causal', action='store_true')
    p.add_argument('--no-bf16', action='store_true')
    p.add_argument('--ckpt', type=str, default=None)
    p.add_argument('--skip-test', action='store_true')
    p.add_argument('--no-compile', action='store_true')
    a = p.parse_args()

    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {dev}")
    bf = not a.no_bf16 and torch.cuda.is_available()
    if bf: print("bf16 ON"); torch.backends.cuda.matmul.allow_tf32 = True

    cd = os.path.join(OUT, 'checkpoints'); od = os.path.join(OUT, 'results_wav')
    td = os.path.join(OUT, 'logs')
    for d in [cd, od, td]: os.makedirs(d, exist_ok=True)

    model = Aec7Model().to(dev)
    print_summary(model)

    if not a.no_compile and sys.platform == 'win32':
        try:
            import pathlib, shutil, tempfile
            _vs_roots = [pathlib.Path(p) for p in [
                r'C:\Program Files\Microsoft Visual Studio',
                r'C:\Program Files (x86)\Microsoft Visual Studio']]
            _msvc_ok = False
            for _vs_root in _vs_roots:
                if not _vs_root.exists(): continue
                for _ver_dir in sorted(_vs_root.iterdir()):
                    if not _ver_dir.is_dir(): continue
                    for _ed in ['Community', 'Professional', 'Enterprise', 'BuildTools']:
                        _vcvars = _ver_dir / _ed / 'VC' / 'Auxiliary' / 'Build' / 'vcvars64.bat'
                        if _vcvars.exists():
                            _bat = tempfile.NamedTemporaryFile(suffix='.bat', mode='w', delete=False)
                            _bat.write(f'call "{_vcvars}" >nul 2>&1\r\nset\r\n')
                            _bat.close()
                            _out = subprocess.run(['cmd', '/c', _bat.name], capture_output=True, text=True)
                            os.unlink(_bat.name)
                            for _line in _out.stdout.splitlines():
                                if '=' in _line and not _line.startswith('_'):
                                    k, v = _line.split('=', 1); os.environ[k] = v
                            if not shutil.which('cl.exe'):
                                _d = _ver_dir / _ed / 'VC' / 'Tools' / 'MSVC'
                                if _d.exists():
                                    for _v in sorted(_d.iterdir(), reverse=True):
                                        _cl = _v / 'bin' / 'Hostx64' / 'x64' / 'cl.exe'
                                        if _cl.exists(): os.environ['PATH'] = str(_cl.parent) + ';' + os.environ.get('PATH', ''); break
                            if shutil.which('cl.exe'): _msvc_ok = True; print(f"MSVC activated: {_vcvars}"); break
                    if _msvc_ok: break
                if _msvc_ok: break
            if _msvc_ok:
                pass  # compile deferred to after state restoration
            else:
                print("torch.compile SKIPPED (no MSVC)")
        except Exception as e:
            print(f"MSVC setup failed: {e}"); print("torch.compile SKIPPED")

    if a.causal:
        print("\nCausal check...")
        a_ = torch.randn(1, 48000, device=dev); b_ = torch.randn(1, 48000, device=dev)
        c_ = torch.randn(1, 48000, device=dev); fa_ = torch.randn(1, 48000, device=dev)
        x1 = torch.cat([a_, b_], 1); x2 = torch.cat([a_, c_], 1)
        model.eval()
        with torch.no_grad(): y1 = model(x1, fa_); y2 = model(x2, fa_)
        lat = WIN + HOP; ch = 48000 - lat
        d_ = (y1[0, lat:lat + ch] - y2[0, lat:lat + ch]).abs().max()
        print(f"  diff={d_:.2e} {'OK' if d_ < 1e-3 else 'FAIL'}"); return

    if a.ckpt and os.path.exists(a.ckpt):
        print(f"Load {a.ckpt}")
        ck = torch.load(a.ckpt, map_location=dev, weights_only=False)
        sd = {k.replace('_orig_mod.', ''): v for k, v in ck['model'].items()}
        raw = getattr(model, '_orig_mod', model)
        raw.load_state_dict(sd); model.eval()
        if os.path.exists(TEST_MIC) and os.path.exists(TEST_FAR):
            m = load_wav(TEST_MIC).to(dev); f = load_wav(TEST_FAR).to(dev)
            y_st = stream_infer(model, m, f, dev).squeeze(0)
            save_wav(os.path.join(od, 'result.wav'), y_st)
            print(f"  result.wav saved")
        return

    start_tensorboard(td)

    random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)

    try:
        opt = optim.AdamW(model.parameters(), lr=LR0, weight_decay=1e-2)
        sc = torch.amp.GradScaler('cuda', enabled=bf)
        scheduler = torch.optim.lr_scheduler.LambdaLR(opt,
            lr_lambda=lambda step: lr_fn(step) / LR0)

        lc, se = find_ckpt(cd); gs = 0; best_si = -float('inf')
        if lc:
            ck = torch.load(lc, map_location=dev, weights_only=False)
            sd = {k.replace('_orig_mod.', ''): v for k, v in ck['model'].items()}
            raw = getattr(model, '_orig_mod', model)
            raw.load_state_dict(sd)
            if 'optimizer' in ck:
                opt.load_state_dict(ck['optimizer'])
            else:
                print("WARNING: No optimizer in checkpoint.")
            if 'scaler' in ck:
                sc.load_state_dict(ck['scaler'])
            if 'scheduler' in ck:
                scheduler.load_state_dict(ck['scheduler'])
            se = ck.get('epoch', 0) + 1
            gs = ck.get('global_step', 0)
            best_si = ck.get('best_sisnr', -float('inf'))
            # DIAG: confirm optimizer state loaded
            p0 = list(model.parameters())[0]
            st = opt.state.get(p0, {})
            print(f"[RESUME] ep={se} gs={gs} best_si={best_si:.2f} "
                  f"opt_step={st.get('step',0)} exp_avg_mean={st.get('exp_avg',torch.tensor(0.)).mean().item():.6e}")
        else:
            print("No checkpoint — training from scratch.")

        # ─── torch.compile AFTER all state is restored ───
        if _msvc_ok and dev == 'cuda':
            print("torch.compile ON (reduce-overhead, dynamic=False)")
            _t0 = time.time()
            model = torch.compile(model, mode="reduce-overhead", dynamic=False)
            print(f"  compile took {time.time()-_t0:.1f}s")

        w = SummaryWriter(log_dir=td, purge_step=gs if se > 0 else None)
        tds = Aec7Dataset(DATA, train=True, chunk_sec=CHUNK)
        vds = Aec7Dataset(DATA, train=False, chunk_sec=CHUNK)
        vl = DataLoader(vds, BS, shuffle=False, num_workers=4, pin_memory=True,
                        drop_last=True, collate_fn=collate_aec7, prefetch_factor=2)

        for ep in range(se, EPO + 1):
            idx = np.random.choice(len(tds), SPE, replace=False)
            sm = SubsetRandomSampler(idx)
            tl = DataLoader(tds, BS, sampler=sm, num_workers=4, pin_memory=True,
                            drop_last=True, collate_fn=collate_aec7, prefetch_factor=2)
            model.train()
            el_ri, el_mg, el_si, el_amp, el_vad, el_tot = 0, 0, 0, 0, 0, 0
            pb = tqdm(tl, total=len(tl), desc=f'E{ep}/{EPO}', ncols=80)
            for si, (mc, fa, ne) in enumerate(pb):
                scheduler.step()
                w.add_scalar(f'{G_ALL}/0_LR', scheduler.get_last_lr()[0], gs)
                mc = mc.to(dev); ne = ne.to(dev); fa = fa.to(dev)
                with torch.amp.autocast('cuda', enabled=bf, dtype=torch.bfloat16):
                    es = model(mc, fa)
                    tot, ri, mg, sis, amp, vad = criterion(es, ne)
                if torch.isnan(tot):
                    tqdm.write(f'  NaN at step {gs} (ep {ep}) -- SKIPPING batch'); gs += 1; continue
                opt.zero_grad()
                if bf: sc.scale(tot).backward()
                else: tot.backward()
                if CGN > 0:
                    if bf: sc.unscale_(opt)
                    gn = nn.utils.clip_grad_norm_(model.parameters(), CGN)
                    w.add_scalar(f'{G_ALL}/1_GN', gn.item(), gs)
                if bf: sc.step(opt); sc.update()
                else: opt.step()
                el_si += sis.item(); el_ri += ri.item(); el_mg += mg.item()
                el_amp += amp.item(); el_vad += vad.item(); el_tot += tot.item()
                w.add_scalar(f'{G_TRAIN}/0_Total', tot.item(), gs)
                w.add_scalar(f'{G_TRAIN}/1_SISNR', sis.item(), gs)
                w.add_scalar(f'{G_TRAIN}/2_RI', ri.item(), gs)
                w.add_scalar(f'{G_TRAIN}/3_Mag', mg.item(), gs)
                w.add_scalar(f'{G_TRAIN}/4_AMP', amp.item(), gs)
                w.add_scalar(f'{G_TRAIN}/5_VAD', vad.item(), gs)
                pb.set_postfix(si=f'{sis.item():.1f}')
                gs += 1

            nt = len(tl)
            w.add_scalar(f'{G_TRAIN_Epoch}/0_Total', el_tot / nt, ep)
            w.add_scalar(f'{G_TRAIN_Epoch}/1_SISNR', el_si / nt, ep)
            w.add_scalar(f'{G_TRAIN_Epoch}/2_RI', el_ri / nt, ep)
            w.add_scalar(f'{G_TRAIN_Epoch}/3_Mag', el_mg / nt, ep)
            w.add_scalar(f'{G_TRAIN_Epoch}/4_AMP', el_amp / nt, ep)
            w.add_scalar(f'{G_TRAIN_Epoch}/5_VAD', el_vad / nt, ep)
            print(f"  SI-SNR: {el_si / nt:.2f}")

            # 验证
            model.eval()
            vl_si, vl_ri, vl_mg, vl_amp, vl_vad = 0, 0, 0, 0, 0
            with torch.no_grad():
                for mc, fa, ne in tqdm(vl, desc='Val', ncols=80):
                    mc, ne, fa = mc.to(dev), ne.to(dev), fa.to(dev)
                    with torch.amp.autocast('cuda', enabled=bf, dtype=torch.bfloat16):
                        es = model(mc, fa)
                        tot, ri, mg, sis, amp, vad = criterion(es, ne)
                    vl_si += sis.item(); vl_ri += ri.item(); vl_mg += mg.item()
                    vl_amp += amp.item(); vl_vad += vad.item()
            nv = len(vl); vsi = vl_si / nv
            w.add_scalar(f'{G_VAL}/0_Total', (vl_ri + vl_mg + vl_amp + vl_si + vl_vad) / nv, ep)
            w.add_scalar(f'{G_VAL}/1_SISNR', vsi, ep)
            w.add_scalar(f'{G_VAL}/2_RI', vl_ri / nv, ep)
            w.add_scalar(f'{G_VAL}/3_Mag', vl_mg / nv, ep)
            w.add_scalar(f'{G_VAL}/4_AMP', vl_amp / nv, ep)
            w.add_scalar(f'{G_VAL}/5_VAD', vl_vad / nv, ep)

            cd2 = {'model': model.state_dict(), 'optimizer': opt.state_dict(),
                    'scheduler': scheduler.state_dict(), 'scaler': sc.state_dict(),
                    'epoch': ep, 'global_step': gs, 'val_sisnr': vsi, 'best_sisnr': best_si}
            torch.save(cd2, os.path.join(cd, f'checkpoint_epoch_{ep}.tar'))
            if vsi > best_si: best_si = vsi; print(f"  Best SI-SNR: {best_si:.2f}")

            if not a.skip_test and os.path.exists(TEST_MIC) and os.path.exists(TEST_FAR):
                print(f"\n[Epoch {ep} test]")
                mc_ = load_wav(TEST_MIC).to(dev); fa_ = load_wav(TEST_FAR).to(dev)

                # 1. 离线 PyTorch
                t0 = time.time()
                with torch.no_grad():
                    y_pt = model(mc_.unsqueeze(0), fa_.unsqueeze(0)).squeeze(0)
                n = mc_.shape[-1]
                if y_pt.shape[-1] < n: y_pt = F.pad(y_pt, (0, n - y_pt.shape[-1]))
                else: y_pt = y_pt[:n]
                save_wav(os.path.join(od, f'torch_e{ep}.wav'), y_pt)
                print(f"  Offline: {time.time() - t0:.1f}s")

                # 2. ONNX 导出
                onnx_dir = os.path.join(OUT, 'onnx'); os.makedirs(onnx_dir, exist_ok=True)
                onnx_path = os.path.join(onnx_dir, f'aec7_ep{ep:04d}.onnx')
                try:
                    raw_m = getattr(model, '_orig_mod', model)
                    export_aec7_onnx(raw_m, onnx_path, dev)
                except Exception as e:
                    print(f"  ONNX export FAILED: {e}")
                    w.add_scalar(f'{G_TEST}/1_ONNX_Cosine', 0.0, ep)
                    w.add_scalar(f'{G_TEST}/2_ONNX_SNR', -999.0, ep)
                else:
                    # 3. ONNX 流式推理
                    try:
                        t2 = time.time()
                        y_onnx = infer_aec7_onnx(onnx_path, mc_, fa_, dev).squeeze(0)
                        if y_onnx.shape[-1] < n: y_onnx = F.pad(y_onnx, (0, n - y_onnx.shape[-1]))
                        else: y_onnx = y_onnx[:n]
                        save_wav(os.path.join(od, f'onnx_e{ep}.wav'), y_onnx)
                        print(f"  ONNX Stream: {time.time() - t2:.1f}s")
                    except Exception as e:
                        print(f"  ONNX inference FAILED: {e}")
                        w.add_scalar(f'{G_TEST}/1_ONNX_Cosine', 0.0, ep)
                        w.add_scalar(f'{G_TEST}/2_ONNX_SNR', -999.0, ep)
                    else:
                        # 4. 对比
                        ml = min(y_pt.shape[-1], y_onnx.shape[-1])
                        ya = y_pt[:ml].cpu(); yb = y_onnx[:ml].cpu()
                        cos = F.cosine_similarity(ya.unsqueeze(0), yb.unsqueeze(0)).item()
                        snr = 10 * np.log10((ya.pow(2).sum() / ((ya - yb).pow(2).sum().clamp(1e-12))).item())
                        print(f"  torch_e{ep}.wav  onnx_e{ep}.wav  cos={cos:.6f}  snr={snr:.1f}dB")
                        w.add_scalar(f'{G_TEST}/1_ONNX_Cosine', cos, ep)
                        w.add_scalar(f'{G_TEST}/2_ONNX_SNR', snr, ep)

            if ep % 10 == 0: w.flush()

        torch.save({'model': model.state_dict(), 'optimizer': opt.state_dict(),
                    'scheduler': scheduler.state_dict(), 'scaler': sc.state_dict(),
                     'epoch': EPO, 'global_step': gs, 'best_sisnr': best_si}, os.path.join(cd, 'final_model.tar'))
        w.close()
        print(f"Done. Best SI-SNR: {best_si:.2f}")

    finally:
        stop_tensorboard()


if __name__ == "__main__":
    main()
