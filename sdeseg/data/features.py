from __future__ import annotations
import torch
import soundfile as sf
import torchaudio

def load_audio(path: str, train: bool = False) -> torch.Tensor:
    wav, sr = sf.read(path, always_2d=False); wav = torch.as_tensor(wav, dtype=torch.float32)
    if wav.ndim == 2: wav = wav.mean(1)
    if sr != 16000: wav = torchaudio.functional.resample(wav, sr, 16000)
    peak = wav.abs().max(); wav = wav / peak.clamp_min(1e-8)
    n=64000
    if wav.numel() < n: wav = wav.repeat((n+wav.numel()-1)//wav.numel())
    start = int(torch.randint(0, wav.numel()-n+1, (1,))) if train and wav.numel()>n else max(0,(wav.numel()-n)//2)
    return wav[start:start+n]

# A real-valued one-sided STFT with n_fft=512 has n_stft=257 bins.
_MEL = torchaudio.transforms.MelScale(64, 16000, 0, 8000, 257)
_WINDOW = torch.hann_window(400)

def logmel(path: str, train: bool = False) -> torch.Tensor:
    wav = load_audio(path, train)
    spec = torch.stft(wav, 512, 160, 400, _WINDOW, center=True, return_complex=True)
    spec = spec[:, :400]
    mel = _MEL(spec.abs().pow(2))
    x = torch.log(mel + 1e-6)
    return ((x - x.mean()) / x.std().clamp_min(1e-6)).unsqueeze(0)
