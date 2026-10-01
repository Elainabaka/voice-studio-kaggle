"""Tien ich am thanh: doc/ghi, resample, mono, trim silent, chuan hoa am luong."""

from __future__ import annotations

import os
import subprocess
from typing import Tuple

import numpy as np

try:
    import soundfile as sf
except Exception:  # pragma: no cover
    sf = None

try:
    import librosa
except Exception:  # pragma: no cover
    librosa = None


def load_audio(path: str, target_sr: int | None = None) -> Tuple[np.ndarray, int]:
    """Doc file am thanh -> (float32 mono [-1,1], sample_rate)."""
    if sf is not None:
        data, sr = sf.read(path, dtype="float32", always_2d=True)
        wav = data.mean(axis=1)
    elif librosa is not None:
        wav, sr = librosa.load(path, sr=target_sr, mono=True)
    else:
        raise RuntimeError("Can `soundfile` hoac `librosa` de doc am thanh.")
    if target_sr and sr != target_sr:
        wav = resample(wav, sr, target_sr)
        sr = target_sr
    return wav.astype(np.float32), int(sr)


def resample(wav: np.ndarray, sr: int, target_sr: int) -> np.ndarray:
    if sr == target_sr:
        return wav
    if librosa is not None:
        return librosa.resample(wav, orig_sr=sr, target_sr=target_sr)
    # Fallback: linear interpolation (kem chat luong hon, chi dung khi thieu librosa).
    n = int(round(len(wav) * target_sr / sr))
    return np.interp(np.linspace(0, len(wav) - 1, n), np.arange(len(wav)), wav).astype(np.float32)


def to_mono(wav: np.ndarray) -> np.ndarray:
    return wav if wav.ndim == 1 else wav.mean(axis=1).astype(np.float32)


def trim_silence(wav: np.ndarray, sr: int, top_db: float = 35.0) -> np.ndarray:
    """Cat bot khoang lang o 2 dau. Giu am thanh noi."""
    if librosa is None:
        return wav
    idx = librosa.effects.split(wav, top_db=top_db)
    if len(idx) == 0:
        return wav
    return wav[idx[0][0] : idx[-1][1]]


def normalize(wav: np.ndarray, peak: float = 0.97) -> np.ndarray:
    m = float(np.max(np.abs(wav))) if wav.size else 0.0
    if m < 1e-9:
        return wav
    return (wav * (peak / m)).astype(np.float32)


def save_wav(path: str, wav: np.ndarray, sr: int) -> str:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    if sf is None:
        raise RuntimeError("Can `soundfile` de ghi wav.")
    sf.write(path, np.clip(wav, -1.0, 1.0), sr)
    return path


def save_mp3(path: str, wav: np.ndarray, sr: int, bitrate: str = "192k") -> str:
    """WAV -> MP3 qua ffmpeg (mac dinh co tren Kaggle)."""
    tmp = path + ".tmp.wav"
    save_wav(tmp, wav, sr)
    try:
        subprocess.run(
            ["ffmpeg", "-y", "-loglevel", "error", "-i", tmp, "-b:a", bitrate, path],
            check=True,
        )
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)
    return path


def concat(wavs: list, sr: int, gap_seconds: float = 0.25) -> np.ndarray:
    """Noi nhieu doan am thanh lai, chen khoang lang giua cac doan."""
    gap = np.zeros(int(sr * gap_seconds), dtype=np.float32)
    parts = []
    for i, w in enumerate(wavs):
        parts.append(np.asarray(w, dtype=np.float32))
        if i < len(wavs) - 1:
            parts.append(gap)
    return np.concatenate(parts) if parts else np.zeros(0, dtype=np.float32)
