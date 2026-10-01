"""Dong goi san pham: ghi wav/mp3, tao zip tai ve."""

from __future__ import annotations

import os
import zipfile
from typing import List

import numpy as np

from . import audio


def write_output(path: str, wav: np.ndarray, sr: int, fmt: str = "wav") -> str:
    if fmt == "mp3":
        return audio.save_mp3(path if path.endswith(".mp3") else path + ".mp3", wav, sr)
    return audio.save_wav(path if path.endswith(".wav") else path + ".wav", wav, sr)


def zip_outputs(out_dir: str, zip_path: str, patterns: tuple = (".wav", ".mp3")) -> str:
    """Nen toan bo san pham trong out_dir thanh .zip de tai 1 lan."""
    os.makedirs(os.path.dirname(zip_path) or ".", exist_ok=True)
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as z:
        for fn in sorted(os.listdir(out_dir)):
            if fn.lower().endswith(patterns):
                z.write(os.path.join(out_dir, fn), arcname=fn)
    return zip_path


def list_outputs(out_dir: str) -> List[str]:
    if not os.path.isdir(out_dir):
        return []
    return [
        os.path.join(out_dir, f)
        for f in sorted(os.listdir(out_dir))
        if f.lower().endswith((".wav", ".mp3"))
    ]
