"""Render script da giong (podcast / audiobook / hoi thoai) -> am thanh.

Moi dong trong script = {voice, text, style}. Engine sinh tung dong roi noi lai.
Cach nay hoat dong tren MOI engine (khong phu thuoc API rieng), vi vay de
"lap rap them engine khac" ma khong phai sua render.
"""

from __future__ import annotations

import os
from typing import Callable, List

import numpy as np

from . import audio
from .config import CFG

# Ham sinh am chuan hoa: (text, voice_or_ref, style) -> (wav, sr)
SynthFn = Callable[[str, str, str], tuple]


def parse_script(text: str) -> List[dict]:
    """Phân tích kịch bản: mỗi dòng 'Giọng: nội dung' (hoặc 'Giọng | nội dung')."""
    lines: List[dict] = []
    for raw in (text or "").splitlines():
        raw = raw.strip()
        if not raw or raw.startswith("//"):
            continue
        if ":" in raw:
            v, tx = raw.split(":", 1)
        elif "|" in raw:
            v, tx = raw.split("|", 1)
        else:
            v, tx = "", raw
        lines.append({"voice": v.strip(), "text": tx.strip(), "style": ""})
    return lines


def render_lines(synth: SynthFn, lines: List[dict], gap_seconds: float = 0.25) -> tuple:
    """Sinh tung dong va noi lai -> (wav, sr). lines = [{voice, text, style}]."""
    wavs: List[np.ndarray] = []
    sr = CFG["sample_rate"]
    for i, ln in enumerate(lines):
        text = (ln.get("text") or "").strip()
        if not text:
            continue
        w, s = synth(text, ln.get("voice", ""), ln.get("style", ""))
        w = audio.normalize(audio.to_mono(np.asarray(w, dtype=np.float32)))
        sr = int(s)
        wavs.append(w)
    merged = audio.concat(wavs, sr, gap_seconds=gap_seconds)
    return merged, sr


def save_script(out_dir: str, name: str, wav: np.ndarray, sr: int,
                per_line: bool = False, wavs: List[np.ndarray] | None = None) -> List[str]:
    """Luu script da render thanh wav. Tra ve danh sach duong dan file."""
    os.makedirs(out_dir, exist_ok=True)
    paths = []
    main = os.path.join(out_dir, f"{name}.wav")
    audio.save_wav(main, wav, sr)
    paths.append(main)
    if per_line and wavs:
        sub = os.path.join(out_dir, f"{name}_lines")
        os.makedirs(sub, exist_ok=True)
        for i, w in enumerate(wavs, 1):
            p = os.path.join(sub, f"{i:03d}.wav")
            audio.save_wav(p, w, sr)
            paths.append(p)
    return paths
