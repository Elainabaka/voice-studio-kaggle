"""Sinh am theo giong (preset / clone / design) — dung chung cho app va kernel offload.

Tach khoi app.py de kernel batch (kaggle_offload) dung duoc ma khong keo theo gradio.
"""
from __future__ import annotations

from typing import Tuple

import numpy as np

from .config import CFG
from .engines import get_engine
from .voicebank import list_profiles, load_profile

WavOut = Tuple[np.ndarray, int]


def synth_text(text: str, mode: str, desc: str, ref_path: str, ref_text: str,
               preset: str, style: str) -> WavOut:
    """Sinh âm theo chế độ trực tiếp. Voice Design luôn chạy bằng VoxCPM2."""
    eng = get_engine("voxcpm") if mode == "design" else get_engine()
    return eng.tts(
        text,
        desc=desc if mode == "design" else "",
        ref_path=ref_path if mode == "clone" else "",
        ref_text=ref_text,
        preset=preset if mode == "preset" else "",
        style=style,
    )


def synth_profile(profile_name: str, text: str, style: str = "") -> WavOut:
    """Sinh âm bằng một giọng đã lưu trong kho (hoặc giọng mẫu)."""
    bank = list_profiles(CFG["bank_dir"])
    if profile_name not in bank:
        return synth_text(text, "preset", "", "", "", profile_name, style)  # giọng mẫu
    eng = get_engine()
    prof = load_profile(CFG["bank_dir"], profile_name)
    st = style or prof.style
    if prof.kind == "design":
        return eng.tts(text, desc=prof.desc, style=st)
    if prof.kind == "clone":
        return eng.tts(text, ref_path=prof.ref_audio, ref_text=prof.ref_text, style=st)
    return eng.tts(text, preset=prof.name, style=st)
