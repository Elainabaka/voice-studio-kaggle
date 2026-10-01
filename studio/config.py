"""Cau hinh trung tam Voice Studio AI. Doi CFG o day la doi toan bo notebook."""

from __future__ import annotations

import os

# --- Duong dan lam viec ----------------------------------------------------
# Mac dinh (local): ~/.voice_studio. Tren Kaggle, dat env VOICE_STUDIO_WORK
# = /kaggle/working/voice_studio TRUOC khi import studio.
WORK = os.environ.get("VOICE_STUDIO_WORK", os.path.expanduser("~/.voice_studio"))

CFG: dict = {
    # Engine dang su dung: "vieneu" | "voxcpm" | "chatterbox" | ...
    "engine": "vieneu",
    # "cuda" | "cpu". Entry local tu do GPU; doi bang env VOICE_STUDIO_DEVICE.
    "device": os.environ.get("VOICE_STUDIO_DEVICE", "cuda"),
    # Thu muc goc chua moi san pham.
    "work_dir": WORK,
    # Kho giong (voice bank) - TRU COT: vuot qua session Kaggle.
    "bank_dir": os.path.join(WORK, "voicebank"),
    # San pham sinh ra (wav/mp3).
    "out_dir": os.path.join(WORK, "output"),
    # Tham so sinh am mac dinh.
    "sample_rate": 48000,
    "cfg_value": 2.0,            # classifier-free guidance (VoxCPM2)
    "inference_timesteps": 10,   # CFM steps (VoxCPM2)
    "default_style": "doc_truyen",  # VieNeu: natural | news | doc_truyen
    # Gioi han do dai file de ghi ra (giay) - bao ve session.
    "max_seconds": 600,
}

# Ten engine hien thi + ghi chu.
ENGINE_INFO: dict = {
    "vieneu": {
        "label": "VieNeu v3-Turbo (Tieng Viet)",
        "license": "Apache-2.0",
        "source": "https://github.com/pnnbao97/VieNeu-TTS",
        "notes": "23 giong 3 mien, clone 3-8s, emotion cues, podcast mode, 48kHz, ~1GB VRAM",
    },
    "voxcpm": {
        "label": "VoxCPM2 (Design + Clone + Style)",
        "license": "Apache-2.0",
        "source": "https://github.com/OpenBMB/VoxCPM",
        "notes": "Voice Design tu mo ta, Controllable/Ultimate Cloning, 30 ngon ngu (co TV), 48kHz, ~8GB VRAM",
    },
    "chatterbox": {
        "label": "Chatterbox Multilingual (roadmap)",
        "license": "MIT",
        "source": "https://github.com/resemble-ai/chatterbox",
        "notes": "23 ngon ngu, zero-shot clone, paralinguistic tags - dang roadmap",
    },
}


def ensure_dirs() -> None:
    """Tao thu muc lam viec. Goi 1 lan khi khoi dong."""
    for k in ("work_dir", "bank_dir", "out_dir"):
        os.makedirs(CFG[k], exist_ok=True)
