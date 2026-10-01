"""Engine TTS + hot-swap.

Nguyen tac: chi nap DUNG MOT engine vao VRAM moi luc (hot-swap) de khong OOM
tren Kaggle T4. Muon them engine moi -> ke thua BaseEngine va dang ky vao REGISTRY
("lap rap duoc them cai khac vao").

Moi engine chuan hoa ve 1 ham:
    tts(text, *, desc="", ref_path="", ref_text="", preset="", style="") -> (wav, sr)
- desc     : mo ta giong (VOICE DESIGN - tao giong moi tu chu)
- ref_path : file mau (VOICE CLONE)
- ref_text : transcript cua file mau (ultimate cloning, tuy chon)
- preset   : ten giong san co (gallery)
- style    : huong dien (toc do, cam xuc, doc truyen...)
"""

from __future__ import annotations

import gc
from typing import List, Optional, Tuple

import numpy as np

from . import audio
from .config import CFG

WavOut = Tuple[np.ndarray, int]


class BaseEngine:
    name: str = "base"
    label: str = ""
    license: str = ""
    source: str = ""

    def load(self) -> None: ...
    def unload(self) -> None:
        gc.collect()
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass

    @property
    def sample_rate(self) -> int:
        return CFG["sample_rate"]

    def tts(self, text: str, *, desc: str = "", ref_path: str = "",
            ref_text: str = "", preset: str = "", style: str = "") -> WavOut:
        raise NotImplementedError

    def list_presets(self) -> List[str]:
        return []


# ---------------------------------------------------------------- VieNeu ---
class VieneuEngine(BaseEngine):
    """VieNeu-TTS v3-Turbo - tieng Viet so 1. Apache-2.0."""

    name = "vieneu"
    label = "VieNeu v3-Turbo (Tieng Viet)"
    license = "Apache-2.0"
    source = "https://github.com/pnnbao97/VieNeu-TTS"

    def __init__(self) -> None:
        self._tts = None

    def load(self) -> None:
        if self._tts is not None:
            return
        from vieneu import Vieneu  # noqa

        self._tts = Vieneu()  # mac dinh v3-Turbo, tu chuyen PyTorch tren GPU

    def unload(self) -> None:
        self._tts = None
        super().unload()

    def list_presets(self) -> List[str]:
        self.load()
        try:
            return list(self._tts.list_preset_voices())
        except Exception:
            return []

    def tts(self, text: str, *, desc: str = "", ref_path: str = "",
            ref_text: str = "", preset: str = "", style: str = "") -> WavOut:
        self.load()
        style = style or CFG["default_style"]
        # Emotion cues: gan [cuoi], [tho dai] truc tiep vao text la duoc (v3-Turbo).
        try:
            if ref_path:
                wav = self._tts.infer(text, ref_audio=ref_path, denoise=True, style=style)
            elif preset:
                wav = self._tts.infer(text, voice=preset, style=style)
            else:
                wav = self._tts.infer(text, style=style)  # giong mac dinh
        except TypeError:
            # Phien ban SDK khong nhan style= -> goi lai thuan.
            if ref_path:
                wav = self._tts.infer(text, ref_audio=ref_path, denoise=True)
            elif preset:
                wav = self._tts.infer(text, voice=preset)
            else:
                wav = self._tts.infer(text)
        return _to_mono_wav(wav), self.sample_rate


# ---------------------------------------------------------------- VoxCPM ---
class VoxCPMEngine(BaseEngine):
    """VoxCPM2 - Design + Controllable/Ultimate Cloning + Style. Apache-2.0."""

    name = "voxcpm"
    label = "VoxCPM2 (Design + Clone + Style)"
    license = "Apache-2.0"
    source = "https://github.com/OpenBMB/VoxCPM"

    def __init__(self) -> None:
        self._model = None

    def load(self) -> None:
        if self._model is not None:
            return
        from voxcpm import VoxCPM  # noqa

        self._model = VoxCPM.from_pretrained("openbmb/VoxCPM2", load_denoiser=False)

    def unload(self) -> None:
        self._model = None
        super().unload()

    def tts(self, text: str, *, desc: str = "", ref_path: str = "",
            ref_text: str = "", preset: str = "", style: str = "") -> WavOut:
        self.load()
        kw = dict(cfg_value=CFG["cfg_value"], inference_timesteps=CFG["inference_timesteps"])
        if desc and ref_path:
            # Controllable cloning: ref quyet dinh "ai noi", (style+desc) quyet dinh "noi the nao".
            prompt = f"({desc}) {text}" if desc else text
            if style:
                prompt = f"({style}) {prompt}"
            wav = self._model.generate(text=prompt, reference_wav_path=ref_path, **kw)
        elif desc:
            # Voice design: tao giong moi tu mo ta, khong can ref.
            wav = self._model.generate(text=f"({desc}) {text}", **kw)
        elif ref_path and ref_text:
            # Ultimate cloning: ref + transcript -> trung thuc nhat.
            wav = self._model.generate(text=text, prompt_wav_path=ref_path, prompt_text=ref_text, **kw)
        elif ref_path:
            # Reference-only cloning.
            prompt = f"({style}) {text}" if style else text
            wav = self._model.generate(text=prompt, reference_wav_path=ref_path, **kw)
        else:
            wav = self._model.generate(text=text, **kw)
        sr = getattr(self._model.tts_model, "sample_rate", self.sample_rate)
        return _to_mono_wav(wav), int(sr)


def _to_mono_wav(wav) -> np.ndarray:
    if isinstance(wav, (list, tuple)):
        wav = wav[0]
    arr = np.asarray(wav, dtype=np.float32)
    return audio.normalize(audio.to_mono(arr))


# ---------------------------------------------------------------- Registry --
REGISTRY = {
    "vieneu": VieneuEngine,
    "voxcpm": VoxCPMEngine,
}

_CURRENT: Optional[BaseEngine] = None


def available_engines() -> List[str]:
    return list(REGISTRY.keys())


def get_engine(name: Optional[str] = None, swap: bool = True) -> BaseEngine:
    """Lay engine (mac dinh: engine dang chon). swap=True -> thay engine cu, giai phong VRAM."""
    global _CURRENT
    name = name or CFG["engine"]
    if name not in REGISTRY:
        raise ValueError(f"Engine '{name}' chua duoc dang ky. Co: {available_engines()}")
    if _CURRENT is not None and _CURRENT.name == name and not swap:
        return _CURRENT
    if swap and _CURRENT is not None and _CURRENT.name != name:
        _CURRENT.unload()
        _CURRENT = None
    if _CURRENT is None:
        _CURRENT = REGISTRY[name]()
        _CURRENT.load()
    return _CURRENT
