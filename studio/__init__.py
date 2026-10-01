"""Voice Studio AI - core package (Kaggle GPU).

Modules
-------
config    CFG + ensure_dirs
audio     load/save/resample/mono/trim/loudness
voicebank VoiceProfile + bank save/load/export/import  (TRU COT - luu giong vuot session)
render    multi-speaker script -> audio
export    wav/mp3/zip packaging
engines   engine registry (vieneu / voxcpm) + hot-swap loader
"""

from . import audio, config, export, render, voicebank
from .config import CFG, ensure_dirs

# engines: import nhe (torch/vieneu/voxcpm chi nap khi load()).
from . import engines  # noqa: E402

# app: keo theo gradio -> chi import khi can (trong notebook/app).

__all__ = [
    "config",
    "audio",
    "voicebank",
    "render",
    "export",
    "engines",
    "CFG",
    "ensure_dirs",
]
