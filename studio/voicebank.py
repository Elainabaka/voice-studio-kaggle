"""VOICE BANK - TRU COT cua Voice Studio.

Kaggle session la tam thoi (het session = mat file). Vi vay moi giong duoc
thiet ke/clone deu duoc luu thanh *voice profile* va export ra .zip de tai ve,
import lai khi mo session moi. Day la tinh nang "tuong lai khong the thieu"
ma hau het notebook TTS tren Kaggle bo qua.

Mot profile luu: ten, loai (preset|clone|design), engine, file mau (neu clone),
mo ta (neu design), transcript mau, style mac dinh.
"""

from __future__ import annotations

import json
import os
import shutil
import zipfile
from dataclasses import asdict, dataclass, field
from typing import List, Optional

META_VERSION = 1


@dataclass
class VoiceProfile:
    name: str
    kind: str                       # "preset" | "clone" | "design"
    engine: str                     # "vieneu" | "voxcpm" | ...
    ref_audio: str = ""             # duong dan file mau (clone) - luu ben trong bank
    ref_text: str = ""              # transcript cua file mau (ultimate cloning)
    desc: str = ""                  # mo ta giong (design)
    style: str = ""                 # style mac dinh (natural|news|doc_truyen|...)
    meta: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)


def _profile_dir(bank_dir: str, name: str) -> str:
    safe = "".join(c if c.isalnum() or c in "-_ " else "_" for c in name).strip() or "voice"
    return os.path.join(bank_dir, safe)


def save_profile(bank_dir: str, profile: VoiceProfile, ref_src: Optional[str] = None) -> str:
    """Luu profile + copy file mau (neu co) vao bank. Tra ve thu muc profile."""
    pdir = _profile_dir(bank_dir, profile.name)
    os.makedirs(pdir, exist_ok=True)

    if ref_src and os.path.exists(ref_src):
        ext = os.path.splitext(ref_src)[1] or ".wav"
        dst = os.path.join(pdir, "ref" + ext)
        shutil.copy2(ref_src, dst)
        profile.ref_audio = "ref" + ext  # duong dan tuong doi ben trong profile

    with open(os.path.join(pdir, "profile.json"), "w", encoding="utf-8") as f:
        json.dump(profile.to_dict(), f, ensure_ascii=False, indent=2)
    return pdir


def load_profile(bank_dir: str, name: str) -> VoiceProfile:
    pdir = _profile_dir(bank_dir, name)
    with open(os.path.join(pdir, "profile.json"), encoding="utf-8") as f:
        d = json.load(f)
    prof = VoiceProfile(**{k: d.get(k, getattr(VoiceProfile, k, "")) for k in d})
    if prof.ref_audio and not os.path.isabs(prof.ref_audio):
        prof.ref_audio = os.path.join(pdir, prof.ref_audio)
    return prof


def list_profiles(bank_dir: str) -> List[str]:
    if not os.path.isdir(bank_dir):
        return []
    out = []
    for entry in sorted(os.listdir(bank_dir)):
        if os.path.isfile(os.path.join(bank_dir, entry, "profile.json")):
            out.append(entry)
    return out


def delete_profile(bank_dir: str, name: str) -> bool:
    pdir = _profile_dir(bank_dir, name)
    if os.path.isdir(pdir):
        shutil.rmtree(pdir)
        return True
    return False


def export_bank(bank_dir: str, zip_path: str) -> str:
    """Dong toan bo kho giong thanh .zip de tai ve vuot qua session."""
    os.makedirs(os.path.dirname(zip_path) or ".", exist_ok=True)
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as z:
        for name in list_profiles(bank_dir):
            pdir = _profile_dir(bank_dir, name)
            for fn in os.listdir(pdir):
                z.write(os.path.join(pdir, fn), arcname=os.path.join(name, fn))
    return zip_path


def import_bank(zip_path: str, bank_dir: str) -> List[str]:
    """Nap .zip kho giong vao bank. Tra ve danh sach ten da import."""
    imported = []
    with zipfile.ZipFile(zip_path, "r") as z:
        z.extractall(bank_dir)
        for info in z.infolist():
            top = info.filename.split("/")[0]
            if top and top not in imported:
                imported.append(top)
    return [p for p in imported if os.path.isfile(os.path.join(bank_dir, p, "profile.json"))]
