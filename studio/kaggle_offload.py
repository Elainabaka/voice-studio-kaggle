"""Offload render lên Kaggle GPU (batch) qua Kaggle CLI.

Kaggle CLI là BATCH: đóng gói job thành kernel notebook → `kaggle kernels push`
(chạy trên GPU) → poll `kaggle kernels status` → tải `kaggle kernels output`.
Không phải GPU tương tác — hợp để render audiobook/kịch bản nặng khi máy không
có GPU local. Muốn gõ chữ nghe liền thì dùng chế độ Local.
"""
from __future__ import annotations

import base64
import io
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Optional

STUDIO_DIR = Path(__file__).resolve().parent


# ------------------------------------------------------------------ CLI utils --
def _kaggle_exe() -> Optional[str]:
    from shutil import which
    return which("kaggle")


def _kaggle(*args: str, timeout: int = 600) -> str:
    exe = _kaggle_exe()
    if not exe:
        raise RuntimeError("Chưa cài Kaggle CLI. Chạy `pip install kaggle` hoặc bấm nút 'Cài Kaggle CLI'.")
    try:
        r = subprocess.run([exe, *args], capture_output=True, text=True, timeout=timeout)
    except FileNotFoundError:
        raise RuntimeError("Không tìm thấy lệnh `kaggle`. Hãy cài: pip install kaggle")
    if r.returncode != 0:
        raise RuntimeError(f"`kaggle {' '.join(args)}` lỗi:\n{(r.stderr or r.stdout).strip()}")
    return r.stdout


def install_cli() -> str:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "kaggle"])
    return "Đã cài Kaggle CLI."


def check_setup() -> dict:
    """Kiểm tra Kaggle CLI + đăng nhập. Trả về dict để hiển thị."""
    info = {"cli": bool(_kaggle_exe()), "auth": False, "user": "", "msg": ""}
    if not info["cli"]:
        info["msg"] = "Chưa cài Kaggle CLI. Chạy `pip install kaggle` hoặc bấm nút 'Cài Kaggle CLI'."
        return info
    kaggle_json = Path.home() / ".kaggle" / "kaggle.json"
    if kaggle_json.exists() or os.environ.get("KAGGLE_KEY"):
        info["auth"] = True
        try:
            out = _kaggle("config", "view", timeout=30)
            for ln in out.splitlines():
                if "username" in ln.lower():
                    info["user"] = ln.split(":")[-1].strip()
                    break
        except Exception:
            pass
        info["msg"] = f"Sẵn sàng. Người dùng: {info['user'] or '(không rõ)'}."
    else:
        info["msg"] = ("Chưa đăng nhập Kaggle. Chạy `kaggle auth login` (mở trình duyệt) "
                       "hoặc đặt ~/.kaggle/kaggle.json.")
    return info


# ------------------------------------------------------------------ packaging --
def _embed_studio() -> dict:
    """Đọc toàn bộ studio/*.py → base64 để nhúng vào kernel."""
    files = {}
    for p in sorted(STUDIO_DIR.glob("*.py")):
        files["studio/" + p.name] = base64.b64encode(p.read_bytes()).decode("ascii")
    return files


def _voicebank_b64(bank_dir: Optional[str]) -> str:
    """Nén kho giọng (dùng export_bank để đúng format import_bank)."""
    if not bank_dir or not os.path.isdir(bank_dir) or not os.listdir(bank_dir):
        return ""
    from .voicebank import export_bank
    with tempfile.TemporaryDirectory() as td:
        zp = os.path.join(td, "voicebank.zip")
        export_bank(bank_dir, zp)
        return base64.b64encode(Path(zp).read_bytes()).decode("ascii")


def _md(text: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": text.splitlines(keepends=True)}


def _code(text: str) -> dict:
    return {"cell_type": "code", "metadata": {}, "execution_count": None,
            "outputs": [], "source": text.splitlines(keepends=True)}


def _kernel_notebook(files: dict, job_b64: str) -> dict:
    files_json = json.dumps(files, indent=0)
    cells = [
        _md("# Voice Studio — offload render (batch) trên Kaggle GPU\n\n"
            "Kernel này được tạo tự động từ Voice Studio local. Chạy trên GPU, "
            "xuất WAV vào `/kaggle/working/output`."),
        _code(f"""# 1 · JOB + cài thư viện
import base64, json, subprocess, sys
JOB = json.loads(base64.b64decode(r'''{job_b64}''').decode("utf-8"))

def pip(*pkgs):
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", *pkgs])

pip("soundfile", "librosa")
pip("voxcpm" if JOB.get("engine") == "voxcpm" else "vieneu")
print("Đã cài. Engine:", JOB.get("engine"))
"""),
        _code(f"""# 2 · unpack studio + kho giọng
import base64, os, sys, json, tempfile
PKG = "/kaggle/working/voice_studio_pkg"
os.makedirs(PKG, exist_ok=True)
_FILES = json.loads(r'''{files_json}''')
for _rel, _b64 in _FILES.items():
    _path = os.path.join(PKG, _rel)
    os.makedirs(os.path.dirname(_path), exist_ok=True)
    with open(_path, "wb") as f:
        f.write(base64.b64decode(_b64))
sys.path.insert(0, PKG)
os.environ["VOICE_STUDIO_WORK"] = "/kaggle/working/voice_studio"

from studio.config import CFG, ensure_dirs
from studio.engines import get_engine
from studio.synth import synth_profile
from studio.render import render_lines, save_script, parse_script
from studio import audio
CFG["engine"] = JOB.get("engine", "vieneu")
ensure_dirs()
if JOB.get("voicebank_b64"):
    from studio.voicebank import import_bank
    zp = os.path.join(tempfile.gettempdir(), "vb.zip")
    with open(zp, "wb") as f:
        f.write(base64.b64decode(JOB["voicebank_b64"]))
    import_bank(zp, CFG["bank_dir"])
print("Studio sẵn sàng. Bank:", os.listdir(CFG["bank_dir"]))
"""),
        _code("""# 3 · render trên GPU
get_engine(CFG["engine"])   # load model vào VRAM

def _synth(text, voice, style):
    return synth_profile(voice, text, style)

task = JOB.get("task", "script")
if task == "script":
    lines = parse_script(JOB.get("script", ""))
    if not lines:
        raise SystemExit("Script rỗng.")
    wav, sr = render_lines(_synth, lines, gap_seconds=float(JOB.get("gap", 0.25)))
    paths = save_script(CFG["out_dir"], JOB.get("out_name", "audio"), wav, sr)
else:
    wav, sr = _synth(JOB.get("text", ""), JOB.get("voice", ""), JOB.get("style", ""))
    p = os.path.join(CFG["out_dir"], JOB.get("out_name", "audio") + ".wav")
    audio.save_wav(p, wav, sr)
    paths = [p]
print("DONE:", paths)
"""),
    ]
    return {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.10.12"},
            "kaggle": {"accelerator": "gpu", "isGpuEnabled": True, "isInternetEnabled": True,
                       "language": "python", "sourceType": "notebook"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


def build_kernel(job: dict, bank_dir: Optional[str], out_dir: str) -> str:
    """Đóng gói job thành kernel folder (kernel-metadata.json + notebook)."""
    os.makedirs(out_dir, exist_ok=True)
    files = _embed_studio()
    job = dict(job)
    vb = _voicebank_b64(bank_dir) if bank_dir else ""
    if vb:
        job["voicebank_b64"] = vb
    job_b64 = base64.b64encode(json.dumps(job, ensure_ascii=False).encode("utf-8")).decode("ascii")
    nb = _kernel_notebook(files, job_b64)
    with open(os.path.join(out_dir, "voice_studio_offload.ipynb"), "w", encoding="utf-8") as f:
        json.dump(nb, f, ensure_ascii=False, indent=1)
    meta = {
        "id": job.get("slug", "voice-studio-offload"),
        "title": "Voice Studio Offload",
        "code_file": "voice_studio_offload.ipynb",
        "language": "python",
        "kernel_type": "notebook",
        "is_private": True,
        "enable_gpu": True,
        "enable_internet": True,
        "keywords": ["tts", "voice-studio", "offload"],
    }
    with open(os.path.join(out_dir, "kernel-metadata.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    return out_dir


# ------------------------------------------------------------------ run --
def push(kernel_dir: str, accelerator: str = "NvidiaTeslaT4", timeout: int = 600) -> str:
    return _kaggle("kernels", "push", "-p", kernel_dir,
                   "--accelerator", accelerator, timeout=timeout)


def status(kernel_ref: str) -> str:
    return _kaggle("kernels", "status", kernel_ref, timeout=60)


def download(kernel_ref: str, dest: str, timeout: int = 300) -> str:
    os.makedirs(dest, exist_ok=True)
    return _kaggle("kernels", "output", kernel_ref, "-p", dest, "-o", timeout=timeout)
