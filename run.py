"""One-click launcher for Voice Studio AI (local).

Usage: double-click run.bat (Win) / run.sh (Mac-Linux), or: python run.py
First run creates .venv + installs deps (pulls torch/model, slow). Later runs open fast.
"""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
VENV = ROOT / ".venv"
REQ = ROOT / "requirements.txt"
MARK = VENV / ".deps_installed"


def venv_python() -> Path:
    return VENV / ("Scripts/python.exe" if os.name == "nt" else "bin/python")


def ensure_env() -> Path:
    py = venv_python()
    if not py.exists():
        print("-> Tao venv moi ...")
        subprocess.run([sys.executable, "-m", "venv", str(VENV)], check=True)
        subprocess.run([str(py), "-m", "pip", "install", "-q", "--upgrade", "pip"], check=True)
    need = (not MARK.exists()) or (REQ.stat().st_mtime > MARK.stat().st_mtime)
    if need:
        print("-> Cai thu vien (lan dau keo torch + model, vui long doi) ...")
        subprocess.run([str(py), "-m", "pip", "install", "-q", "-r", str(REQ)], check=True)
        MARK.touch()
    return py


def main() -> None:
    py = ensure_env()
    os.chdir(ROOT)
    print("-> Mo Voice Studio (trinh duyet se tu mo) ...")
    raise SystemExit(subprocess.call([str(py), "-m", "studio"]))


if __name__ == "__main__":
    main()
