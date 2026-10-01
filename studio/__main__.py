"""Entry local: `python -m studio` -> mo Voice Studio tren may (localhost)."""
from __future__ import annotations

import os


def detect_device() -> str:
    if os.environ.get("VOICE_STUDIO_DEVICE"):
        return os.environ["VOICE_STUDIO_DEVICE"]
    try:
        import torch
        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"


def main() -> None:
    from .config import CFG, ensure_dirs
    CFG["device"] = detect_device()
    ensure_dirs()
    from .app import build_ui
    print(f"Voice Studio | device: {CFG['device']} | data: {CFG['work_dir']}")
    demo = build_ui()
    # share=False: chay local (nhanh, khong di qua gradio.live). inbrowser: tu mo trinh duyet.
    demo.launch(share=False, inbrowser=True, show_error=False)


if __name__ == "__main__":
    main()
