"""Test UI: thuc thi body cua @gr.render (ca VI lan EN) de bat loi constructor / KeyError.

Cach lam: thay `gr.render` bang decorator goi ngay fn(), va thay `get_engine` bang
dummy (khong nap model). Gia lap render truc tiep trong Blocks context.
"""

import sys
import warnings

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
warnings.filterwarnings("ignore")
sys.path.insert(0, ".")

import gradio as gr


class DummyEngine:
    name = "dummy"

    def list_presets(self):
        return ["Giọng Mẫu A", "Giọng Mẫu B"]


def _fake_get_engine(name=None, swap=True):
    return DummyEngine()


def _immediate_render(*a, **k):
    def deco(fn):
        fn("🇻🇳 Tiếng Việt")   # render tieng Viet
        fn("🇺🇸 English")      # render tieng Anh
        return fn
    return deco


gr.render = _immediate_render          # ep goi ngay thay vi dang ky callback

import studio.app as A                  # noqa: E402

A.get_engine = _fake_get_engine         # khong nap model

ok = True
for lang in ["vi", "en"]:
    missing = [k for k in ("app_title", "tab_design", "tab_clone", "tab_bank",
                           "tab_script", "tab_export", "tab_guide", "guide", "footer")
               if not A.TR[lang].get(k)]
    if missing:
        ok = False
        print(f"TR[{lang}] thieu key: {missing}")

try:
    demo = A.build_ui()                 # se thuc thi render body x2 (VI+EN)
    print("BUILD UI (render VI+EN) OK:", type(demo).__name__)
except Exception as e:
    ok = False
    print("BUILD UI ERROR:", type(e).__name__, e)

print("UI TEST", "PASS" if ok else "FAIL")
sys.exit(0 if ok else 1)
