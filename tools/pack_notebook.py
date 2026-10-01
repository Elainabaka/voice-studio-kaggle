"""Dong goi source `studio/` vao notebook Kaggle TU CHUA.

Tinh nang giong `mimo-v26-kaggle-tpu`: nguon goc nam o `studio/*.py` (de bao tri,
test), notebook sinh ra se nhung toan bo nguon dang base64 vao 1 cell, ghi ra dia
roi import. Vi vay moi nguoi copy&edit notebook la chay duoc, khong can git clone.

Chay lai moi khi sua `studio/`:
    python tools/pack_notebook.py
"""

from __future__ import annotations

import base64
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
STUDIO = os.path.join(ROOT, "studio")
OUT = os.path.join(ROOT, "notebook", "Voice_Studio_AI_Kaggle_GPU.ipynb")

PKG_FILES = [
    "__init__.py", "config.py", "audio.py", "voicebank.py",
    "render.py", "export.py", "engines.py", "synth.py", "app.py",
    "kaggle_offload.py",
]


def load_sources() -> dict:
    files = {}
    for fn in PKG_FILES:
        path = os.path.join(STUDIO, fn)
        with open(path, "rb") as f:
            files["studio/" + fn] = base64.b64encode(f.read()).decode("ascii")
    return files


def md(text: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": text.splitlines(keepends=True)}


def code(text: str) -> dict:
    return {"cell_type": "code", "metadata": {}, "execution_count": None,
            "outputs": [], "source": text.splitlines(keepends=True)}


def build() -> dict:
    files = load_sources()
    files_json = json.dumps(files, indent=0)

    cells = [
        md("""# Voice Studio AI — Kaggle GPU

**Tạo và sở hữu giọng nói bằng AI trên GPU miễn phí của Kaggle.** Không chỉ "gõ chữ → ra tiếng":
đây là một **studio thu nhỏ** để *thiết kế* và *nhân bản* giọng nói, lưu kho giọng tái sử dụng,
dựng podcast/audiobook nhiều giọng — rồi xuất file.

| Tab | Làm được gì |
|---|---|
| 🎨 **Thiết kế giọng** | Chọn giới tính / tuổi / vùng miền / chất giọng / cảm xúc → tạo giọng **chưa từng tồn tại** |
| 🎙️ **Nhân bản giọng** | Tải clip 5–15 giây → nhân bản giọng + điều khiển phong cách |
| 🖼️ **Thư viện giọng** | Giọng mẫu + giọng của bạn · **export/import .zip giữ giọng qua session** |
| 🎧 **Kịch bản** | Kịch bản nhiều giọng → podcast / audiobook |
| 📦 **Xuất file** | Tải WAV/MP3 + nén zip |
| 📖 **Hướng dẫn** | Bảng hướng dẫn chi tiết |

**Cách dùng:** chọn *Accelerator = GPU (T4 x2)*, *Internet = ON* → **Run All** →
giao diện **tự mở** trong cell cuối. Đổi ngôn ngữ **🇻🇳 / 🇺🇸** ở góc trên giao diện.

**Engine:** VieNeu v3-Turbo (tiếng Việt) hoặc VoxCPM2 (Design + Clone + Style, 30 ngôn ngữ).
Đổi engine ở cell cấu hình. Cả hai **Apache-2.0**. *Voice Design chạy bằng VoxCPM2.*

> **Author: Elainabaka** © 2026 — thiết kế & dựng notebook Voice Studio AI cho cộng đồng.
> Code MIT · model giữ license Apache-2.0 của tác giả gốc. Xem mục Credit ở cuối notebook.
> **Chỉ clone giọng bạn có quyền sử dụng.**
"""),

        md("""## 🧭 Chỉ đường

1. **Cấu hình & GPU** — đổi `ENGINE` nếu muốn (`vieneu` = tiếng Việt, `voxcpm` = Design + 30 ngôn ngữ).
2. **Cài thư viện** — chỉ cài engine đang chọn (tránh xung đột dependency).
3. **Nạp Voice Studio** — giải nén gói `studio/` ra đĩa rồi import.
4. **Nạp model + test nhanh** — hot-swap, chỉ 1 model vào VRAM.
5. **Gradio Studio** — giao diện **tự mở** ngay trong cell này (đổi ngôn ngữ 🇻🇳/🇺🇸 ở góc trên).
"""),

        code("""# ===== 1 · CẤU HÌNH & KIỂM TRA GPU =====
ENGINE = "vieneu"   # "vieneu" (tiếng Việt) | "voxcpm" (Design+Clone+Style, 30 ngôn ngữ)

import torch
assert torch.cuda.is_available(), "Bật Accelerator = GPU (T4 x2) trong Settings rồi chạy lại!"
print(f"GPU: {torch.cuda.get_device_name(0)} | VRAM: {torch.cuda.get_device_properties(0).total_memory/1e9:.1f} GB")
print(f"Engine đã chọn: {ENGINE}")
"""),

        code("""# ===== 2 · CÀI THƯ VIỆN (chỉ engine đang chọn) =====
import subprocess, sys

def pip(*pkgs):
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", *pkgs])

pip("gradio>=4.44", "soundfile", "librosa")
if ENGINE == "vieneu":
    pip("vieneu")                       # VieNeu v3-Turbo (Apache-2.0)
elif ENGINE == "voxcpm":
    pip("voxcpm")                       # VoxCPM2 (Apache-2.0)
    try:                                # flash-attn = tăng tốc (tùy chọn), lỗi cũng không sao
        pip("flash-attn==2.7.4.post1", "--no-build-isolation")
    except Exception as e:
        print("flash-attn bỏ qua (không bắt buộc):", e)
print("Cài xong.")
"""),

        code(f"""# ===== 3 · NẠP VOICE STUDIO (giải nén gói studio/ ra đĩa) =====
import base64, os, sys, json

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

from studio import CFG, ensure_dirs
from studio.engines import available_engines, get_engine
from studio.config import ENGINE_INFO
ensure_dirs()
CFG["engine"] = ENGINE
print("Voice Studio đã nạp. Engine có sẵn:", available_engines())
"""),

        code("""# ===== 4 · NẠP MODEL + TEST NHANH =====
eng = get_engine(ENGINE)          # hot-swap: chỉ 1 model vào VRAM
print(f"Engine: {eng.label} | {eng.sample_rate} Hz | license: {eng.license}")
print("Gọng preset:", (eng.list_presets() or ["(engine tự chọn giọng mặc định)"])[:8])

# Test nhanh 1 câu (đổi text nếu muốn)
try:
    wav, sr = eng.tts("Xin chào, đây là Voice Studio AI trên Kaggle.",
                      desc="giọng nữ miền Bắc, ấm, tự nhiên" if ENGINE == "voxcpm" else "",
                      preset=(eng.list_presets() or [""])[0] if ENGINE == "vieneu" else "")
    from studio import audio
    audio.save_wav("/kaggle/working/voice_studio/output/smoke.wav", wav, sr)
    print("✅ Test OK — đã ghi output/smoke.wav")
except Exception as e:
    print("⚠️ Test lỗi (có thể do model chưa tải xong):", e)
"""),

        code("""# ===== 5 · GRADIO STUDIO — TỰ MỞ giao diện ngay trong cell =====
from studio.app import build_ui
from IPython.display import HTML, display
import time

demo = build_ui()
demo.launch(share=True, prevent_thread_lock=True, quiet=True, show_error=False)
time.sleep(0.5)
url = demo.share_url or demo.server_url
print("🎙️ Voice Studio AI đang chạy:", url)

# Tự hiện giao diện bên dưới (iframe) + tự mở tab mới + link dự phòng.
display(HTML(f'''
<div style="font-family:system-ui,sans-serif; border:2px solid #6366f1; border-radius:12px; overflow:hidden; background:#eef2ff">
  <div style="background:#312e81; color:#fff; padding:10px 14px; font-weight:700">
    🎙️ Voice Studio AI — giao diện đã sẵn sàng (tự hiện bên dưới)
  </div>
  <iframe src="{url}?embed=true" style="width:100%; height:760px; border:0; display:block; background:#fff"></iframe>
  <div style="padding:8px 14px">
    <a href="{url}" target="_blank" style="color:#4f46e5; font-weight:700">↗️ Mở trong tab mới</a>
    <span style="color:#64748b; font-size:13px"> · đổi ngôn ngữ 🇻🇳/🇺🇸 ở góc trên giao diện</span>
  </div>
</div>
<script>try{{window.open("{url}","_blank")}}catch(e){{}}</script>
'''))
"""),

        code("""# (Tuỳ chọn) Dùng API Python trực tiếp — không qua giao diện:
# eng = get_engine()
# wav, sr = eng.tts("Xin chào", preset="Minh Quân")           # giọng mẫu
# audio.save_wav("/kaggle/working/voice_studio/output/demo.wav", wav, sr)
"""),

        md("""## 📥 Giữ kho giọng qua session (Voice Bank)

Session Kaggle **tạm thời** — hết phiên là mất file. Muốn giữ giọng đã thiết kế/clone:

1. Tab 🖼️ **Thư viện giọng** → bấm **Xuất kho giọng → .zip** → tải `voice_bank.zip` về máy.
2. Lần chạy sau (session mới, Run All) → tab 🖼️ **Thư viện giọng** → **Nhập .zip vào kho**.

Đây là điểm khác biệt lớn nhất so với các notebook TTS thông thường: giọng nói của bạn
trở thành **tài sản tái sử dụng**, không mất theo phiên.

**Tip clone chất lượng cao:** clip 5–15 giây, thu sát mic, phòng yên, **không nhạc**,
một người nói, đọc đúng tông bạn muốn đầu ra (vì model copy cả *cách nói*, không chỉ chất giọng).
"""),

        md("""## ⚠️ Đạo đức, consent & giấy phép

- **Chỉ clone giọng khi bạn có quyền** (giọng của chính bạn hoặc được phép rõ ràng).
- **Voice Design** tạo giọng *tổng hợp* từ mô tả — không sao chép ai — là lựa chọn an toàn
  về pháp lý và đạo đức nếu bạn cần giọng nói mới.
- Không dùng để mạo danh, lừa đảo, hoặc tạo nội dung sai sự thật.

---

## 📄 Credit & License

**Voice Studio AI** được thiết kế và dựng bởi **Elainabaka** © 2026 —
notebook tạo giọng nói cho cộng đồng trên GPU Kaggle miễn phí.

- **Code notebook này:** MIT (dùng cá nhân & thương mại, giữ nguyên credit).
- **Dựng trên model mã nguồn mở** của tác giả gốc — cảm ơn và **giữ nguyên credit của họ**:

| Thành phần | Tác giả gốc | License |
|---|---|---|
| VieNeu-TTS (v3-Turbo) | Phạm Nguyễn Ngọc Bảo & cộng sự · github.com/pnnbao97/VieNeu-TTS | Apache-2.0 |
| VoxCPM2 (OpenBMB) | github.com/OpenBMB/VoxCPM | Apache-2.0 |
| Giao diện | Gradio | Apache-2.0 |

*Cảm ơn các tác giả gốc đã mở model. Vui lòng giữ nguyên mục credit khi chia sẻ lại.*
"""),

        md("""## 🗺️ Roadmap
- [ ] Thêm engine **Chatterbox Multilingual** (MIT, paralinguistic tags) — registry đã sẵn sàng.
- [ ] Thêm engine **Qwen3-TTS-VoiceDesign** (voice design SOTA).
- [ ] Tự động ASR transcript cho clip clone (Whisper) để tăng độ giống.
- [ ] Chấm điểm độ giống giọng (ECAPA cosine) sau mỗi lần clone.
- [ ] Lưu bank thẳng lên Kaggle Dataset (tùy chọn, cần token).
"""),
    ]

    return {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.10.12"},
            "kaggle": {
                "accelerator": "gpu",
                "isGpuEnabled": True,
                "isInternetEnabled": True,
                "language": "python",
                "sourceType": "notebook",
            },
            "colab": {"provenance": [], "gpuType": "T4"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


def main() -> None:
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    nb = build()
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(nb, f, ensure_ascii=False, indent=1)
    n_code = sum(1 for c in nb["cells"] if c["cell_type"] == "code")
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
    print(f"Wrote {os.path.basename(OUT)}")
    print(f"  {len(nb['cells'])} cells ({n_code} code) | {os.path.getsize(OUT)/1024:.1f} KB")


if __name__ == "__main__":
    main()
