"""Voice Studio — điều phối cấp cao + Gradio Studio UI.

Thiết kế bám theo workspace của VoiceStudio (debpalash/VoiceStudio):
  🎨 Thiết kế giọng (Voice Design) — chọn giới tính/tuổi/accent/cao độ/nhịp/cảm xúc/chất giọng
  🎙️ Nhân bản giọng (Voice Cloning) — clip 5–15s + điều khiển phong cách
  🖼️ Thư viện giọng (Gallery) — giọng mẫu + giọng của bạn, export/import .zip (vượt phiên)
  🎧 Kịch bản (Stories) — nhiều giọng → podcast / audiobook
  📦 Xuất file (Export) — WAV/MP3 + nén zip
  📖 Hướng dẫn (Guide)

Giao diện song ngữ 🇻🇳 Tiếng Việt / 🇺️ English — đổi bằng công tắc cờ ở góc trên.
"""

from __future__ import annotations

import os
import time
from typing import List, Tuple

import numpy as np

from . import audio, export as ex, render
from .config import CFG, ENGINE_INFO, ensure_dirs
from .engines import get_engine
from .synth import synth_text, synth_profile
from . import kaggle_offload as ko
from .voicebank import (
    VoiceProfile,
    delete_profile,
    export_bank,
    import_bank,
    list_profiles,
    load_profile,
    save_profile,
)

WavOut = Tuple[np.ndarray, int]


# ------------------------------------------------------------------ synth --
def synth_text_note():
    """Xem studio/synth.py — synth_text/synth_profile dùng chung app + kernel offload."""


def compose_desc(gender, age, accent, pitch, pace, emotion, texture, extra) -> str:
    """Ghép các thuộc tính đã chọn thành mô tả giọng tự nhiên (Voice Design)."""
    pitch_w = ["cao độ thấp", "cao độ trung bình", "cao độ cao"][int(pitch)]
    pace_w = ["nhịp chậm", "nhịp vừa", "nhịp nhanh"][int(pace)]
    parts = [gender, age, f"miền {accent}", f"chất giọng {texture}", emotion, pitch_w, pace_w]
    desc = "giọng " + ", ".join(str(p).lower() for p in parts if p)
    return (desc + ". " + extra.strip()).strip()


def _save_out(wav: np.ndarray, sr: int, stem: str, fmt: str = "wav") -> str:
    ensure_dirs()
    path = os.path.join(CFG["out_dir"], f"{stem}_{int(time.time())}.{fmt}")
    return ex.write_output(path, wav, sr, fmt=fmt)


# ------------------------------------------------------------------- i18n --
TR = {
    "vi": {
        "app_title": "## 🎙️ Voice Studio AI — Studio tạo giọng nói",
        "app_desc": ("**Tạo và sở hữu giọng nói bằng AI trên GPU Kaggle miễn phí.** "
                     "Thiết kế giọng từ mô tả, nhân bản giọng từ clip, lưu kho giọng tái "
                     "sử dụng, dựng podcast nhiều giọng — rồi xuất WAV/MP3."),
        "engine_badge": "Engine",
        "tab_design": "🎨 Thiết kế giọng",
        "tab_clone": "🎙️ Nhân bản giọng",
        "tab_bank": "🖼️ Thư viện giọng",
        "tab_script": "🎧 Kịch bản",
        "tab_export": "📦 Xuất file",
        "tab_guide": "📖 Hướng dẫn",
        # design
        "d_note": "Tạo một giọng **chưa từng tồn tại** bằng cách mô tả. Chạy bằng engine **VoxCPM2**.",
        "d_attrs": "Thuộc tính giọng nói",
        "d_gender": "Giới tính", "d_age": "Độ tuổi", "d_accent": "Vùng miền",
        "d_texture": "Chất giọng", "d_emotion": "Cảm xúc / phong cách",
        "d_pitch": "Cao độ", "d_pace": "Nhịp độ", "d_extra": "Mô tả thêm (tùy chọn)",
        "d_text": "Nội dung cần đọc", "d_name": "Tên lưu vào thư viện (bỏ trống nếu không lưu)",
        "d_fmt": "Định dạng", "d_btn": "Thiết kế giọng", "d_out": "Kết quả",
        # clone
        "c_note": "Tải clip 5–15 giây → nhân bản giọng. Clip sạch (sát mic, không nhạc) sẽ giống hơn.",
        "c_ref": "Clip mẫu", "c_ref_info": "5–15 giây · một người · không nhạc",
        "c_reftext": "Lời thoại của clip (tùy chọn)", "c_reftext_info": "Giúp giọng giống hơn (ultimate cloning)",
        "c_text": "Nội dung cần đọc", "c_style": "Phong cách (tùy chọn)",
        "c_name": "Tên lưu vào thư viện", "c_fmt": "Định dạng",
        "c_btn": "Nhân bản giọng", "c_out": "Kết quả",
        # bank
        "b_filter": "Lọc", "b_filter_all": "Tất cả", "b_filter_preset": "Giọng mẫu", "b_filter_mine": "Giọng của tôi",
        "b_pick": "Chọn giọng", "b_text": "Nội dung cần đọc", "b_style": "Phong cách",
        "b_fmt": "Định dạng", "b_prev": "Đọc thử", "b_del": "Xóa giọng", "b_out": "Kết quả",
        "b_persist": "### 💾 Giữ giọng qua phiên (session)",
        "b_persist_desc": "Kaggle xóa file khi hết phiên. Xuất kho giọng ra `.zip` để giữ lại.",
        "b_exp": "Xuất kho giọng → .zip", "b_imp": "Nhập .zip vào kho", "b_file": "Tệp voice_bank.zip",
        # script
        "s_note": "Mỗi dòng một câu: `Tên giọng: nội dung`. Dòng bắt đầu `//` bị bỏ qua. Có thể trộn giọng mẫu và giọng của bạn.",
        "s_script": "Kịch bản nhiều giọng", "s_gap": "Khoảng lặng giữa các câu (giây)",
        "s_fmt": "Định dạng", "s_btn": "Tạo bản thu", "s_out": "Kết quả",
        "s_example": ("Minh Quân: Chào mừng đến với Voice Studio AI.\n"
                      "Ngọc Trân: Đây là kịch bản nhiều giọng, mỗi dòng một câu.\n"
                      "Minh Quân: Bấm nút để xuất thành một tệp âm thanh duy nhất."),
        # export
        "e_refresh": "Làm mới danh sách", "e_list": "Các tệp đã tạo",
        "e_zip": "Nén toàn bộ → .zip", "e_out": "Tệp .zip",
        # messages
        "err_text": "Vui lòng nhập nội dung cần đọc.",
        "err_ref": "Vui lòng tải lên clip mẫu.",
        "err_pick": "Vui lòng chọn một giọng.",
        "err_zip": "Vui lòng chọn tệp voice_bank.zip.",
        "err_script": "Kịch bản trống. Mỗi dòng: `Tên giọng: nội dung`.",
        "saved": "✅ Đã lưu giọng **{name}** vào thư viện.",
        "deleted": "🗑️ Đã xóa **{name}**.",
        "exported_bank": "💾 Đã đóng gói **{n}** giọng. Tải tệp `.zip` để giữ lại qua phiên.",
        "imported_bank": "📥 Đã nhập **{n}** giọng.",
        # guide
        "guide": "",  # filled below
        "footer": ("---\n**Đạo đức & consent:** Chỉ clone giọng khi bạn có quyền. *Voice Design* tạo giọng "
                   "tổng hợp (không sao chép ai) là lựa chọn an toàn. Không dùng để mạo danh hay lừa đảo.\n\n"
                   "**Voice Studio AI** © 2026 **Elainabaka** · MIT · Dựa trên VieNeu-TTS & VoxCPM2 (Apache-2.0)."),
    },
    "en": {
        "app_title": "## 🎙️ Voice Studio AI — AI Voice Studio",
        "app_desc": ("**Create and own AI voices on Kaggle's free GPU.** Design a voice from a "
                     "description, clone from a clip, keep a reusable voice bank, render "
                     "multi-speaker podcasts — then export WAV/MP3."),
        "engine_badge": "Engine",
        "tab_design": "🎨 Voice Design",
        "tab_clone": "🎙️ Voice Cloning",
        "tab_bank": "🖼️ Voice Gallery",
        "tab_script": "🎧 Stories",
        "tab_export": "📦 Export",
        "tab_guide": "📖 Guide",
        "d_note": "Create a voice that **never existed** by describing it. Runs on the **VoxCPM2** engine.",
        "d_attrs": "Voice attributes",
        "d_gender": "Gender", "d_age": "Age", "d_accent": "Accent",
        "d_texture": "Timbre", "d_emotion": "Emotion / style",
        "d_pitch": "Pitch", "d_pace": "Pace", "d_extra": "Extra description (optional)",
        "d_text": "Text to read", "d_name": "Save as (blank to skip)",
        "d_fmt": "Format", "d_btn": "Design voice", "d_out": "Result",
        "c_note": "Upload a 5–15s clip to clone the voice. A clean clip (close mic, no music) clones better.",
        "c_ref": "Reference clip", "c_ref_info": "5–15s · one speaker · no music",
        "c_reftext": "Clip transcript (optional)", "c_reftext_info": "Improves fidelity (ultimate cloning)",
        "c_text": "Text to read", "c_style": "Style (optional)",
        "c_name": "Save as", "c_fmt": "Format",
        "c_btn": "Clone voice", "c_out": "Result",
        "b_filter": "Filter", "b_filter_all": "All", "b_filter_preset": "Presets", "b_filter_mine": "My voices",
        "b_pick": "Pick a voice", "b_text": "Text to read", "b_style": "Style",
        "b_fmt": "Format", "b_prev": "Preview", "b_del": "Delete", "b_out": "Result",
        "b_persist": "### 💾 Keep voices across sessions",
        "b_persist_desc": "Kaggle wipes files when a session ends. Export your bank to `.zip` to keep it.",
        "b_exp": "Export bank → .zip", "b_imp": "Import .zip into bank", "b_file": "voice_bank.zip file",
        "s_note": "One line per sentence: `Voice name: text`. Lines starting `//` are skipped. Mix presets and your voices.",
        "s_script": "Multi-speaker script", "s_gap": "Gap between lines (s)",
        "s_fmt": "Format", "s_btn": "Render", "s_out": "Result",
        "s_example": ("Minh Quân: Welcome to Voice Studio AI.\n"
                      "Ngọc Trân: This is a multi-speaker script, one line per sentence.\n"
                      "Minh Quân: Press render to export a single audio file."),
        "e_refresh": "Refresh list", "e_list": "Generated files",
        "e_zip": "Zip all → .zip", "e_out": "Zip file",
        "err_text": "Please enter text to read.",
        "err_ref": "Please upload a reference clip.",
        "err_pick": "Please pick a voice.",
        "err_zip": "Please choose voice_bank.zip.",
        "err_script": "Empty script. Use: `Voice name: text`.",
        "saved": "✅ Saved **{name}** to your library.",
        "deleted": "🗑️ Deleted **{name}**.",
        "exported_bank": "💾 Packaged **{n}** voices. Download the `.zip` to keep them.",
        "imported_bank": "📥 Imported **{n}** voices.",
        "guide": "",  # filled below
        "footer": ("---\n**Ethics & consent:** Only clone voices you have rights to. *Voice Design* makes "
                   "synthetic voices (copies no one) and is the safer path. Never impersonate or defraud.\n\n"
                   "**Voice Studio AI** © 2026 **Elainabaka** · MIT · Built on VieNeu-TTS & VoxCPM2 (Apache-2.0)."),
    },
}

_GUIDE_VI = """## 📖 Hướng dẫn sử dụng

### Các workspace
| Tab | Làm gì | Engine |
|---|---|---|
| 🎨 **Thiết kế giọng** | Chọn giới tính / tuổi / vùng miền / chất giọng / cảm xúc… → tạo giọng **mới hoàn toàn** | VoxCPM2 |
| 🎙️ **Nhân bản giọng** | Tải clip 5–15s → nhân bản giọng + chỉnh phong cách | VieNeu / VoxCPM2 |
| 🖼️ **Thư viện giọng** | Nghe thử giọng mẫu + giọng của bạn · xuất/nhập `.zip` | VieNeu |
| 🎧 **Kịch bản** | Kịch bản nhiều giọng → podcast / audiobook | Đang chọn |
| 📦 **Xuất file** | Tải WAV/MP3, nén toàn bộ ra `.zip` | — |

### Các bước nhanh
1. **Tạo giọng mới:** tab 🎨 → chọn thuộc tính → nhập nội dung → **Thiết kế giọng** → đặt tên lưu.
2. **Nhân bản giọng:** tab 🎙️ → tải clip sạch → nhập nội dung → **Nhân bản giọng**.
3. **Dùng lại giọng:** tab 🖼️ → chọn giọng → **Đọc thử**; hoặc tab 🎧 cho kịch bản nhiều giọng.
4. **Giữ giọng qua phiên:** tab 🖼️ → **Xuất kho giọng → `.zip`** → lần sau **Nhập `.zip` vào kho**.

### Mẹo chất lượng
- **Clone:** clip 5–15 giây, sát mic, phòng yên, **không nhạc**, một người, đọc đúng tông bạn muốn.
- **Design:** mô tả càng cụ thể càng tốt. Chất giọng *ấm/khan/sáng* + cảm xúc *kể chuyện* rất hợp làm narrator.
- **Giọng Việt chuẩn** → để engine `vieneu`. **Design + 30 ngôn ngữ** → `voxcpm`.
- Kaggle có hạn ~30 giờ GPU/tuần — chỉ bật GPU khi chạy, đừng để idle.
"""

_GUIDE_EN = """## 📖 How to use

### Workspaces
| Tab | What it does | Engine |
|---|---|---|
| 🎨 **Voice Design** | Pick gender / age / accent / timbre / emotion… → a voice that **never existed** | VoxCPM2 |
| 🎙️ **Voice Cloning** | Upload a 5–15s clip → clone + steer the style | VieNeu / VoxCPM2 |
| 🖼️ **Voice Gallery** | Preview presets + your voices · export/import `.zip` | VieNeu |
| 🎧 **Stories** | Multi-speaker script → podcast / audiobook | Active engine |
| 📦 **Export** | Download WAV/MP3, zip everything | — |

### Quick steps
1. **Design a voice:** 🎨 tab → pick attributes → type text → **Design voice** → name & save.
2. **Clone a voice:** 🎙️ tab → upload a clean clip → type text → **Clone voice**.
3. **Reuse a voice:** 🖼️ tab → pick a voice → **Preview**; or 🎧 tab for multi-speaker scripts.
4. **Keep voices across sessions:** 🖼️ tab → **Export bank → `.zip`** → next run **Import `.zip`**.

### Quality tips
- **Clone:** 5–15s clip, close mic, quiet room, **no music**, one speaker, read in the tone you want.
- **Design:** the more specific the better. A *warm/slightly raspy* narrator with *storytelling* style works great.
- **Pure Vietnamese** → engine `vieneu`. **Design + 30 languages** → `voxcpm`.
- Kaggle caps ~30 GPU hours/week — only enable GPU while running.
"""

TR["vi"]["guide"] = _GUIDE_VI
TR["en"]["guide"] = _GUIDE_EN

# Offload lên Kaggle (batch) — chỉ dùng khi chạy local.
TR["vi"].update({
    "tab_kgl": "☁️ Chạy trên Kaggle (GPU free)",
    "k_note": ("**Offload lên GPU Kaggle miễn phí** cho kịch bản/audiobook nặng hoặc máy không có GPU. "
               "⚠️ Kaggle chạy **batch** (đẩy lên → chờ → tải về), **không phải** tương tác như Local."),
    "k_slug": "Kernel slug (owner/tên)",
    "k_accel": "GPU", "k_engine": "Engine",
    "k_script": "Kịch bản (mỗi dòng: `Giọng: nội dung`)",
    "k_setup": "Kiểm tra Kaggle CLI", "k_install": "Cài Kaggle CLI",
    "k_push": "Đóng gói & đẩy lên GPU", "k_status": "Xem trạng thái",
    "k_download": "Tải kết quả về", "k_out": "Nhật ký / trạng thái",
})
TR["en"].update({
    "tab_kgl": "☁️ Run on Kaggle (free GPU)",
    "k_note": ("**Offload to Kaggle's free GPU** for heavy scripts/audiobooks or machines without a GPU. "
               "⚠️ Kaggle is **batch** (push → wait → download), **not** interactive like Local mode."),
    "k_slug": "Kernel slug (owner/name)",
    "k_accel": "GPU", "k_engine": "Engine",
    "k_script": "Script (one line: `Voice: text`)",
    "k_setup": "Check Kaggle CLI", "k_install": "Install Kaggle CLI",
    "k_push": "Package & push to GPU", "k_status": "Check status",
    "k_download": "Download results", "k_out": "Log / status",
})


def _on_kaggle() -> bool:
    return os.path.isdir("/kaggle")


# -------------------------------------------------------------------- UI ---
def build_ui():
    import warnings

    import gradio as gr

    # Gradio 6 cảnh báo theme/css đã chuyển sang launch() — vẫn chạy tốt trên Blocks,
    # chỉ tắt warning để cell trên Kaggle trông sạch.
    warnings.filterwarnings("ignore", message="The parameters have been moved from the Blocks constructor")

    ensure_dirs()

    def _names():
        return list_profiles(CFG["bank_dir"])

    def _presets():
        try:
            return get_engine().list_presets()
        except Exception:
            return []

    def _gallery(filter_):
        presets = _presets()
        mine = _names()
        items = []
        if filter_ in (None, "all"):
            items += [(p, "preset") for p in presets] + [(m, "mine") for m in mine]
        elif filter_ == "preset":
            items += [(p, "preset") for p in presets]
        else:
            items += [(m, "mine") for m in mine]
        return items

    theme = gr.themes.Soft(
        primary_hue="indigo",
        font=gr.themes.GoogleFont("Inter"),
    )

    with gr.Blocks(title="Voice Studio AI", theme=theme, css="""
        .vs-header { padding: 6px 0 2px; }
        .vs-title { font-size: 1.9em; font-weight: 800; margin: 0; }
        footer {display: none !important}
    """) as demo:
        lang = gr.Radio(
            ["🇻🇳 Tiếng Việt", "🇺🇸 English"], value="🇻🇳 Tiếng Việt",
            label="🌐 Ngôn ngữ / Language", container=False, scale=1,
        )

        @gr.render(inputs=[lang])
        def render_ui(sel):
            L = "en" if "English" in (sel or "") else "vi"
            t = TR[L]
            vi = (L == "vi")
            pitch_words = (["Thấp", "Trung bình", "Cao"] if vi else ["Low", "Mid", "High"])
            pace_words = (["Chậm", "Vừa", "Nhanh"] if vi else ["Slow", "Medium", "Fast"])
            g_opts = (["Nam", "Nữ", "Trung tính"] if vi else ["Male", "Female", "Neutral"])
            a_opts = (["Trẻ", "Trưởng thành", "Cao tuổi"] if vi else ["Young", "Adult", "Senior"])
            ac_opts = (["Miền Bắc", "Miền Trung", "Miền Nam"] if vi else ["Northern", "Central", "Southern"])
            tx_opts = (["Ấm áp", "Khàn", "Sáng rõ", "Mềm mại", "Trầm ấm"]
                       if vi else ["Warm", "Raspy", "Bright", "Soft", "Deep"])
            em_opts = (["Tự nhiên", "Vui tươi", "Trầm buồn", "Thì thầm", "Kể chuyện", "Bản tin", "Hào hứng"]
                       if vi else ["Natural", "Cheerful", "Somber", "Whisper", "Storytelling", "News", "Excited"])

            def compose(gender, age, accent, pitch, pace, emotion, texture, extra):
                accent_w = accent.replace("Miền ", "").replace("ern", "").lower() if vi else accent.lower()
                pitch_w = pitch_words[int(pitch)].lower()
                pace_w = pace_words[int(pace)].lower()
                parts = [gender, age, f"miền {accent_w}" if vi else f"{accent_w} accent",
                         f"chất giọng {texture}" if vi else f"{texture} timbre", emotion, pitch_w, pace_w]
                base = ("giọng " if vi else "") + ", ".join(str(p).lower() for p in parts if p)
                return (base + ". " + (extra or "").strip()).strip()

            # ---- handlers (đóng gói t) ----
            def do_design(gender, age, accent, pitch, pace, emotion, texture, extra, text, name, fmt):
                if not text.strip():
                    raise gr.Error(t["err_text"])
                desc = compose(gender, age, accent, pitch, pace, emotion, texture, extra)
                wav, sr = synth_text(text, "design", desc, "", "", "", "")
                path = _save_out(wav, sr, "design", fmt)
                msg = ""
                if name.strip():
                    save_profile(CFG["bank_dir"], VoiceProfile(
                        name=name.strip(), kind="design", engine="voxcpm", desc=desc), ref_src=None)
                    msg = t["saved"].format(name=name.strip())
                return path, msg, gr.update(choices=[n for n, _ in _gallery("mine")])

            def do_clone(ref, ref_text, text, style, name, fmt):
                if ref is None:
                    raise gr.Error(t["err_ref"])
                if not text.strip():
                    raise gr.Error(t["err_text"])
                ref_path = ref if isinstance(ref, str) else ref.name
                wav, sr = synth_text(text, "clone", "", ref_path, ref_text or "", "", style)
                path = _save_out(wav, sr, "clone", fmt)
                msg = ""
                if name.strip():
                    save_profile(CFG["bank_dir"], VoiceProfile(
                        name=name.strip(), kind="clone", engine=CFG["engine"],
                        ref_text=ref_text or "", style=style), ref_src=ref_path)
                    msg = t["saved"].format(name=name.strip())
                return path, msg, gr.update(choices=[n for n, _ in _gallery("mine")])

            def do_preview(pick, text, style, fmt):
                if not pick:
                    raise gr.Error(t["err_pick"])
                if not text.strip():
                    raise gr.Error(t["err_text"])
                wav, sr = synth_profile(pick, text, style)
                return _save_out(wav, sr, f"bank_{pick}", fmt)

            def do_delete(pick):
                if pick and pick in _names():
                    delete_profile(CFG["bank_dir"], pick)
                return (gr.update(choices=[n for n, _ in _gallery("mine")], value=None),
                        t["deleted"].format(name=pick))

            def do_export_bank():
                ensure_dirs()
                zp = os.path.join(CFG["work_dir"], "voice_bank.zip")
                export_bank(CFG["bank_dir"], zp)
                return zp, t["exported_bank"].format(n=len(_names()))

            def do_import_bank(zf):
                if zf is None:
                    raise gr.Error(t["err_zip"])
                zp = zf if isinstance(zf, str) else zf.name
                imported = import_bank(zp, CFG["bank_dir"])
                return gr.update(choices=[n for n, _ in _gallery("mine")]), t["imported_bank"].format(n=len(imported))

            def do_gallery_filter(f):
                items = _gallery(f)
                labels = [n + ("  · mẫu" if k == "preset" else "  · của tôi") if vi
                          else n + ("  · preset" if k == "preset" else "  · mine")
                          for n, k in items]
                # map label -> name
                return gr.update(choices=[n for n, _ in items])

            def do_script(script, gap, fmt):
                lines: List[dict] = render.parse_script(script)
                if not lines:
                    raise gr.Error(t["err_script"])

                def synth(text, voice, style):
                    return synth_profile(voice, text, style) if voice else \
                        synth_text(text, "preset", "", "", "", (_presets() or [""])[0], style)

                wav, sr = render.render_lines(synth, lines, gap_seconds=gap)
                return _save_out(wav, sr, "script", fmt)

            def do_list_outputs():
                return ex.list_outputs(CFG["out_dir"])

            def do_zip_outputs():
                ensure_dirs()
                zp = os.path.join(CFG["work_dir"], "audio_outputs.zip")
                ex.zip_outputs(CFG["out_dir"], zp)
                return zp

            def do_kgl_setup():
                info = ko.check_setup()
                cli = "✅" if info["cli"] else "❌"
                auth = "✅" if info["auth"] else "❌"
                lbl = "Đăng nhập" if vi else "Auth"
                return f"Kaggle CLI: {cli} · {lbl}: {auth}\n{info['msg']}"

            def do_kgl_install():
                return ko.install_cli()

            def do_kgl_push(slug, accel, engine, script):
                ensure_dirs()
                job = {"slug": slug or "voice-studio-offload", "engine": engine,
                       "task": "script", "script": script or "", "gap": 0.25, "out_name": "audio"}
                kdir = os.path.join(CFG["work_dir"], "kaggle_kernel")
                ko.build_kernel(job, CFG["bank_dir"], kdir)
                log = ("Đã đóng gói: " if vi else "Packaged: ") + kdir + "\n"
                log += ko.push(kdir, accelerator=accel or "NvidiaTeslaT4")
                log += ("\nĐã gửi lên Kaggle. Bấm 'Xem trạng thái' rồi 'Tải kết quả'."
                        if vi else "\nSent to Kaggle. Use 'Check status' then 'Download results'.")
                return log

            def do_kgl_status(slug):
                return ko.status(slug or "voice-studio-offload")

            def do_kgl_download(slug):
                dest = os.path.join(CFG["out_dir"], "kaggle_out")
                out = ko.download(slug or "voice-studio-offload", dest)
                return out + "\n" + ("Đã tải về: " if vi else "Downloaded to: ") + dest

            # ---- giao diện ----
            gr.HTML(f"<div class='vs-header'><p class='vs-title'>{t['app_title'].replace('## ','')}</p></div>")
            gr.Markdown(t["app_desc"])
            gr.Markdown(f"> **{t['engine_badge']}:** `{CFG['engine']}` · "
                        f"{ENGINE_INFO.get(CFG['engine'], {}).get('notes', '')}")

            with gr.Tabs():
                # 1 · Design
                with gr.Tab(t["tab_design"]):
                    gr.Markdown(t["d_note"])
                    with gr.Accordion(t["d_attrs"], open=True):
                        with gr.Row():
                            d_gender = gr.Dropdown(g_opts, value=g_opts[0], label=t["d_gender"])
                            d_age = gr.Dropdown(a_opts, value=a_opts[1], label=t["d_age"])
                            d_accent = gr.Dropdown(ac_opts, value=ac_opts[0], label=t["d_accent"])
                        with gr.Row():
                            d_texture = gr.Dropdown(tx_opts, value=tx_opts[0], label=t["d_texture"])
                            d_emotion = gr.Dropdown(em_opts, value=em_opts[4], label=t["d_emotion"])
                        with gr.Row():
                            d_pitch = gr.Slider(0, 2, value=1, step=1, label=t["d_pitch"], info="/".join(pitch_words))
                            d_pace = gr.Slider(0, 2, value=1, step=1, label=t["d_pace"], info="/".join(pace_words))
                        d_extra = gr.Textbox(label=t["d_extra"], placeholder="ví dụ: hơi khàn, như phát thanh viên…")
                    d_text = gr.Textbox(label=t["d_text"], lines=4,
                                        value="Xin chào, đây là giọng nói được thiết kế hoàn toàn từ mô tả." if vi
                                        else "Hello, this voice was designed entirely from a description.")
                    with gr.Row():
                        d_name = gr.Textbox(label=t["d_name"], value="")
                        d_fmt = gr.Radio(["wav", "mp3"], value="wav", label=t["d_fmt"])
                    d_btn = gr.Button(t["d_btn"], variant="primary")
                    d_out = gr.Audio(label=t["d_out"], type="filepath")
                    d_msg = gr.Markdown()

                # 2 · Clone
                with gr.Tab(t["tab_clone"]):
                    gr.Markdown(t["c_note"])
                    c_ref = gr.Audio(label=t["c_ref"], type="filepath")
                    gr.Markdown(f"<small>ℹ️ {t['c_ref_info']}</small>")
                    c_reftext = gr.Textbox(label=t["c_reftext"], info=t["c_reftext_info"], value="")
                    c_text = gr.Textbox(label=t["c_text"], lines=4,
                                        value="Đây là giọng nói được nhân bản từ clip mẫu." if vi
                                        else "This is a voice cloned from the reference clip.")
                    c_style = gr.Textbox(label=t["c_style"], placeholder="ví dụ: thì thầm, nhanh, vui…" if vi
                                         else "e.g. whisper, fast, cheerful…")
                    with gr.Row():
                        c_name = gr.Textbox(label=t["c_name"], value="")
                        c_fmt = gr.Radio(["wav", "mp3"], value="wav", label=t["c_fmt"])
                    c_btn = gr.Button(t["c_btn"], variant="primary")
                    c_out = gr.Audio(label=t["c_out"], type="filepath")
                    c_msg = gr.Markdown()

                # 3 · Gallery / Bank
                with gr.Tab(t["tab_bank"]):
                    with gr.Row():
                        b_filter = gr.Dropdown(
                            [(t["b_filter_all"], "all"), (t["b_filter_preset"], "preset"), (t["b_filter_mine"], "mine")],
                            value="all", label=t["b_filter"])
                        b_pick = gr.Dropdown([n for n, _ in _gallery("all")], label=t["b_pick"])
                    b_text = gr.Textbox(label=t["b_text"], lines=3,
                                        value="Đây là giọng trong thư viện." if vi else "This is a voice from the gallery.")
                    with gr.Row():
                        b_style = gr.Textbox(label=t["b_style"], value="")
                        b_fmt = gr.Radio(["wav", "mp3"], value="wav", label=t["b_fmt"])
                    with gr.Row():
                        b_prev = gr.Button(t["b_prev"], variant="primary")
                        b_del = gr.Button(t["b_del"])
                    b_out = gr.Audio(label=t["b_out"], type="filepath")
                    b_msg = gr.Markdown()
                    gr.Markdown(t["b_persist"] + "\n\n" + t["b_persist_desc"])
                    with gr.Row():
                        b_exp = gr.Button(t["b_exp"])
                        b_imp = gr.Button(t["b_imp"])
                    b_file = gr.File(label=t["b_file"], file_types=[".zip"])

                # 4 · Stories
                with gr.Tab(t["tab_script"]):
                    gr.Markdown(t["s_note"])
                    s_script = gr.Textbox(label=t["s_script"], lines=10, value=t["s_example"])
                    with gr.Row():
                        s_gap = gr.Slider(0.0, 2.0, value=0.3, step=0.05, label=t["s_gap"])
                        s_fmt = gr.Radio(["wav", "mp3"], value="wav", label=t["s_fmt"])
                    s_btn = gr.Button(t["s_btn"], variant="primary")
                    s_out = gr.Audio(label=t["s_out"], type="filepath")

                # 5 · Export
                with gr.Tab(t["tab_export"]):
                    e_refresh = gr.Button(t["e_refresh"])
                    e_list = gr.File(label=t["e_list"], file_count="multiple")
                    e_zip = gr.Button(t["e_zip"], variant="primary")
                    e_out = gr.File(label=t["e_out"])

                # 6 · Guide
                with gr.Tab(t["tab_guide"]):
                    gr.Markdown(t["guide"])

                # 7 · Kaggle offload (chỉ khi chạy local)
                if not _on_kaggle():
                    with gr.Tab(t["tab_kgl"]):
                        gr.Markdown(t["k_note"])
                        with gr.Row():
                            k_slug = gr.Textbox(label=t["k_slug"], value="elainabaka/voice-studio-offload")
                            k_accel = gr.Dropdown(["NvidiaTeslaT4", "NvidiaL4", "NvidiaTeslaP100"],
                                                  value="NvidiaTeslaT4", label=t["k_accel"])
                            k_engine = gr.Dropdown(["vieneu", "voxcpm"], value=CFG["engine"], label=t["k_engine"])
                        k_script = gr.Textbox(label=t["k_script"], lines=8, value=t["s_example"])
                        with gr.Row():
                            k_setup = gr.Button(t["k_setup"])
                            k_install = gr.Button(t["k_install"])
                            k_push = gr.Button(t["k_push"], variant="primary")
                        with gr.Row():
                            k_status = gr.Button(t["k_status"])
                            k_download = gr.Button(t["k_download"])
                        k_log = gr.Textbox(label=t["k_out"], lines=8)
                        k_setup.click(do_kgl_setup, [], [k_log])
                        k_install.click(do_kgl_install, [], [k_log])
                        k_push.click(do_kgl_push, [k_slug, k_accel, k_engine, k_script], [k_log])
                        k_status.click(do_kgl_status, [k_slug], [k_log])
                        k_download.click(do_kgl_download, [k_slug], [k_log])

            gr.Markdown(t["footer"])

            # ---- wiring ----
            d_btn.click(do_design,
                        [d_gender, d_age, d_accent, d_pitch, d_pace, d_emotion, d_texture,
                         d_extra, d_text, d_name, d_fmt], [d_out, d_msg, b_pick])
            c_btn.click(do_clone, [c_ref, c_reftext, c_text, c_style, c_name, c_fmt],
                        [c_out, c_msg, b_pick])
            b_filter.change(do_gallery_filter, [b_filter], [b_pick])
            b_prev.click(do_preview, [b_pick, b_text, b_style, b_fmt], [b_out])
            b_del.click(do_delete, [b_pick], [b_pick, b_msg])
            b_exp.click(do_export_bank, [], [b_file, b_msg])
            b_imp.click(do_import_bank, [b_file], [b_pick, b_msg])
            s_btn.click(do_script, [s_script, s_gap, s_fmt], [s_out])
            e_refresh.click(do_list_outputs, [], [e_list])
            e_zip.click(do_zip_outputs, [], [e_out])

    return demo


def launch(share: bool = True, **kw):
    demo = build_ui()
    return demo.launch(share=share, **kw)
