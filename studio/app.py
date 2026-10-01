"""Voice Studio - dieu phoi cap cao + Gradio Studio UI (5 tab).

Tab:
  1. Design   - tao giong MOI tu mo ta bang chu (khong can clip)
  2. Clone    - nhan ban giong tu clip 3-15s + dieu khien style
  3. Bank     - kho giong: xem/doi/xoa, export/import .zip (vuot session)
  4. Script   - script da giong -> podcast / audiobook / hoi thoai
  5. Export   - tai san pham (wav/mp3) + nen zip
"""

from __future__ import annotations

import os
import time
from typing import List, Tuple

import numpy as np

from . import audio, export as ex, render
from .config import CFG, ENGINE_INFO, ensure_dirs
from .engines import get_engine
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
def synth_text(text: str, mode: str, desc: str, ref_path: str, ref_text: str,
               preset: str, style: str) -> WavOut:
    """Sinh am theo che do truc tiep (khong qua profile)."""
    eng = get_engine()
    return eng.tts(
        text,
        desc=desc if mode == "design" else "",
        ref_path=ref_path if mode == "clone" else "",
        ref_text=ref_text,
        preset=preset if mode == "preset" else "",
        style=style,
    )


def synth_profile(profile_name: str, text: str, style: str = "") -> WavOut:
    """Sinh am bang mot giong da luu trong bank."""
    eng = get_engine()
    prof = load_profile(CFG["bank_dir"], profile_name)
    st = style or prof.style
    if prof.kind == "design":
        return eng.tts(text, desc=prof.desc, style=st)
    if prof.kind == "clone":
        return eng.tts(text, ref_path=prof.ref_audio, ref_text=prof.ref_text, style=st)
    return eng.tts(text, preset=prof.name, style=st)


def _save_out(wav: np.ndarray, sr: int, stem: str, fmt: str = "wav") -> str:
    ensure_dirs()
    path = os.path.join(CFG["out_dir"], f"{stem}_{int(time.time())}.{fmt}")
    return ex.write_output(path, wav, sr, fmt=fmt)


# -------------------------------------------------------------------- UI ---
def build_ui():
    import gradio as gr

    ensure_dirs()

    def _names():
        return list_profiles(CFG["bank_dir"])

    def _presets():
        try:
            return get_engine().list_presets()
        except Exception:
            return []

    # ---- handlers -----------------------------------------------------
    def do_design(desc, text, style, name, fmt):
        if not text.strip():
            raise gr.Error("Nhap noi dung can doc.")
        wav, sr = synth_text(text, "design", desc, "", "", "", style)
        path = _save_out(wav, sr, "design", fmt)
        saved = ""
        if name.strip():
            save_profile(CFG["bank_dir"], VoiceProfile(
                name=name.strip(), kind="design", engine=CFG["engine"],
                desc=desc.strip(), style=style), ref_src=None)
            saved = f"Da luu giong **{name.strip()}** vao Voice Bank."
        return path, saved, gr.update(choices=_names())

    def do_clone(ref, ref_text, text, style, name, fmt):
        if ref is None:
            raise gr.Error("Tai len clip mau (3-15s, sach, khong nhac).")
        if not text.strip():
            raise gr.Error("Nhap noi dung can doc.")
        ref_path = ref if isinstance(ref, str) else ref.name
        wav, sr = synth_text(text, "clone", "", ref_path, ref_text or "", "", style)
        path = _save_out(wav, sr, "clone", fmt)
        saved = ""
        if name.strip():
            save_profile(CFG["bank_dir"], VoiceProfile(
                name=name.strip(), kind="clone", engine=CFG["engine"],
                ref_text=ref_text or "", style=style), ref_src=ref_path)
            saved = f"Da luu giong **{name.strip()}** vao Voice Bank."
        return path, saved, gr.update(choices=_names())

    def do_preview(profile_name, text, style, fmt):
        if not profile_name:
            raise gr.Error("Chon mot giong trong bank.")
        if not text.strip():
            raise gr.Error("Nhap noi dung can doc.")
        wav, sr = synth_profile(profile_name, text, style)
        return _save_out(wav, sr, f"bank_{profile_name}", fmt)

    def do_delete(profile_name):
        if profile_name:
            delete_profile(CFG["bank_dir"], profile_name)
        return gr.update(choices=_names(), value=None), f"Da xoa **{profile_name}**."

    def do_export_bank():
        ensure_dirs()
        zp = os.path.join(CFG["work_dir"], "voice_bank.zip")
        export_bank(CFG["bank_dir"], zp)
        return zp, f"Da dong goi {len(_names())} giong. Tai .zip de giu lai vuot session."

    def do_import_bank(zipfile_obj):
        if zipfile_obj is None:
            raise gr.Error("Chon file voice_bank.zip.")
        zp = zipfile_obj if isinstance(zipfile_obj, str) else zipfile_obj.name
        imported = import_bank(zp, CFG["bank_dir"])
        return gr.update(choices=_names()), f"Da import {len(imported)} giong."

    def do_script(script, gap, fmt):
        """Moi dong: `TEN_GIONG: noi dung` hoac `TEN_GIONG | noi dung`."""
        lines: List[dict] = []
        for raw in (script or "").splitlines():
            raw = raw.strip()
            if not raw or raw.startswith("//"):
                continue
            if ":" in raw:
                v, t = raw.split(":", 1)
            elif "|" in raw:
                v, t = raw.split("|", 1)
            else:
                v, t = "", raw
            lines.append({"voice": v.strip(), "text": t.strip(), "style": ""})
        if not lines:
            raise gr.Error("Script rong. Moi dong: `TEN_GIONG: noi dung`.")
        presets = _presets()

        def synth(text, voice, style):
            if not voice:
                return synth_text(text, "preset", "", "", "", presets[0] if presets else "", style)
            try:
                return synth_profile(voice, text, style)          # ten trong Bank / preset
            except Exception:
                return synth_text(text, "clone", "", voice, "", "", style)  # voice = duong dan clip

        wav, sr = render.render_lines(synth, lines, gap_seconds=gap)
        return _save_out(wav, sr, "script", fmt)

    def do_list_outputs():
        return ex.list_outputs(CFG["out_dir"])

    def do_zip_outputs():
        ensure_dirs()
        zp = os.path.join(CFG["work_dir"], "audio_outputs.zip")
        ex.zip_outputs(CFG["out_dir"], zp)
        return zp

    # ---- giao dien ----------------------------------------------------
    with gr.Blocks(title="Voice Studio AI") as demo:
        gr.Markdown(
            "# Voice Studio AI\n"
            f"**Engine:** `{CFG['engine']}` · "
            f"{ENGINE_INFO.get(CFG['engine'], {}).get('notes', '')}\n\n"
            "Design (tao giong) · Clone (nhan ban) · Bank (kho giong) · "
            "Script (podcast) · Export (tai san pham)."
        )

        # 1 · Design
        with gr.Tab("1 · Design (tao giong moi)"):
            d_desc = gr.Textbox(label="Mo ta giong (tu do)",
                                value="Giong nam trung nien mien Nam, am, hoi khan, ke chuyen cham")
            d_text = gr.Textbox(label="Noi dung can doc", lines=4,
                                value="Xin chao, day la giong duoc thiet ke hoan toan tu mo ta.")
            d_style = gr.Textbox(label="Style (tuy chon)", value="doc_truyen")
            d_name = gr.Textbox(label="Ten luu vao Bank (de trong neu khong luu)", value="")
            d_fmt = gr.Radio(["wav", "mp3"], value="wav", label="Dinh dang")
            d_btn = gr.Button("Thiet ke giong", variant="primary")
            d_out = gr.Audio(label="Ket qua", type="filepath")
            d_msg = gr.Markdown()

        # 2 · Clone
        with gr.Tab("2 · Clone (nhan ban tu clip)"):
            c_ref = gr.Audio(label="Clip mau (3-15s, sach, khong nhac)", type="filepath")
            c_reftext = gr.Textbox(label="Transcript clip (tuy chon, giong trung hon)", value="")
            c_text = gr.Textbox(label="Noi dung can doc", lines=4, value="Day la giong duoc nhan ban.")
            c_style = gr.Textbox(label="Style (tuy chon)", value="")
            c_name = gr.Textbox(label="Ten luu vao Bank", value="")
            c_fmt = gr.Radio(["wav", "mp3"], value="wav", label="Dinh dang")
            c_btn = gr.Button("Nhan ban giong", variant="primary")
            c_out = gr.Audio(label="Ket qua", type="filepath")
            c_msg = gr.Markdown()

        # 3 · Bank
        with gr.Tab("3 · Bank (kho giong)"):
            with gr.Row():
                b_pick = gr.Dropdown(choices=_names(), label="Giong da luu")
                b_text = gr.Textbox(label="Noi dung can doc", value="Thu giong trong kho.")
            b_style = gr.Textbox(label="Style (tuy chon)", value="")
            b_fmt = gr.Radio(["wav", "mp3"], value="wav", label="Dinh dang")
            with gr.Row():
                b_prev = gr.Button("Doc thu", variant="primary")
                b_del = gr.Button("Xoa giong")
            b_out = gr.Audio(label="Ket qua", type="filepath")
            b_msg = gr.Markdown()
            gr.Markdown("### Vuot session Kaggle")
            with gr.Row():
                b_exp = gr.Button("Export kho giong -> .zip")
                b_imp = gr.Button("Import .zip vao kho")
            b_file = gr.File(label="File voice_bank.zip", file_types=[".zip"])

        # 4 · Script
        with gr.Tab("4 · Script (podcast / audiobook)"):
            gr.Markdown("Moi dong mot cau: `TEN_GIONG: noi dung`. Dong `//` bi bo qua.")
            s_script = gr.Textbox(label="Script da giong", lines=10, value=(
                "Minh Quân: Chào mừng đến với Voice Studio AI.\n"
                "Ngọc Trân: Đây là kịch bản nhiều giọng, mỗi dòng một câu.\n"
                "Minh Quân: Bấm nút để xuất thành một file âm thanh duy nhất."
            ))
            s_gap = gr.Slider(0.0, 2.0, value=0.3, step=0.05, label="Khoang lang giua cau (giay)")
            s_fmt = gr.Radio(["wav", "mp3"], value="wav", label="Dinh dang")
            s_btn = gr.Button("Render script", variant="primary")
            s_out = gr.Audio(label="Ket qua", type="filepath")

        # 5 · Export
        with gr.Tab("5 · Export (tai san pham)"):
            e_refresh = gr.Button("Lam moi danh sach")
            e_list = gr.File(label="San pham da tao", file_count="multiple")
            e_zip = gr.Button("Nen toan bo -> .zip", variant="primary")
            e_out = gr.File(label="File .zip")

        gr.Markdown(
            "---\n"
            "**Dao duc & consent:** chi clone giong ban duoc phep. Voice Design tao giong "
            "tong hop (khong copy ai) la cach an toan ve phap ly.\n"
            "**License engine:** VieNeu (Apache-2.0) · VoxCPM2 (Apache-2.0). Xem README."
        )

        # ---- wiring (sau khi tao het component de doi duoc b_pick) ----
        d_btn.click(do_design, [d_desc, d_text, d_style, d_name, d_fmt],
                    [d_out, d_msg, b_pick])
        c_btn.click(do_clone, [c_ref, c_reftext, c_text, c_style, c_name, c_fmt],
                    [c_out, c_msg, b_pick])
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
