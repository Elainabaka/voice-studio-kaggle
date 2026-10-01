"""Smoke test phan loi Voice Studio (khong can GPU / engine TTS nang).

Chay:  python -m tests.smoke
Kiem tra: audio concat/save, voicebank roundtrip + export/import, render da giong.
"""

from __future__ import annotations

import json
import os
import sys
import tempfile

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from studio import audio, export as ex, render  # noqa: E402
from studio.config import CFG  # noqa: E402
from studio.voicebank import (  # noqa: E402
    VoiceProfile,
    delete_profile,
    export_bank,
    import_bank,
    list_profiles,
    load_profile,
    save_profile,
)

SR = 24000


def _fake_synth(text, voice, style):
    """Synth gia lap: chuoi -> am sine do dai ~0.3s, tan so tuy voice."""
    freq = 220 + (hash(voice) % 400)
    t = np.linspace(0, 0.3, int(SR * 0.3), endpoint=False)
    return (0.2 * np.sin(2 * np.pi * freq * t)).astype(np.float32), SR


def test_audio_concat_save(tmp):
    w1 = np.random.uniform(-0.3, 0.3, SR).astype(np.float32)
    w2 = np.random.uniform(-0.3, 0.3, SR).astype(np.float32)
    merged = audio.concat([w1, w2], SR, gap_seconds=0.25)
    assert len(merged) == len(w1) + len(w2) + int(SR * 0.25)
    p = audio.save_wav(os.path.join(tmp, "a.wav"), merged, SR)
    back, sr = audio.load_audio(p, target_sr=SR)
    assert sr == SR and back.ndim == 1 and len(back) > 0
    print("  [ok] audio concat + save/load roundtrip")


def test_voicebank_roundtrip(tmp):
    bank = os.path.join(tmp, "bank")
    # tao file mau gia
    ref = os.path.join(tmp, "ref.wav")
    audio.save_wav(ref, np.random.uniform(-0.2, 0.2, SR).astype(np.float32), SR)

    save_profile(bank, VoiceProfile(name="Giong Cua Toi", kind="clone",
                                    engine="vieneu", ref_text="hello", style="news"), ref_src=ref)
    save_profile(bank, VoiceProfile(name="Giong Thiet Ke", kind="design",
                                    engine="voxcpm", desc="nam tram am", style="doc_truyen"))

    names = list_profiles(bank)
    assert "Giong Cua Toi" in names and "Giong Thiet Ke" in names, names

    p = load_profile(bank, "Giong Cua Toi")
    assert p.kind == "clone" and os.path.exists(p.ref_audio), p
    d = load_profile(bank, "Giong Thiet Ke")
    assert d.kind == "design" and "nam" in d.desc

    # export / import (vuot session)
    zp = os.path.join(tmp, "bank.zip")
    export_bank(bank, zp)
    bank2 = os.path.join(tmp, "bank2")
    imported = import_bank(zp, bank2)
    assert set(imported) == set(names), (imported, names)
    assert os.path.exists(load_profile(bank2, "Giong Cua Toi").ref_audio)

    assert delete_profile(bank2, "Giong Thiet Ke")
    assert "Giong Thiet Ke" not in list_profiles(bank2)
    print("  [ok] voicebank save/load + export/import vuot session + delete")


def test_render(tmp):
    lines = [{"voice": "A", "text": "Xin chao"}, {"voice": "B", "text": "Moi nguoi"}]
    wav, sr = render.render_lines(_fake_synth, lines, gap_seconds=0.25)
    assert sr == SR and len(wav) > 0
    paths = render.save_script(tmp, "podcast", wav, sr)
    assert os.path.exists(paths[0])
    print("  [ok] render script da giong + save")


def test_export_zip(tmp):
    out = os.path.join(tmp, "out")
    os.makedirs(out, exist_ok=True)
    audio.save_wav(os.path.join(out, "x.wav"), np.zeros(SR, np.float32), SR)
    assert len(ex.list_outputs(out)) == 1
    zp = ex.zip_outputs(out, os.path.join(tmp, "o.zip"))
    assert os.path.exists(zp)
    print("  [ok] export list + zip")


def test_offload_build(tmp):
    lines = render.parse_script("A: Xin chao\n// bo qua\nB: Moi nguoi")
    assert lines == [{"voice": "A", "text": "Xin chao", "style": ""},
                     {"voice": "B", "text": "Moi nguoi", "style": ""}], lines
    from studio import kaggle_offload as ko
    job = {"slug": "u/voice-studio-offload", "engine": "vieneu", "task": "script",
           "script": "A: Xin chao", "gap": 0.25, "out_name": "audio"}
    kdir = os.path.join(tmp, "kernel")
    ko.build_kernel(job, None, kdir)
    with open(os.path.join(kdir, "kernel-metadata.json"), encoding="utf-8") as f:
        meta = json.load(f)
    assert meta["id"] == "u/voice-studio-offload" and meta["enable_gpu"]
    with open(os.path.join(kdir, "voice_studio_offload.ipynb"), encoding="utf-8") as f:
        nb = json.load(f)
    assert nb["nbformat"] == 4
    src = "\n".join("".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code")
    for needle in ["studio/synth.py", "synth_profile", "parse_script", "JOB = json.loads"]:
        assert needle in src, needle
    print("  [ok] parse_script + kaggle_offload.build_kernel")


def main():
    print("Voice Studio smoke test:")
    with tempfile.TemporaryDirectory() as tmp:
        test_audio_concat_save(tmp)
        test_voicebank_roundtrip(tmp)
        test_render(tmp)
        test_export_zip(tmp)
        test_offload_build(tmp)
    print("ALL PASS")


if __name__ == "__main__":
    main()
