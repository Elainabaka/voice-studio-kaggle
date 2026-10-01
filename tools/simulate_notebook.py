"""Mo phong DUNG cell thu vien cua notebook: giai nen base64 ra thu muc tam,
import studio tu do roi chay smoke test. Chứng minh notebook tu chua va chay duoc."""

import base64, json, os, re, subprocess, sys, tempfile

root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
nb = json.load(open(os.path.join(root, "notebook", "Voice_Studio_AI_Kaggle_GPU.ipynb"), encoding="utf-8"))
src = next("".join(c["source"]) for c in nb["cells"] if "_FILES = json.loads" in "".join(c["source"]))
files = json.loads(re.search(r"json\.loads\(r'''(.*?)'''\)", src, re.S).group(1))

with tempfile.TemporaryDirectory() as pkg:
    for rel, b64 in files.items():
        p = os.path.join(pkg, rel)
        os.makedirs(os.path.dirname(p), exist_ok=True)
        open(p, "wb").write(base64.b64decode(b64))

    code = (
        "import sys; sys.path.insert(0, %r)\n" % pkg
        + "from studio import CFG, ensure_dirs\n"
        + "from studio.engines import available_engines\n"
        + "from studio.voicebank import VoiceProfile, save_profile, list_profiles\n"
        + "CFG['work_dir']=%r; CFG['bank_dir']=%r; CFG['out_dir']=%r\n"
          % (pkg, os.path.join(pkg, "bank"), os.path.join(pkg, "out"))
        + "ensure_dirs()\n"
        + "print('engines:', available_engines())\n"
        + "save_profile(CFG['bank_dir'], VoiceProfile(name='t', kind='design', engine='voxcpm', desc='x'))\n"
        + "print('bank:', list_profiles(CFG['bank_dir']))\n"
        + "print('IMPORT + RUN OK')\n"
    )
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    print(r.stdout, end="")
    if r.returncode != 0:
        print(r.stderr)
        sys.exit(1)
print("NOTEBOOK SELF-CONTAINED OK")
