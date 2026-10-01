import base64, json, os, re, sys

root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
nb = json.load(open(os.path.join(root, "notebook", "Voice_Studio_AI_Kaggle_GPU.ipynb"), encoding="utf-8"))
src = None
for c in nb["cells"]:
    s = "".join(c["source"])
    if "_FILES = json.loads" in s:
        src = s
        break
assert src, "khong tim thay cell thu vien"
m = re.search(r"json\.loads\(r'''(.*?)'''\)", src, re.S)
files = json.loads(m.group(1))
ok = True
for rel, b64 in files.items():
    dec = base64.b64decode(b64)
    orig = open(os.path.join(root, "studio", os.path.basename(rel)), "rb").read()
    match = dec == orig
    ok = ok and match
    print(("OK  " if match else "FAIL"), rel, len(dec), "bytes")
print("ALL MATCH" if ok else "MISMATCH!")
sys.exit(0 if ok else 1)
