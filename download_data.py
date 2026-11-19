"""下载 MR 数据集 + GloVe 100d 词向量"""
import gzip
import os
import shutil
import sys
import time

import requests

ROOT = r"C:\Users\z2069\Desktop\CNN论文解析\repro"
DATA = os.path.join(ROOT, "data")
os.makedirs(DATA, exist_ok=True)

MR_FILES = {
    "rt-polarity.pos": "https://raw.githubusercontent.com/yoonkim/CNN_sentence/master/data/rt-polarity.pos",
    "rt-polarity.neg": "https://raw.githubusercontent.com/yoonkim/CNN_sentence/master/data/rt-polarity.neg",
}
GLOVE_URLS = [
    "https://huggingface.co/datasets/SLU-CSCI4750/glove.6B.100d.txt/resolve/main/glove.6B.100d.txt.gz",
    "https://hf-mirror.com/datasets/SLU-CSCI4750/glove.6B.100d.txt/resolve/main/glove.6B.100d.txt.gz",
]


def fetch(url, dest, desc):
    if os.path.exists(dest) and os.path.getsize(dest) > 0:
        print(f"[skip] {desc} already exists ({os.path.getsize(dest)/1e6:.1f} MB)")
        return True
    t0 = time.time()
    try:
        with requests.get(url, stream=True, timeout=60) as r:
            r.raise_for_status()
            total = int(r.headers.get("Content-Length", 0))
            done = 0
            tmp = dest + ".part"
            with open(tmp, "wb") as f:
                for chunk in r.iter_content(1 << 20):
                    f.write(chunk)
                    done += len(chunk)
                    if total and done % (20 << 20) < (1 << 20):
                        print(f"  {desc}: {done/1e6:.0f}/{total/1e6:.0f} MB "
                              f"({time.time()-t0:.0f}s)", flush=True)
            os.replace(tmp, dest)
        print(f"[ok] {desc} {os.path.getsize(dest)/1e6:.1f} MB in {time.time()-t0:.1f}s")
        return True
    except Exception as e:
        print(f"[fail] {desc}: {type(e).__name__} {e}", flush=True)
        return False


for name, url in MR_FILES.items():
    fetch(url, os.path.join(DATA, name), name)

gz = os.path.join(DATA, "glove.6B.100d.txt.gz")
txt = os.path.join(DATA, "glove.6B.100d.txt")
if not os.path.exists(txt):
    ok = False
    for u in GLOVE_URLS:
        if fetch(u, gz, "glove.6B.100d.txt.gz"):
            ok = True
            break
    if ok:
        print("unzipping...", flush=True)
        with gzip.open(gz, "rb") as fin, open(txt, "wb") as fout:
            shutil.copyfileobj(fin, fout)
        print(f"[ok] glove txt {os.path.getsize(txt)/1e6:.1f} MB")
else:
    print("[skip] glove txt exists")

print("\n=== data dir ===")
for f in sorted(os.listdir(DATA)):
    print(f"{os.path.getsize(os.path.join(DATA, f))/1e6:8.1f} MB  {f}")

# 快速校验
pos = os.path.join(DATA, "rt-polarity.pos")
neg = os.path.join(DATA, "rt-polarity.neg")
if os.path.exists(pos) and os.path.exists(neg):
    npos = sum(1 for _ in open(pos, encoding="latin-1"))
    nneg = sum(1 for _ in open(neg, encoding="latin-1"))
    print(f"MR: pos={npos} neg={nneg} total={npos+nneg}")
if os.path.exists(txt):
    with open(txt, encoding="utf-8") as f:
        first = f.readline().split()
        print(f"GloVe: dim={len(first)-1} first_word={first[0]}")
