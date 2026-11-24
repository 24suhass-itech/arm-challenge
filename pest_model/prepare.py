"""KrishiRakshak pest model - step 1: index IP102, cache resized images, hash them, build leakage-free splits.

IP102 official split (train/val/test) is kept, but
  - exact (SHA-256) and near-duplicate (DCT perceptual hash, Hamming <= 4) images are linked into components,
  - a component touching the test set loses its train/val members; one touching val loses its train members,
  - duplicates inside the same split keep one copy,
  - components whose members disagree on the label are reported (and dropped from train/val).
Outputs: index.csv, manifest.csv, dropped.csv, classes.csv
"""
import hashlib
import io
import os
import re
import zipfile
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd
from PIL import Image
from scipy.fft import dctn

OUT = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(OUT)
ZIP = os.path.join(ROOT, "datasets", "ip102.zip")
CACHE = os.path.join(OUT, "cache")
SHORT = 320
THRESH = 4
PRIORITY = {"test": 3, "val": 2, "train": 1}
# IP102 class ranges per crop (Wu et al., CVPR 2019)
CROPS = [("Rice", 14), ("Corn", 13), ("Wheat", 9), ("Beet", 8), ("Alfalfa", 13), ("Vitis", 16), ("Citrus", 19),
         ("Mango", 10)]


def phash(im):
    g = np.asarray(im.convert("L").resize((32, 32), Image.LANCZOS), dtype=np.float64)
    d = dctn(g, type=2, norm="ortho")[:8, :8]
    return int(np.packbits((d > np.median(d)).flatten()).view(">u8")[0])


_Z = None


def work(args):
    global _Z
    if _Z is None:
        _Z = zipfile.ZipFile(ZIP)
    i, member = args
    data = _Z.read(member)
    sha = hashlib.sha256(data).hexdigest()
    out = os.path.join(CACHE, f"{i:06d}.jpg")
    try:
        im = Image.open(io.BytesIO(data))
        im.load()
        im = im.convert("RGB")
    except Exception:
        return i, sha, None, None
    ph = phash(im)
    if not os.path.exists(out):
        w, h = im.size
        s = SHORT / min(w, h)
        if s < 1:
            im = im.resize((max(1, round(w * s)), max(1, round(h * s))), Image.BICUBIC)
        im.save(out, "JPEG", quality=93)
    return i, sha, ph, f"cache/{i:06d}.jpg"


class DSU:
    def __init__(self, n):
        self.p = np.arange(n)

    def find(self, x):
        r = x
        while self.p[r] != r:
            r = self.p[r]
        while self.p[x] != r:
            self.p[x], x = r, self.p[x]
        return r

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.p[max(ra, rb)] = min(ra, rb)


def main():
    os.makedirs(CACHE, exist_ok=True)
    with zipfile.ZipFile(ZIP) as z:
        names = [re.sub(r"\s+", " ", l).strip() for l in z.read("classes.txt").decode().splitlines() if l.strip()]
        members = [m for m in z.namelist() if m.startswith("classification/") and m.lower().endswith(".jpg")]
    names = [re.sub(r"^\d+\s*", "", n) for n in names]
    crop_of = [c for c, k in CROPS for _ in range(k)]
    assert len(names) == 102 == len(crop_of), len(names)
    pd.DataFrame({"class_index": range(102), "class_name": names, "crop": crop_of}).to_csv(
        os.path.join(OUT, "classes.csv"), index=False)

    df = pd.DataFrame({"member": sorted(members)})
    parts = df.member.str.split("/")
    df["split"] = parts.str[1]
    df["class_index"] = parts.str[2].astype(int)
    df["label"] = df.class_index.map(dict(enumerate(names)))
    df["crop"] = df.class_index.map(dict(enumerate(crop_of)))
    df["id"] = range(len(df))

    res = [None] * len(df)
    with ProcessPoolExecutor(4) as ex:
        for k, r in enumerate(ex.map(work, zip(df.id, df.member), chunksize=256)):
            res[r[0]] = r
            if k % 5000 == 0:
                print(f"cached {k}/{len(df)}", flush=True)
    df["sha256"] = [r[1] for r in res]
    df["phash"] = [r[2] for r in res]
    df["cache_path"] = [r[3] for r in res]
    bad = df.cache_path.isna()
    print("unreadable images:", int(bad.sum()))
    df = df[~bad].reset_index(drop=True)
    df["phash"] = df.phash.astype("uint64")
    df.to_csv(os.path.join(OUT, "index.csv"), index=False)

    # near-duplicate components
    n = len(df)
    h = df.phash.values.astype(np.uint64)
    dsu = DSU(n)
    npairs = 0
    for s in range(0, n, 512):
        d = np.bitwise_count(h[s:s + 512, None] ^ h[None, :])
        qi, j = np.nonzero(d <= THRESH)
        i = qi + s
        for a, b in zip(i[i < j], j[i < j]):
            dsu.union(a, b)
            npairs += 1
    for _, idx in df.groupby("sha256").indices.items():
        for k in idx[1:]:
            dsu.union(idx[0], k)
    df["comp"] = [dsu.find(i) for i in range(n)]
    print("near/exact duplicate pairs:", npairs)

    df["prio"] = df.split.map(PRIORITY)
    cp = df.groupby("comp").prio.transform("max")
    df["drop_reason"] = ""
    inv = {v: k for k, v in PRIORITY.items()}
    lower = df.prio < cp
    df.loc[lower, "drop_reason"] = "duplicate of a " + cp[lower].map(inv) + " image"
    live = df.drop_reason == ""
    nlab = df[live].groupby("comp").class_index.nunique()
    conflict = df.comp.isin(nlab[nlab > 1].index) & live
    print("images in label-conflict components:", int(conflict.sum()), "(test ones kept, train/val dropped)")
    df.loc[conflict & (df.split != "test"), "drop_reason"] = "near-duplicate with a different label"
    live = df.drop_reason == ""
    dup = live & df[live].duplicated(["comp", "split", "class_index"]).reindex(df.index, fill_value=False)
    df.loc[dup & (df.split != "test"), "drop_reason"] = "duplicate within split"
    df["test_label_conflict"] = conflict & (df.split == "test")

    df[df.drop_reason != ""].drop(columns=["phash"]).to_csv(os.path.join(OUT, "dropped.csv"), index=False)
    man = df[df.drop_reason == ""].drop(columns=["phash", "prio", "drop_reason"])
    man.to_csv(os.path.join(OUT, "manifest.csv"), index=False)
    print("\ndropped:\n", df[df.drop_reason != ""].groupby(["split", "drop_reason"]).size().to_string())
    print("\nfinal:\n", man.split.value_counts().to_string())


if __name__ == "__main__":
    main()
