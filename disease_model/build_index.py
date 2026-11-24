"""KrishiRakshak disease model - step 1: index every source image, map it to a class, cache it, and hash it.

Output: index.csv with one row per image:
  id, dataset, source, member, label, class_index, fixed_split, group, sha256, phash, cache_path

Classes: the 38 PlantVillage classes + "Background" (not a leaf).
fixed_split pre-assigns splits that must not change:
  - PlantDoc official test images -> "test"
  - custom test images ("test images" folder) -> "custom" (never trained on)
  - PlantVillage / PlantDoc-train / Corn / Potato -> the split used by earlier unified model (unified_training) when available,
    so the earlier unified model and this model can be compared fairly on the same test images
Everything else gets its split later in make_splits.py (grouped, near-duplicate aware).
"""
import hashlib
import io
import os
import re
import tarfile
import threading
import zipfile
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd
from PIL import Image
from scipy.fft import dctn

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "disease_model")
CACHE = os.path.join(OUT, "cache")
DS = os.path.join(ROOT, "datasets")
CACHE_SHORT_SIDE = 256

ZIPS = {
    "PlantVillage": os.path.join(DS, "PlantVillage-Dataset-master.zip"),
    "PlantDoc": os.path.join(DS, "PlantDoc-Dataset.zip"),
    "Corn": os.path.join(DS, "corn-leaf-disease-data.zip"),
    "Potato": os.path.join(DS, "potato-leaf-disease-dataset.zip"),
    "TomatoVillage": os.path.join(DS, "tomato.zip"),
    "PlantWild": os.path.join(DS, "plantwild_v2.zip"),
}
CALTECH_TAR = os.path.join(DS, "background", "256_ObjectCategories.tar")
DTD_TAR = os.path.join(DS, "background", "dtd-r1.0.1.tar.gz")
CUSTOM_DIRS = ["test images"]

PLANTDOC_MAP = {
    "Apple Scab Leaf": "Apple___Apple_scab", "Apple leaf": "Apple___healthy", "Apple rust leaf": "Apple___Cedar_apple_rust",
    "Bell_pepper leaf": "Pepper,_bell___healthy", "Bell_pepper leaf spot": "Pepper,_bell___Bacterial_spot",
    "Blueberry leaf": "Blueberry___healthy", "Cherry leaf": "Cherry_(including_sour)___healthy",
    "Corn Gray leaf spot": "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot",
    "Corn leaf blight": "Corn_(maize)___Northern_Leaf_Blight", "Corn rust leaf": "Corn_(maize)___Common_rust_",
    "Peach leaf": "Peach___healthy", "Potato leaf early blight": "Potato___Early_blight",
    "Potato leaf late blight": "Potato___Late_blight", "Raspberry leaf": "Raspberry___healthy",
    "Soyabean leaf": "Soybean___healthy", "Squash Powdery mildew leaf": "Squash___Powdery_mildew",
    "Strawberry leaf": "Strawberry___healthy", "Tomato Early blight leaf": "Tomato___Early_blight",
    "Tomato Septoria leaf spot": "Tomato___Septoria_leaf_spot", "Tomato leaf": "Tomato___healthy",
    "Tomato leaf bacterial spot": "Tomato___Bacterial_spot", "Tomato leaf late blight": "Tomato___Late_blight",
    "Tomato leaf mosaic virus": "Tomato___Tomato_mosaic_virus", "Tomato leaf yellow virus": "Tomato___Tomato_Yellow_Leaf_Curl_Virus",
    "Tomato mold leaf": "Tomato___Leaf_Mold", "grape leaf": "Grape___healthy", "grape leaf black rot": "Grape___Black_rot",
}
PLANTWILD_MAP = {
    "apple black rot": "Apple___Black_rot", "apple rust": "Apple___Cedar_apple_rust", "apple scab": "Apple___Apple_scab",
    "bell pepper bacterial spot": "Pepper,_bell___Bacterial_spot",
    "cherry powdery mildew": "Cherry_(including_sour)___Powdery_mildew",
    "citrus greening disease": "Orange___Haunglongbing_(Citrus_greening)",
    "corn gray leaf spot": "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot",
    "corn northern leaf blight": "Corn_(maize)___Northern_Leaf_Blight", "corn rust": "Corn_(maize)___Common_rust_",
    "grape black rot": "Grape___Black_rot",
    "grape leaf spot": "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)",  # closest match; flagged in the report
    "potato early blight": "Potato___Early_blight", "potato late blight": "Potato___Late_blight",
    "squash powdery mildew": "Squash___Powdery_mildew", "strawberry leaf scorch": "Strawberry___Leaf_scorch",
    "tomato bacterial leaf spot": "Tomato___Bacterial_spot", "tomato early blight": "Tomato___Early_blight",
    "tomato late blight": "Tomato___Late_blight", "tomato leaf mold": "Tomato___Leaf_Mold",
    "tomato mosaic virus": "Tomato___Tomato_mosaic_virus", "tomato septoria leaf spot": "Tomato___Septoria_leaf_spot",
    "tomato yellow leaf curl virus": "Tomato___Tomato_Yellow_Leaf_Curl_Virus",
}
CORN_MAP = {"Blight": "Corn_(maize)___Northern_Leaf_Blight",
            "Gray_Leaf_Spot": "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot", "Healthy": "Corn_(maize)___healthy"}
POTATO_MAP = {"Phytopthora": "Potato___Late_blight", "Healthy": "Potato___healthy"}
TOMATO_VILLAGE_MAP = {"Early_blight": "Tomato___Early_blight", "Late_blight": "Tomato___Late_blight", "Healthy": "Tomato___healthy"}

# Background sources: skip anything that shows plants/leaves so "Background" never means "a leaf"
CALTECH_EXCLUDE = re.compile(r"bonsai|cactus|fern|flower|hibiscus|iris|palm|sunflower|grape|tomato|watermelon|mushroom|"
                             r"lettuce|leaf|tree|plant|\bcorn|cabbage|broccoli|strawberr|pineapple|banana|fruit|lotus|lily|"
                             r"lilly|mango|cereal|eggplant|vine|garden|ivy|grasshopper|mantis|butterfly|frog|snail|"
                             r"ladybug|caterpillar|iguana|chameleon|hummingbird", re.I)
DTD_EXCLUDE = {"veined", "frilly"}  # these categories are often leaves / lettuce
CALTECH_PER_CLASS, DTD_PER_CLASS = 14, 25


def phash(im):
    g = np.asarray(im.convert("L").resize((32, 32), Image.LANCZOS), dtype=np.float64)
    d = dctn(g, type=2, norm="ortho")[:8, :8]
    bits = (d > np.median(d)).flatten()
    return int(np.packbits(bits).view(">u8")[0])


def process(i, data):
    sha = hashlib.sha256(data).hexdigest()
    out = os.path.join(CACHE, f"{i:06d}.jpg")
    try:
        im = Image.open(io.BytesIO(data))
        im.load()
        im = im.convert("RGB")
    except Exception:
        return sha, None, None
    ph = phash(im)
    if not os.path.exists(out):
        w, h = im.size
        s = CACHE_SHORT_SIDE / min(w, h)
        if s < 1:
            im = im.resize((max(1, round(w * s)), max(1, round(h * s))), Image.BICUBIC)
        im.save(out, "JPEG", quality=93)
    return sha, ph, f"cache/{i:06d}.jpg"


def enumerate_sources(classes):
    prev_path = os.path.join(ROOT, "unified_training", "unified_manifest.csv")
    prev = pd.read_csv(prev_path) if os.path.exists(prev_path) else pd.DataFrame(columns=["dataset", "member", "split"])
    prev_split = {(d, m): s for d, m, s in zip(prev.dataset, prev.member, prev.split)}
    recs = []  # dataset, source, member, label, fixed_split, group

    def add(ds, src, member, label, fixed=None, group=None):
        recs.append(dict(dataset=ds, source=src, member=member, label=label, fixed_split=fixed, group=group))

    z = zipfile.ZipFile(ZIPS["PlantVillage"])
    for n in z.namelist():
        p = n.split("/")
        if len(p) == 5 and p[1] == "raw" and p[2] == "color" and p[4] and p[3] in classes:
            leaf = re.sub(r"\.\w+$", "", p[4].split("___")[-1]).strip()
            add("PlantVillage", "PlantVillage", n, p[3], prev_split.get(("PlantVillage", n)), f"PV/{p[3]}/{leaf}")

    z = zipfile.ZipFile(ZIPS["PlantDoc"])
    for n in z.namelist():
        p = n.split("/")
        if len(p) == 4 and p[3] and p[1] in ("train", "test") and p[2] in PLANTDOC_MAP:
            fixed = "test" if p[1] == "test" else prev_split.get(("PlantDoc", n))
            add("PlantDoc", "PlantDoc", n, PLANTDOC_MAP[p[2]], fixed)

    for ds, mp, depth_cls in [("Corn", CORN_MAP, 0), ("Potato", POTATO_MAP, 1)]:
        z = zipfile.ZipFile(ZIPS[ds])
        for n in z.namelist():
            p = n.split("/")
            if not n.endswith("/") and len(p) == depth_cls + 2 and p[depth_cls] in mp:
                add(ds, ds, n, mp[p[depth_cls]], prev_split.get((ds, n)))

    z = zipfile.ZipFile(ZIPS["TomatoVillage"])
    for n in z.namelist():
        p = n.split("/")
        if p[0].startswith("Variant-a") and len(p) == 4 and p[3] and p[2] in TOMATO_VILLAGE_MAP:
            base = re.sub(r"_\d+$", "", p[3].rsplit(".", 1)[0])  # IMG..._3.jpg is a crop of IMG....jpg
            add("TomatoVillage", "TomatoVillage", n, TOMATO_VILLAGE_MAP[p[2]], None, f"TV/{base}")

    z = zipfile.ZipFile(ZIPS["PlantWild"])
    for n in z.namelist():
        p = n.split("/")
        if len(p) == 3 and p[2] and p[1] in PLANTWILD_MAP:
            add("PlantWild", "PlantWild", n, PLANTWILD_MAP[p[1]])

    for d in CUSTOM_DIRS:
        for f in sorted(os.listdir(os.path.join(ROOT, d))):
            if f.lower().endswith((".jpg", ".jpeg", ".png", ".webp")):
                add("Custom", "file", os.path.join(d, f), "", "custom")
    return recs


def main():
    os.makedirs(CACHE, exist_ok=True)
    classes = sorted(n.split("/")[3] for n in zipfile.ZipFile(ZIPS["PlantVillage"]).namelist()
                     if n.count("/") == 4 and n.startswith("PlantVillage-Dataset-master/raw/color/") and n.endswith("/"))
    assert len(classes) == 38, len(classes)
    classes = classes + ["Background"]
    recs = enumerate_sources(set(classes))

    local = threading.local()

    def read(r):
        if r["source"] == "file":
            with open(os.path.join(ROOT, r["member"]), "rb") as f:
                return f.read()
        if not hasattr(local, "z"):
            local.z = {}
        if r["source"] not in local.z:
            local.z[r["source"]] = zipfile.ZipFile(ZIPS[r["source"]])
        return local.z[r["source"]].read(r["member"])

    results = {}
    with ThreadPoolExecutor(max_workers=8) as ex:
        futs = {}
        for i, r in enumerate(recs):
            futs[ex.submit(lambda i=i, r=r: process(i, read(r)))] = i
        for k, f in enumerate(futs):
            results[futs[f]] = f.result()
            if k % 5000 == 0:
                print(f"zip/file images: {k}/{len(recs)}", flush=True)

        # Background: stream the tar archives, sample a fixed number per category
        def tar_items(path, mode, cat_of, per_class, exclude):
            seen = {}
            with tarfile.open(path, mode) as t:
                for m in t:
                    if not m.isfile() or not m.name.lower().endswith((".jpg", ".jpeg", ".png")):
                        continue
                    cat = cat_of(m.name)
                    if cat is None or exclude(cat) or seen.get(cat, 0) >= per_class:
                        continue
                    seen[cat] = seen.get(cat, 0) + 1
                    yield m.name, cat, t.extractfile(m).read()

        bg = [
            (CALTECH_TAR, "r:", lambda n: n.split("/")[1] if n.count("/") == 2 else None, CALTECH_PER_CLASS,
             lambda c: bool(CALTECH_EXCLUDE.search(c)), "Caltech256"),
            (DTD_TAR, "r|gz", lambda n: n.split("/")[2] if n.startswith("dtd/images/") and n.count("/") == 3 else None,
             DTD_PER_CLASS, lambda c: c in DTD_EXCLUDE, "DTD"),
        ]
        bg_futs = []
        for path, mode, cat_of, per, excl, name in bg:
            for member, cat, data in tar_items(path, mode, cat_of, per, excl):
                i = len(recs)
                recs.append(dict(dataset="Background", source=name, member=member, label="Background",
                                 fixed_split=None, group=f"{name}/{cat}/{member}"))
                bg_futs.append((i, ex.submit(process, i, data)))
        for i, f in bg_futs:
            results[i] = f.result()
        print(f"background images: {len(bg_futs)}", flush=True)

    df = pd.DataFrame(recs)
    df.insert(0, "id", range(len(df)))
    df["sha256"] = [results[i][0] for i in df.id]
    df["phash"] = [results[i][1] for i in df.id]
    df["cache_path"] = [results[i][2] for i in df.id]
    cidx = {c: k for k, c in enumerate(classes)}
    df["class_index"] = df.label.map(cidx).fillna(-1).astype(int)
    bad = df.cache_path.isna()
    print(f"unreadable images dropped: {bad.sum()}")
    df = df[~bad].copy()
    df["phash"] = df.phash.astype("uint64")
    df.to_csv(os.path.join(OUT, "index.csv"), index=False)
    pd.Series(classes).to_csv(os.path.join(OUT, "classes.csv"), index=False, header=["class_name"])
    print(df.groupby(["dataset", df.fixed_split.fillna("-")]).size().unstack(fill_value=0))


if __name__ == "__main__":
    main()
