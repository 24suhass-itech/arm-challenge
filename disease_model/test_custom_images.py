"""Run this disease model and the two earlier models (unified, six_dataset) on the custom test images and flag images an older model was trained on.

Labels come from filenames (e.g. 'apple_black_rot_2.jpg'); 'images (N).jpg' files are unlabeled.
Overlap flags come from near_duplicate_pairs.csv (distance <= 4): a custom image that matches a
PlantDoc-train image was seen by unified and six_dataset; one that matches PlantWild was seen by six_dataset.
This model excluded both.
"""
import os
import re

import pandas as pd
import torch
from PIL import Image

from evaluate import MODELS, load
from train import EVAL_TF, OUT

ROOT = os.path.dirname(OUT)


def label_from_name(fname, classes):
    stem = re.sub(r"(_\d+)?(\s*\(\d+\))?$", "", os.path.splitext(os.path.basename(fname))[0]).lower()
    norm = lambda c: re.sub(r"[^a-z]+", "_", c.lower().replace("(maize)", "")).strip("_")
    for c in classes:
        if "___" not in c:
            continue
        crop, disease = (norm(p) for p in c.split("___"))
        if stem == f"{crop}_{disease}" or (disease.startswith(crop) and stem == disease):
            return c
    return None


def main():
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    man = pd.read_csv(os.path.join(OUT, "manifest.csv"))
    idx = pd.read_csv(os.path.join(OUT, "index.csv"), usecols=["id", "dataset", "member"])
    cust = man[man.dataset == "Custom"].reset_index(drop=True)

    pairs = pd.read_csv(os.path.join(OUT, "near_duplicate_pairs.csv"))
    pairs = pairs[pairs.dist <= 4]
    ds_of = idx.set_index("id").dataset
    member_of = idx.set_index("id").member
    seen_by = {}
    for r in pairs.itertuples():
        for c, o in [(r.a, r.b), (r.b, r.a)]:
            if ds_of[c] == "Custom" and ds_of[o] in ("PlantDoc", "PlantWild"):
                if ds_of[o] == "PlantDoc" and "/train/" in member_of[o]:
                    seen_by.setdefault(c, set()).update({"unified", "six_dataset"})
                elif ds_of[o] == "PlantWild":
                    seen_by.setdefault(c, set()).add("six_dataset")

    classes = load(next(iter(MODELS.values())))[1]
    cust["expected"] = [label_from_name(m, classes) for m in cust.member]
    cust["seen_in_training_by"] = [",".join(sorted(seen_by.get(i, []))) for i in cust.id]
    x = torch.stack([EVAL_TF(Image.open(os.path.join(ROOT, m)).convert("RGB")) for m in cust.member])
    for name, path in MODELS.items():
        m, cls = load(path)
        m.eval().to(dev)
        with torch.no_grad():
            pr = torch.softmax(m(x.to(dev)), 1).cpu()
        conf, top = pr.max(1)
        top3 = pr.topk(3, 1).indices
        cust[f"{name}_pred"] = [cls[i] for i in top]
        cust[f"{name}_conf"] = (100 * conf.numpy()).round(1)
        cust[f"{name}_correct"] = [(e == cls[i]) if e else None for e, i in zip(cust.expected, top)]
        cust[f"{name}_top3"] = [(e in [cls[j] for j in t]) if e else None for e, t in zip(cust.expected, top3)]
    os.makedirs(os.path.join(OUT, "results"), exist_ok=True)
    cust.drop(columns=["split", "comp", "sha256", "cache_path", "class_index", "label", "source"]) \
        .to_csv(os.path.join(OUT, "results", "custom_image_results.csv"), index=False)

    pd.set_option("display.width", 250, "display.max_colwidth", 40)
    show = cust.assign(image=cust.member.map(os.path.basename))
    print(show[["image", "expected", "seen_in_training_by", ] + [f"{n}_pred" for n in MODELS]].to_string(index=False))
    lab = cust[cust.expected.notna()]
    for name in MODELS:
        ok = lab[f"{name}_correct"].astype(bool)
        clean = ~lab.seen_in_training_by.str.contains(name)
        print(f"{name:13s} all labelled: {ok.sum()}/{len(lab)}  top3 {lab[f'{name}_top3'].astype(bool).sum()}/{len(lab)}"
              f"  | only images it never trained on: {ok[clean].sum()}/{clean.sum()}")


if __name__ == "__main__":
    main()
