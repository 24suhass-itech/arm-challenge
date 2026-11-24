"""KrishiRakshak disease model - step 2: leakage-free train/val/test splits.

Images are linked into connected components when they are
  - byte-identical (same SHA-256),
  - in the same group (same PlantVillage leaf, or crops of the same Tomato Village photo), or
  - near-duplicates (perceptual-hash Hamming distance <= --threshold), except PlantVillage-vs-PlantVillage
    (PlantVillage photos share near-identical backgrounds; its leaf grouping is used instead).
A whole component always lands in one split, so no photo can appear in both training and evaluation.

Rules
  - component split = highest-priority fixed split in it (custom > test > val > train)
  - members fixed to a lower split are dropped (e.g. a PlantDoc-train image that duplicates a PlantDoc-test image)
  - components touching a custom test image are removed from train/val/test entirely
  - components whose free (non-fixed) members disagree on the label drop those members
  - free components are split 70/15/15 per dataset, stratified by label
  - in val/test, only one free image per component per dataset is kept (no double counting of crops/copies)
"""
import argparse
import os

import numpy as np
import pandas as pd

OUT = os.path.dirname(os.path.abspath(__file__))
PRIORITY = {"custom": 4, "test": 3, "val": 2, "train": 1}
FIXED_EVAL_DATASETS = {"PlantVillage", "PlantDoc", "Corn", "Potato"}
SEED = 42


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


def near_pairs(h, query_idx, max_d, chunk=256):
    """All (i, j, d) with i in query_idx, j any index != i, hamming(h[i], h[j]) <= max_d."""
    out = []
    for s in range(0, len(query_idx), chunk):
        q = query_idx[s:s + chunk]
        d = np.bitwise_count(h[q][:, None] ^ h[None, :])
        qi, j = np.nonzero(d <= max_d)
        i = q[qi]
        keep = i != j
        out.append(np.stack([i[keep], j[keep], d[qi[keep], j[keep]]], 1))
    return np.concatenate(out) if out else np.zeros((0, 3), int)


def main():
    ap = argparse.ArgumentParser()
    # 4: true copies sit at 0-4; at 6-8 plain-background photos (PlantVillage, white studio shots) collide
    ap.add_argument("--threshold", type=int, default=4)
    args = ap.parse_args()

    df = pd.read_csv(os.path.join(OUT, "index.csv"), dtype={"phash": "uint64"}, keep_default_na=False,
                     na_values={"fixed_split": [""], "group": [""]})
    n = len(df)
    pos = np.arange(n)
    h = df.phash.values.astype(np.uint64)
    is_pv = (df.dataset == "PlantVillage").values

    # candidate near-duplicate pairs (saved up to distance 12 for review / threshold calibration)
    pairs = near_pairs(h, pos[~is_pv], 12)
    pairs = pairs[~(is_pv[pairs[:, 0]] & is_pv[pairs[:, 1]])]
    pairs = pairs[(pairs[:, 0] < pairs[:, 1]) | is_pv[pairs[:, 1]]]  # each unordered pair once
    pdf = pd.DataFrame({"a": df.id.values[pairs[:, 0]], "b": df.id.values[pairs[:, 1]], "dist": pairs[:, 2],
                        "a_dataset": df.dataset.values[pairs[:, 0]], "b_dataset": df.dataset.values[pairs[:, 1]],
                        "a_label": df.label.values[pairs[:, 0]], "b_label": df.label.values[pairs[:, 1]]})
    pdf.to_csv(os.path.join(OUT, "near_duplicate_pairs.csv"), index=False)
    print("near-duplicate pairs by distance:\n", pdf.dist.value_counts().sort_index().to_string())

    dsu = DSU(n)
    for a, b, d in pairs:
        if d <= args.threshold:
            dsu.union(a, b)
    for key in ["sha256", "group"]:
        for _, idx in df.groupby(key).indices.items():
            for k in idx[1:]:
                dsu.union(idx[0], k)
    df["comp"] = [dsu.find(i) for i in range(n)]

    df["prio"] = df.fixed_split.map(PRIORITY).fillna(0).astype(int)
    comp_prio = df.groupby("comp").prio.transform("max")
    inv = {v: k for k, v in PRIORITY.items()}
    df["split"] = df.fixed_split
    df["drop_reason"] = ""

    fixed = df.prio > 0
    lower = fixed & (df.prio < comp_prio) & (df.dataset != "Custom")
    df.loc[lower, "drop_reason"] = "duplicate of a " + comp_prio[lower].map(inv) + " image"
    free = ~fixed
    # free images only JOIN a fixed component when it is a training component; copies of val/test/custom
    # images are dropped so the official / earlier-model evaluation sets stay exactly as they were
    joins_train = free & (comp_prio == PRIORITY["train"])
    df.loc[joins_train, "split"] = "train"
    to_eval = free & (comp_prio > PRIORITY["train"])
    df.loc[to_eval, "drop_reason"] = "duplicate of a " + comp_prio[to_eval].map(inv) + " image"

    # label conflicts among free members
    live = (df.drop_reason == "") & (df.dataset != "Custom")
    nlab = df[live].groupby("comp").label.nunique()
    conflict = df.comp.isin(nlab[nlab > 1].index) & free & live
    df.loc[conflict, "drop_reason"] = "near-duplicate with a different label"

    # split the remaining free components per dataset, stratified by label
    rng = np.random.default_rng(SEED)
    todo = df[free & (df.drop_reason == "") & df.split.isna()]
    comps = todo.groupby("comp").agg(dataset=("dataset", "first"), label=("label", "first"))
    assign = {}
    for (ds, _), g in comps.groupby(["dataset", "label"]):
        c = g.index.values.copy()
        if ds in FIXED_EVAL_DATASETS:  # their val/test sets are fixed (official / earlier model); extras only add training data
            assign.update({comp: "train" for comp in c})
            continue
        rng.shuffle(c)
        k = len(c)
        n_val = int(round(k * 0.15)) if k >= 4 else 0
        n_test = int(round(k * 0.15)) if k >= 4 else 0
        for t, comp in enumerate(c):
            assign[comp] = "val" if t < n_val else ("test" if t < n_val + n_test else "train")
    m = free & (df.drop_reason == "") & df.split.isna()
    df.loc[m, "split"] = df.loc[m, "comp"].map(assign)

    # one free image per component per dataset in val/test (prefer the un-cropped original)
    ev = free & (df.drop_reason == "") & df.split.isin(["val", "test"])
    order = df[ev].assign(cropped=df.member.str.contains(r"_\d+\.\w+$", regex=True)).sort_values(["cropped", "id"])
    dup = order.duplicated(["comp", "dataset"])
    df.loc[order.index[dup], "drop_reason"] = "extra copy in val/test"

    # exact duplicates inside the same split
    live = df.drop_reason == ""
    exact = live & df[live].duplicated(["sha256", "split"]).reindex(df.index, fill_value=False)
    df.loc[exact, "drop_reason"] = "exact duplicate"

    keep = (df.drop_reason == "") & (df.dataset != "Custom")
    df[~keep & (df.dataset != "Custom")][["id", "dataset", "member", "label", "fixed_split", "drop_reason"]] \
        .to_csv(os.path.join(OUT, "dropped.csv"), index=False)
    man = df[keep | (df.dataset == "Custom")][["id", "dataset", "source", "member", "label", "class_index", "split",
                                                "comp", "sha256", "cache_path"]]
    man.to_csv(os.path.join(OUT, "manifest.csv"), index=False)

    print(f"\nthreshold = {args.threshold}")
    print("\ndropped:\n", df[~keep & (df.dataset != "Custom")].groupby(["dataset", "drop_reason"]).size().to_string())
    print("\nfinal manifest:\n", man.groupby(["dataset", "split"]).size().unstack(fill_value=0).to_string())


if __name__ == "__main__":
    main()
