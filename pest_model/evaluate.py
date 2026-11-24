"""KrishiRakshak pest model - step 3: final evaluation on the official IP102 test split (22,619 images).

Reports top-1/3/5, macro (per-class mean) accuracy, crop-level accuracy, per-crop accuracy, and
accuracy vs. coverage at confidence thresholds. Writes results/test_summary.csv, test_predictions.csv,
per_class.csv, confidence_thresholds.csv.
"""
import argparse
import os

import pandas as pd
import torch
from torch.utils.data import DataLoader

from train import ImgDS, OUT, build_model, tfs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=os.path.join(OUT, "checkpoints", "efficientnet_b0_best.pth"))
    args = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    m = build_model(ck["architecture"], len(ck["classes"]), pretrained=False)
    m.load_state_dict(ck["model_state_dict"])
    m.eval().to(dev)
    print(f"checkpoint epoch {ck['epoch']}  val top1 {ck['val_top1']:.4f}", flush=True)

    cls = pd.read_csv(os.path.join(OUT, "classes.csv"))
    crop_of = cls.crop.values
    te = pd.read_csv(os.path.join(OUT, "manifest.csv")).query("split == 'test'").reset_index(drop=True)
    dl = DataLoader(ImgDS(te, tfs(ck["input_size"])[1]), batch_size=128, num_workers=2)
    out = []
    with torch.no_grad():
        for x, _ in dl:
            with torch.autocast("cuda", enabled=dev == "cuda"):
                out.append(torch.softmax(m(x.to(dev)).float(), 1).cpu())
    pr = torch.cat(out)
    y = torch.tensor(te.class_index.values)
    conf, top1 = pr.max(1)
    tk = pr.topk(5, 1).indices
    te["predicted"] = cls.class_name.values[top1.numpy()]
    te["confidence"] = conf.numpy()
    te["correct"] = (top1 == y).numpy()
    te["top3"] = (tk[:, :3] == y[:, None]).any(1).numpy()
    te["top5"] = (tk == y[:, None]).any(1).numpy()
    te["crop_correct"] = crop_of[top1.numpy()] == te.crop.values
    for k in range(5):
        te[f"pred{k + 1}"] = cls.class_name.values[tk[:, k].numpy()]

    res = os.path.join(OUT, "results")
    te.drop(columns=["member", "sha256", "comp"], errors="ignore").to_csv(
        os.path.join(res, "test_predictions.csv"), index=False)
    summ = dict(test_images=len(te), top1=te.correct.mean(), top3=te.top3.mean(), top5=te.top5.mean(),
                macro_top1=te.groupby("class_index").correct.mean().mean(), crop_level=te.crop_correct.mean())
    pd.DataFrame([summ]).to_csv(os.path.join(res, "test_summary.csv"), index=False)
    print("\nOVERALL TEST")
    for k, v in summ.items():
        print(f"  {k:12s} {v if k == 'test_images' else f'{100 * v:.2f}%'}")

    print("\nPER CROP")
    pc = te.groupby("crop").agg(images=("correct", "size"), top1=("correct", "mean"), top3=("top3", "mean"),
                                top5=("top5", "mean"))
    print((pc.assign(**{c: (100 * pc[c]).round(1) for c in ["top1", "top3", "top5"]})).to_string())

    per = te.groupby(["class_index", "label", "crop"]).agg(images=("correct", "size"), top1=("correct", "mean"),
                                                             top5=("top5", "mean")).reset_index()
    per.sort_values("top1").to_csv(os.path.join(res, "per_class.csv"), index=False)
    print(f"\nclasses >=90% top1: {(per.top1 >= .9).sum()}   70-90%: {((per.top1 >= .7) & (per.top1 < .9)).sum()}"
          f"   50-70%: {((per.top1 >= .5) & (per.top1 < .7)).sum()}   <50%: {(per.top1 < .5).sum()}")
    print("\nworst 10 classes:\n", per.sort_values("top1").head(10).round(3).to_string(index=False))
    print("\nbest 10 classes:\n", per.sort_values("top1", ascending=False).head(10).round(3).to_string(index=False))

    rows = []
    for th in [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
        k = te[te.confidence >= th]
        rows.append(dict(threshold=th, coverage=len(k) / len(te), accuracy=k.correct.mean()))
    ct = pd.DataFrame(rows)
    ct.to_csv(os.path.join(res, "confidence_thresholds.csv"), index=False)
    print("\nCONFIDENCE THRESHOLDS\n", ct.assign(coverage=(100 * ct.coverage).round(1),
                                                accuracy=(100 * ct.accuracy).round(1)).to_string(index=False))
    # smallest threshold reaching 90% accuracy
    for th in [i / 100 for i in range(20, 100)]:
        k = te[te.confidence >= th]
        if len(k) and k.correct.mean() >= 0.90:
            print(f"\n>=90% accuracy reached at confidence >= {th:.2f}: answers {100 * len(k) / len(te):.1f}% of photos "
                  f"with {100 * k.correct.mean():.1f}% accuracy")
            break
    print("\ntop confusions:\n", te[~te.correct].groupby(["label", "predicted"]).size()
          .sort_values(ascending=False).head(10).to_string())


if __name__ == "__main__":
    main()
