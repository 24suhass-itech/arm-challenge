"""KrishiRakshak disease model - step 4: test-set evaluation and comparison with earlier models.

Models compared on these test splits:
  disease_model : disease_model/checkpoints/disease_model_best.pth      (39 classes incl. Background)
  unified       : unified_training/checkpoints/unified_best.pth        (earlier model, 38 classes; PlantVillage/PlantDoc/Corn/Potato)
  six_dataset   : six_dataset_training/krishirakshak_6dataset_best.pth (earlier model, 38 classes)
Fairness:
  - unified is fair everywhere (these splits reuse its splits; it never saw Tomato Village / PlantWild at all).
  - six_dataset trained on ALL its Corn, Potato, Tomato Village and PlantWild records, so those scores are marked
    contaminated. Its PlantVillage and PlantDoc-test scores are fair.
  - 38-class models cannot predict Background, so they are not scored on the Background test set.
"""
import os

import pandas as pd
import torch

from train import OUT, build_model, predict

ROOT = os.path.dirname(OUT)
MODELS = {
    "disease_model": os.path.join(OUT, "checkpoints", "disease_model_best.pth"),
    "unified": os.path.join(ROOT, "unified_training", "checkpoints", "unified_best.pth"),
    "six_dataset": os.path.join(ROOT, "six_dataset_training", "krishirakshak_6dataset_best.pth"),
}
# the older models are not part of the published repository; compare against them only when present locally
MODELS = {k: v for k, v in MODELS.items() if os.path.exists(v)}
CONTAMINATED = {"six_dataset": {"Corn", "Potato", "TomatoVillage", "PlantWild"}}


def load(path):
    ck = torch.load(path, map_location="cpu", weights_only=False)
    m = build_model(len(ck["classes"]), pretrained=False)
    m.load_state_dict(ck["model_state_dict"])
    return m, ck["classes"]


def main():
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    man = pd.read_csv(os.path.join(OUT, "manifest.csv"))
    te = man[man.split == "test"].reset_index(drop=True)
    rows, preds = [], {}
    for name, path in MODELS.items():
        m, classes = load(path)
        m.to(dev)
        sub = te if len(classes) == 39 else te[te.dataset != "Background"].reset_index(drop=True)
        pr = predict(m, sub, dev)
        conf, idx = pr.max(1)
        top3 = pr.topk(3, 1).indices
        p = sub.copy()
        p["predicted"] = [classes[i] for i in idx]
        p["confidence"] = conf.numpy()
        p["correct"] = idx.numpy() == sub.class_index.values
        p["top3_correct"] = (top3 == torch.tensor(sub.class_index.values)[:, None]).any(1).numpy()
        p["crop_correct"] = p.predicted.str.split("___").str[0] == p.label.str.split("___").str[0]
        preds[name] = p
        for ds, g in p.groupby("dataset"):
            rows.append(dict(model=name, dataset=ds, test_images=len(g), top1=100 * g.correct.mean(),
                             top3=100 * g.top3_correct.mean(), crop=100 * g.crop_correct.mean(),
                             macro_class_acc=100 * g.groupby("label").correct.mean().mean(),
                             note="contaminated: trained on these" if ds in CONTAMINATED.get(name, ()) else ""))
        print(f"evaluated {name}", flush=True)
        del m
        torch.cuda.empty_cache()

    s = pd.DataFrame(rows).round(2)
    os.makedirs(os.path.join(OUT, "results"), exist_ok=True)
    s.to_csv(os.path.join(OUT, "results", "test_summary.csv"), index=False)
    print(s.pivot_table(index="dataset", columns="model", values="top1").round(2).to_string())
    print("\n(contaminated six_dataset cells:", s[s.note != ""][["dataset"]].dataset.tolist(), ")")

    dm = preds["disease_model"]
    dm.drop(columns=["member"]).to_csv(os.path.join(OUT, "results", "test_predictions.csv"), index=False)
    plants = dm[dm.dataset != "Background"]
    bg = dm[dm.dataset == "Background"]
    print(f"\nBackground test: {100 * bg.correct.mean():.1f}% of {len(bg)} non-leaf photos rejected as Background")
    print(f"Plant test images wrongly called Background: {(plants.predicted == 'Background').sum()} / {len(plants)}")

    pdc = pd.DataFrame({k: preds[k].query("dataset == 'PlantDoc'").groupby("label").correct.mean() * 100
                        for k in MODELS}).round(1)
    pdc.insert(0, "images", dm.query("dataset == 'PlantDoc'").groupby("label").size())
    pdc.sort_values("disease_model").to_csv(os.path.join(OUT, "results", "plantdoc_per_class.csv"))
    print("\nPlantDoc per class (%):\n", pdc.sort_values("disease_model").to_string())

    field = plants[plants.dataset != "PlantVillage"]
    print("\nTop confusions on field photos (disease_model):")
    print(field[~field.correct].groupby(["label", "predicted"]).size().sort_values(ascending=False).head(12).to_string())
    for th in [0.5, 0.7, 0.9]:
        k = field[field.confidence >= th]
        print(f"field photos conf>={th}: coverage {100 * len(k) / len(field):.0f}%  accuracy {100 * k.correct.mean():.1f}%")


if __name__ == "__main__":
    main()
