"""KrishiRakshak disease model - step 3: train one model on all six datasets + background.

MobileNetV3-Small (ImageNet init), 39 classes (38 PlantVillage + Background), 224x224 input.
Balanced sampling across datasets, strong augmentation, label smoothing, cosine LR, AMP.
Best checkpoint = highest mean of per-dataset validation accuracy.
Memory-lean for a 7 GB machine (2 loader workers, validation in the main process). Resume with --resume.
"""
import argparse
import json
import math
import os
import time

import pandas as pd
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision import models
from torchvision import transforms as T

OUT = os.path.dirname(os.path.abspath(__file__))
MEAN, STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
TRAIN_TF = T.Compose([
    T.RandomResizedCrop(224, scale=(0.35, 1.0)),
    T.RandomHorizontalFlip(),
    T.RandomVerticalFlip(0.2),
    T.TrivialAugmentWide(),
    T.ToTensor(),
    T.Normalize(MEAN, STD),
    T.RandomErasing(p=0.25),
])
EVAL_TF = T.Compose([T.Resize((224, 224)), T.ToTensor(), T.Normalize(MEAN, STD)])
DEFAULT_WEIGHTS = ("PlantVillage=0.35,PlantDoc=0.22,PlantWild=0.18,TomatoVillage=0.08,"
                   "Corn=0.06,Potato=0.06,Background=0.05")


class ImgDS(Dataset):
    def __init__(self, df, tf):
        self.p = [os.path.join(OUT, p) for p in df.cache_path]
        self.y = df.class_index.tolist()
        self.tf = tf

    def __len__(self):
        return len(self.p)

    def __getitem__(self, i):
        return self.tf(Image.open(self.p[i]).convert("RGB")), self.y[i]


def build_model(num_classes, pretrained=True):
    m = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.IMAGENET1K_V1 if pretrained else None)
    m.classifier[3] = nn.Linear(m.classifier[3].in_features, num_classes)
    return m


@torch.no_grad()
def predict(model, df, dev, bs=128):
    dl = DataLoader(ImgDS(df, EVAL_TF), batch_size=bs, num_workers=0)
    model.eval()
    out = []
    for x, _ in dl:
        with torch.autocast("cuda", enabled=dev == "cuda"):
            out.append(torch.softmax(model(x.to(dev)).float(), 1).cpu())
    return torch.cat(out)


def per_dataset_acc(model, df, dev):
    correct = predict(model, df, dev).argmax(1).numpy() == df.class_index.values
    return {ds: float(c.mean()) for ds, c in pd.Series(correct, index=df.index).groupby(df.dataset)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--samples_per_epoch", type=int, default=30000)
    ap.add_argument("--bs", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--wd", type=float, default=0.02)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--weights", default=DEFAULT_WEIGHTS)
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()

    torch.manual_seed(42)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    df = pd.read_csv(os.path.join(OUT, "manifest.csv"))
    df = df[df.dataset != "Custom"]
    classes = pd.read_csv(os.path.join(OUT, "classes.csv")).class_name.tolist()
    tr = df[df.split == "train"].reset_index(drop=True)
    va = df[df.split == "val"].reset_index(drop=True)

    ds_w = {k: float(v) for k, v in (kv.split("=") for kv in args.weights.split(","))}
    cnt = tr.groupby(["dataset", "class_index"]).size()
    raw = pd.Series([cnt[(d, c)] ** -0.5 for d, c in zip(tr.dataset, tr.class_index)])
    sw = raw / raw.groupby(tr.dataset).transform("sum") * tr.dataset.map(ds_w)
    sampler = WeightedRandomSampler(torch.tensor(sw.values, dtype=torch.double), args.samples_per_epoch, replacement=True)
    dl = DataLoader(ImgDS(tr, TRAIN_TF), batch_size=args.bs, sampler=sampler, num_workers=args.workers,
                    pin_memory=True, persistent_workers=True, drop_last=True)

    model = build_model(len(classes)).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
    steps, warm = args.epochs * len(dl), len(dl)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: (s + 1) / warm if s < warm else 0.5 * (1 + math.cos(math.pi * (s - warm) / max(1, steps - warm))))
    scaler = torch.amp.GradScaler(enabled=dev == "cuda")
    crit = nn.CrossEntropyLoss(label_smoothing=0.1)

    os.makedirs(os.path.join(OUT, "checkpoints"), exist_ok=True)
    os.makedirs(os.path.join(OUT, "results"), exist_ok=True)
    last_path = os.path.join(OUT, "checkpoints", "disease_model_last.pth")
    hist, best, start = [], -1.0, 1
    if args.resume and os.path.exists(last_path):
        ck = torch.load(last_path, map_location="cpu", weights_only=False)
        model.load_state_dict(ck["model_state_dict"])
        opt.load_state_dict(ck["optimizer_state_dict"])
        sched.load_state_dict(ck["scheduler_state_dict"])
        scaler.load_state_dict(ck["scaler_state_dict"])
        torch.set_rng_state(ck["torch_rng_state"])
        hist, best, start = ck["history"], ck["best_score"], ck["epoch"] + 1
        print(f"resumed from epoch {ck['epoch']} (best val_mean {best:.4f})", flush=True)
    print(f"train {len(tr)}  val {len(va)}  classes {len(classes)}  steps/epoch {len(dl)}  device {dev}", flush=True)

    for ep in range(start, args.epochs + 1):
        model.train()
        t0, tl, tc, tn = time.time(), 0.0, 0, 0
        for x, y in dl:
            x, y = x.to(dev, non_blocking=True), y.to(dev, non_blocking=True)
            with torch.autocast("cuda", enabled=dev == "cuda"):
                out = model(x)
                loss = crit(out, y)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            sched.step()
            tl += loss.item() * len(y)
            tc += (out.argmax(1) == y).sum().item()
            tn += len(y)
        vacc = per_dataset_acc(model, va, dev)
        score = sum(vacc.values()) / len(vacc)
        row = dict(epoch=ep, train_loss=tl / tn, train_acc=tc / tn, **{f"val_{k}": v for k, v in vacc.items()},
                   val_mean=score, lr=sched.get_last_lr()[0], minutes=(time.time() - t0) / 60)
        hist.append(row)
        pd.DataFrame(hist).to_csv(os.path.join(OUT, "results", "training_history.csv"), index=False)
        print(json.dumps({k: round(v, 4) if isinstance(v, float) else v for k, v in row.items()}), flush=True)
        ckpt = dict(model_state_dict=model.state_dict(), classes=classes, epoch=ep, architecture="mobilenet_v3_small",
                    input_size=224, preprocessing="Resize((224,224)) + ImageNet normalize",
                    background_class="Background", val_accuracy_per_dataset=vacc, val_mean_accuracy=score,
                    datasets=sorted(df.dataset.unique()), training_records=len(tr), args=vars(args),
                    experiment_status="EXPERIMENTAL - datasets pending label/license review")
        if score > best:
            best = score
            torch.save(ckpt, os.path.join(OUT, "checkpoints", "disease_model_best.pth"))
            print(f"  -> new best (val_mean {score:.4f})", flush=True)
        torch.save(dict(ckpt, optimizer_state_dict=opt.state_dict(), scheduler_state_dict=sched.state_dict(),
                        scaler_state_dict=scaler.state_dict(), torch_rng_state=torch.get_rng_state(),
                        history=hist, best_score=best), last_path)


if __name__ == "__main__":
    main()
