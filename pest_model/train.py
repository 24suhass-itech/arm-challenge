"""KrishiRakshak pest model - step 2: train a 102-class IP102 pest classifier.

Backbone selectable (--arch): efficientnet_b0 (default), mobilenet_v3_large, convnext_tiny; ImageNet init.
Natural class distribution (the test set has the same long tail), strong augmentation + CutMix/MixUp,
label smoothing, AdamW, warmup + cosine, AMP. Best epoch = highest val top-1. Resume with --resume.
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
from torch.utils.data import DataLoader, Dataset
from torchvision import models
from torchvision import transforms as T
from torchvision.transforms import v2

OUT = os.path.dirname(os.path.abspath(__file__))
MEAN, STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]


def tfs(size):
    train = T.Compose([
        T.RandomResizedCrop(size, scale=(0.25, 1.0)),
        T.RandomHorizontalFlip(),
        T.RandomVerticalFlip(0.2),
        T.TrivialAugmentWide(),
        T.ToTensor(),
        T.Normalize(MEAN, STD),
        T.RandomErasing(p=0.2),
    ])
    ev = T.Compose([T.Resize((size, size)), T.ToTensor(), T.Normalize(MEAN, STD)])
    return train, ev


class ImgDS(Dataset):
    def __init__(self, df, tf):
        self.p = [os.path.join(OUT, p) for p in df.cache_path]
        self.y = df.class_index.tolist()
        self.tf = tf

    def __len__(self):
        return len(self.p)

    def __getitem__(self, i):
        return self.tf(Image.open(self.p[i]).convert("RGB")), self.y[i]


def build_model(arch, num_classes, pretrained=True):
    if arch == "efficientnet_b0":
        m = models.efficientnet_b0(weights="IMAGENET1K_V1" if pretrained else None)
        m.classifier[1] = nn.Linear(m.classifier[1].in_features, num_classes)
    elif arch == "mobilenet_v3_large":
        m = models.mobilenet_v3_large(weights="IMAGENET1K_V2" if pretrained else None)
        m.classifier[3] = nn.Linear(m.classifier[3].in_features, num_classes)
    elif arch == "convnext_tiny":
        m = models.convnext_tiny(weights="IMAGENET1K_V1" if pretrained else None)
        m.classifier[2] = nn.Linear(m.classifier[2].in_features, num_classes)
    else:
        raise ValueError(arch)
    return m


@torch.no_grad()
def predict(model, df, dev, tf, bs=128):
    dl = DataLoader(ImgDS(df, tf), batch_size=bs, num_workers=0)
    model.eval()
    out = []
    for x, _ in dl:
        with torch.autocast("cuda", enabled=dev == "cuda"):
            out.append(torch.softmax(model(x.to(dev)).float(), 1).cpu())
    return torch.cat(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arch", default="efficientnet_b0")
    ap.add_argument("--size", type=int, default=256)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--bs", type=int, default=48)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--wd", type=float, default=0.05)
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--mix_p", type=float, default=0.5)
    ap.add_argument("--tag", default="")
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()
    tag = args.tag or args.arch

    torch.manual_seed(42)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    df = pd.read_csv(os.path.join(OUT, "manifest.csv"))
    classes = pd.read_csv(os.path.join(OUT, "classes.csv")).class_name.tolist()
    tr = df[df.split == "train"].reset_index(drop=True)
    va = df[df.split == "val"].reset_index(drop=True)
    train_tf, eval_tf = tfs(args.size)
    dl = DataLoader(ImgDS(tr, train_tf), batch_size=args.bs, shuffle=True, num_workers=args.workers,
                    pin_memory=True, persistent_workers=True, drop_last=True)
    mix = v2.RandomChoice([v2.CutMix(num_classes=len(classes)), v2.MixUp(num_classes=len(classes), alpha=0.2)])

    model = build_model(args.arch, len(classes)).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
    steps, warm = args.epochs * len(dl), len(dl)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: (s + 1) / warm if s < warm else 0.5 * (1 + math.cos(math.pi * (s - warm) / max(1, steps - warm))))
    scaler = torch.amp.GradScaler(enabled=dev == "cuda")
    crit = nn.CrossEntropyLoss(label_smoothing=0.1)

    ck_dir = os.path.join(OUT, "checkpoints")
    os.makedirs(ck_dir, exist_ok=True)
    os.makedirs(os.path.join(OUT, "results"), exist_ok=True)
    last_path = os.path.join(ck_dir, f"{tag}_last.pth")
    hist, best, start = [], -1.0, 1
    if args.resume and os.path.exists(last_path):
        ck = torch.load(last_path, map_location="cpu", weights_only=False)
        model.load_state_dict(ck["model_state_dict"])
        opt.load_state_dict(ck["optimizer_state_dict"])
        sched.load_state_dict(ck["scheduler_state_dict"])
        scaler.load_state_dict(ck["scaler_state_dict"])
        torch.set_rng_state(ck["torch_rng_state"])
        hist, best, start = ck["history"], ck["best_score"], ck["epoch"] + 1
        print(f"resumed from epoch {ck['epoch']} (best val {best:.4f})", flush=True)
    print(f"arch {args.arch}  size {args.size}  train {len(tr)}  val {len(va)}  steps/epoch {len(dl)}  {dev}",
          flush=True)

    for ep in range(start, args.epochs + 1):
        model.train()
        t0, tl, tn = time.time(), 0.0, 0
        for x, y in dl:
            x, y = x.to(dev, non_blocking=True), y.to(dev, non_blocking=True)
            if torch.rand(1).item() < args.mix_p:
                x, y = mix(x, y)
            with torch.autocast("cuda", enabled=dev == "cuda"):
                loss = crit(model(x), y)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            sched.step()
            tl += loss.item() * len(x)
            tn += len(x)
        pr = predict(model, va, dev, eval_tf)
        yv = torch.tensor(va.class_index.values)
        top1 = (pr.argmax(1) == yv).float().mean().item()
        top5 = (pr.topk(5, 1).indices == yv[:, None]).any(1).float().mean().item()
        row = dict(epoch=ep, train_loss=tl / tn, val_top1=top1, val_top5=top5, lr=sched.get_last_lr()[0],
                   minutes=(time.time() - t0) / 60)
        hist.append(row)
        pd.DataFrame(hist).to_csv(os.path.join(OUT, "results", f"{tag}_history.csv"), index=False)
        print(json.dumps({k: round(v, 4) if isinstance(v, float) else v for k, v in row.items()}), flush=True)
        ckpt = dict(model_state_dict=model.state_dict(), classes=classes, epoch=ep, architecture=args.arch,
                    input_size=args.size, preprocessing=f"Resize(({args.size},{args.size})) + ImageNet normalize",
                    val_top1=top1, val_top5=top5, dataset="IP102 (deduplicated)", args=vars(args))
        if top1 > best:
            best = top1
            torch.save(ckpt, os.path.join(ck_dir, f"{tag}_best.pth"))
            print(f"  -> new best (val top1 {top1:.4f})", flush=True)
        torch.save(dict(ckpt, optimizer_state_dict=opt.state_dict(), scheduler_state_dict=sched.state_dict(),
                        scaler_state_dict=scaler.state_dict(), torch_rng_state=torch.get_rng_state(),
                        history=hist, best_score=best), last_path)


if __name__ == "__main__":
    main()
