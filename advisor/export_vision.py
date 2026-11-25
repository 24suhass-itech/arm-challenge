"""Export the disease and pest classifiers to ExecuTorch (.pte, XNNPACK = Arm CPU) and check them against PyTorch.

    C:\\Users\\HP\\et-venv\\Scripts\\python.exe advisor\\export_vision.py

Outputs advisor/models/disease_model.pte and advisor/models/pest_model.pte plus *_meta.json (classes, input size,
normalisation), so the phone / Raspberry Pi app needs nothing but the .pte files and the JSON.
"""
import importlib.util
import json
import os

import sys

import torch
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
OUT = os.path.join(HERE, "models")
MEAN, STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]


def _module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def load_torch(which):
    if which == "disease_model":
        tr = _module(os.path.join(ROOT, "disease_model", "train.py"), "disease_train")
        ck = torch.load(os.path.join(ROOT, "disease_model", "checkpoints", "disease_model_best.pth"),
                        map_location="cpu", weights_only=False)
        m = tr.build_model(len(ck["classes"]), pretrained=False)
        size = 224
    else:
        tr = _module(os.path.join(ROOT, "pest_model", "train.py"), "pest_train")
        ck = torch.load(os.path.join(ROOT, "pest_model", "checkpoints", "efficientnet_b0_best.pth"),
                        map_location="cpu", weights_only=False)
        m = tr.build_model(ck["architecture"], len(ck["classes"]), pretrained=False)
        size = ck["input_size"]
    m.load_state_dict(ck["model_state_dict"])
    return m.eval(), ck["classes"], size


def export(which):
    from export_llama import _use_bundled_flatc
    _use_bundled_flatc()
    from executorch.backends.xnnpack.partition.xnnpack_partitioner import XnnpackPartitioner
    from executorch.exir import to_edge_transform_and_lower
    m, classes, size = load_torch(which)
    ep = torch.export.export(m, (torch.randn(1, 3, size, size),))
    prog = to_edge_transform_and_lower(ep, partitioner=[XnnpackPartitioner()]).to_executorch()
    path = os.path.join(OUT, f"{which}.pte")
    with open(path, "wb") as f:
        f.write(prog.buffer)
    meta = dict(classes=classes, input_size=size, mean=MEAN, std=STD, resize="squash to input_size x input_size",
                output="logits (apply softmax)")
    with open(os.path.join(OUT, f"{which}_meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=1)
    return m, path, size


def check(which, m, path, size, images):
    from executorch.runtime import Runtime
    import torchvision.transforms as T
    tf = T.Compose([T.Resize((size, size)), T.ToTensor(), T.Normalize(MEAN, STD)])
    method = Runtime.get().load_program(path).load_method("forward")
    agree, max_diff = 0, 0.0
    for p in images:
        x = tf(Image.open(p).convert("RGB"))[None]
        ref = torch.softmax(m(x), 1)
        got = torch.softmax(method.execute([x])[0], 1)
        agree += int(ref.argmax() == got.argmax())
        max_diff = max(max_diff, float((ref - got).abs().max()))
    print(f"{which}: {os.path.getsize(path) / 1e6:.1f} MB, same top-1 on {agree}/{len(images)} images, "
          f"max probability difference {max_diff:.2e}")


if __name__ == "__main__":
    import glob
    os.makedirs(OUT, exist_ok=True)
    imgs = sorted(glob.glob(os.path.join(ROOT, "test images", "*.jpg")))
    with torch.no_grad():
        for w in ["disease_model", "pest_model"]:
            m, path, size = export(w)
            check(w, m, path, size, imgs)
