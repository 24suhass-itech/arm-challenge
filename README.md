# KrishiRakshak AI – Arm Challenge

Offline crop **disease** and **pest** identification from a single photo, designed to run on Arm devices
(Android phones, Raspberry Pi 5) without internet.

Full write-up with charts: [docs/KrishiRakshak_AI_Project_Report.pdf](docs/KrishiRakshak_AI_Project_Report.pdf)

## Models

| Model | Classes | Architecture | Size | File |
|---|---|---|---|---|
| Disease model | 38 crop disease / healthy classes + `Background` (not a leaf) | MobileNetV3-Small, 224×224 | 6 MB | `disease_model/checkpoints/disease_model_best.pth` |
| Pest model | 102 insect pests (IP102) | EfficientNet-B0, 256×256 | 16 MB | `pest_model/checkpoints/efficientnet_b0_best.pth` |

## Results (held-out test sets)

**Disease model**

| Test set | Images | Top-1 |
|---|---|---|
| PlantVillage (lab photos) | 8,175 | 99.4% |
| PlantDoc (field photos, official test) | 236 | 65.3% |
| PlantWild (field photos) | 356 | 71.3% |
| Tomato Village (field photos) | 58 | 93.1% |
| Corn | 142 | 94.4% |
| Potato | 76 | 97.4% |
| Background (non-leaf photos rejected) | 664 | 98.6% |

Field photos answered at confidence ≥ 0.7: 91.3% accuracy on 66% of photos.

**Pest model (IP102 official test, 22,619 images)**

| Measure | Result |
|---|---|
| Top-1 | 70.1% |
| Top-5 | 91.0% |
| Top-1 at confidence ≥ 0.71 | 90.1% (answers 54% of photos) |

All test sets were checked for exact (SHA-256) and near-duplicate (perceptual hash) copies of training
images; leaked copies were removed before training (4,419 in IP102 alone).

## Repository layout

```
disease_model/           disease model
  build_index.py         index + hash every source image, map labels to 39 classes
  make_splits.py         leakage-free train/val/test splits (manifest.csv)
  train.py               training (balanced multi-dataset sampling, AMP, --resume)
  evaluate.py            test-set evaluation
  test_custom_images.py  run the model on the photos in "test images/"
pest_model/              pest model
  prepare.py             cache + deduplicate IP102, build manifest.csv
  train.py               training (--arch efficientnet_b0 | mobilenet_v3_large | convnext_tiny)
  evaluate.py            test-set evaluation
docs/                    project report (PDF) and the script that builds it
test images/             19 web photos used as an extra real-world check
```

## Usage

```bash
pip install -r requirements.txt

# predict with a saved model
python - <<'EOF'
import torch
from PIL import Image
from disease_model.train import build_model, EVAL_TF
ck = torch.load("disease_model/checkpoints/disease_model_best.pth", map_location="cpu", weights_only=False)
m = build_model(len(ck["classes"]), pretrained=False); m.load_state_dict(ck["model_state_dict"]); m.eval()
p = torch.softmax(m(EVAL_TF(Image.open("test images/apple_scab_1.jpg").convert("RGB"))[None]), 1)[0]
print(ck["classes"][p.argmax()], float(p.max()))
EOF
```

Retraining needs the source datasets (not included) in `datasets/`: PlantVillage, PlantDoc, PlantWild,
Tomato Village, Corn and Potato leaf disease sets, Caltech-256 and DTD (background), and IP102.

```bash
python disease_model/build_index.py && python disease_model/make_splits.py
python disease_model/train.py && python disease_model/evaluate.py
python pest_model/prepare.py && python pest_model/train.py && python pest_model/evaluate.py
python docs/make_report.py
```

## Status

Done: disease detection, non-leaf rejection, pest detection.
Next: "no pest" class, ExecuTorch export, Llama 3.2 (ExecuTorch) advisory module, severity levels, local languages.

## Licences

The datasets used for training have their own licences (several are research / non-commercial only).
Review them before any commercial use of the trained weights.
