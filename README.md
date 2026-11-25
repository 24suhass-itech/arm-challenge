# KrishiRakshak AI – Arm Challenge

Offline crop doctor for farmers: take one photo of a leaf or insect and get the **disease or pest**, its
**severity**, and **treatment advice for the farmer's region and season, in their own language**, with no
internet. Every model runs on the Arm CPU of a **Raspberry Pi 5** through **ExecuTorch**, and the advice is
written by **Llama 3.2 1B-Instruct**.

Benchmark report (timing, memory and power on the laptop and the Raspberry Pi 5):
[docs/KrishiRakshak_AI_Benchmark_Results.pdf](docs/KrishiRakshak_AI_Benchmark_Results.pdf)

## How it works

```
photo ──► VISION PIPELINE ─────────────────────────────►  ADVISORY PIPELINE ─────────────────────────────► answer
          disease_model.pte  (39 classes)                  knowledge base  (38 diseases, 102 pests, 19 states)
          pest_model.pte     (102 pests)                   + region + season (kharif / rabi / zaid)
          → disease / pest / "retake photo"                → Llama 3.2 1B-Instruct .pte  (English advice)
          → OpenCV severity (% of leaf affected)           → IndicTrans2  (22 Indian languages)
```

| | Vision pipeline | Advisory pipeline |
|---|---|---|
| Input | camera photo | diagnosis + confidence + severity + region + month + language |
| Models | `disease_model.pte` (MobileNetV3-Small, 6 MB), `pest_model.pte` (EfficientNet-B0, 16 MB) | `llama3_2_1b_instruct_8da4w.pte` (Llama 3.2 1B-Instruct, 4-bit, 0.8 GB), IndicTrans2 200M |
| Runtime | ExecuTorch 1.6, XNNPACK backend, Arm CPU | ExecuTorch 1.6, XNNPACK backend, Arm CPU (Llama); PyTorch CPU (translator) |
| Output | label, confidence, severity level | short structured advice in the farmer's language |

### 1. Vision pipeline
- Both models run on every photo. The pest model is used only if it is confident (≥ 0.71) and beats the
  disease model, or the disease model says the photo is not a leaf (`Background`).
- **Confidence**: ≥ 0.70 → direct answer; 0.20–0.70 → "possible", with up to two alternatives of the
  **same crop**; < 0.20 or not a leaf → ask the farmer to retake the photo.
- **Severity** ([advisor/severity.py](advisor/severity.py)): OpenCV finds the leaf, then the diseased area
  (brown, yellow, dark or pale spots), and reports the % of the leaf affected:
  none < 3% ≤ mild < 10% ≤ moderate < 25% ≤ severe. Viral diseases and pests are not area-based.

### 2. Advisory pipeline
- **Knowledge base** ([advisor/knowledge/](advisor/knowledge/)): for each disease, its symptoms, the conditions
  that favour it, and cultural, organic and chemical control (with doses) plus prevention; 16 pest groups with
  action thresholds; 19 states with their language, agro-climatic zone and crop notes; Indian seasons.
- **Llama 3.2 1B-Instruct** gets only the facts for this diagnosis, region and season, and writes the steps
  in simple English. To keep a small model correct:
  - the *Problem* line (crop, disease, severity) is written by code, so Llama cannot name the wrong disease;
  - the output is cut to *What to do* (at most 5 steps); *Prevention*, *Safety* and the "not fully sure"
    note come straight from the knowledge base;
  - decoding is greedy, stops at Llama's end-of-turn tokens, and has a repetition guard.
- **IndicTrans2** (AI4Bharat, distilled 200M) translates the English advice line by line. Doses, product
  codes, the disease name and the Kisan Call Centre number are protected from translation; crop names use a
  fixed local glossary ([crop_names.json](advisor/knowledge/crop_names.json)); output is converted to the
  native script (Kannada, Tamil, Telugu, …). Llama writes English because the 1B model's Indian-language
  output was poor.

### Example (laptop, `advisor/app.py`)

```
Photo     : test images/tomato_early_blight_1.jpg
Disease   : Tomato: Early blight 99.1%
Diagnosis : Tomato: Early blight (disease, confidence 99%, confident)
Severity  : severe (44% of leaf)
Region    : Karnataka | season: rabi | language: English
Advice    : [llama-executorch]
Problem: Your tomato has Early blight. About 44% of the leaf is affected, so the damage is severe.
What to do:
1. Remove the infected lower leaves.
2. Mulch and stake the plants to prevent soil from splashing on the leaves.
3. Use Trichoderma or Neem-based sprays to control the fungus.
4. Use Mancozeb 75 WP to control the fungus.
Prevention: Spray at first spots, repeat every 10-15 days.
Safety: Wear gloves and a mask when spraying, ... Kisan Call Centre 1800-180-1551.
```

### Benchmarks (`docs/benchmark.py`, CPU only)

| | Raspberry Pi 5 (8 GB, 4× Cortex-A76) | Laptop (Ryzen 5 5600H) |
|---|---|---|
| Disease model `.pte` (median) | 6.1 ms (3.1× faster than PyTorch eager) | 16.1 ms (same as eager) |
| Pest model `.pte` (median) | 39.4 ms (2.8× faster than PyTorch eager) | 33.1 ms (1.3× faster than eager) |
| Llama decode speed | 9.6 tokens/s | 15.2 tokens/s |
| Photo → full English advice (mean of 3) | 21.2 s | 20.4 s |
| Translation (IndicTrans2, 4 beams) | not measured | ≈ 19 s per answer |
| Peak memory | 2.0 GB | 1.4 GB (with translator) |
| Power / energy per answered photo | ≈ 5.3 W, 112 J (PMIC rails) | ≈ 24.6 W, ≈ 500 J (whole laptop, from battery) |

Almost all of the answer time is Llama (prompt reading + writing); vision is under 1%. The ExecuTorch
speed-up is largest on the Arm CPU of the Pi. Full results and method: the
[benchmark report](docs/KrishiRakshak_AI_Benchmark_Results.pdf).

## Models and results (held-out test sets)

| Model | Classes | Architecture | Size |
|---|---|---|---|
| Disease model | 38 crop disease / healthy classes + `Background` (not a leaf) | MobileNetV3-Small, 224×224 | 6 MB |
| Pest model | 102 insect pests (IP102) | EfficientNet-B0, 256×256 | 16 MB |

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
images; leaked copies were removed before training (4,419 in IP102 alone). The ExecuTorch `.pte` models give
the same predictions as PyTorch on all 19 web test photos.

## Repository layout

```
advisor/                 on-device pipeline (ExecuTorch)
  app.py                 command-line demo: photo -> diagnosis -> severity -> advice
  server.py              kiosk web server (Pi camera, touch UI in web/)
  pipeline.py            loads all models once; vision + advisory pipelines
  advisor.py             knowledge base lookup, Llama prompt, decoding and output clean-up
  severity.py            OpenCV lesion-area severity
  translate.py           IndicTrans2 translation with protected terms
  export_vision.py       disease / pest models -> .pte (XNNPACK)
  export_llama.py        Llama 3.2 1B-Instruct -> 4-bit .pte (XNNPACK, KV cache)
  knowledge/             diseases, pests, regions, crop names (JSON)
  models/                disease_model.pte, pest_model.pte + metadata
deploy/raspberry_pi/     setup script, systemd service, kiosk auto-start, bundle builder
disease_model/           disease model: index, leakage-free splits, training, evaluation
pest_model/              pest model: IP102 preparation, training, evaluation
docs/                    benchmark script, its results (benchmark.json) and the benchmark report (PDF)
test images/             42 held-out dataset photos for a quick check (one per disease class, 4 pests)
```

## Run it

The advisory pipeline uses its own environment (tested with Python 3.14 on Windows 10; ExecuTorch also has
aarch64 wheels for the Raspberry Pi).

```bash
python -m venv et-venv && et-venv/Scripts/activate        # Pi / Linux: source et-venv/bin/activate
pip install -r advisor/requirements.txt

# Llama 3.2 1B-Instruct (accept Meta's licence on Hugging Face first, then `hf auth login`)
python advisor/export_llama.py                  # download + export, about 16 min, needs about 8 GB RAM
# IndicTrans2 translator (accept the AI4Bharat terms on Hugging Face first)
hf download ai4bharat/indictrans2-en-indic-dist-200M --local-dir advisor/translator/indictrans2-en-indic-dist-200M

python advisor/app.py "test images/apple_scab_1.jpg" --region Karnataka --language Kannada
python advisor/server.py                        # kiosk UI at http://localhost:8000 (--no_camera on a laptop)
```

Options: `--region` (any state in `regions.json`, default "India (general)"), `--language` (default: the
region's language), `--month`, `--mode auto|disease|pest`, `--no_llm` (template answer without Llama).
The Llama and translator weights are not in this repository (size and licence terms).

**Raspberry Pi 5 kiosk** (8 GB, Pi camera + screen, CPU only): see
[deploy/raspberry_pi/README.md](deploy/raspberry_pi/README.md).

## Retraining the vision models

Needs the source datasets (not included) in `datasets/`: PlantVillage, PlantDoc, PlantWild, Tomato Village,
Corn and Potato leaf disease sets, Caltech-256 and DTD (background), and IP102.

```bash
pip install -r requirements.txt
python disease_model/build_index.py && python disease_model/make_splits.py
python disease_model/train.py && python disease_model/evaluate.py
python pest_model/prepare.py && python pest_model/train.py && python pest_model/evaluate.py
python advisor/export_vision.py                 # refresh the .pte files
python docs/benchmark.py                        # advisory env; timing + power -> docs/benchmark.json
```

Photos in `test images/` come from the models' own held-out test splits: one PlantVillage photo per disease
class (`<crop>_<disease>_1.jpg`) and four IP102 pest photos (`pest_<name>_1.jpg`). The models never trained on
them, and the deployed pipeline gets all 42 right in auto mode (93.6–99.5% confidence). They are copies of
dataset images, so move the folder out of the way before rebuilding the index.

## Status

Done: disease and pest detection, non-leaf rejection, severity, treatment advice with Llama 3.2 1B through
ExecuTorch, region and season awareness, 22 Indian languages, Raspberry Pi 5 kiosk, benchmarks on the
laptop and the Raspberry Pi 5.
Next: measure the translator on the Pi 5, a "no pest" class for the pest model, more field photos.

Known limits: the 1B model sometimes calls a chemical "organic", repeats a step, or leaves out a dose;
severity can be over-estimated on busy backgrounds; the pest model has no "no pest" class yet, so it is only
trusted when very confident.

## Licences

Llama 3.2 is under the Llama 3.2 Community Licence; IndicTrans2 is MIT. The datasets used for training have
their own licences (several are research / non-commercial only). Review them before any commercial use of the
trained weights. The advice is general guidance: farmers should confirm chemical use with their local KVK.
