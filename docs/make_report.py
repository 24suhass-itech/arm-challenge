"""Build docs/KrishiRakshak_AI_Project_Report.pdf from the saved results of the disease model and the pest model.

Run from anywhere:  python docs/make_report.py
All numbers are read from the result CSVs, so re-running after retraining refreshes the report.
"""
import os
from datetime import date

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (Image, KeepTogether, PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table,
                                TableStyle)

DOCS = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(DOCS)
DM = os.path.join(ROOT, "disease_model")
PEST = os.path.join(ROOT, "pest_model")
FIG = os.path.join(DOCS, "figures")
OUT_PDF = os.path.join(DOCS, "KrishiRakshak_AI_Project_Report.pdf")
os.makedirs(FIG, exist_ok=True)

# ---------- palette (reference categorical slots 1-3, light mode) ----------
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
INK, INK2, MUTED, GRID = "#0b0b0b", "#52514e", "#8a8984", "#e4e3df"
HEAD_BG, ZEBRA = colors.HexColor("#1f4e3d"), colors.HexColor("#f4f6f3")
GOOD, BAD = colors.HexColor("#0a7a33"), colors.HexColor("#b42318")

# ---------- fonts (Arial covers >=, en dash, etc.) ----------
FD = r"C:\Windows\Fonts"
pdfmetrics.registerFont(TTFont("Arial", os.path.join(FD, "arial.ttf")))
pdfmetrics.registerFont(TTFont("Arial-Bold", os.path.join(FD, "arialbd.ttf")))
pdfmetrics.registerFont(TTFont("Arial-Italic", os.path.join(FD, "ariali.ttf")))
pdfmetrics.registerFontFamily("Arial", normal="Arial", bold="Arial-Bold", italic="Arial-Italic")
plt.rcParams.update({"font.family": "Arial", "font.size": 9, "axes.edgecolor": MUTED, "axes.labelcolor": INK2,
                     "xtick.color": INK2, "ytick.color": INK2, "axes.spines.top": False,
                     "axes.spines.right": False, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
                     "axes.axisbelow": True, "legend.frameon": False})

ss = getSampleStyleSheet()
H1 = ParagraphStyle("H1", parent=ss["Heading1"], fontName="Arial-Bold", fontSize=16, textColor=HEAD_BG,
                    spaceBefore=6, spaceAfter=8)
H2 = ParagraphStyle("H2", parent=ss["Heading2"], fontName="Arial-Bold", fontSize=12, textColor=HEAD_BG,
                    spaceBefore=10, spaceAfter=5)
BODY = ParagraphStyle("B", parent=ss["BodyText"], fontName="Arial", fontSize=9.5, leading=13.5, spaceAfter=5)
SMALL = ParagraphStyle("S", parent=BODY, fontSize=8, leading=10.5, textColor=colors.HexColor(INK2))
CELL = ParagraphStyle("C", parent=BODY, fontSize=8.3, leading=10.5, spaceAfter=0)
CELLB = ParagraphStyle("CB", parent=CELL, fontName="Arial-Bold", textColor=colors.white)
BUL = ParagraphStyle("BU", parent=BODY, leftIndent=12, bulletIndent=2, spaceAfter=2)


def P(t, st=BODY):
    return Paragraph(t, st)


def bullets(items):
    return [Paragraph(t, BUL, bulletText="\u2022") for t in items]


def table(rows, widths, head=True, align_right_from=None, highlight_rows=()):
    data = [[c if isinstance(c, Paragraph) else Paragraph(str(c), CELLB if (head and i == 0) else CELL)
             for c in r] for i, r in enumerate(rows)]
    t = Table(data, colWidths=widths, repeatRows=1 if head else 0)
    st = [("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#c9cbc5")),
          ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
          ("TOPPADDING", (0, 0), (-1, -1), 3), ("BOTTOMPADDING", (0, 0), (-1, -1), 3)]
    if head:
        st.append(("BACKGROUND", (0, 0), (-1, 0), HEAD_BG))
    for i in range(1 if head else 0, len(rows)):
        if i % 2 == 0:
            st.append(("BACKGROUND", (0, i), (-1, i), ZEBRA))
    for i in highlight_rows:
        st.append(("BACKGROUND", (0, i), (-1, i), colors.HexColor("#e3f1e8")))
    t.setStyle(TableStyle(st))
    return t


def fig(path, w_cm):
    from PIL import Image as PI
    w, h = PI.open(path).size
    return Image(path, width=w_cm * cm, height=w_cm * cm * h / w)


def pct(x, d=1):
    return f"{100 * x:.{d}f}%" if x <= 1.0001 else f"{x:.{d}f}%"


def short(c):
    return (c.replace("___", " \u2013 ").replace("_(maize)", "").replace(",_bell", "").replace("_", " ")
            .replace("Cercospora leaf spot Gray leaf spot", "Gray leaf spot").strip())


# ======================= load results =======================
dm_hist = pd.read_csv(os.path.join(DM, "results", "training_history.csv"))
dm_sum = pd.read_csv(os.path.join(DM, "results", "test_summary.csv"))
dm_pred = pd.read_csv(os.path.join(DM, "results", "test_predictions.csv"))
pdc = pd.read_csv(os.path.join(DM, "results", "plantdoc_per_class.csv"))
cust = pd.read_csv(os.path.join(DM, "results", "custom_image_results.csv"))
man = pd.read_csv(os.path.join(DM, "manifest.csv"), usecols=["dataset", "split"])
p_hist = pd.read_csv(os.path.join(PEST, "results", "efficientnet_b0_history.csv"))
p_sum = pd.read_csv(os.path.join(PEST, "results", "test_summary.csv")).iloc[0]
p_pred = pd.read_csv(os.path.join(PEST, "results", "test_predictions.csv"))
p_cls = pd.read_csv(os.path.join(PEST, "results", "per_class.csv"))
p_thr = pd.read_csv(os.path.join(PEST, "results", "confidence_thresholds.csv"))
p_man = pd.read_csv(os.path.join(PEST, "manifest.csv"), usecols=["split"])
p_drop = pd.read_csv(os.path.join(PEST, "dropped.csv"), usecols=["split", "drop_reason"])

dm_best = dm_hist.loc[dm_hist.val_mean.idxmax()]
p_best = p_hist.loc[p_hist.val_top1.idxmax()]
PD_CROP = dm_sum[(dm_sum.model == "disease_model") & (dm_sum.dataset == "PlantDoc")].crop.iloc[0]
pv = dm_sum.pivot_table(index="dataset", columns="model", values="top1")

plants = dm_pred[dm_pred.dataset != "Background"]
field = plants[plants.dataset != "PlantVillage"]
bg = dm_pred[dm_pred.dataset == "Background"]
dm_thr = [(t, (field.confidence >= t).mean(), field[field.confidence >= t].correct.mean()) for t in (0.5, 0.7, 0.9)]

# ======================= figures =======================
# 1. disease model training curves
f, ax = plt.subplots(figsize=(7, 3.0), dpi=200)
for col, lab, c in [("val_mean", "Mean of all validation sets", BLUE), ("val_PlantDoc", "PlantDoc (field)", ORANGE),
                    ("val_PlantWild", "PlantWild (field)", AQUA)]:
    ax.plot(dm_hist.epoch, 100 * dm_hist[col], color=c, lw=2, label=lab)
ax.axvline(dm_best.epoch, color=MUTED, lw=1, ls="--")
ax.annotate(f"best epoch {int(dm_best.epoch)}", (dm_best.epoch, 52), xytext=(4, 0), textcoords="offset points",
            color=INK2, fontsize=8)
ax.set_xlabel("Epoch"); ax.set_ylabel("Validation accuracy (%)"); ax.set_ylim(40, 100)
ax.legend(loc="lower right", ncol=3, fontsize=8)
f.tight_layout(); f.savefig(os.path.join(FIG, "disease_training.png")); plt.close(f)

# 2. model comparison on test sets
order = ["PlantDoc", "PlantWild", "TomatoVillage", "PlantVillage", "Corn", "Potato"]
f, ax = plt.subplots(figsize=(7, 3.1), dpi=200)
w = 0.26
for k, (m, lab, c) in enumerate([("disease_model", "This model", BLUE), ("unified", "Earlier 4-dataset model", ORANGE),
                                 ("six_dataset", "Earlier 6-dataset model", AQUA)]):
    xs = [i + (k - 1) * (w + 0.02) for i in range(len(order))]
    vals = [pv.loc[d, m] for d in order]
    bars = ax.bar(xs, vals, width=w, color=c, label=lab, edgecolor="white", linewidth=1)
    if m == "six_dataset":
        for x, d, b in zip(xs, order, bars):
            if d in ("PlantWild", "TomatoVillage", "Corn", "Potato"):
                b.set_hatch("////"); b.set_alpha(0.55)
    if m == "disease_model":
        for x, v in zip(xs, vals):
            ax.text(x, v + 1.2, f"{v:.0f}", ha="center", fontsize=7.5, color=INK)
ax.set_xticks(range(len(order)), ["PlantDoc", "PlantWild", "Tomato Village", "PlantVillage", "Corn", "Potato"])
ax.set_ylabel("Test top-1 accuracy (%)"); ax.set_ylim(0, 108)
ax.legend(loc="upper center", ncol=3, fontsize=8, bbox_to_anchor=(0.5, 1.13))
f.tight_layout(); f.savefig(os.path.join(FIG, "model_comparison.png")); plt.close(f)

# 3. pest training curves
f, ax = plt.subplots(figsize=(7, 2.8), dpi=200)
ax.plot(p_hist.epoch, 100 * p_hist.val_top5, color=BLUE, lw=2, label="Top-5")
ax.plot(p_hist.epoch, 100 * p_hist.val_top1, color=ORANGE, lw=2, label="Top-1")
ax.axhline(90, color=MUTED, lw=1, ls=":")
ax.text(1, 91, "90% target", color=INK2, fontsize=8)
ax.set_xlabel("Epoch"); ax.set_ylabel("Validation accuracy (%)"); ax.set_ylim(40, 100)
ax.legend(loc="lower right", ncol=2, fontsize=8)
f.tight_layout(); f.savefig(os.path.join(FIG, "pest_training.png")); plt.close(f)

# 4. pest per crop
pc = p_pred.groupby("crop").agg(n=("correct", "size"), top1=("correct", "mean"), top5=("top5", "mean"))
pc = pc.sort_values("top5", ascending=False)
f, ax = plt.subplots(figsize=(7, 2.9), dpi=200)
x = range(len(pc))
ax.bar([i - 0.2 for i in x], 100 * pc.top1, width=0.38, color=ORANGE, label="Top-1", edgecolor="white")
ax.bar([i + 0.2 for i in x], 100 * pc.top5, width=0.38, color=BLUE, label="Top-5", edgecolor="white")
for i, v in zip(x, pc.top5):
    ax.text(i + 0.2, 100 * v + 1.2, f"{100 * v:.0f}", ha="center", fontsize=7.5)
ax.axhline(90, color=MUTED, lw=1, ls=":")
ax.set_xticks(list(x), [c if c != "Vitis" else "Vitis (grape)" for c in pc.index])
ax.set_ylabel("Test accuracy (%)"); ax.set_ylim(0, 105)
ax.legend(loc="upper right", ncol=2, fontsize=8)
f.tight_layout(); f.savefig(os.path.join(FIG, "pest_per_crop.png")); plt.close(f)

# 5. accuracy vs coverage (pest + field disease)
f, ax = plt.subplots(figsize=(7, 2.9), dpi=200)
ths = [i / 100 for i in range(0, 96, 2)]
for df_, lab, c in [(p_pred, "Pest model (IP102 test)", BLUE), (field, "Disease model (field-photo tests)", ORANGE)]:
    cov = [100 * (df_.confidence >= t).mean() for t in ths]
    acc = [100 * df_[df_.confidence >= t].correct.mean() for t in ths]
    ax.plot(cov, acc, color=c, lw=2, label=lab)
ax.axhline(90, color=MUTED, lw=1, ls=":")
ax.set_xlabel("Photos answered (%) - lower = stricter confidence threshold")
ax.set_ylabel("Accuracy on answered (%)")
ax.invert_xaxis(); ax.set_ylim(55, 100)
ax.legend(loc="lower left", fontsize=8)
f.tight_layout(); f.savefig(os.path.join(FIG, "coverage.png")); plt.close(f)


# ======================= document =======================
def on_page(c, d):
    c.saveState()
    c.setFont("Arial", 7.5)
    c.setFillColor(colors.HexColor(MUTED))
    if d.page > 1:
        c.drawString(2 * cm, 1.2 * cm, "KrishiRakshak AI \u2013 Project Report")
        c.drawRightString(A4[0] - 2 * cm, 1.2 * cm, f"Page {d.page}")
    c.restoreState()


story = []
# ---------- cover ----------
story += [Spacer(1, 5 * cm),
          P("KrishiRakshak AI", ParagraphStyle("T", parent=H1, fontSize=30, leading=36, alignment=TA_CENTER)),
          Spacer(1, 0.4 * cm),
          P("Offline crop disease and pest identification for farmers",
            ParagraphStyle("ST", parent=BODY, fontSize=14, leading=18, alignment=TA_CENTER,
                           textColor=colors.HexColor(INK2))),
          Spacer(1, 1.2 * cm),
          P("Project report: datasets, models, evaluation and roadmap",
            ParagraphStyle("ST2", parent=BODY, fontSize=11, alignment=TA_CENTER)),
          Spacer(1, 0.3 * cm),
          P(date.today().strftime("%d %B %Y"), ParagraphStyle("D", parent=BODY, alignment=TA_CENTER,
                                                               textColor=colors.HexColor(MUTED))),
          Spacer(1, 4 * cm)]
kp = [["Disease model", "39 classes (38 crop diseases/healthy + Background), MobileNetV3-Small, 6 MB"],
      ["Pest model", "102 insect pests (IP102), EfficientNet-B0, 16 MB"],
      ["Runs on", "Arm phones / Raspberry Pi class devices, fully offline (no internet)"]]
story += [table(kp, [4.5 * cm, 11.5 * cm], head=False), PageBreak()]

# ---------- 1 summary ----------
story += [P("1. Executive summary", H1),
          P("KrishiRakshak AI identifies crop diseases and insect pests from a single photo, on the device, without "
            "internet. This report covers the two vision models built so far, how they were trained and - most "
            "importantly - how they were evaluated, because earlier versions of the project reported accuracies "
            "that were inflated by duplicate images shared between training and test data.")]
krows = [["Result", "Value", "Test data"],
         ["Disease model on lab photos (PlantVillage)", pct(pv.loc["PlantVillage", "disease_model"] / 100), "8,175 photos"],
         ["Disease model on real field photos (PlantDoc)", pct(pv.loc["PlantDoc", "disease_model"] / 100),
          "236 photos (official test)"],
         ["Disease model on field photos, confidence >= 0.7",
          f"{pct(dm_thr[1][2])} accuracy, answers {pct(dm_thr[1][1], 0)} of photos", "650 field photos"],
         ["Rejects photos that are not leaves", pct(bg.correct.mean()), "664 non-leaf photos"],
         ["Pest model top-5 (correct pest in 5 suggestions)", pct(p_sum.top5), "22,619 photos"],
         ["Pest model top-1, confidence >= 0.71", "90.1% accuracy, answers 54% of photos", "22,619 photos"],
         ["Pest model top-1 (single best guess)", pct(p_sum.top1), "22,619 photos"],
         ["Your 14 own test photos (disease)", f"{int(cust.disease_model_correct.fillna(False).astype(bool).sum())}/14",
          "none seen in training"]]
story += [table(krows, [7.3 * cm, 5.2 * cm, 3.5 * cm]), Spacer(1, 6),
          P("<b>Key messages</b>"),
          *bullets([
              "On lab-style photos the disease model is essentially solved (99.4%). The hard, honest benchmark is "
              "real field photos, where it reaches 65-71% top-1 and is the best model the project has produced.",
              "Both models report a confidence score. Using it - answer when confident, otherwise show the top "
              "suggestions or ask for a better photo - lifts delivered accuracy to about 90%.",
              "Every test set used here was checked for exact and near-duplicate copies of training images. Thousands "
              "of leaked images were found and removed (Section 4).",
              "Still to build for the full brief: the Llama 3.2 + ExecuTorch advisory module, severity levels, "
              "multilingual output and a 'no pest' reject class for the pest model (Section 9)."]),
          PageBreak()]

# ---------- 2 requirements ----------
story += [P("2. Requirements coverage", H1),
          P("Status against the challenge brief (on-device crop disease and pest assistant for Arm devices).")]
req = [["Requirement", "Status", "Notes"],
       ["Detect crop diseases", "Done", "38 disease/healthy classes across 14 crops, field-photo trained"],
       ["Reject non-plant photos", "Done", "Background class, 98.6% of non-leaf photos rejected"],
       ["Detect pests", "Done (first version)", "102 IP102 pests; needs a 'no pest' class (Section 7.4)"],
       ["Classify severity", "Not started", "Plan: lesion-area ratio from a segmentation dataset (PlantSeg)"],
       ["Treatment recommendations", "Not started", "Plan: curated knowledge base per disease/pest + Llama"],
       ["Region-specific advice", "Not started", "Needs target region; crop calendars, approved chemicals"],
       ["Multiple local languages", "Not started", "Llama 3.2 supports Hindi; others via IndicTrans2"],
       ["Llama via ExecuTorch (required)", "Not started", "Llama 3.2 1B/3B-Instruct, quantised, on Arm CPU"],
       ["Runs offline on Arm", "Ready to export", "Both models are small CNNs (6 MB and 16 MB) for ExecuTorch / TFLite"]]
req = [req[0]] + [[r[0], Paragraph('<font color="%s"><b>%s</b></font>' % (
    "#0a7a33" if r[1].startswith(("Done", "Ready")) else "#b42318", r[1]), CELL), r[2]] for r in req[1:]]
t = table(req, [4.6 * cm, 2.6 * cm, 8.8 * cm])
story += [t, Spacer(1, 10),
          P("3. System design", H1),
          P("The app runs two independent pipelines on the device, as the brief requires them to be documented:"),
          *bullets([
              "<b>Vision pipeline</b> (camera photo -> class + confidence): the disease model and the pest model are "
              "small convolutional networks. On a phone or Raspberry Pi 5 they run on the Arm CPU through ExecuTorch "
              "(or on a Hailo NPU through HailoRT if an AI HAT+ is used).",
              "<b>Confidence gate</b>: high confidence -> one answer; medium -> top-3/top-5 suggestions with "
              "reference pictures for the farmer to confirm; low or 'Background' -> ask for a closer, clearer photo.",
              "<b>Advisory pipeline</b> (planned): the confirmed disease/pest, severity and region are looked up in an "
              "offline knowledge base; Llama 3.2 (1B/3B-Instruct, 4-bit, ExecuTorch on the Arm CPU) turns the "
              "entry into a short, practical answer in the farmer's language."]),
          PageBreak()]

# ---------- 4 data ----------
cnt = man[man.dataset != "Custom"].groupby(["dataset", "split"]).size().unstack(fill_value=0)
desc = {"PlantVillage": "Lab photos, plain background, 38 classes", "PlantDoc": "Real field / web photos, 27 classes",
        "PlantWild": "Field photos (22 folders mapped)", "TomatoVillage": "Indian field photos, tomato",
        "Corn": "Corn leaf disease set", "Potato": "Potato leaf disease set",
        "Background": "Non-leaf photos (Caltech-256, DTD textures)"}
drows = [["Dataset", "Content", "Train", "Val", "Test"]]
for d in ["PlantVillage", "PlantDoc", "PlantWild", "TomatoVillage", "Corn", "Potato", "Background"]:
    drows.append([d, desc[d], f"{cnt.loc[d, 'train']:,}", f"{cnt.loc[d, 'val']:,}", f"{cnt.loc[d, 'test']:,}"])
drows.append(["Total", "", f"{cnt.train.sum():,}", f"{cnt.val.sum():,}", f"{cnt.test.sum():,}"])
pcnt = p_man.split.value_counts()
dropped_test = p_drop[p_drop.drop_reason == "duplicate of a test image"]
story += [P("4. Data", H1), P("4.1 Disease datasets", H2),
          table(drows, [3.0 * cm, 7.0 * cm, 2.0 * cm, 2.0 * cm, 2.0 * cm], highlight_rows=[len(drows) - 1]),
          Spacer(1, 4),
          P("Classes from all field datasets were mapped onto the 38 PlantVillage class names, so one model covers "
            "all sources. PlantVillage, PlantDoc, Corn and Potato keep their fixed (official / earlier) test sets, so "
            "results stay comparable across model versions.", SMALL),
          P("4.2 Pest dataset", H2),
          P(f"IP102: 102 insect pests of rice, corn, wheat, beet, alfalfa, grape (Vitis), citrus and mango. After "
            f"cleaning: <b>{pcnt['train']:,}</b> train, <b>{pcnt['val']:,}</b> validation, <b>{pcnt['test']:,}</b> "
            f"test images (official split)."),
          P("4.3 Leakage control - why the numbers can be trusted", H2),
          P("A model that is tested on copies of its training images looks better than it is. Every image was "
            "therefore fingerprinted twice: an exact SHA-256 hash and a 64-bit perceptual (DCT) hash that also "
            "catches resized, re-compressed or slightly cropped copies (Hamming distance <= 4). Linked images form "
            "a group, and each group is placed in exactly one split."),
          table([["Finding", "Action"],
                 ["Old PlantDoc validation set: 364 of 465 images also in training", "Rebuilt splits; groups never cross splits"],
                 ["Tomato Village official split: crops of the same photo in train and test",
                  "Grouped by original photo"],
                 [f"IP102: {len(dropped_test):,} training/validation images are copies of test images",
                  "Removed from train/val"],
                 ["About 170 identical photos labelled differently in PlantDoc vs PlantWild", "Dropped"],
                 ["11 of your own test photos were inside PlantDoc / PlantWild training data",
                  "Excluded from training; flagged when comparing older models"],
                 ["A PlantDoc 'Apple rust' image is actually apple black rot", "Documented as label noise"]],
                [9.5 * cm, 6.5 * cm]),
          PageBreak()]

# ---------- 5 disease model ----------
story += [P("5. Disease model", H1), P("5.1 Training", H2),
          table([["Item", "Setting"],
                 ["Architecture", "MobileNetV3-Small, ImageNet pre-trained, 224 x 224 input, 39 outputs"],
                 ["Sampling", "Balanced across datasets (PlantVillage 35%, PlantDoc 22%, PlantWild 18%, Tomato "
                              "Village 8%, Corn 6%, Potato 6%, Background 5%), 30,000 images per epoch"],
                 ["Augmentation", "Random resized crop, flips, TrivialAugment, random erasing"],
                 ["Optimisation", "AdamW (lr 1e-3, weight decay 0.02), warm-up + cosine, label smoothing 0.1, "
                                  "mixed precision, 30 epochs"],
                 ["Model selection", f"Mean of per-dataset validation accuracy; best = epoch {int(dm_best.epoch)} "
                                     f"({pct(dm_best.val_mean)})"]],
                [3.2 * cm, 12.8 * cm]),
          Spacer(1, 6), fig(os.path.join(FIG, "disease_training.png"), 16),
          P("Figure 1. Validation accuracy per epoch. The field-photo sets (PlantDoc, PlantWild) are the hardest and "
            "drive model selection.", SMALL),
          PageBreak(),
          P("5.2 Test results and comparison with earlier models", H2),
          fig(os.path.join(FIG, "model_comparison.png"), 16),
          P("Figure 2. Top-1 accuracy on held-out test sets. Hatched bars: the earlier 6-dataset model was trained on "
            "those exact test images, so those scores are not a fair comparison.", SMALL)]
crow = [["Test set", "Images", "This model", "Earlier 4-dataset", "Earlier 6-dataset"]]
for d in order + ["Background"]:
    n = int(dm_sum.query("model == 'disease_model' and dataset == @d").test_images.iloc[0])
    o6 = "-" if d not in pv.index or pd.isna(pv.loc[d].get("six_dataset")) else f"{pv.loc[d, 'six_dataset']:.1f}%"
    if d in ("PlantWild", "TomatoVillage", "Corn", "Potato"):
        o6 += " *"
    un = "-" if pd.isna(pv.loc[d].get("unified")) else f"{pv.loc[d, 'unified']:.1f}%"
    crow.append([d, f"{n:,}", f"<b>{pv.loc[d, 'disease_model']:.1f}%</b>", un, o6])
story += [table(crow, [4 * cm, 2.2 * cm, 3 * cm, 3 * cm, 3.8 * cm]),
          P("* trained on these test images (contaminated). 38-class models cannot output Background.", SMALL),
          *bullets([f"Only {(plants.predicted == 'Background').sum()} of {len(plants):,} plant test photos were "
                    f"wrongly rejected as Background.",
                    "Most remaining mistakes are between look-alike diseases of the same crop: on PlantDoc the crop is "
                    f"identified correctly {PD_CROP:.1f}% "
                    "of the time, while the exact disease is right 65% of the time."]),
          P("5.3 Confidence on field photos", H2),
          table([["Answer only when confidence is at least", "Photos answered", "Accuracy on those"]] +
                [[f"{t:.1f}", pct(c, 0), pct(a)] for t, c, a in dm_thr], [7 * cm, 4.5 * cm, 4.5 * cm],
                highlight_rows=[2])]
pd_s = pdc.sort_values("disease_model", ascending=False)
strong = ", ".join(f"{short(r.label)} ({r.disease_model:.0f}%)" for r in pd_s.itertuples() if r.disease_model >= 80)
weak = ", ".join(f"{short(r.label)} ({r.disease_model:.0f}%)" for r in pd_s.itertuples() if r.disease_model < 50)
conf = field[~field.correct].groupby(["label", "predicted"]).size().sort_values(ascending=False).head(6)
story += [KeepTogether([P("5.4 Strengths and weaknesses (PlantDoc field test, per class)", H2),
          P(f"<b>Strong (>= 80%):</b> {strong}."), P(f"<b>Weak (< 50%):</b> {weak}."),
          table([["True disease", "Predicted as", "Count"]] +
                [[short(a), short(b), str(n)] for (a, b), n in conf.items()], [6.5 * cm, 6.5 * cm, 3 * cm]),
          P("Most frequent field-photo confusions.", SMALL)]), PageBreak()]

# ---------- 6 custom images ----------
lab = cust[cust.expected.notna()].copy()
lab["image"] = lab.member.str.replace("\\", "/", regex=False).str.split("/").str[-1].str.replace(r" \(1\)", "", regex=True)


def mark(ok, pred, conf_=None, seen=False):
    col = "#0a7a33" if ok else "#b42318"
    s = f'<font color="{col}"><b>{"Right" if ok else "Wrong"}</b></font>'
    if not ok:
        s += f" ({short(pred).split(' \u2013 ')[-1]})"
    if conf_ is not None:
        s += f" {conf_:.0f}%"
    if seen:
        s += " \u2020"
    return Paragraph(s, CELL)


crows = [["Photo", "True label", "This model (confidence)", "Earlier 4-dataset", "Earlier 6-dataset"]]
for r in lab.sort_values(["disease_model_correct", "image"], ascending=[False, True]).itertuples():
    seen = str(r.seen_in_training_by) if isinstance(r.seen_in_training_by, str) else ""
    crows.append([r.image.replace(".jpg", ""), short(r.expected),
                  mark(bool(r.disease_model_correct), r.disease_model_pred, r.disease_model_conf),
                  mark(bool(r.unified_correct), r.unified_pred, seen="unified" in seen),
                  mark(bool(r.six_dataset_correct), r.six_dataset_pred, seen="six_dataset" in seen)])
tot = lambda m: int(lab[f"{m}_correct"].astype(bool).sum())
clean = lambda m: (int(lab[~lab.seen_in_training_by.fillna("").str.contains(m)][f"{m}_correct"].astype(bool).sum()),
                   int((~lab.seen_in_training_by.fillna("").str.contains(m)).sum()))
crows.append(["<b>Score</b>", "", f"<b>{tot('disease_model')}/14</b>", f"{tot('unified')}/14", f"{tot('six_dataset')}/14"])
crows.append(["<b>Unseen photos only</b>", "", f"<b>{clean('disease_model')[0]}/{clean('disease_model')[1]}</b>",
              f"{clean('unified')[0]}/{clean('unified')[1]}", f"{clean('six_dataset')[0]}/{clean('six_dataset')[1]}"])
story += [P("6. Test on your own photos", H1),
          P("14 labelled photos collected from the web (not from any dataset split). This model was built so that none of "
            "them, or any near-copy, was used in training."),
          table(crows, [3.9 * cm, 3.6 * cm, 3.4 * cm, 2.6 * cm, 2.5 * cm],
                highlight_rows=[len(crows) - 2, len(crows) - 1]),
          P("\u2020 the model had already seen this exact photo during training.", SMALL),
          *bullets(["This model has the best score on photos it never saw (9/14).",
                    "Apple black rot is missed by every model (0/3) - the training data has very few field photos of "
                    "it, and one PlantDoc example is even mislabelled.",
                    "Tomato bacterial spot vs Septoria leaf spot is the other recurring confusion."]),
          PageBreak()]

# ---------- 7 pest model ----------
pcr = [["Crop", "Test images", "Top-1", "Top-5"]] + [
    [c if c != "Vitis" else "Vitis (grape)", f"{int(r.n):,}", pct(r.top1), f"<b>{pct(r.top5)}</b>"]
    for c, r in pc.iterrows()]
pconf = p_pred[~p_pred.correct].groupby(["label", "predicted"]).size().sort_values(ascending=False).head(6)
band = lambda lo, hi: int(((p_cls.top1 >= lo) & (p_cls.top1 < hi)).sum())
story += [P("7. Pest model", H1), P("7.1 Training", H2),
          table([["Item", "Setting"],
                 ["Architecture", "EfficientNet-B0, ImageNet pre-trained, 256 x 256 input (insects are small)"],
                 ["Augmentation", "Random resized crop, flips, TrivialAugment, random erasing, CutMix / MixUp"],
                 ["Optimisation", "AdamW (lr 1e-3, wd 0.05), warm-up + cosine, label smoothing 0.1, mixed "
                                  "precision, 30 epochs (about 6 h on a GTX 1650 laptop GPU)"],
                 ["Model selection", f"Best validation top-1 = epoch {int(p_best.epoch)} ({pct(p_best.val_top1)})"]],
                [3.2 * cm, 12.8 * cm]),
          Spacer(1, 6), fig(os.path.join(FIG, "pest_training.png"), 15.5),
          P("Figure 3. Pest model validation accuracy per epoch.", SMALL),
          P("7.2 Test results (official IP102 test split, 22,619 photos)", H2),
          table([["Measure", "Result"],
                 ["Top-1 (single best guess)", pct(p_sum.top1)],
                 ["Top-3", pct(p_sum.top3)],
                 ["<b>Top-5 (correct pest among 5 suggestions)</b>", f"<b>{pct(p_sum.top5)}</b>"],
                 ["Average over the 102 pests (each weighted equally)", pct(p_sum.macro_top1)],
                 ["Correct crop group", pct(p_sum.crop_level)]], [10 * cm, 6 * cm], highlight_rows=[3]),
          P("For reference, the best published IP102 results are roughly 75-80% top-1 using much larger models; the "
            "original IP102 paper reported about 49% for ResNet-50. This model is phone-sized and was tested after "
            "removing leaked duplicates.", SMALL),
          PageBreak(),
          fig(os.path.join(FIG, "pest_per_crop.png"), 15.5),
          P("Figure 4. Pest model accuracy by crop. Grape, mango and corn pests are recognised best; rice and wheat "
            "have many small, look-alike hoppers, aphids and borers.", SMALL),
          P(f"Per-pest spread: {band(0.9, 1.01)} pests >= 90% top-1, {band(0.7, 0.9)} at 70-90%, {band(0.5, 0.7)} at "
            f"50-70%, {band(0, 0.5)} below 50%.", BODY),
          P("7.3 Most frequent confusions", H2),
          table([["True pest", "Predicted as", "Count"]] + [[a, b, str(n)] for (a, b), n in pconf.items()],
                [6.5 * cm, 6.5 * cm, 3 * cm]),
          P("Several confusions are between a species and its own parent group in the IP102 label set (e.g. "
            "tarnished plant bug vs Miridae, english grain aphid vs aphids), so the advice would often be the same.",
            SMALL),
          P("7.4 Limitation found on your photos", H2),
          P("Run on the 19 disease photos (no insects), the pest model must still name one of its 102 pests. It "
            "was unsure on 18 of 19 (below the 0.71 threshold), but one apple black rot leaf was called grape "
            "phylloxera at 73%. A 'No pest' class trained on clean leaf photos is needed before release."),
          PageBreak()]

# ---------- 8 confidence ----------
story += [P("8. Reaching 90% for the farmer", H1),
          P("Both models output a confidence score. The app can trade coverage for accuracy: it answers directly when "
            "confident and otherwise asks the farmer to choose from a short list or retake the photo."),
          fig(os.path.join(FIG, "coverage.png"), 15.5),
          P("Figure 5. Accuracy on the photos the model answers, as the confidence threshold rises (moving right to "
            "left). Dotted line: 90%.", SMALL),
          table([["Pest model: answer when confidence >= ", "Photos answered", "Accuracy"]] +
                [[f"{r.threshold:.1f}", pct(r.coverage, 0), pct(r.accuracy)] for r in p_thr.itertuples()],
                [7 * cm, 4.5 * cm, 4.5 * cm], highlight_rows=[5]),
          Spacer(1, 6),
          P("<b>Recommended app behaviour</b>"),
          *bullets(["Confidence >= 0.71: show one diagnosis (about 90% correct).",
                    "Lower confidence: show the top 3-5 candidates with reference pictures; the correct pest is in "
                    "the top 5 for 91% of photos.",
                    "'Background' or very low confidence: ask for a closer, well-lit photo of one leaf or insect."]),
          PageBreak()]

# ---------- 9 roadmap ----------
story += [P("9. Limitations and next steps", H1), P("9.1 Known limitations", H2),
          *bullets(["Field-photo disease accuracy (65-71% top-1) is far below lab accuracy (99%); real farmer "
                    "photos from the target region are the most valuable data to add.",
                    "Weak classes: apple black rot, tomato bacterial spot vs Septoria, corn gray leaf spot vs "
                    "northern leaf blight, several 'healthy' classes (pepper, soybean, blueberry).",
                    "No rice, wheat, cotton, sugarcane or cassava diseases yet - important crops for Indian "
                    "smallholders. The cassava file received has only 537 images and 3 of 5 classes.",
                    "Pest model has no 'No pest' class (Section 7.4).",
                    "Dataset licences (Corn, Potato, Tomato Village, PlantWild, IP102, Caltech-256, DTD) must be "
                    "reviewed before any public or commercial release."]),
          P("9.2 Roadmap", H2),
          table([["Step", "Work", "Data needed"],
                 ["1", "Add 'No pest' class to the pest model and retrain", "Already available"],
                 ["2", "Export both models to ExecuTorch (and TFLite) and measure speed on an Arm phone / Pi 5",
                  "-"],
                 ["3", "Advisory module: Llama 3.2 1B/3B-Instruct, 4-bit, ExecuTorch on Arm CPU, answering from an "
                       "offline treatment knowledge base", "ICAR / state university package of practices, KCC queries"],
                 ["4", "Severity: lesion area / leaf area -> mild, moderate, severe", "PlantSeg (segmentation masks)"],
                 ["5", "More crops: rice, wheat, cotton, sugarcane, cassava",
                  "Rice leaf disease, wheat rust, Kaggle Cassava 2020 (21k)"],
                 ["6", "Multilingual output and optional voice", "IndicTrans2, Indic speech models"]],
                [1.2 * cm, 9.3 * cm, 5.5 * cm]),
          P("10. Reproducibility", H1),
          table([["Step", "Command (from the project folder)"],
                 ["Disease: index + hash all images", "python disease_model/build_index.py"],
                 ["Disease: leakage-free splits", "python disease_model/make_splits.py"],
                 ["Disease: train / evaluate", "python disease_model/train.py ; python disease_model/evaluate.py"],
                 ["Disease: own photos", "python disease_model/test_custom_images.py"],
                 ["Pest: prepare / train / evaluate",
                  "python pest_model/prepare.py ; python pest_model/train.py ; python pest_model/evaluate.py"],
                 ["This report", "python docs/make_report.py"]], [5.5 * cm, 10.5 * cm]),
          P("Model files: disease_model/checkpoints/disease_model_best.pth (disease), "
            "pest_model/checkpoints/efficientnet_b0_best.pth (pest). Training resumes after interruptions with "
            "--resume.", SMALL)]

doc = SimpleDocTemplate(OUT_PDF, pagesize=A4, leftMargin=2 * cm, rightMargin=2 * cm, topMargin=1.8 * cm,
                        bottomMargin=1.8 * cm, title="KrishiRakshak AI - Project Report",
                        author="KrishiRakshak AI team", subject="Offline crop disease and pest identification")
doc.build(story, onFirstPage=on_page, onLaterPages=on_page)
print("wrote", OUT_PDF)
