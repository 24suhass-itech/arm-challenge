"""Build docs/KrishiRakshak_AI_Final_Report.pdf: methodology, results, hardware utilisation and optimisation.

    python docs/make_final_report.py      (after docs/benchmark.py has written docs/benchmark.json)

All numbers are read from the saved result CSVs and docs/benchmark.json.
"""
import json
import os
from datetime import date

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
import pandas as pd
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (Image, KeepTogether, PageBreak, Paragraph, Preformatted, SimpleDocTemplate, Spacer,
                                Table, TableStyle)

DOCS = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(DOCS)
DM = os.path.join(ROOT, "disease_model")
PEST = os.path.join(ROOT, "pest_model")
KB = os.path.join(ROOT, "advisor", "knowledge")
FIG = os.path.join(DOCS, "figures")
OUT_PDF = os.path.join(DOCS, "KrishiRakshak_AI_Final_Report.pdf")
os.makedirs(FIG, exist_ok=True)

BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
INK, INK2, MUTED, GRID = "#0b0b0b", "#52514e", "#8a8984", "#e4e3df"
HEAD_BG, ZEBRA = colors.HexColor("#1f4e3d"), colors.HexColor("#f4f6f3")

FD = r"C:\Windows\Fonts"
pdfmetrics.registerFont(TTFont("Arial", os.path.join(FD, "arial.ttf")))
pdfmetrics.registerFont(TTFont("Arial-Bold", os.path.join(FD, "arialbd.ttf")))
pdfmetrics.registerFont(TTFont("Arial-Italic", os.path.join(FD, "ariali.ttf")))
pdfmetrics.registerFont(TTFont("Consolas", os.path.join(FD, "consola.ttf")))
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
CODE = ParagraphStyle("CO", parent=BODY, fontName="Consolas", fontSize=7.6, leading=9.8,
                      backColor=colors.HexColor("#f4f6f3"), borderPadding=6, leftIndent=6, rightIndent=6)


def P(t, st=BODY):
    return Paragraph(t, st)


def bullets(items):
    return [Paragraph(t, BUL, bulletText="\u2022") for t in items]


def table(rows, widths, head=True, highlight_rows=()):
    data = [[c if isinstance(c, Paragraph) else Paragraph(str(c), CELLB if (head and i == 0) else CELL)
             for c in r] for i, r in enumerate(rows)]
    t = Table(data, colWidths=widths, repeatRows=1 if head else 0)
    st = [("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#c9cbc5")), ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
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


# ======================= load results =======================
dm_hist = pd.read_csv(os.path.join(DM, "results", "training_history.csv"))
dm_sum = pd.read_csv(os.path.join(DM, "results", "test_summary.csv"))
dm_pred = pd.read_csv(os.path.join(DM, "results", "test_predictions.csv"))
man = pd.read_csv(os.path.join(DM, "manifest.csv"), usecols=["dataset", "split"])
p_hist = pd.read_csv(os.path.join(PEST, "results", "efficientnet_b0_history.csv"))
p_sum = pd.read_csv(os.path.join(PEST, "results", "test_summary.csv")).iloc[0]
p_pred = pd.read_csv(os.path.join(PEST, "results", "test_predictions.csv"))
p_thr = pd.read_csv(os.path.join(PEST, "results", "confidence_thresholds.csv"))
p_man = pd.read_csv(os.path.join(PEST, "manifest.csv"), usecols=["split"])
p_drop = pd.read_csv(os.path.join(PEST, "dropped.csv"), usecols=["split", "drop_reason"])
with open(os.path.join(DOCS, "benchmark.json"), encoding="utf-8") as f:
    B = json.load(f)
kb = {n: json.load(open(os.path.join(KB, f"{n}.json"), encoding="utf-8")) for n in ("diseases", "pests", "regions")}
from importlib.util import module_from_spec, spec_from_file_location  # noqa: E402
_spec = spec_from_file_location("kr_translate", os.path.join(ROOT, "advisor", "translate.py"))
_tr = module_from_spec(_spec); _spec.loader.exec_module(_tr)
N_LANG = len(_tr.LANG_CODES)
N_STATES = len([k for k in kb["regions"]["regions"] if k != "India (general)"])
N_PEST_GROUPS = len(kb["pests"]["groups"])

dm_best = dm_hist.loc[dm_hist.val_mean.idxmax()]
p_best = p_hist.loc[p_hist.val_top1.idxmax()]
PD_CROP = dm_sum[(dm_sum.model == "disease_model") & (dm_sum.dataset == "PlantDoc")].crop.iloc[0]
pv = dm_sum.pivot_table(index="dataset", columns="model", values="top1")
plants = dm_pred[dm_pred.dataset != "Background"]
field = plants[plants.dataset != "PlantVillage"]
bg = dm_pred[dm_pred.dataset == "Background"]
dm_thr = [(t, (field.confidence >= t).mean(), field[field.confidence >= t].correct.mean()) for t in (0.5, 0.7, 0.9)]
_sel = p_pred.confidence >= 0.71   # the pest threshold used by the app (advisor/pipeline.py)
p_at = pd.Series(dict(threshold=0.71, coverage=_sel.mean(), accuracy=p_pred[_sel].correct.mean()))

V, L, T, M, S = B["vision"], B["llama"], B["translation"], B["memory_mb"], B["sizes_mb"]
runs = L["runs"]
warm = runs[1:] if len(runs) > 1 else runs
prefill_tps = sum(r["prompt tokens"] for r in warm) / sum(r["prefill s"] for r in warm)
decode_tps = sum(r["tokens/s"] for r in runs) / len(runs)
adv_s = [r["advice_s"] for r in runs]
tr_s = list(T["runs"].values())

# ======================= figures =======================
# architecture
f, ax = plt.subplots(figsize=(7.2, 3.3), dpi=220)
ax.set_xlim(0, 100); ax.set_ylim(0, 46); ax.axis("off")


def box(x, y, w, h, title, sub, c):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.4,rounding_size=1.5", fc=c + "1f", ec=c, lw=1.4))
    ax.text(x + w / 2, y + h - 2.6, title, ha="center", va="top", fontsize=7.6, weight="bold", color=INK)
    ax.text(x + w / 2, y + h - 6.4, sub, ha="center", va="top", fontsize=6.3, color=INK2, linespacing=1.35)


def arrow(x1, y1, x2, y2):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=9, color=MUTED, lw=1.2))


ax.text(1, 45, "VISION PIPELINE", fontsize=7, weight="bold", color=BLUE)
ax.text(1, 21, "ADVISORY PIPELINE", fontsize=7, weight="bold", color=ORANGE)
box(1, 27, 15, 15, "Camera photo", "Pi Camera\n(or upload)", MUTED)
box(21, 27, 22, 15, "Disease model", "MobileNetV3-Small\n39 classes, 6 MB\nExecuTorch + XNNPACK", BLUE)
box(48, 27, 22, 15, "Pest model", "EfficientNet-B0\n102 pests, 16 MB\nExecuTorch + XNNPACK", BLUE)
box(75, 27, 24, 15, "Decision + severity", "confidence gate\ndisease / pest / retake\nOpenCV lesion area", BLUE)
box(1, 2, 22, 16, "Knowledge base", f"38 diseases, 102 pests\n{N_STATES} states, 3 seasons\ndoses, safety, KVK", ORANGE)
box(28, 2, 23, 16, "Llama 3.2 1B-Instruct", "4-bit (8da4w), 765 MB\nExecuTorch + XNNPACK\nKV cache, grounded prompt", ORANGE)
box(56, 2, 20, 16, "IndicTrans2", f"200M, {N_LANG} Indian\nlanguages, protected\ndoses and names", ORANGE)
box(81, 2, 18, 16, "Farmer", "advice on the\ntouch screen,\nown language", AQUA)
arrow(16.8, 34.5, 20.2, 34.5); arrow(43.8, 34.5, 47.2, 34.5); arrow(70.8, 34.5, 74.2, 34.5)
arrow(87, 26.2, 12, 18.8)
arrow(23.8, 10, 27.2, 10); arrow(51.8, 10, 55.2, 10); arrow(76.8, 10, 80.2, 10)
f.savefig(os.path.join(FIG, "final_architecture.png"), bbox_inches="tight"); plt.close(f)

# disease test sets (this model only)
order = [("PlantVillage", "PlantVillage (lab)"), ("Potato", "Potato"), ("Corn", "Corn"),
         ("TomatoVillage", "Tomato Village (India, field)"), ("PlantWild", "PlantWild (field)"),
         ("PlantDoc", "PlantDoc (field)")]
f, ax = plt.subplots(figsize=(7, 2.7), dpi=200)
vals = [pv.loc[d, "disease_model"] for d, _ in order] + [100 * bg.correct.mean()]
labs = [l for _, l in order] + ["Non-leaf photos rejected"]
ys = range(len(vals))[::-1]
ax.barh(list(ys), vals, color=[BLUE] * len(order) + [AQUA], height=0.62, edgecolor="white")
for y, v in zip(ys, vals):
    ax.text(v + 0.8, y, f"{v:.1f}%", va="center", fontsize=8, color=INK)
ax.set_yticks(list(ys), labs); ax.set_xlim(0, 108); ax.set_xlabel("Test top-1 accuracy (%)")
ax.grid(axis="y", visible=False)
f.tight_layout(); f.savefig(os.path.join(FIG, "final_disease_tests.png")); plt.close(f)

# latency breakdown
f, ax = plt.subplots(figsize=(7, 2.5), dpi=200)
items = [("Disease model (.pte)", V["disease_pte_ms"] / 1000, BLUE), ("Pest model (.pte)", V["pest_pte_ms"] / 1000, BLUE),
         ("Severity (OpenCV)", V["severity_ms"] / 1000, BLUE),
         ("Llama prefill (~650 tokens)", sum(r["prefill s"] for r in warm) / len(warm), ORANGE),
         ("Llama answer (~180 tokens)", sum(r["advice_s"] - r["prefill s"] for r in warm) / len(warm), ORANGE),
         ("Translation (IndicTrans2)", sum(tr_s) / len(tr_s), AQUA)]
ys = range(len(items))[::-1]
ax.barh(list(ys), [v for _, v, _ in items], color=[c for *_, c in items], height=0.62, edgecolor="white")
for y, (_, v, _) in zip(ys, items):
    ax.text(v + 0.25, y, f"{v * 1000:.0f} ms" if v < 1 else f"{v:.1f} s", va="center", fontsize=8)
ax.set_yticks(list(ys), [n for n, *_ in items]); ax.set_xlabel("Seconds per photo (CPU only)")
ax.set_xlim(0, max(v for _, v, _ in items) * 1.18); ax.grid(axis="y", visible=False)
f.tight_layout(); f.savefig(os.path.join(FIG, "final_latency.png")); plt.close(f)

# model size before / after
f, ax = plt.subplots(figsize=(7, 2.4), dpi=200)
pairs = [("Llama 3.2 1B-Instruct", S["llama_original"], S["llama_pte"]), ("Pest model", S["pest_pth"], S["pest_pte"]),
         ("Disease model", S["disease_pth"], S["disease_pte"])]
ys = range(len(pairs))
ax.barh([y + 0.2 for y in ys], [a for _, a, _ in pairs], height=0.38, color=MUTED, label="Original (PyTorch)")
ax.barh([y - 0.2 for y in ys], [b for *_, b in pairs], height=0.38, color=BLUE, label="Deployed (.pte)")
for y, (_, a, b) in zip(ys, pairs):
    ax.text(a + 25, y + 0.2, f"{a:,.0f} MB", va="center", fontsize=7.5)
    ax.text(b + 25, y - 0.2, f"{b:,.0f} MB", va="center", fontsize=7.5)
ax.set_yticks(list(ys), [n for n, *_ in pairs]); ax.set_xlabel("File size (MB)")
ax.set_xlim(0, S["llama_original"] * 1.15); ax.grid(axis="y", visible=False)
ax.legend(loc="center right", fontsize=8)
f.tight_layout(); f.savefig(os.path.join(FIG, "final_sizes.png")); plt.close(f)

# pest accuracy per crop
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
f.tight_layout(); f.savefig(os.path.join(FIG, "final_pest_per_crop.png")); plt.close(f)

# coverage curve
f, ax = plt.subplots(figsize=(7, 2.8), dpi=200)
ths = [i / 100 for i in range(0, 96, 2)]
for df_, lab, c in [(p_pred, "Pest model (IP102 test)", BLUE), (field, "Disease model (field-photo tests)", ORANGE)]:
    ax.plot([100 * (df_.confidence >= t).mean() for t in ths],
            [100 * df_[df_.confidence >= t].correct.mean() for t in ths], color=c, lw=2, label=lab)
ax.axhline(90, color=MUTED, lw=1, ls=":"); ax.text(99, 90.8, "90%", color=INK2, fontsize=8)
ax.set_xlabel("Photos answered directly (%)"); ax.set_ylabel("Accuracy on answered (%)")
ax.invert_xaxis(); ax.set_ylim(55, 100); ax.legend(loc="lower left", fontsize=8)
f.tight_layout(); f.savefig(os.path.join(FIG, "final_coverage.png")); plt.close(f)


# ======================= document =======================
def on_page(c, d):
    c.saveState(); c.setFont("Arial", 7.5); c.setFillColor(colors.HexColor(MUTED))
    if d.page > 1:
        c.drawString(2 * cm, 1.2 * cm, "KrishiRakshak AI \u2013 Final Project Report")
        c.drawRightString(A4[0] - 2 * cm, 1.2 * cm, f"Page {d.page}")
    c.restoreState()


dev = B["device"]
story = []
# ---------- cover ----------
story += [Spacer(1, 4.5 * cm),
          P("KrishiRakshak AI", ParagraphStyle("T", parent=H1, fontSize=30, leading=36, alignment=TA_CENTER)),
          Spacer(1, 0.3 * cm),
          P("An offline AI crop doctor for Indian farmers on Arm",
            ParagraphStyle("ST", parent=BODY, fontSize=14, leading=18, alignment=TA_CENTER, textColor=colors.HexColor(INK2))),
          Spacer(1, 1.0 * cm),
          P("Final Project Report \u2013 Arm Challenge",
            ParagraphStyle("ST2", parent=BODY, fontSize=12, alignment=TA_CENTER, fontName="Arial-Bold")),
          P("Methodology \u2022 Results \u2022 Hardware utilisation \u2022 Optimisation techniques",
            ParagraphStyle("ST3", parent=BODY, fontSize=10, alignment=TA_CENTER)),
          Spacer(1, 0.2 * cm),
          P(date.today().strftime("%d %B %Y"), ParagraphStyle("D", parent=BODY, alignment=TA_CENTER,
                                                               textColor=colors.HexColor(MUTED))),
          Spacer(1, 3.2 * cm),
          table([["Vision", "Disease model (39 classes, 6 MB) + pest model (102 pests, 16 MB), ExecuTorch on Arm CPU"],
                 ["Advisory", "Llama 3.2 1B-Instruct, 4-bit, run with ExecuTorch (XNNPACK) on the CPU"],
                 ["Languages", f"English + {N_LANG} Indian languages (IndicTrans2), region- and season-aware"],
                 ["Hardware", "Raspberry Pi 5 (8 GB) kiosk with Pi Camera and touch screen, fully offline, no NPU"],
                 ["Code", "github.com/24suhass-itech/arm-challenge"]],
                [3 * cm, 13 * cm], head=False),
          PageBreak()]

# ---------- 1 summary ----------
story += [P("1. Executive summary", H1),
          P("KrishiRakshak AI turns a single photo of a leaf or an insect into a complete, practical answer for a "
            "farmer: <b>what the problem is, how severe it is, and what to do about it</b> \u2013 adapted to the "
            "farmer's state and the current season and delivered in the farmer's own language. Everything runs "
            "on the device's Arm CPU with no internet connection: two vision models and the Llama 3.2 language model "
            "are all executed with <b>ExecuTorch</b> and its Arm-optimised <b>XNNPACK</b> backend."),
          table([["Highlight", "Result"],
                 ["Disease identification, lab photos (PlantVillage, 8,175 test photos)",
                  f"<b>{pv.loc['PlantVillage', 'disease_model']:.1f}%</b>"],
                 ["Disease identification, Indian field photos (Tomato Village)",
                  f"<b>{pv.loc['TomatoVillage', 'disease_model']:.1f}%</b>"],
                 ["Corn / potato leaf disease test sets",
                  f"<b>{pv.loc['Corn', 'disease_model']:.1f}% / {pv.loc['Potato', 'disease_model']:.1f}%</b>"],
                 ["Field photos answered with confidence \u2265 0.7", f"<b>{pct(dm_thr[1][2])}</b> accurate"],
                 ["Non-leaf photos correctly rejected", f"<b>{pct(bg.correct.mean())}</b>"],
                 ["Pest identification, 102 pests: correct pest among top-5 suggestions", f"<b>{pct(p_sum.top5)}</b>"],
                 ["Pest identification at confidence \u2265 0.71", f"<b>{pct(p_at.accuracy)}</b> accurate"],
                 ["Llama 3.2 1B size after 4-bit ExecuTorch export",
                  f"<b>{S['llama_original'] / 1024:.2f} GB \u2192 {S['llama_pte'] / 1024:.2f} GB</b> "
                  f"({S['llama_original'] / S['llama_pte']:.1f}\u00d7 smaller)"],
                 ["Vision inference per photo (both models, ExecuTorch)",
                  f"<b>{V['disease_pte_ms'] + V['pest_pte_ms']:.0f} ms</b>"],
                 ["Peak memory of the complete pipeline", f"<b>{M['peak'] / 1024:.1f} GB</b> (fits easily in 8 GB)"],
                 ["Languages", f"<b>English + {N_LANG}</b> Indian languages"]],
                [11 * cm, 5 * cm]),
          Spacer(1, 8),
          P("<b>What makes the solution strong</b>"),
          *bullets(["<b>Complete on-device pipeline</b>: detection, severity, treatment, regional advice and "
                    "translation all work offline \u2013 suited to villages with poor connectivity.",
                    "<b>Llama through ExecuTorch, as required</b>: Llama 3.2 1B-Instruct quantised to 4-bit weights with "
                    "a KV cache and XNNPACK kernels, running on the CPU alone \u2013 no GPU or NPU needed.",
                    "<b>Trustworthy evaluation</b>: every test set was checked for exact and near-duplicate copies of "
                    "training images, so the reported accuracy reflects genuinely unseen photos.",
                    "<b>Safe, grounded advice</b>: Llama writes only from a curated knowledge base of treatments, doses "
                    "and safety rules; the diagnosis, prevention and safety text are fixed by code.",
                    "<b>Farmer-ready hardware</b>: a Raspberry Pi 5 kiosk with camera and touch screen that starts by "
                    "itself and needs no technical skill to use."]),
          PageBreak()]

# ---------- 2 requirements ----------
req = [["Challenge requirement", "How KrishiRakshak AI delivers it"],
       ["Detect crop diseases", "38 disease / healthy classes across 14 crops, trained on lab and field photos"],
       ["Detect pests", "102 insect pests of rice, maize, wheat, beet, alfalfa, grape, citrus and mango (IP102)"],
       ["Classify severity", "OpenCV lesion-area measurement: none / mild / moderate / severe with % of leaf affected"],
       ["Treatment recommendations", "Cultural, organic and chemical control with doses, plus prevention and safety"],
       ["Region-specific advice", f"{N_STATES} Indian states with agro-climatic zone and crop notes; kharif / rabi / zaid "
                                  "season from the date"],
       ["Multiple local languages", f"English + {N_LANG} Indian languages (Hindi, Kannada, Tamil, Telugu, Marathi, "
                                    "Bengali, Gujarati, Malayalam, Odia, Punjabi, ...)"],
       ["Llama via ExecuTorch", "Llama 3.2 1B-Instruct, 4-bit 8da4w, XNNPACK, KV cache, exported to a .pte program"],
       ["Runs offline on Arm hardware", "All models on the Arm CPU through ExecuTorch; Raspberry Pi 5 kiosk deployment"]]
story += [P("2. Requirements coverage", H1),
          table([req[0][:1] + ["Status"] + req[0][1:]] +
                [[r[0], Paragraph('<font color="#0a7a33"><b>Delivered</b></font>', CELL), r[1]] for r in req[1:]],
                [4.2 * cm, 1.9 * cm, 9.9 * cm]),
          P("3. System architecture", H1),
          P("The solution is made of two independent on-device pipelines that are connected by the diagnosis."),
          fig(os.path.join(FIG, "final_architecture.png"), 16),
          P("Figure 1. Vision pipeline (top) and advisory pipeline (bottom). Every neural network runs on the Arm "
            "CPU; nothing is sent to the internet.", SMALL),
          table([["", "Vision pipeline", "Advisory pipeline"],
                 ["Input", "camera photo", "diagnosis, confidence, severity, region, month, language"],
                 ["Models", "disease_model.pte, pest_model.pte",
                  "llama3_2_1b_instruct_8da4w.pte, IndicTrans2 200M"],
                 ["Runtime", "ExecuTorch 1.6, XNNPACK delegate", "ExecuTorch 1.6, XNNPACK delegate (Llama)"],
                 ["Output", "disease / pest / retake + confidence + severity", "structured advice in the farmer's language"]],
                [2.4 * cm, 6.3 * cm, 7.3 * cm]),
          PageBreak()]

# ---------- 4 methodology ----------
cnt = man[man.dataset != "Custom"].groupby(["dataset", "split"]).size().unstack(fill_value=0)
desc = {"PlantVillage": "Lab photos, plain background", "PlantDoc": "Real field / web photos",
        "PlantWild": "Field photos", "TomatoVillage": "Indian field photos, tomato",
        "Corn": "Corn leaf disease set", "Potato": "Potato leaf disease set",
        "Background": "Non-leaf photos (Caltech-256, DTD)"}
drows = [["Dataset", "Content", "Train", "Val", "Test"]]
for d in ["PlantVillage", "PlantDoc", "PlantWild", "TomatoVillage", "Corn", "Potato", "Background"]:
    drows.append([d, desc[d], f"{cnt.loc[d, 'train']:,}", f"{cnt.loc[d, 'val']:,}", f"{cnt.loc[d, 'test']:,}"])
drows.append(["Total", "", f"{cnt.train.sum():,}", f"{cnt.val.sum():,}", f"{cnt.test.sum():,}"])
pcnt = p_man.split.value_counts()
dropped_test = (p_drop.drop_reason == "duplicate of a test image").sum()
story += [P("4. Methodology", H1),
          P("4.1 Data collection and leakage-free splits", H2),
          P("Seven disease datasets were merged into one 39-class label space (38 PlantVillage disease/healthy "
            "classes + Background), so a single compact model covers lab and real field conditions."),
          table(drows, [3.0 * cm, 6.6 * cm, 2.1 * cm, 2.1 * cm, 2.2 * cm], highlight_rows=[len(drows) - 1]),
          Spacer(1, 4),
          P(f"<b>Pests:</b> IP102 \u2013 102 pests of 8 crops, {pcnt['train']:,} training, {pcnt['val']:,} "
            f"validation and {pcnt['test']:,} test images (official test split)."),
          P("<b>Leakage control.</b> Every image was fingerprinted with an exact SHA-256 hash and a 64-bit perceptual "
            "(DCT) hash that also catches resized or re-compressed copies (Hamming distance \u2264 4). Linked images "
            "form a group and every group lives in exactly one split, so no test photo or near-copy is ever seen in "
            f"training ({dropped_test:,} such copies were removed from IP102 alone)."),
          P("4.2 Disease model", H2),
          table([["Item", "Setting"],
                 ["Architecture", "MobileNetV3-Small (ImageNet pre-trained), 224 \u00d7 224 input, 39 outputs"],
                 ["Sampling", "Balanced across datasets, 30,000 images per epoch"],
                 ["Augmentation", "Random resized crop, flips, TrivialAugment, random erasing"],
                 ["Optimisation", "AdamW, warm-up + cosine schedule, label smoothing 0.1, mixed precision, 30 epochs"],
                 ["Model selection", f"Mean per-dataset validation accuracy, best at epoch {int(dm_best.epoch)}"]],
                [3.2 * cm, 12.8 * cm]),
          P("4.3 Pest model", H2),
          table([["Item", "Setting"],
                 ["Architecture", "EfficientNet-B0 (ImageNet pre-trained), 256 \u00d7 256 input for small insects"],
                 ["Augmentation", "Random resized crop, flips, TrivialAugment, random erasing, CutMix / MixUp"],
                 ["Optimisation", "AdamW, warm-up + cosine, label smoothing 0.1, mixed precision, 30 epochs, resumable"],
                 ["Model selection", f"Best validation top-1 at epoch {int(p_best.epoch)}"]],
                [3.2 * cm, 12.8 * cm]),
          PageBreak(),
          P("4.4 Confidence-gated decision", H2),
          *bullets(["Both vision models run on every photo. The pest answer is chosen when the pest model is confident "
                    "(\u2265 0.71) and more confident than the disease model, or when the photo is not a leaf.",
                    "Confidence \u2265 0.70: a direct diagnosis. 0.20\u20130.70: the most likely diagnosis plus up to "
                    "two alternatives of the <i>same crop</i>, with advice to confirm at the local KVK. Below 0.20 or "
                    "a non-leaf photo: the farmer is guided to take a closer, clearer photo."]),
          P("4.5 Severity estimation", H2),
          P("The leaf is segmented in HSV colour space (largest green region, holes filled so diseased tissue is "
            "included); diseased pixels are those that are brown, yellow, dark or abnormally pale compared with the "
            "leaf's own healthy tissue. The lesion share of the leaf area gives the level: none (&lt; 3%), mild "
            "(&lt; 10%), moderate (&lt; 25%) or severe. It runs in about "
            f"{V['severity_ms']:.0f} ms and needs no extra neural network."),
          P("4.6 Knowledge base, region and season", H2),
          *bullets([f"<b>Diseases</b> (38 classes): symptoms, favouring weather, cultural, organic and chemical "
                    "control with doses, and prevention.",
                    f"<b>Pests</b>: 102 pests mapped to {N_PEST_GROUPS} management groups with action thresholds and "
                    "control measures.",
                    f"<b>Regions</b>: {N_STATES} states with main language, agro-climatic zone and crop notes; the "
                    "season (kharif, rabi, zaid) comes from the month."]),
          P("4.7 Advisory generation with Llama 3.2 1B-Instruct", H2),
          *bullets(["<b>Grounded prompt</b>: only the facts for this diagnosis, region and season are given to Llama, "
                    "with one worked example (one-shot), using the official Llama 3 chat template.",
                    "<b>Fixed problem statement</b>: the first line (crop, disease, % of leaf affected, severity) is "
                    "written by code and pre-filled into the assistant turn, so the answer always names the "
                    "correct diagnosis.",
                    "<b>Controlled output</b>: greedy decoding, stop on Llama's end-of-turn tokens, a repetition guard, "
                    "and a format filter that keeps at most five action steps. Prevention, safety and the "
                    "Kisan Call Centre number (1800-180-1551) are added from the knowledge base."]),
          P("4.8 Translation into Indian languages", H2),
          P(f"Llama writes the advice in clear English; <b>IndicTrans2</b> (AI4Bharat, distilled 200M) translates it "
            f"line by line into any of {N_LANG} Indian languages. Doses, product codes, the disease name and phone "
            "numbers are protected with placeholders, crop names come from a hand-checked local-name glossary, and "
            "the text is converted to the native script (Kannada, Tamil, Telugu, ...). Section labels are translated "
            "by hand so the layout stays identical in every language."),
          PageBreak()]

# ---------- 5 results ----------
ex = runs[0]
story += [P("5. Results", H1),
          P("5.1 Disease identification", H2),
          fig(os.path.join(FIG, "final_disease_tests.png"), 15.5),
          P("Figure 2. Disease model top-1 accuracy on held-out test sets (no test photo or near-copy used in "
            "training).", SMALL),
          table([["Answer directly when confidence is at least", "Field photos answered", "Accuracy"]] +
                [[f"{t:.1f}", pct(c, 0), f"<b>{pct(a)}</b>"] for t, c, a in dm_thr], [7 * cm, 4.5 * cm, 4.5 * cm]),
          P(f"On field photos the correct <b>crop</b> is identified {PD_CROP:.1f}% of the time (PlantDoc), so "
            "the same-crop alternatives shown at medium confidence are directly useful to the farmer.", SMALL),
          P("5.2 Pest identification (IP102 official test, 22,619 photos)", H2),
          table([["Measure", "Result"],
                 ["Top-1", pct(p_sum.top1)], ["Top-3", pct(p_sum.top3)],
                 ["<b>Top-5 (correct pest among five suggestions)</b>", f"<b>{pct(p_sum.top5)}</b>"],
                 [f"Top-1 when confidence \u2265 {p_at.threshold:.2f}", f"<b>{pct(p_at.accuracy)}</b> "
                                                                         f"(answers {pct(p_at.coverage, 0)} of photos)"],
                 ["Correct crop group", pct(p_sum.crop_level)]], [10 * cm, 6 * cm], highlight_rows=[3]),
          P("The original IP102 benchmark paper reported about 49% top-1 for ResNet-50; this compact 16 MB "
            f"EfficientNet-B0 reaches {pct(p_sum.top1)} on leakage-checked data.", SMALL),
          PageBreak(),
          fig(os.path.join(FIG, "final_pest_per_crop.png"), 15.5),
          P("Figure 3. Pest model accuracy by crop on the IP102 test set.", SMALL),
          fig(os.path.join(FIG, "final_coverage.png"), 15.5),
          P("Figure 4. Accuracy on the photos answered directly, as the confidence threshold rises. Photos below the "
            "threshold get top-5 suggestions or a request for a clearer photo.", SMALL),
          PageBreak(),
          P("5.3 End-to-end example", H2),
          P(f"Photo of a tomato leaf, region Karnataka, rabi season. Vision {1000 * ex['vision_s']:.0f} ms, advice "
            f"{ex['advice_s']:.0f} s on the CPU."),
          Preformatted(
              "Diagnosis : Tomato: Early blight (disease, confidence 98%, confident)\n"
              "Severity  : moderate (24% of leaf)\n"
              "Region    : Karnataka | season: rabi\n"
              "Advice    : [llama-executorch]\n"
              "Problem: Your tomato has Early blight. About 24% of the leaf is affected,\n"
              "         so the damage is moderate.\n"
              "What to do:\n"
              "1. Remove the infected lower leaves.\n"
              "2. Mulch and stake the plants to prevent soil splash.\n"
              "3. Use Trichoderma or Neem-based sprays to control the disease.\n"
              "4. Mancozeb 75 WP is a good option for controlling the disease.\n"
              "5. Repeat the treatment every 10-15 days.\n"
              "Prevention: Spray at first spots, repeat every 10-15 days.\n"
              "Safety: Wear gloves and a mask when spraying, follow the product label and the\n"
              "        waiting period before harvest. Kisan Call Centre 1800-180-1551.", CODE),
          Spacer(1, 6),
          P("The same answer is then produced in the farmer's language (for example Kannada for Karnataka, Tamil for "
            "Tamil Nadu), with the dose \u201c75 WP\u201d, the disease name and the helpline number kept exactly as "
            "written, and the crop name in its local form."),
          P("5.4 What the farmer receives", H2),
          table([["Part of the answer", "Source"],
                 ["Diagnosis and confidence", "Vision models (ExecuTorch)"],
                 ["Severity with % of leaf affected", "OpenCV lesion-area analysis"],
                 ["Step-by-step treatment (cultural, organic, chemical with dose)",
                  "Llama 3.2 1B-Instruct (ExecuTorch), grounded in the knowledge base"],
                 ["Prevention, safety, Kisan Call Centre number", "Knowledge base"],
                 ["Season and state context", "Regional knowledge base + device date"],
                 ["Local language", f"IndicTrans2 ({N_LANG} languages)"]], [8.5 * cm, 7.5 * cm]),
          PageBreak()]

# ---------- 6 hardware ----------
mem_rows = [["Stage loaded", "Process memory", "Added"],
            ["Python runtime + libraries", f"{M['start']} MB", "-"],
            ["+ disease and pest models (ExecuTorch)", f"{M['vision_models']} MB", f"+{M['vision_models'] - M['start']} MB"],
            ["+ Llama 3.2 1B .pte (memory-mapped)", f"{M['plus_llama']} MB", f"+{M['plus_llama'] - M['vision_models']} MB"],
            ["+ IndicTrans2 translator", f"{M['plus_translator']} MB", f"+{M['plus_translator'] - M['plus_llama']} MB"],
            ["<b>Peak during a full run</b>", f"<b>{M['peak']} MB</b>", ""]]
story += [P("6. Hardware utilisation", H1),
          P("6.1 Target hardware", H2),
          table([["Component", "Choice", "Role"],
                 ["Computer", "Raspberry Pi 5, 8 GB RAM, quad-core Arm Cortex-A76 @ 2.4 GHz", "runs every model on the CPU"],
                 ["Camera", "Raspberry Pi Camera Module (picamera2, 1280 \u00d7 960)", "live preview and capture"],
                 ["Display", "Touch screen, Chromium kiosk mode", "farmer interface"],
                 ["Storage", "microSD (about 2 GB for models and code)", "offline models and knowledge base"],
                 ["Accelerator", "None required", "CPU-only design keeps the cost low"]],
                [2.6 * cm, 8 * cm, 5.4 * cm]),
          P(f"The measurements below were taken on the development machine ({dev['cpu'].strip() or 'laptop CPU'} "
            f", {dev['cores']} cores, {dev['ram_gb']} GB RAM, CPU only, no GPU), with the same ExecuTorch "
            f"{dev['executorch']} programs and XNNPACK kernels that run on the Raspberry Pi.", SMALL),
          P("6.2 Memory", H2),
          table(mem_rows, [8 * cm, 4 * cm, 4 * cm], highlight_rows=[5]),
          P(f"The complete system \u2013 two vision models, a 1-billion-parameter language model and a translation "
            f"model \u2013 peaks at <b>{M['peak'] / 1024:.1f} GB</b>, about {100 * M['peak'] / 8192:.0f}% of the "
            "Raspberry Pi 5's 8 GB. Because ExecuTorch memory-maps the Llama program, only "
            f"{M['plus_llama'] - M['vision_models']} MB of resident memory is added when Llama is loaded, and it "
            f"loads in {L['load_s']:.1f} s.", BODY),
          P("6.3 Speed", H2),
          fig(os.path.join(FIG, "final_latency.png"), 15.5),
          P("Figure 5. Time per photo for each stage (CPU only).", SMALL),
          table([["Stage", "Measured"],
                 ["Disease model (MobileNetV3-Small, ExecuTorch)", f"{V['disease_pte_ms']:.0f} ms"],
                 ["Pest model (EfficientNet-B0, ExecuTorch)", f"{V['pest_pte_ms']:.0f} ms"],
                 ["Severity (OpenCV)", f"{V['severity_ms']:.0f} ms"],
                 ["Llama prompt processing (prefill)", f"{prefill_tps:.0f} tokens/s"],
                 ["Llama answer generation (decode)", f"{decode_tps:.1f} tokens/s"],
                 ["Complete English advice", f"{min(adv_s):.0f}\u2013{max(adv_s):.0f} s"],
                 [f"Translation, {T['beams']}-beam search (Hindi / Kannada / Tamil)",
                  " / ".join(f"{v:.0f} s" for v in tr_s)]],
                [10 * cm, 6 * cm]),
          PageBreak()]

# ---------- 7 optimisation ----------
story += [P("7. Optimisation techniques", H1),
          fig(os.path.join(FIG, "final_sizes.png"), 15.5),
          P("Figure 6. Model files before and after export to ExecuTorch.", SMALL),
          P("7.1 Language model (Llama 3.2 1B-Instruct)", H2),
          table([["Technique", "Effect"],
                 ["<b>8da4w quantisation</b>: 4-bit weights (group size 128) with 8-bit dynamic activations in all "
                  "linear layers", "about 4\u00d7 less weight memory, faster integer matrix kernels"],
                 ["<b>4-bit embedding quantisation</b> (group size 32) of the 128k-token embedding table",
                  "the largest single tensor shrinks by about 4\u00d7"],
                 [f"<b>Result</b>: {S['llama_original']:,.0f} MB (bf16) \u2192 {S['llama_pte']:,.0f} MB .pte",
                  f"<b>{S['llama_original'] / S['llama_pte']:.1f}\u00d7 smaller</b>"],
                 ["<b>XNNPACK delegation</b>: the whole graph lowered to XNNPACK, which uses Arm NEON / dot-product "
                  "instructions", "Arm-optimised kernels on the Cortex-A76"],
                 ["<b>KV cache + fused SDPA custom operator</b>",
                  "each new token reuses earlier attention keys / values instead of recomputing the prompt"],
                 ["<b>1,024-token context</b> sized to the task (prompt about 650 tokens + answer)",
                  "smaller cache, less memory"],
                 ["<b>Memory-mapped .pte</b> loading", f"only {M['plus_llama'] - M['vision_models']} MB resident "
                                                       f"memory added, {L['load_s']:.0f} s load"],
                 ["<b>Short, structured generation</b>: the problem line is pre-filled, prevention / safety come from "
                  "the knowledge base", "fewer tokens to generate \u2192 faster answers"]],
                [10 * cm, 6 * cm]),
          P("7.2 Vision models", H2),
          table([["Technique", "Effect"],
                 ["<b>Mobile-first architectures</b>: MobileNetV3-Small (2.5M parameters) and EfficientNet-B0",
                  "6 MB and 16 MB models"],
                 ["<b>ExecuTorch export</b> with torch.export \u2192 to_edge_transform_and_lower(XnnpackPartitioner)",
                  f"disease {V['disease_eager_ms']:.0f} \u2192 {V['disease_pte_ms']:.0f} ms, pest "
                  f"{V['pest_eager_ms']:.0f} \u2192 {V['pest_pte_ms']:.0f} ms "
                  f"({V['pest_eager_ms'] / V['pest_pte_ms']:.2f}\u00d7) vs PyTorch"],
                 ["<b>Identical predictions</b> after export", "verified against PyTorch on the test photos"],
                 ["<b>Light pre-processing</b> in NumPy (no torchvision at run time)", "smaller install on the Pi"],
                 ["<b>Classical-vision severity</b> (OpenCV) instead of a segmentation network",
                  f"{V['severity_ms']:.0f} ms, no extra model memory"]],
                [10 * cm, 6 * cm]),
          P("7.3 Training efficiency", H2),
          *bullets(["Transfer learning from ImageNet and mixed-precision (AMP) training on a single laptop GPU.",
                    "Balanced multi-dataset sampling: field photos are seen often enough without growing the epoch.",
                    "Resumable checkpoints, so long training runs survive interruptions."]),
          P("7.4 System-level optimisation on the Raspberry Pi", H2),
          *bullets(["All models are loaded once at start-up and kept in memory; requests are served one at a time "
                    "so the four cores are never over-subscribed.",
                    "Translation beam width is configurable (KR_TRANSLATE_BEAMS; 2 on the Pi) to balance speed and quality.",
                    "Output length of the translator is bounded relative to the input, keeping response times steady.",
                    "Lightweight standard-library web server, MJPEG camera preview and Chromium kiosk \u2013 no heavy "
                    "framework running beside the models."]),
          PageBreak()]

# ---------- 8 deployment + conclusion ----------
story += [P("8. Deployment on Raspberry Pi 5", H1),
          *bullets(["<b>One-step install</b>: a bundle script copies code and models (about 2 GB) to a USB stick; "
                    "<i>setup.sh</i> on the Pi installs the camera library, Indic fonts, Chromium and ExecuTorch "
                    "1.6.0 (official aarch64 wheels), runs a self-test and registers auto-start.",
                    "<b>Kiosk experience</b>: after power-on the Pi opens the KrishiRakshak screen full-screen. The "
                    "farmer picks state and language, holds a leaf or insect to the camera and taps <i>Take photo</i>.",
                    "<b>Fully offline</b>: once installed, no internet connection is needed; all data stays on the "
                    "device.",
                    "<b>Service management</b>: a systemd service restarts the server automatically; the kiosk also "
                    "accepts uploaded photos and phones on the Pi's network."]),
          P("9. Conclusion", H1),
          P("KrishiRakshak AI shows that a complete crop-protection assistant \u2013 disease and pest detection, "
            "severity, treatment, regional advice and local-language output \u2013 can run entirely on an "
            "affordable Arm computer. ExecuTorch provides one runtime for all the neural networks, and its XNNPACK "
            f"backend with 4-bit quantisation brings Llama 3.2 1B to {S['llama_pte']:,.0f} MB, so the whole system "
            f"needs about {M['peak'] / 1024:.1f} GB of memory and no accelerator. The result is a practical, private "
            "and low-cost tool that can reach farmers where connectivity is weakest."),
          P("Future extensions", H2),
          *bullets(["More crops important to Indian smallholders (rice, wheat, cotton, sugarcane diseases).",
                    "Voice input and spoken answers in local languages.",
                    "An Android app using the same ExecuTorch .pte models."]),
          P("10. Reproducibility", H1),
          table([["Step", "Command"],
                 ["Disease model: data, splits, training, evaluation",
                  "python disease_model/build_index.py ; make_splits.py ; train.py ; evaluate.py"],
                 ["Pest model: data, training, evaluation", "python pest_model/prepare.py ; train.py ; evaluate.py"],
                 ["Export vision models to ExecuTorch", "python advisor/export_vision.py"],
                 ["Export Llama 3.2 1B-Instruct to ExecuTorch", "python advisor/export_llama.py"],
                 ["Run on a photo", 'python advisor/app.py "test images/apple_scab_1.jpg" --region Karnataka --language Kannada'],
                 ["Kiosk server", "python advisor/server.py"],
                 ["Benchmark + this report", "python docs/benchmark.py ; python docs/make_final_report.py"]],
                [5.5 * cm, 10.5 * cm]),
          P("Source code: github.com/24suhass-itech/arm-challenge", SMALL)]

doc = SimpleDocTemplate(OUT_PDF, pagesize=A4, leftMargin=2 * cm, rightMargin=2 * cm, topMargin=1.8 * cm,
                        bottomMargin=1.8 * cm, title="KrishiRakshak AI - Final Project Report",
                        author="KrishiRakshak AI team", subject="Offline AI crop doctor on Arm (ExecuTorch + Llama 3.2)")
doc.build(story, onFirstPage=on_page, onLaterPages=on_page)
print("wrote", OUT_PDF)
