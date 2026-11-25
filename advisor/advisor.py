"""KrishiRakshak advisory module: diagnosis + severity + region + language -> practical advice from Llama (ExecuTorch).

Pipeline (all offline):
    vision result (disease or pest class, confidence)
      -> knowledge base entry (advisor/knowledge/*.json)
      -> severity (advisor/severity.py, lesion area from the photo)
      -> region + season context (advisor/knowledge/regions.json)
      -> compact prompt -> Llama 3.2 1B-Instruct exported with ExecuTorch (.pte, XNNPACK, 4-bit)
      -> short answer in the farmer's language
If the Llama .pte is not available, a template answer is built from the same facts, so the app always responds.
"""
import datetime
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
KB_DIR = os.path.join(HERE, "knowledge")
LLAMA_DIR = os.path.join(HERE, "llama")
DEFAULT_PTE = os.path.join(LLAMA_DIR, "llama3_2_1b_instruct_8da4w.pte")
DEFAULT_TOKENIZER = os.path.join(LLAMA_DIR, "original", "tokenizer.model")

CONFIDENT = 0.70          # answer directly at or above this confidence
UNSURE = 0.20             # below this, ask for a better photo; in between, show the top candidates
SAFETY = ("Wear gloves and a mask when spraying, follow the product label and the waiting period before harvest, "
          "and confirm products with your local KVK or Kisan Call Centre 1800-180-1551.")


def _load(name):
    with open(os.path.join(KB_DIR, name), encoding="utf-8") as f:
        return json.load(f)


DISEASES = _load("diseases.json")
PESTS = _load("pests.json")
REGIONS = _load("regions.json")


def season_of(month):
    for name, s in REGIONS["seasons"].items():
        if month in s["months"]:
            return name, s["risk"]
    return "", ""


def facts_for(kind, key):
    """Return (title, facts dict) from the knowledge base for a disease class name or a pest class index."""
    if kind == "disease":
        e = DISEASES[key]
        on_crop = "" if e["crop"].split()[0].lower() in e["name"].lower() else f" on {e['crop']}"
        return e["name"] + on_crop, e
    name, crop, group = PESTS["pests"][str(key)]
    g = PESTS["groups"][group]
    return f"{name} ({g['name'].lower()}) on {crop}", dict(g, crop=crop, name=name, type="pest")


def build_context(kind, key, confidence, severity=None, region="India (general)", language=None, month=None,
                  alternatives=()):
    """Collect everything the advice is based on. kind: 'disease' | 'pest' | 'background'."""
    month = month or datetime.date.today().month
    reg = REGIONS["regions"].get(region, REGIONS["regions"]["India (general)"])
    language = language or reg["language"]
    season, season_risk = season_of(month)
    ctx = dict(kind=kind, key=key, confidence=float(confidence), region=region, region_info=reg, language=language,
               season=season, season_risk=season_risk, severity=severity or {}, alternatives=list(alternatives))
    if kind == "background" or confidence < UNSURE:
        ctx["status"] = "retake"
        return ctx
    ctx["status"] = "confident" if confidence >= CONFIDENT else "possible"
    ctx["title"], ctx["facts"] = facts_for(kind, key)
    return ctx


def _bullet(xs, n=3):
    return "; ".join(xs[:n])


def facts_text(ctx, for_llm=False):
    """Compact fact sheet. for_llm=True drops general region/season notes that a 1B model tends to mix in."""
    f = ctx["facts"]
    lines = [f"Problem: {ctx['title']}" + ("" if ctx["status"] == "confident" else " (possible; not certain)")]
    if f.get("type") == "healthy":
        lines.append("Advice: " + _bullet(f["advice"]))
    else:
        lines.append(f"Type: {f.get('type', '')}")
        lines.append("Signs: " + (f.get("symptoms") or f.get("damage", "")))
        sev = ctx["severity"]
        if sev.get("percent") is not None:
            lines.append(f"Severity: {sev['level']} (about {sev['percent']:.0f}% of the leaf affected)")
        lines.append("Field practices: " + _bullet(f["cultural"]))
        lines.append("Organic options: " + _bullet(f["organic"], 2))
        lines.append("Chemical options: " + _bullet(f["chemical"], 2))
        extra = f.get("prevention") or f.get("threshold")
        if extra:
            lines.append("When to act: " + extra)
    lines.append(f"Region: {ctx['region']} ({ctx['region_info']['zone']}). {ctx['region_info']['notes']}")
    if ctx["season"]:
        lines.append(f"Season now: {ctx['season']}. {ctx['season_risk']}")
    lines.append("Safety: " + SAFETY)
    return "\n".join(lines)


SYSTEM = ("You are KrishiRakshak, an agricultural extension assistant for small farmers. "
          "Use ONLY the FACTS. Never add products, doses, spray intervals or other details that are not in the FACTS. "
          "Write short, simple sentences a farmer can follow today. Use exactly this format:\n"
          "Problem: <one sentence>\n"
          "What to do:\n1. ...\n2. ...\n3. ...\n(up to 4 steps; non-chemical steps first; include one chemical "
          "option with its dose only if severity is moderate or severe)\n"
          "Prevention: <one sentence>\n"
          "Safety: <one sentence>\n"
          "Maximum 120 words. No markdown, no bold text.")


EXAMPLE_FACTS = """Problem: Septoria leaf spot on Tomato
Severity: moderate (about 15% of the leaf affected)
Field practices: Remove infected lower leaves; Mulch to stop soil splash
Organic options: Copper oxychloride 50 WP 3 g/L
Chemical options: Chlorothalonil 75 WP 2 g/L
When to act: Spray at first spots and repeat every 7-10 days in wet weather."""
EXAMPLE_ANSWER = """Problem: Your tomato has Septoria leaf spot, and about 15% of the leaf is affected.
What to do:
1. Remove the spotted lower leaves and burn them.
2. Put mulch under the plants so soil does not splash on the leaves.
3. Spray Chlorothalonil 75 WP at 2 g per litre of water.
Prevention: Spray at the first spots and repeat every 7-10 days in wet weather.
Safety: Wear gloves and a mask, and follow the label."""


def build_prompt(ctx):
    """Llama 3 chat prompt with one worked example (keeps a 1B model on format). The tokenizer adds BOS."""
    # Llama 3.2 1B writes reliable English but weak Indian languages, so it always answers in English and
    # advisor/translate.py (IndicTrans2) translates the answer into the farmer's language.
    # Alternatives are NOT put in the prompt (a 1B model copies them into the answer); advise() adds that note.
    user = f"FACTS:\n{facts_text(ctx, for_llm=True)}\nWrite the answer in simple English."

    def turn(role, text):
        return f"<|start_header_id|>{role}<|end_header_id|>\n\n{text}<|eot_id|>"

    return (turn("system", SYSTEM) + turn("user", f"FACTS:\n{EXAMPLE_FACTS}\nWrite the answer in English.") +
            turn("assistant", EXAMPLE_ANSWER) + turn("user", user) +
            "<|start_header_id|>assistant<|end_header_id|>\n\n" + problem_line(ctx) + "\nWhat to do:\n1.")


def problem_line(ctx):
    """The diagnosis sentence is written by code, never by the LLM, so the disease / pest name is always the one
    the vision model found. Llama continues the answer from 'What to do:'."""
    f, sev = ctx["facts"], ctx["severity"]
    crop = f.get("crop", "crop").split(" /")[0].lower()
    if f.get("type") == "healthy":
        return f"Problem: No disease is seen; your {crop} looks healthy."
    name = f["name"]
    if name.lower().startswith(crop + " "):  # "Potato early blight" on potato -> "early blight"
        name = name[len(crop) + 1:]
    what = f"{name} (a pest)" if f.get("type") == "pest" else name
    line = f"Problem: Your {crop} has {what}"
    if sev.get("percent") is not None:
        line += f". About {sev['percent']:.0f}% of the leaf is affected, so the damage is {sev['level']}"
    return line + "."


def template_answer(ctx):
    """Deterministic answer from the same facts (used when the LLM is unavailable, and as a reference)."""
    if ctx["status"] == "retake":
        return ("I could not identify a crop problem in this photo. Please take a close, well-lit photo of one "
                "affected leaf or insect, filling most of the picture.")
    f = ctx["facts"]
    head = ("This looks like " if ctx["status"] == "confident" else "This may be ") + ctx["title"] + "."
    if ctx["status"] == "possible" and ctx["alternatives"]:
        head += (" I am not fully sure - it could also be " + " or ".join(ctx["alternatives"][:2]) +
                 ". Compare with pictures or ask your KVK to confirm.")
    if f.get("type") == "healthy":
        return head + " " + " ".join(f["advice"])
    sev = ctx["severity"]
    out = [head + (f" Severity: {sev['level']} (~{sev['percent']:.0f}% of leaf)." if sev.get("percent") is not None
                   else "")]
    steps = list(f["cultural"][:2]) + list(f["organic"][:1])
    if sev.get("level") in ("moderate", "severe") or sev.get("percent") is None:
        steps.append("If it keeps spreading: " + f["chemical"][0])
    out.append("What to do:\n" + "\n".join(f"{i}. {s}" for i, s in enumerate(steps, 1)))
    if f.get("prevention"):
        out.append("Prevention: " + f["prevention"])
    elif f.get("threshold"):
        out.append("Act when: " + f["threshold"])
    if ctx["season_risk"]:
        out.append(f"This season ({ctx['season']}): {ctx['season_risk']}")
    out.append("Safety: " + SAFETY)
    return "\n".join(out)


class LlamaAdvisor:
    """Runs the ExecuTorch-exported Llama 3.2 1B-Instruct .pte (XNNPACK, 4-bit) with the ExecuTorch runtime.
    Only the runtime is needed (same code on Windows x86 and on the Raspberry Pi 5 Arm CPU)."""

    def __init__(self, pte=DEFAULT_PTE, tokenizer=DEFAULT_TOKENIZER, max_len=1024):
        from executorch.extension.pybindings.portable_lib import _load_for_executorch
        # register the custom SDPA/KV-cache ops and quantized kernels used by the exported Llama graph
        from executorch.extension.llm.custom_ops import custom_ops  # noqa: F401
        from executorch.kernels import quantized  # noqa: F401
        from pytorch_tokenizers import get_tokenizer

        class _Runner:  # same attribute names as ExecuTorch's NativeLlamaRunner
            pass

        self.runner = _Runner()
        self.runner.model = _load_for_executorch(pte)
        self.runner.tokenizer = get_tokenizer(tokenizer)
        self.max_len = max_len

    STOP_IDS = {128001, 128008, 128009}  # <|end_of_text|>, <|eom_id|>, <|eot_id|>

    def generate(self, prompt, max_new_tokens=300, clean=True):
        """Greedy decoding with the ExecuTorch .pte (KV cache): one prefill call, then one token per call."""
        import time
        import torch
        tok = self.runner.tokenizer
        # allowed_special: encode <|start_header_id|>, <|eot_id|>, ... as Llama 3 special tokens, not as plain text
        ids = tok.encode(prompt, bos=True, eos=False, allowed_special="all")
        m = self.runner.model
        t0 = time.time()
        logits = m.forward((torch.tensor([ids], dtype=torch.long), torch.tensor([0], dtype=torch.long)))[0]
        t1 = time.time()
        out, pos = [], len(ids)
        nxt = int(logits.reshape(-1, logits.shape[-1])[-1].argmax())
        while nxt not in self.STOP_IDS and len(out) < max_new_tokens and pos < self.max_len - 1:
            out.append(nxt)
            if len(out) >= 24 and out[-12:] == out[-24:-12]:  # stuck repeating the same phrase
                out = out[:-12]
                break
            logits = m.forward((torch.tensor([[nxt]], dtype=torch.long), torch.tensor([pos], dtype=torch.long)))[0]
            pos += 1
            nxt = int(logits.reshape(-1, logits.shape[-1])[-1].argmax())
        t2 = time.time()
        self.last_stats = {"prompt tokens": len(ids), "prefill s": round(t1 - t0, 2), "new tokens": len(out),
                           "tokens/s": round(len(out) / max(t2 - t1, 1e-6), 1)}
        text = tok.decode(out)
        if not clean:  # continuation of a prefilled answer: keep it raw, just ensure one leading space
            return " " + text.lstrip(" ")
        return self.clean(text)

    @staticmethod
    def clean(text):
        """Keep only the requested format (a 1B model tends to add extra sections such as 'Organic options:' or
        'Region:' with details that are not in the facts): Problem, What to do + numbered steps, Prevention, Safety.
        Stops after the first Safety line; at most 5 steps."""
        import re
        out, steps = [], 0
        for l in text.strip().splitlines():
            s = l.strip()
            low = s.lower()
            if not s:
                continue
            if re.match(r"^\d+\.", s):
                steps += 1
                if steps <= 5:
                    out.append(s)
            elif low.startswith(("problem:", "what to do", "prevention:")):
                out.append(s)
            elif low.startswith("safety:"):
                out.append(s)
                break
        return "\n".join(out).strip()


def advise(ctx, llm=None):
    """Return (answer text, source) where source is 'llama-executorch' or 'template'."""
    if ctx["status"] == "retake" or llm is None:
        return template_answer(ctx), "template"
    head = problem_line(ctx) + "\nWhat to do:\n1."
    text = LlamaAdvisor.clean(head + llm.generate(build_prompt(ctx), clean=False))
    # Llama writes only the steps. Problem, Prevention, Safety and the uncertainty note are written by code
    # from the knowledge base (the 1B model invented harvest waiting times and repeated lines there).
    lines = [l for l in text.splitlines() if not l.lower().startswith(("prevention:", "safety:"))]
    f = ctx["facts"]
    if f.get("prevention"):
        lines.append("Prevention: " + f["prevention"])
    elif f.get("threshold"):
        lines.append("Prevention: Act when " + f["threshold"][0].lower() + f["threshold"][1:])
    lines.append("Safety: " + SAFETY)
    if ctx["status"] == "possible":
        alt = (" It could also be " + " or ".join(ctx["alternatives"][:2]) + ".") if ctx["alternatives"] else ""
        lines.append("Note: I am not fully sure." + alt + " Please confirm with your local KVK.")
    return "\n".join(lines), "llama-executorch"
