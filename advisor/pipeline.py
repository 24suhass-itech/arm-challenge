"""KrishiRakshak on-device pipeline: load every model once, then photo -> diagnosis -> severity -> advice -> language.

Two separate pipelines (documented separately, as the challenge requires):
  1. Vision   : disease_model.pte + pest_model.pte  (ExecuTorch, XNNPACK backend, Arm CPU)
                -> decide disease / pest / retake -> OpenCV severity (lesion area)
  2. Advisory : knowledge base + region + season -> Llama 3.2 1B-Instruct .pte (ExecuTorch, XNNPACK, Arm CPU)
                -> English advice -> IndicTrans2 (PyTorch, CPU) -> farmer's language
"""
import datetime
import json
import os
import sys
import threading
import time

import numpy as np
import torch
from PIL import Image, ImageOps

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import advisor  # noqa: E402
import severity  # noqa: E402

MODELS = os.path.join(HERE, "models")
PEST_CONFIDENT = 0.71   # the pest model has no "no pest" class yet: only trust it when confident


class VisionModel:
    def __init__(self, name):
        from executorch.runtime import Runtime
        with open(os.path.join(MODELS, f"{name}_meta.json"), encoding="utf-8") as f:
            self.meta = json.load(f)
        self.method = Runtime.get().load_program(os.path.join(MODELS, f"{name}.pte")).load_method("forward")
        self.classes = self.meta["classes"]

    def __call__(self, img):
        s = self.meta["input_size"]
        x = np.asarray(img.convert("RGB").resize((s, s), Image.BILINEAR), np.float32) / 255.0
        x = (x - np.array(self.meta["mean"], np.float32)) / np.array(self.meta["std"], np.float32)
        x = torch.from_numpy(x.transpose(2, 0, 1)[None].copy())
        return torch.softmax(self.method.execute([x])[0], 1)[0]


def pretty(c):
    return c.replace("___", ": ").replace("_", " ").replace(",", "").strip()


class KrishiRakshak:
    def __init__(self, use_llm=True, use_translator=True, preload=False):
        self.disease = VisionModel("disease_model")
        self.pest = VisionModel("pest_model")
        self.use_llm = use_llm and os.path.exists(advisor.DEFAULT_PTE)
        self.use_translator = use_translator
        self._llm = self._tr = None
        self._lock = threading.Lock()  # one request at a time (Raspberry Pi: 4 cores, 8 GB)
        if preload:
            self.llm()
            self.translator()

    def llm(self):
        if self._llm is None and self.use_llm:
            self._llm = advisor.LlamaAdvisor()
        return self._llm

    def translator(self):
        from translate import MODEL_DIR, Translator
        if self._tr is None and self.use_translator and os.path.exists(MODEL_DIR):
            self._tr = Translator()
        return self._tr

    def diagnose(self, img, mode="auto"):
        dp, pp = self.disease(img), self.pest(img)
        d_conf, d_i = float(dp.max()), int(dp.argmax())
        p_conf, p_i = float(pp.max()), int(pp.argmax())
        d_cls = self.disease.classes[d_i]
        # alternatives for the "not fully sure" note: only other problems of the SAME crop (the farmer knows the
        # crop; e.g. tomato early blight is the same fungus as potato early blight and only differs by leaf shape)
        crop = self.disease.classes[d_i].split("___")[0]
        d_alt = [pretty(self.disease.classes[i]) for i in dp.topk(6).indices[1:].tolist()
                 if self.disease.classes[i].split("___")[0] == crop and float(dp[i]) >= 0.05][:2]
        p_alt = [advisor.PESTS["pests"][str(i)][0] for i in pp.topk(3).indices[1:].tolist()]
        self.last_top3 = dict(
            disease=[(pretty(self.disease.classes[i]), float(dp[i])) for i in dp.topk(3).indices.tolist()],
            pest=[(advisor.PESTS["pests"][str(i)][0], float(pp[i])) for i in pp.topk(3).indices.tolist()])
        pest_wins = p_conf >= PEST_CONFIDENT and (d_cls == "Background" or p_conf > d_conf)
        if mode == "pest" or (mode == "auto" and pest_wins):
            return dict(kind="pest", key=p_i, confidence=p_conf, alternatives=p_alt,
                        label=advisor.PESTS["pests"][str(p_i)][0])
        if d_cls == "Background":
            return dict(kind="background", key=None, confidence=d_conf, alternatives=[], label="not a crop leaf")
        return dict(kind="disease", key=d_cls, confidence=d_conf, alternatives=d_alt, label=pretty(d_cls))

    def run(self, image, region="India (general)", language=None, month=None, mode="auto"):
        img = ImageOps.exif_transpose(image if isinstance(image, Image.Image) else Image.open(image)).convert("RGB")
        with self._lock:
            t0 = time.time()
            d = self.diagnose(img, mode)
            sev = severity.estimate(img, advisor.DISEASES[d["key"]]["type"]) if d["kind"] == "disease" else None
            t_vision = time.time() - t0
            ctx = advisor.build_context(d["kind"], d["key"], d["confidence"], sev, region, language,
                                        month or datetime.date.today().month, d["alternatives"])
            t1 = time.time()
            llm = self.llm() if ctx["status"] != "retake" else None
            english, source = advisor.advise(ctx, llm)
            t_llm = time.time() - t1
            answer, t_tr = english, 0.0
            if ctx["language"] != "English":
                from translate import LANG_CODES
                tr = self.translator() if ctx["language"] in LANG_CODES else None
                if tr is not None:
                    t2 = time.time()
                    terms = []
                    if ctx.get("facts"):
                        crop = ctx["facts"].get("crop", "").split(" /")[0].lower()
                        name = ctx["facts"]["name"]
                        terms = [name, name[len(crop) + 1:] if name.lower().startswith(crop + " ") else ""]
                    answer = tr.translate_advice(english, ctx["language"], terms)
                    source += " + indictrans2"
                    t_tr = time.time() - t2
        return dict(diagnosis=d, top3=self.last_top3, severity=sev, status=ctx["status"], region=ctx["region"], season=ctx["season"],
                    language=ctx["language"], answer=answer, english=english, source=source,
                    timing=dict(vision_s=round(t_vision, 2), advice_s=round(t_llm, 1), translate_s=round(t_tr, 1),
                                **(getattr(llm, "last_stats", {}) if llm else {})))
