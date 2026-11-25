"""Measure on-device performance of the full KrishiRakshak pipeline (CPU only) -> docs/benchmark.json.

    python docs/benchmark.py          (run in the ExecuTorch environment, advisor/requirements.txt)

Records memory after each model is loaded, vision latency (ExecuTorch .pte vs PyTorch eager .pth), severity,
Llama prefill / decode speed, and translation time. Used by docs/make_final_report.py.
"""
import importlib.util
import json
import os
import platform
import statistics
import sys
import time

import psutil
import torch
from PIL import Image

DOCS = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(DOCS)
sys.path.insert(0, os.path.join(ROOT, "advisor"))
import advisor  # noqa: E402
import severity  # noqa: E402
from pipeline import KrishiRakshak  # noqa: E402

PROC = psutil.Process()
IMAGES = ["tomato_early_blight_1.jpg", "apple_scab_1.jpg", "potato_early_blight_1.jpg"]


def rss_mb():
    return round(PROC.memory_info().rss / 2**20)


def timeit(fn, n=30, warm=5):
    for _ in range(warm):
        fn()
    ts = []
    for _ in range(n):
        t = time.perf_counter(); fn(); ts.append(1000 * (time.perf_counter() - t))
    return round(statistics.median(ts), 1)


def load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
    return m


def cpu_name():
    try:
        if sys.platform == "win32":
            import winreg
            k = winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, r"HARDWARE\DESCRIPTION\System\CentralProcessor\0")
            return winreg.QueryValueEx(k, "ProcessorNameString")[0].strip()
        with open("/proc/cpuinfo") as f:  # Linux / Raspberry Pi
            for line in f:
                if line.lower().startswith(("model name", "model")):
                    return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor()


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    out = dict(device=dict(cpu=cpu_name(),cores=psutil.cpu_count(logical=False),
                           threads=psutil.cpu_count(), ram_gb=round(psutil.virtual_memory().total / 2**30, 1),
                           os=platform.platform(), python=platform.python_version(), torch=torch.__version__))
    import executorch
    out["device"]["executorch"] = getattr(executorch, "__version__", "1.6.0")
    mem = dict(start=rss_mb())

    kr = KrishiRakshak(use_llm=True, use_translator=True)
    mem["vision_models"] = rss_mb()
    img = Image.open(os.path.join(ROOT, "test images", IMAGES[0])).convert("RGB")

    # vision: ExecuTorch (.pte, XNNPACK) vs PyTorch eager (.pth, fp32)
    vis = dict(disease_pte_ms=timeit(lambda: kr.disease(img)), pest_pte_ms=timeit(lambda: kr.pest(img)))
    torch.set_num_threads(psutil.cpu_count(logical=False))
    dm = load_module(os.path.join(ROOT, "disease_model", "train.py"), "dm_train")
    ck = torch.load(os.path.join(ROOT, "disease_model", "checkpoints", "disease_model_best.pth"), map_location="cpu",
                    weights_only=False)
    m = dm.build_model(len(ck["classes"]), pretrained=False); m.load_state_dict(ck["model_state_dict"]); m.eval()
    x = dm.EVAL_TF(img)[None]
    with torch.no_grad():
        vis["disease_eager_ms"] = timeit(lambda: m(x))
    pm = load_module(os.path.join(ROOT, "pest_model", "train.py"), "pm_train")
    ck = torch.load(os.path.join(ROOT, "pest_model", "checkpoints", "efficientnet_b0_best.pth"), map_location="cpu",
                    weights_only=False)
    p = pm.build_model(ck.get("arch", "efficientnet_b0"), len(ck["classes"]), pretrained=False)
    p.load_state_dict(ck["model_state_dict"]); p.eval()
    xp = torch.randn(1, 3, 256, 256)
    with torch.no_grad():
        vis["pest_eager_ms"] = timeit(lambda: p(xp))
    del m, p
    vis["severity_ms"] = timeit(lambda: severity.estimate(img, "fungal"), n=10, warm=2)
    out["vision"] = vis

    # advisory: Llama 3.2 1B-Instruct (ExecuTorch) on three photos
    t = time.perf_counter(); kr.llm(); load_s = time.perf_counter() - t
    mem["plus_llama"] = rss_mb()
    runs = []
    for name in IMAGES:
        r = kr.run(os.path.join(ROOT, "test images", name), "Karnataka", "English")
        runs.append(dict(image=name, diagnosis=r["diagnosis"]["label"], confidence=round(r["diagnosis"]["confidence"], 3),
                         severity=(r["severity"] or {}).get("level"), **r["timing"]))
        print(runs[-1], flush=True)
    out["llama"] = dict(load_s=round(load_s, 1), runs=runs)
    english = r["english"]

    # translation (IndicTrans2 200M), beams as on the Pi
    t = time.perf_counter(); tr = kr.translator(); tr_load = time.perf_counter() - t
    mem["plus_translator"] = rss_mb()
    trans = dict(load_s=round(tr_load, 1), beams=tr.num_beams, runs={})
    for lang in ["Hindi", "Kannada", "Tamil"]:
        t = time.perf_counter(); tr.translate_advice(english, lang, ["Early blight"]); trans["runs"][lang] = round(time.perf_counter() - t, 1)
        print(lang, trans["runs"][lang], flush=True)
    out["translation"] = trans
    mem["peak"] = round(getattr(PROC.memory_info(), "peak_wset", PROC.memory_info().rss) / 2**20)
    out["memory_mb"] = mem

    sizes = {}
    for k, pth in dict(llama_original=r"advisor/llama/original/consolidated.00.pth",
                       llama_pte=r"advisor/llama/llama3_2_1b_instruct_8da4w.pte",
                       disease_pth="disease_model/checkpoints/disease_model_best.pth",
                       disease_pte="advisor/models/disease_model.pte",
                       pest_pth="pest_model/checkpoints/efficientnet_b0_best.pth",
                       pest_pte="advisor/models/pest_model.pte",
                       translator="advisor/translator/indictrans2-en-indic-dist-200M/model.safetensors").items():
        fp = os.path.join(ROOT, pth)
        sizes[k] = round(os.path.getsize(fp) / 2**20, 1) if os.path.exists(fp) else None
    out["sizes_mb"] = sizes
    with open(os.path.join(DOCS, "benchmark.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(json.dumps(out, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
