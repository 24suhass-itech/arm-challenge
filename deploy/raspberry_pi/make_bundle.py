"""Collect everything the Raspberry Pi needs into one folder (about 2 GB) to copy with a USB stick.

    python deploy/raspberry_pi/make_bundle.py            # -> KrishiRakshak_Pi/ in the project folder
    python deploy/raspberry_pi/make_bundle.py E:\\        # -> straight onto a USB drive

Included: advisor code + knowledge base + kiosk page, the three ExecuTorch models (disease, pest, Llama 3.2 1B),
the Llama tokenizer, the IndicTrans2 translator, the test photos and the Pi setup scripts.
Not included: the 2.5 GB original Llama checkpoint, datasets, training code and caches.
"""
import os
import shutil
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
ITEMS = [
    "advisor/advisor.py", "advisor/pipeline.py", "advisor/severity.py", "advisor/translate.py", "advisor/app.py",
    "advisor/server.py", "advisor/requirements.txt", "advisor/web", "advisor/knowledge", "advisor/models",
    "advisor/llama/llama3_2_1b_instruct_8da4w.pte", "advisor/llama/original/tokenizer.model",
    "advisor/llama/original/params.json", "advisor/translator/indictrans2-en-indic-dist-200M",
    "deploy/raspberry_pi", "test images", "README.md",
]
SKIP = {"__pycache__", ".cache", "pytorch_model.bin"}


def copy(src, dst):
    if os.path.isdir(src):
        for name in os.listdir(src):
            if name not in SKIP:
                copy(os.path.join(src, name), os.path.join(dst, name))
    else:
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        if not (os.path.exists(dst) and os.path.getsize(dst) == os.path.getsize(src)):
            shutil.copy2(src, dst)


def main():
    out = os.path.join(sys.argv[1] if len(sys.argv) > 1 else ROOT, "KrishiRakshak_Pi")
    missing = [i for i in ITEMS if not os.path.exists(os.path.join(ROOT, i))]
    if missing:
        sys.exit("missing (run advisor/export_llama.py and download the translator first): " + ", ".join(missing))
    total = 0
    for item in ITEMS:
        copy(os.path.join(ROOT, item), os.path.join(out, item))
    for dp, _, fs in os.walk(out):
        total += sum(os.path.getsize(os.path.join(dp, f)) for f in fs)
    print(f"bundle ready: {out} ({total / 1e9:.2f} GB)")
    print("On the Pi:  copy the folder to /home/<user>/KrishiRakshak_Pi, then run:")
    print("            cd ~/KrishiRakshak_Pi && bash deploy/raspberry_pi/setup.sh")


if __name__ == "__main__":
    main()
