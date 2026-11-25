"""KrishiRakshak command-line demo (everything offline; vision + Llama through ExecuTorch).

    python advisor/app.py "test images/apple_scab_1.jpg" --region Karnataka --language Kannada

On the Raspberry Pi the kiosk (advisor/server.py) uses the same pipeline with the camera.
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pipeline import KrishiRakshak  # noqa: E402


def main():
    sys.stdout.reconfigure(encoding="utf-8")  # Indic text on the Windows console
    ap = argparse.ArgumentParser()
    ap.add_argument("image")
    ap.add_argument("--region", default="India (general)")
    ap.add_argument("--language", default=None, help="default: the region's main language")
    ap.add_argument("--month", type=int, default=None)
    ap.add_argument("--mode", choices=["auto", "disease", "pest"], default="auto")
    ap.add_argument("--no_llm", action="store_true", help="use the template answer instead of Llama")
    a = ap.parse_args()

    kr = KrishiRakshak(use_llm=not a.no_llm)
    if not kr.use_llm and not a.no_llm:
        print("[Llama .pte not found - run advisor/export_llama.py; using the template answer]\n")
    r = kr.run(a.image, a.region, a.language, a.month, a.mode)
    d, sev = r["diagnosis"], r["severity"]
    print(f"Photo     : {a.image}")
    for name, rows in r["top3"].items():  # top prediction of each model
        lab, c = rows[0]
        used = (name == r["diagnosis"]["kind"])
        print(f"{name.capitalize():<10}: {lab} {100 * c:.1f}%" + ("" if used else "  (not used)"))
    print(f"Diagnosis : {d['label']} ({d['kind']}, confidence {100 * d['confidence']:.0f}%, {r['status']})")
    if sev:
        pct = "" if sev["percent"] is None else f" ({sev['percent']:.0f}% of leaf)"
        print(f"Severity  : {sev['level']}{pct}")
    print(f"Region    : {r['region']} | season: {r['season']} | language: {r['language']}")
    print(f"Advice    : [{r['source']}]\n{r['answer']}")
    if r["language"] != "English":
        print(f"\n(English original)\n{r['english']}")
    print("\n(" + ", ".join(f"{k} {v}" for k, v in r["timing"].items()) + ")")


if __name__ == "__main__":
    main()
