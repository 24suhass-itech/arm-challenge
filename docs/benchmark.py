"""Measure on-device performance of the full KrishiRakshak pipeline (CPU only) -> docs/benchmark.json.

    python docs/benchmark.py          (run in the ExecuTorch environment, advisor/requirements.txt)

Records memory after each model is loaded, vision latency (ExecuTorch .pte vs PyTorch eager .pth), severity,
Llama prefill / decode speed, and translation time. Used by docs/make_final_report.py.

Also writes two extra sections (not used by the report yet):
  "timing" : full latency statistics (min / mean / median / p90 / p99 / max / std, throughput) for each vision stage,
             run back-to-back for --seconds, plus per-request end-to-end time, time to first token and token rates.
  "power"  : average power, energy and energy per inference / per token for each stage, measured against an idle
             baseline. The sensor is picked automatically:
               Raspberry Pi 5   -> PMIC rails (vcgencmd pmic_read_adc)
               Linux x86        -> RAPL CPU package counter (may need sudo), else battery power while unplugged
               Windows laptop   -> battery discharge rate (WMI), only while running on battery (unplug the charger)
             With no sensor the power numbers are null; timing still works.

    python docs/benchmark.py --seconds 10 --idle 5     (seconds per vision stage, seconds of idle baseline)
"""
import argparse
import glob
import importlib.util
import json
import os
import platform
import re
import shutil
import statistics
import subprocess
import sys
import threading
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


# ---------------------------------------------------------------- timing statistics + power measurement

def parse_args():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--seconds", type=float, default=10, help="seconds each vision stage runs back-to-back")
    ap.add_argument("--idle", type=float, default=5, help="seconds of idle baseline power before the benchmark")
    ap.add_argument("--interval", type=float, default=None, help="power sampling interval in seconds")
    return ap.parse_args()


def stats_ms(ts):
    """Latency statistics (ms) for a list of timings in ms."""
    s, n = sorted(ts), len(ts)
    pct = lambda q: s[min(n - 1, round(q * (n - 1)))]  # noqa: E731
    mean = statistics.fmean(s)
    return dict(n=n, min=round(s[0], 2), mean=round(mean, 2), median=round(statistics.median(s), 2),
                p90=round(pct(0.90), 2), p99=round(pct(0.99), 2), max=round(s[-1], 2),
                std=round(statistics.stdev(s), 2) if n > 1 else 0.0, per_second=round(1000 / mean, 1))


def sustained(fn, pm, name, seconds, warm=5):
    """Run fn back-to-back for `seconds` (at least 3 times): latency statistics + power/energy per inference."""
    for _ in range(warm):
        fn()
    ts = []
    pm.begin(name)
    end = time.perf_counter() + seconds
    while time.perf_counter() < end or len(ts) < 3:
        t = time.perf_counter(); fn(); ts.append(1000 * (time.perf_counter() - t))
    pm.end(name, count=len(ts), unit="inference")
    st = stats_ms(ts)
    print(f"{name}: median {st['median']} ms, p90 {st['p90']} ms, {pm.segments[name].get('avg_w')} W", flush=True)
    return st


def run_cmd(*cmd):
    try:
        return subprocess.run(cmd, capture_output=True, text=True, timeout=3).stdout
    except (OSError, subprocess.SubprocessError):
        return ""


def cpu_temp_c():
    """CPU temperature in C (Linux / Raspberry Pi); None on Windows."""
    try:
        temps = psutil.sensors_temperatures() or {}
    except (AttributeError, OSError):
        return None
    for key in ("cpu_thermal", "k10temp", "coretemp", "soc_thermal", "acpitz"):
        if temps.get(key):
            return round(max(e.current for e in temps[key]), 1)
    return round(next(iter(temps.values()))[0].current, 1) if temps else None


class PowerMeter:
    """Samples power (W), CPU load and temperature in a background thread. begin(name) / end(name) measure a stage:
    average power over the stage's samples x duration = energy; minus the idle baseline = energy the work added."""

    def __init__(self, interval=None):
        self.samples = []          # (time, watts or None, cpu %, temp C)
        self.segments, self._open = {}, {}
        self.idle_w = None
        self._stop = threading.Event()
        self._thread = self._proc = None
        self._latest = None
        self._last_energy = None
        self.source, self.note, self._read, default_interval = self._detect()
        self.interval = interval or default_interval

    # -- sensors: each returns watts (float) or None
    @staticmethod
    def _pi5_pmic():
        amps, volts = {}, {}
        for line in run_cmd("vcgencmd", "pmic_read_adc").splitlines():
            m = re.match(r"\s*(\S+)_([AV])\s+\w+\(\d+\)=([\d.]+)", line)
            if m:
                (amps if m[2] == "A" else volts)[m[1]] = float(m[3])
        return sum(a * volts[k] for k, a in amps.items() if k in volts) or None

    def _rapl(self):
        with open(self._rapl_file) as f:
            e, t = int(f.read()), time.perf_counter()
        prev, self._last_energy = self._last_energy, (e, t)
        if prev is None or t <= prev[1]:
            return None
        de = e - prev[0]
        if de < 0:  # counter wrapped
            de += self._rapl_max
        return de / 1e6 / (t - prev[1])

    def _battery_sysfs(self):
        bat = self._battery_dir
        def rd(n):  # noqa: E306
            with open(os.path.join(bat, n)) as f:
                return f.read().strip()
        if rd("status") != "Discharging":
            return None
        if os.path.exists(os.path.join(bat, "power_now")):
            return int(rd("power_now")) / 1e6
        return int(rd("current_now")) * int(rd("voltage_now")) / 1e12

    def _windows_battery(self):
        return self._latest

    def _start_windows_reader(self):
        """One long-lived PowerShell loop (starting PowerShell per sample would itself load the CPU). Prints the
        discharge rate (mW) and the remaining charge (mWh). Many laptops report the rate as unknown; then the energy
        of each stage comes from the drop in remaining charge (see _capacity_segments)."""
        loop = ("while($true){$b=Get-CimInstance -Namespace root\\wmi -ClassName BatteryStatus;"
                "$r=($b|Where-Object{$_.Discharging -and $_.DischargeRate -gt 0}|Measure-Object -Sum DischargeRate).Sum;"
                "$c=($b|Measure-Object -Sum RemainingCapacity).Sum;"
                "[Console]::Out.WriteLine(\"$r $c\");[Console]::Out.Flush();Start-Sleep -Milliseconds %d}"
                % int(self.interval * 1000))
        self._proc = subprocess.Popen(["powershell", "-NoProfile", "-NonInteractive", "-Command", loop],
                                      stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True,
                                      creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        self._cap_points, self._cap_last = [], None   # (time the reported charge changed, mWh)

        def reader():
            for line in self._proc.stdout:
                toks = line.split()  # "rate cap", or just "cap" when the rate is unknown
                rate, cap = toks if len(toks) == 2 else ("", toks[0]) if toks else ("", "")
                self._latest = float(rate) / 1000 if rate not in ("", "0") else None  # mW -> W
                if cap.isdigit():
                    cap = int(cap)
                    if self._cap_last is not None and cap != self._cap_last:
                        self._cap_points.append((time.perf_counter(), cap))
                    self._cap_last = cap
        threading.Thread(target=reader, daemon=True).start()

    def _cap_at(self, t):
        """Remaining charge (mWh) at time t: linear between the moments the reported value changed."""
        p = self._cap_points
        i = next((k for k in range(1, len(p)) if p[k][0] >= t), len(p) - 1)
        (ta, ca), (tb, cb) = p[i - 1], p[i]
        return ca + (cb - ca) * (t - ta) / (tb - ta)

    def _capacity_segments(self):
        """Fallback when the battery reports no discharge rate: energy = drop in remaining charge over the stage."""
        steps = [a[1] - b[1] for a, b in zip(self._cap_points, self._cap_points[1:])]
        self.source = "battery remaining charge (Windows WMI BatteryStatus.RemainingCapacity)"
        self.note = (f"this laptop reports no discharge rate, so energy = drop in remaining charge (mWh) over each "
                     f"stage, interpolated between the {len(self._cap_points)} moments the charge value changed "
                     f"(median step {statistics.median(steps) if steps else '?'} mWh). Whole-laptop power, screen "
                     f"included. Accurate for long stages (a minute or more); short stages are only indicative.")
        self.idle_w = None
        for name in ["idle"] + [n for n in self._raw if n != "idle"]:
            if name not in self._raw:
                continue
            t0, t1, count, unit = self._raw[name]
            sec = t1 - t0
            ej = max(0.0, (self._cap_at(t0) - self._cap_at(t1)) * 3.6)   # mWh -> J
            seg, avg = self.segments[name], ej / sec
            seg.update(avg_w=round(avg, 2), energy_j=round(ej, 1), samples=None)
            seg.pop("max_w", None)
            if name == "idle":
                self.idle_w = round(avg, 2)
            elif self.idle_w is not None:
                seg["above_idle_w"] = round(avg - self.idle_w, 2)
                seg["above_idle_j"] = round((avg - self.idle_w) * sec, 1)
            if count:
                seg["count"] = count
                seg[f"mj_per_{unit}"] = round(1000 * ej / count, 2)
                if self.idle_w is not None:
                    seg[f"above_idle_mj_per_{unit}"] = round(1000 * (avg - self.idle_w) * sec / count, 2)

    def _detect(self):
        if shutil.which("vcgencmd") and self._pi5_pmic():
            return ("Raspberry Pi 5 PMIC rails (vcgencmd pmic_read_adc)",
                    "sum of V x I over the PMIC rails (SoC, RAM, IO). A USB-C power meter at the input also counts "
                    "conversion losses and USB devices, so it reads higher.", self._pi5_pmic, 0.5)
        for d in sorted(glob.glob("/sys/class/powercap/intel-rapl:[0-9]")):
            f = os.path.join(d, "energy_uj")
            if os.access(f, os.R_OK):
                self._rapl_file = f
                try:
                    with open(os.path.join(d, "max_energy_range_uj")) as fh:
                        self._rapl_max = int(fh.read())
                except OSError:
                    self._rapl_max = 2**32
                return ("RAPL CPU package energy counter", "CPU package only (no RAM / screen / board).",
                        self._rapl, 0.25)
        for d in sorted(glob.glob("/sys/class/power_supply/BAT*")):
            self._battery_dir = d
            return ("battery discharge (sysfs)", "whole-system power; only measured while unplugged.",
                    self._battery_sysfs, 0.5)
        if sys.platform == "win32":
            st = run_cmd("powershell", "-NoProfile", "-NonInteractive", "-Command",
                         "(Get-CimInstance -Namespace root\\wmi -ClassName BatteryStatus).PowerOnline")
            if "False" in st:
                return ("battery discharge rate (Windows WMI BatteryStatus)",
                        "whole-laptop power (screen included) from the battery; Windows refreshes it only every "
                        "few seconds, so short stages are approximate.", self._windows_battery, 1.0)
            note = ("laptop is on the charger: Windows only reports power from the battery. Unplug it and re-run, "
                    "or use a USB-C power meter." if "True" in st else "no battery / power sensor found.")
            return None, note, None, 1.0
        return None, "no power sensor found (no vcgencmd, readable RAPL or battery).", None, 1.0

    def _sample(self):
        try:
            return self._read() if self._read else None
        except (OSError, ValueError):
            return None

    def _loop(self):
        psutil.cpu_percent(None)
        while not self._stop.wait(self.interval):
            w = self._sample()
            self.samples.append((time.perf_counter(), round(w, 3) if w else None, psutil.cpu_percent(None),
                                 cpu_temp_c()))

    def start(self):
        if self._read == self._windows_battery:
            self._start_windows_reader()
        self.temp_start_c = cpu_temp_c()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()
        return self

    def _capacity_mode(self):
        return (self._read == self._windows_battery and not any(s[1] for s in self.samples)
                and len(getattr(self, "_cap_points", [])) >= 2)

    def stop(self):
        if self._read == self._windows_battery and not any(s[1] for s in self.samples):
            # charge-drop fallback: wait (up to 2 min) for the next change so the last stage is bracketed
            n, t_end = len(getattr(self, "_cap_points", [])), time.perf_counter() + 120
            print("waiting for the next battery charge update ...", flush=True)
            while len(self._cap_points) == n and time.perf_counter() < t_end:
                time.sleep(1)
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=5)
        if self._proc:
            self._proc.kill()

    def begin(self, name):
        self._open[name] = time.perf_counter()

    def end(self, name, count=None, unit="item"):
        """Close a stage; count = how many inferences / tokens it produced, for energy per `unit`."""
        t0, t1 = self._open.pop(name), time.perf_counter()
        self._raw = getattr(self, "_raw", {})
        self._raw[name] = (t0, t1, count, unit)
        win = [s for s in self.samples if t0 < s[0] <= t1]
        watts = [s[1] for s in win if s[1] is not None]
        temps = [s[3] for s in win if s[3] is not None]
        sec = t1 - t0
        seg = dict(seconds=round(sec, 2), samples=len(watts),
                   avg_cpu_pct=round(statistics.fmean(s[2] for s in win), 1) if win else None,
                   max_temp_c=max(temps) if temps else None, avg_w=None, energy_j=None)
        if watts:
            avg = statistics.fmean(watts)
            seg.update(avg_w=round(avg, 2), max_w=round(max(watts), 2), energy_j=round(avg * sec, 2))
            if self.idle_w is not None:
                seg["above_idle_w"] = round(avg - self.idle_w, 2)
                seg["above_idle_j"] = round((avg - self.idle_w) * sec, 2)
            if count:
                seg["count"] = count
                seg[f"mj_per_{unit}"] = round(1000 * avg * sec / count, 2)
                if self.idle_w is not None:
                    seg[f"above_idle_mj_per_{unit}"] = round(1000 * (avg - self.idle_w) * sec / count, 2)
        self.segments[name] = seg
        return seg

    def measure_idle(self, seconds):
        print(f"measuring idle power for {seconds:g} s ...", flush=True)
        self.begin("idle")
        time.sleep(seconds)
        self.idle_w = self.end("idle")["avg_w"]
        return self.idle_w

    def report(self):
        if self._capacity_mode():
            self._capacity_segments()
        rep = dict(source=self.source, note=self.note, interval_s=self.interval, idle_w=self.idle_w,
                   temp_start_c=getattr(self, "temp_start_c", None), temp_end_c=cpu_temp_c(), stages=self.segments)
        if shutil.which("vcgencmd"):  # Raspberry Pi: 0x0 = never throttled / under-voltage during the run
            rep["pi_throttled"] = run_cmd("vcgencmd", "get_throttled").strip().split("=")[-1] or None
        return rep


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    args = parse_args()
    out = dict(device=dict(cpu=cpu_name(),cores=psutil.cpu_count(logical=False),
                           threads=psutil.cpu_count(), ram_gb=round(psutil.virtual_memory().total / 2**30, 1),
                           os=platform.platform(), python=platform.python_version(), torch=torch.__version__))
    import executorch
    out["device"]["executorch"] = getattr(executorch, "__version__", "1.6.0")
    mem = dict(start=rss_mb())
    meter = PowerMeter(args.interval).start()
    print(f"power sensor: {meter.source or 'none'} - {meter.note}", flush=True)
    meter.measure_idle(args.idle)
    timing, tv = {}, {}
    meter.begin("whole_run")
    meter.begin("load_vision_models")

    kr = KrishiRakshak(use_llm=True, use_translator=True)
    timing["load_vision_models_s"] = meter.end("load_vision_models")["seconds"]
    mem["vision_models"] = rss_mb()
    img = Image.open(os.path.join(ROOT, "test images", IMAGES[0])).convert("RGB")

    # vision: ExecuTorch (.pte, XNNPACK) vs PyTorch eager (.pth, fp32)
    meter.begin("phase_vision")
    vis = dict(disease_pte_ms=timeit(lambda: kr.disease(img)), pest_pte_ms=timeit(lambda: kr.pest(img)))
    tv["disease_pte"] = sustained(lambda: kr.disease(img), meter, "disease_pte", args.seconds)
    tv["pest_pte"] = sustained(lambda: kr.pest(img), meter, "pest_pte", args.seconds)
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
    with torch.no_grad():
        tv["disease_eager"] = sustained(lambda: m(x), meter, "disease_eager", args.seconds)
        tv["pest_eager"] = sustained(lambda: p(xp), meter, "pest_eager", args.seconds)
    del m, p
    vis["severity_ms"] = timeit(lambda: severity.estimate(img, "fungal"), n=10, warm=2)
    out["vision"] = vis
    tv["severity"] = sustained(lambda: severity.estimate(img, "fungal"), meter, "severity", args.seconds, warm=2)
    tv["diagnose_both_models"] = sustained(lambda: kr.diagnose(img), meter, "diagnose_both_models", args.seconds)
    timing["vision_ms"] = tv
    meter.end("phase_vision")

    # advisory: Llama 3.2 1B-Instruct (ExecuTorch) on three photos
    meter.begin("llama_load")
    t = time.perf_counter(); kr.llm(); load_s = time.perf_counter() - t
    meter.end("llama_load")
    mem["plus_llama"] = rss_mb()
    runs = []
    e2e = []
    meter.begin("phase_advice")
    for name in IMAGES:
        meter.begin(f"advice_{name}")
        t_run = time.perf_counter()
        r = kr.run(os.path.join(ROOT, "test images", name), "Karnataka", "English")
        e2e.append(time.perf_counter() - t_run)
        meter.end(f"advice_{name}", count=(r["timing"].get("new tokens") or None), unit="token")
        runs.append(dict(image=name, diagnosis=r["diagnosis"]["label"], confidence=round(r["diagnosis"]["confidence"], 3),
                         severity=(r["severity"] or {}).get("level"), **r["timing"]))
        print(runs[-1], flush=True)
    meter.end("phase_advice", count=sum(x.get("new tokens") or 0 for x in runs) or None, unit="token")
    out["llama"] = dict(load_s=round(load_s, 1), runs=runs)
    english = r["english"]
    adv = []
    for x, s in zip(runs, e2e):
        pt, ps, nt, tps = x.get("prompt tokens"), x.get("prefill s"), x.get("new tokens"), x.get("tokens/s")
        adv.append(dict(image=x["image"], end_to_end_s=round(s, 2), vision_s=x["vision_s"], advice_s=x["advice_s"],
                        prompt_tokens=pt, prefill_s=ps, prefill_tok_per_s=round(pt / ps, 1) if pt and ps else None,
                        time_to_first_token_s=round(x["vision_s"] + ps, 2) if ps is not None else None,
                        new_tokens=nt, decode_tok_per_s=tps,
                        decode_s=round(nt / tps, 2) if nt and tps else None))
    mean = lambda k: round(statistics.fmean(a[k] for a in adv), 2) if all(a[k] is not None for a in adv) else None  # noqa: E731
    timing["llama_load_s"] = round(load_s, 2)
    timing["advice"] = dict(runs=adv, mean={k: mean(k) for k in ("end_to_end_s", "time_to_first_token_s",
                                                                 "prefill_tok_per_s", "decode_tok_per_s")})

    # translation (IndicTrans2 200M), beams as on the Pi
    meter.begin("translator_load")
    t = time.perf_counter(); tr = kr.translator(); tr_load = time.perf_counter() - t
    meter.end("translator_load")
    mem["plus_translator"] = rss_mb()
    trans = dict(load_s=round(tr_load, 1), beams=tr.num_beams, runs={})
    meter.begin("phase_translate")
    for lang in ["Hindi", "Kannada", "Tamil"]:
        meter.begin(f"translate_{lang}")
        t = time.perf_counter(); tr.translate_advice(english, lang, ["Early blight"]); trans["runs"][lang] = round(time.perf_counter() - t, 1)
        meter.end(f"translate_{lang}")
        print(lang, trans["runs"][lang], flush=True)
    meter.end("phase_translate")
    out["translation"] = trans
    mem["peak"] = round(getattr(PROC.memory_info(), "peak_wset", PROC.memory_info().rss) / 2**20)
    out["memory_mb"] = mem
    timing["translator_load_s"] = round(tr_load, 2)
    timing["translate_s"] = trans["runs"]
    meter.end("whole_run")
    meter.stop()
    out["timing"] = timing
    out["power"] = meter.report()

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
