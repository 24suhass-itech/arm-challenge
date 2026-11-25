# KrishiRakshak on Raspberry Pi 5

Offline crop doctor kiosk: Pi Camera + screen, everything on the Pi's Arm CPU, no internet needed after setup.

## Hardware
- Raspberry Pi 5, 8 GB (CPU only; no AI HAT needed)
- Raspberry Pi Camera Module (v2 / v3 / HQ) on the camera connector
- Screen (official Touch Display 2 or any HDMI monitor), keyboard/mouse only for setup
- 32 GB+ microSD with **64-bit Raspberry Pi OS (desktop)**, official 27 W power supply, case with fan

## Two pipelines (documented separately, as the challenge requires)

| | Vision pipeline | Advisory pipeline |
|---|---|---|
| Input | camera photo | diagnosis + severity + region + season + language |
| Models | `disease_model.pte` (MobileNetV3-Small, 39 classes), `pest_model.pte` (EfficientNet-B0, 102 pests) | `llama3_2_1b_instruct_8da4w.pte` (Llama 3.2 1B-Instruct, 4-bit), IndicTrans2 200M translator |
| Runtime | **ExecuTorch**, XNNPACK backend, Arm CPU | **ExecuTorch**, XNNPACK backend, Arm CPU (Llama); PyTorch CPU (translator) |
| Extra | OpenCV lesion-area severity | offline knowledge base (`advisor/knowledge/*.json`) |
| Output | disease / pest / "retake photo" + confidence | short advice in the farmer's language |

(With an AI HAT+, the vision models could run on the Hailo NPU through HailoRT; Llama would stay on the CPU with
ExecuTorch. This build is CPU-only.)

## Install (once)
1. **On the laptop:** `python deploy/raspberry_pi/make_bundle.py E:\` (E: = USB stick) - copies about 2 GB.
2. **On the Pi:** copy `KrishiRakshak_Pi` from the USB stick to the home folder, connect to the internet, then:
   ```bash
   cd ~/KrishiRakshak_Pi
   bash deploy/raspberry_pi/setup.sh
   sudo reboot
   ```
   Setup installs the camera library, Indic fonts, Chromium and the Python packages (ExecuTorch 1.6.0 has
   official aarch64 wheels), runs a self-test, and registers the auto-start.
3. After the reboot the kiosk opens full-screen by itself. The internet can now be disconnected.

## Use
Pick region and language at the top, hold one leaf (or one insect) close to the camera, tap **Take photo**.
The answer appears in about a minute on the Pi.

## Useful commands
```bash
systemctl status krishirakshak          # server running?
journalctl -u krishirakshak -f          # server log
~/KrishiRakshak_Pi/deploy/raspberry_pi/kiosk.sh   # open the kiosk manually
~/kr-venv/bin/python ~/KrishiRakshak_Pi/advisor/app.py "~/KrishiRakshak_Pi/test images/apple_scab_1.jpg" --language Kannada
```

## Troubleshooting
- *Camera not found*: check the ribbon cable, then `rpicam-hello`. Without a camera the page offers photo upload.
- *Boxes instead of Indian letters*: `sudo apt install fonts-noto-core` and restart Chromium.
- *NumPy errors from picamera2 on Bookworm*: `~/kr-venv/bin/pip install "numpy<2"`.
- *Slow translation*: lower `KR_TRANSLATE_BEAMS` in `/etc/systemd/system/krishirakshak.service` (1 = fastest).
