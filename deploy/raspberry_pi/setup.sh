#!/usr/bin/env bash
# KrishiRakshak - one-time setup on a Raspberry Pi 5 (8 GB, 64-bit Raspberry Pi OS Bookworm or Trixie, desktop).
# Run from the copied bundle folder:   bash deploy/raspberry_pi/setup.sh
# Needs internet ONCE (apt + pip). After that the device works fully offline.
set -euo pipefail

APP_DIR="$(cd "$(dirname "$0")/../.." && pwd)"
VENV="$HOME/kr-venv"
echo "== KrishiRakshak setup in $APP_DIR"

if [ "$(uname -m)" != "aarch64" ]; then
  echo "This script is for the 64-bit Raspberry Pi OS (aarch64)."; exit 1
fi

# 1. system packages: camera library, Indic fonts (so Kannada/Tamil/Hindi... render), Chromium for the kiosk
sudo apt-get update
sudo apt-get install -y python3-venv python3-pip python3-picamera2 \
  fonts-noto-core fonts-noto-ui-core unclutter
sudo apt-get install -y chromium || sudo apt-get install -y chromium-browser

# 2. Python environment (system site-packages so the apt-installed picamera2 is visible)
python3 -m venv --system-site-packages "$VENV"
"$VENV/bin/pip" install --upgrade pip
CODENAME="$(. /etc/os-release && echo "${VERSION_CODENAME:-}")"
NUMPY_PIN="numpy"
if [ "$CODENAME" = "bookworm" ]; then
  NUMPY_PIN="numpy<2"   # Bookworm's picamera2/simplejpeg are built against NumPy 1.x
fi
"$VENV/bin/pip" install "$NUMPY_PIN" "executorch==1.6.0" opencv-python-headless pillow pandas \
  "transformers>=4.46,<4.52" sentencepiece sacremoses indic-nlp-library

# 3. check that the model files from the bundle are in place
for f in advisor/models/disease_model.pte advisor/models/pest_model.pte \
         advisor/llama/llama3_2_1b_instruct_8da4w.pte advisor/llama/original/tokenizer.model \
         advisor/translator/indictrans2-en-indic-dist-200M/model.safetensors; do
  if [ ! -f "$APP_DIR/$f" ]; then echo "MISSING: $f (copy the full bundle from the laptop)"; exit 1; fi
done

# 4. quick self-test (no camera needed)
"$VENV/bin/python" "$APP_DIR/advisor/app.py" "$APP_DIR/test images/apple_scab_1.jpg" --language English

# 5. start the server on boot (systemd) and open the kiosk when the desktop starts
sed -e "s#__APP_DIR__#$APP_DIR#g" -e "s#__VENV__#$VENV#g" -e "s#__USER__#$USER#g" \
  "$APP_DIR/deploy/raspberry_pi/krishirakshak.service" | sudo tee /etc/systemd/system/krishirakshak.service >/dev/null
sudo systemctl daemon-reload
sudo systemctl enable --now krishirakshak.service

mkdir -p "$HOME/.config/autostart"
sed -e "s#__APP_DIR__#$APP_DIR#g" "$APP_DIR/deploy/raspberry_pi/krishirakshak-kiosk.desktop" \
  > "$HOME/.config/autostart/krishirakshak-kiosk.desktop"
chmod +x "$APP_DIR/deploy/raspberry_pi/kiosk.sh"

echo "== Done. The kiosk opens automatically after a reboot (sudo reboot),"
echo "   or start it now with: $APP_DIR/deploy/raspberry_pi/kiosk.sh"
