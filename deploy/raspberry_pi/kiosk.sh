#!/usr/bin/env bash
# Open the KrishiRakshak page full-screen on the Pi's display once the local server is ready.
URL="http://127.0.0.1:8000"
for i in $(seq 1 120); do
  curl -s -o /dev/null "$URL/api/options" && break
  sleep 2
done
unclutter -idle 3 >/dev/null 2>&1 &
BROWSER="$(command -v chromium || command -v chromium-browser)"
exec "$BROWSER" --kiosk --noerrdialogs --disable-infobars --incognito --check-for-update-interval=31536000 \
  --overscroll-history-navigation=0 --disable-pinch "$URL"
