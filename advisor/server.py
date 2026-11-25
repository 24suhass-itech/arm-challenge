"""KrishiRakshak kiosk server for the Raspberry Pi 5 (camera + screen), fully offline.

    python advisor/server.py            # then open http://localhost:8000 (Chromium kiosk on the Pi)

- GET  /                 the touch-friendly page (advisor/web/index.html)
- GET  /api/options      regions, languages, camera availability
- GET  /stream.mjpg      live camera preview (Raspberry Pi Camera via picamera2)
- POST /api/capture      take a photo with the Pi camera and analyse it
- POST /api/analyze      analyse an uploaded photo (body = image bytes); used without a camera / for testing
All models are loaded once at start-up (about 3 GB RAM on the 8 GB Pi 5).
"""
import argparse
import io
import json
import os
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import advisor  # noqa: E402
from pipeline import KrishiRakshak  # noqa: E402
from translate import LANG_CODES  # noqa: E402

LANGUAGES = ["English", "Hindi", "Kannada", "Tamil", "Telugu", "Marathi", "Bengali", "Gujarati", "Malayalam",
             "Odia", "Punjabi", "Assamese"]


class Camera:
    """Pi Camera (picamera2). Keeps the latest frame for the preview stream and for capture."""

    def __init__(self):
        from picamera2 import Picamera2
        self.cam = Picamera2()
        self.cam.configure(self.cam.create_video_configuration(main={"size": (1280, 960), "format": "RGB888"}))
        self.cam.start()
        time.sleep(1.0)  # let auto-exposure settle
        self.lock = threading.Lock()

    def frame(self):
        with self.lock:
            arr = self.cam.capture_array("main")
        return Image.fromarray(arr[..., ::-1])  # picamera2 RGB888 is BGR in memory

    def jpeg(self, size=(640, 480), quality=70):
        buf = io.BytesIO()
        self.frame().resize(size).save(buf, "JPEG", quality=quality)
        return buf.getvalue()


def make_handler(kr, camera):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, fmt, *args):
            pass

        def _send(self, code, body, ctype="application/json; charset=utf-8"):
            if isinstance(body, (dict, list)):
                body = json.dumps(body, ensure_ascii=False).encode("utf-8")
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            path = urlparse(self.path).path
            if path in ("/", "/index.html"):
                with open(os.path.join(HERE, "web", "index.html"), "rb") as f:
                    return self._send(200, f.read(), "text/html; charset=utf-8")
            if path == "/api/options":
                return self._send(200, dict(regions=list(advisor.REGIONS["regions"]),
                                            region_language={k: v["language"] for k, v in advisor.REGIONS["regions"].items()},
                                            languages=[l for l in LANGUAGES if l == "English" or l in LANG_CODES],
                                            camera=camera is not None, llm=kr.use_llm))
            if path == "/stream.mjpg" and camera is not None:
                self.send_response(200)
                self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
                self.end_headers()
                try:
                    while True:
                        jpg = camera.jpeg()
                        self.wfile.write(b"--frame\r\nContent-Type: image/jpeg\r\nContent-Length: " +
                                         str(len(jpg)).encode() + b"\r\n\r\n" + jpg + b"\r\n")
                        time.sleep(0.1)
                except (BrokenPipeError, ConnectionResetError):
                    return
            return self._send(404, {"error": "not found"})

        def do_POST(self):
            u = urlparse(self.path)
            q = {k: v[0] for k, v in parse_qs(u.query).items()}
            try:
                if u.path == "/api/capture":
                    if camera is None:
                        return self._send(400, {"error": "no camera"})
                    img = camera.frame()
                elif u.path == "/api/analyze":
                    n = int(self.headers.get("Content-Length", 0))
                    img = Image.open(io.BytesIO(self.rfile.read(n)))
                else:
                    return self._send(404, {"error": "not found"})
                r = kr.run(img, q.get("region", "India (general)"), q.get("language") or None,
                           int(q["month"]) if q.get("month") else None, q.get("mode", "auto"))
                thumb = io.BytesIO()
                img.convert("RGB").resize((320, int(320 * img.height / img.width))).save(thumb, "JPEG", quality=75)
                import base64
                r["photo"] = "data:image/jpeg;base64," + base64.b64encode(thumb.getvalue()).decode()
                return self._send(200, r)
            except Exception as e:  # keep the kiosk alive; show the error on screen
                return self._send(500, {"error": f"{type(e).__name__}: {e}"})

    return Handler


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1", help="0.0.0.0 to also allow phones on the Pi's Wi-Fi")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--no_camera", action="store_true")
    a = ap.parse_args()
    camera = None
    if not a.no_camera:
        try:
            camera = Camera()
        except Exception as e:
            print(f"camera not available ({type(e).__name__}); upload mode only", flush=True)
    t0 = time.time()
    kr = KrishiRakshak(preload=True)
    print(f"models loaded in {time.time() - t0:.0f}s; open http://{a.host}:{a.port}", flush=True)
    ThreadingHTTPServer((a.host, a.port), make_handler(kr, camera)).serve_forever()


if __name__ == "__main__":
    main()
