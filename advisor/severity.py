"""Severity estimate from a leaf photo with OpenCV: diseased area as a share of the visible leaf.

    leaf   = healthy green tissue + lesions that lie on the leaf
    lesion = brown / yellow / black pixels, or pale grey-white spots, inside the leaf outline
    severity % = lesion / leaf

Bands (common field scale): none < 3% <= mild < 10% <= moderate < 25% <= severe.
Virus and pest classes are not scored by lesion area (their damage is not a spot pattern); the caller reports
that instead. This is a heuristic, not a trained model: it works best on one leaf filling most of the photo.
"""
import cv2
import numpy as np

BANDS = [(3, "none"), (10, "mild"), (25, "moderate"), (101, "severe")]
NOT_AREA_BASED = ("virus", "viral", "healthy", "mite", "pest")


def _masks(bgr):
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    green = (h >= 33) & (h <= 90) & (s >= 45) & (v >= 40)
    brown_yellow = (((h < 33) | (h > 160)) & (s >= 50) & (v >= 35))
    dark = (v < 55)
    return hsv, green, brown_yellow, dark


def leaf_region(green, brown_yellow):
    """Leaf outline: the largest blob of green + yellow/brown tissue that contains green, gaps closed, holes filled.
    Including yellow/brown keeps dead leaf margins inside the outline so they are counted as diseased."""
    plant = (green | brown_yellow).astype(np.uint8) * 255
    k = max(5, int(0.03 * max(plant.shape)) | 1)
    plant = cv2.morphologyEx(plant, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))
    n, lab, stats, _ = cv2.connectedComponentsWithStats(plant, 8)
    if n <= 1:
        return np.zeros(green.shape, bool)
    green_in = np.bincount(lab[green], minlength=n)
    green_in[0] = 0
    best = int(np.argmax(green_in))  # the blob holding the most green tissue
    region = (lab == best).astype(np.uint8) * 255
    cnts, _ = cv2.findContours(region, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    filled = np.zeros_like(region)
    cv2.drawContours(filled, cnts, -1, 255, cv2.FILLED)
    return filled > 0


def lesion_mask(hsv, green, brown_yellow, dark, leaf):
    """Lesions inside the leaf: off-green colour, very dark tissue, or pale / grey spots (much less saturated than
    the leaf's own healthy tissue)."""
    s, v = hsv[..., 1].astype(np.float32), hsv[..., 2].astype(np.float32)
    healthy = leaf & green
    if healthy.sum() < 50:
        return leaf & ~green
    s_ref, v_ref = np.median(s[healthy]), np.median(v[healthy])
    pale = (s < 0.55 * s_ref) & (v > 0.75 * v_ref)
    lesion = leaf & ((~green & (brown_yellow | dark)) | pale)
    return cv2.morphologyEx(lesion.astype(np.uint8), cv2.MORPH_OPEN, np.ones((3, 3), np.uint8)) > 0


def estimate(image, disease_type="fungal", max_side=512):
    """image: path, PIL image or BGR array. Returns dict(percent, level, leaf_fraction, method)."""
    if isinstance(image, str):
        bgr = cv2.imdecode(np.fromfile(image, np.uint8), cv2.IMREAD_COLOR)
    elif isinstance(image, np.ndarray):
        bgr = image
    else:
        bgr = cv2.cvtColor(np.asarray(image.convert("RGB")), cv2.COLOR_RGB2BGR)
    s = max_side / max(bgr.shape[:2])
    if s < 1:
        bgr = cv2.resize(bgr, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)

    if any(w in disease_type.lower() for w in NOT_AREA_BASED):
        return dict(percent=None, level="not area-based", leaf_fraction=None,
                    method="Virus / pest / healthy: judge severity by the share of plants affected")

    hsv, green, brown_yellow, dark = _masks(bgr)
    leaf = leaf_region(green, brown_yellow)
    leaf_frac = float(leaf.mean())
    if leaf_frac < 0.05:
        return dict(percent=None, level="unknown", leaf_fraction=leaf_frac,
                    method="Leaf not found: take a closer photo of one leaf")
    lesion = lesion_mask(hsv, green, brown_yellow, dark, leaf)
    pct = 100.0 * lesion.sum() / max(1, leaf.sum())
    level = next(name for lim, name in BANDS if pct < lim)
    return dict(percent=round(pct, 1), level=level, leaf_fraction=round(leaf_frac, 2), method="lesion area / leaf area")


def overlay(image_path, out_path):
    """Save a debug picture: leaf outline in blue, lesions in red."""
    bgr = cv2.imdecode(np.fromfile(image_path, np.uint8), cv2.IMREAD_COLOR)
    hsv, green, by, dark = _masks(bgr)
    leaf = leaf_region(green, by)
    lesion = lesion_mask(hsv, green, by, dark, leaf)
    vis = bgr.copy()
    vis[lesion] = (0.4 * vis[lesion] + 0.6 * np.array([0, 0, 255])).astype(np.uint8)
    cnts, _ = cv2.findContours(leaf.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(vis, cnts, -1, (255, 0, 0), 2)
    cv2.imencode(".jpg", vis)[1].tofile(out_path)


if __name__ == "__main__":
    import sys
    for p in sys.argv[1:]:
        print(p, estimate(p))
