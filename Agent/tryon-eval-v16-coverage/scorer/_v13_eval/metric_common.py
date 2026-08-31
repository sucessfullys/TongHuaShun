#!/usr/bin/env python3
"""Metric primitives for the try-on evaluator (DINOv2 region cosine, LPIPS,
HSV color-histogram delta, garment-band region masking).

torch.hub dinov2 + pyiqa LPIPS. Each metric is independently optional and the
evaluator degrades gracefully when torch / pyiqa are missing — the metrics are
ADVISORY (VLM-gated) in the shipped crossover-v3 recipe, so a metric-less run
is still valid (pure-VLM judgement).
"""
import threading

DEVICE = "cpu"
_torch = None
DINO = None
DINO_OK = False
DINO_ERR = None
LPIPS = None
LPIPS_OK = False
LPIPS_ERR = None

_INIT_LOCK = threading.Lock()
_INITED = False


def init_metrics():
    """Lazy, idempotent metric load. Safe to call from many threads."""
    global _torch, DEVICE, DINO, DINO_OK, DINO_ERR, LPIPS, LPIPS_OK, LPIPS_ERR, _INITED
    with _INIT_LOCK:
        if _INITED:
            return
        _INITED = True
        try:
            import torch
            _torch = torch
            DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
        except Exception as e:
            DINO_ERR = LPIPS_ERR = f"torch missing: {e}"
            return
        try:
            from torch.hub import load as _hub_load
            DINO = _hub_load('facebookresearch/dinov2', 'dinov2_vitl14', pretrained=True)
            DINO.eval().to(DEVICE)
            DINO_OK = True
        except Exception as e:
            DINO_ERR = str(e)
        try:
            import pyiqa
            LPIPS = pyiqa.create_metric("lpips", device=DEVICE)
            LPIPS_OK = True
        except Exception as e:
            LPIPS_ERR = str(e)


def status():
    return {"device": DEVICE, "dino_ok": DINO_OK, "lpips_ok": LPIPS_OK,
            "dino_err": DINO_ERR, "lpips_err": LPIPS_ERR}


# --- region geometry ------------------------------------------------------
def garment_box(w, h):
    """Central torso garment band (the legitimately-edited region heuristic)."""
    return (int(w * 0.25), int(h * 0.28), int(w * 0.75), int(h * 0.68))


def shoe_box(w, h):
    """Bottom band — footwear region."""
    return (int(w * 0.20), int(h * 0.82), int(w * 0.80), int(h * 1.00))


def background_band(pil_img):
    w, h = pil_img.size
    return pil_img.crop((0, 0, w, int(h * 0.20)))


def crop(pil_img, box):
    w, h = pil_img.size
    x0, y0, x1, y1 = box
    x0 = max(0, min(w, x0)); x1 = max(0, min(w, x1))
    y0 = max(0, min(h, y0)); y1 = max(0, min(h, y1))
    if x1 <= x0 or y1 <= y0:
        return pil_img.copy()
    return pil_img.crop((x0, y0, x1, y1))


def paint_out(pil_img, box):
    """Grey out a box so the embedding/LPIPS focuses on the COMPLEMENT region."""
    from PIL import Image
    img = pil_img.copy()
    x0, y0, x1, y1 = box
    patch = Image.new("RGB", (max(1, x1 - x0), max(1, y1 - y0)), (128, 128, 128))
    img.paste(patch, (x0, max(0, y0)))
    return img


# --- metrics ------------------------------------------------------------
def dino_cosine(img_a, img_b):
    if not DINO_OK:
        return None
    import torchvision.transforms as T
    tx = T.Compose([
        T.Resize((224, 224)), T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    a = tx(img_a.convert("RGB")).unsqueeze(0).to(DEVICE)
    b = tx(img_b.convert("RGB")).unsqueeze(0).to(DEVICE)
    with _torch.no_grad():
        fa = DINO(a)
        fb = DINO(b)
    cos = _torch.nn.functional.cosine_similarity(fa, fb).item()
    return float(max(0.0, min(1.0, (cos + 1.0) / 2.0)))


def lpips_dist(img_a, img_b):
    if not LPIPS_OK:
        return None
    import torchvision.transforms as T
    tx = T.Compose([T.Resize((256, 256)), T.ToTensor()])
    a = tx(img_a.convert("RGB")).unsqueeze(0).to(DEVICE)
    b = tx(img_b.convert("RGB")).unsqueeze(0).to(DEVICE)
    with _torch.no_grad():
        v = LPIPS(a, b)
    return float(v.item() if hasattr(v, "item") else v)


def color_hist_delta(img_a, img_b, bins=32):
    """HSV histogram correlation distance in [0,1] (0 = identical color dist).
    CPU-only (numpy/PIL), no torch needed."""
    import numpy as np
    ha = _hsv_hist(img_a, bins)
    hb = _hsv_hist(img_b, bins)
    # normalized intersection -> distance
    inter = np.minimum(ha, hb).sum()
    denom = max(ha.sum(), hb.sum(), 1e-8)
    sim = float(inter / denom)
    return 1.0 - sim


def _hsv_hist(pil_img, bins=32):
    import numpy as np
    arr = np.asarray(pil_img.convert("HSV"), dtype=np.float32)
    h = arr[..., 0].ravel()
    s = arr[..., 1].ravel()
    # weight hue by saturation so flat/grey areas contribute less
    hist, _ = np.histogram(h, bins=bins, range=(0, 256), weights=(s / 255.0 + 1e-3))
    hist = hist / max(hist.sum(), 1e-8)
    return hist
