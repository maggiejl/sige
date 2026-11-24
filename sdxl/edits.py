"""Synthetic SDEdit-style edits with a controlled edit size, for the edit-size ablation.

An edit is a filled, slightly irregular stroke blob (like a user painting over a region), placed at a
seeded random location and scaled so that it covers ``ratio`` of the image area.
"""

import math
import random

import numpy as np
from PIL import Image, ImageDraw, ImageFilter


def make_stroke_edit(img: Image.Image, ratio: float, seed: int = 0) -> Image.Image:
    rng = random.Random(seed)
    W, H = img.size
    area = ratio * W * H
    aspect = rng.uniform(0.6, 1.6)
    # ellipse area = pi * a * b, with b = a * aspect
    a = math.sqrt(area / (math.pi * aspect))
    b = a * aspect
    a, b = min(a, W / 2 - 1), min(b, H / 2 - 1)
    cx = rng.uniform(a, W - a)
    cy = rng.uniform(b, H - b)

    # irregular blob: polygon around the ellipse with a jittered radius
    pts = []
    n = 48
    for k in range(n):
        th = 2 * math.pi * k / n
        r = 1 + 0.12 * math.sin(3 * th + rng.uniform(0, 6.28)) + rng.uniform(-0.04, 0.04)
        pts.append((cx + a * r * math.cos(th), cy + b * r * math.sin(th)))

    # stroke colour: a colour that is clearly different from the region being covered
    region = np.asarray(img.convert("RGB"))[int(cy - b) : int(cy + b), int(cx - a) : int(cx + a)]
    mean = region.reshape(-1, 3).mean(0) if region.size else np.array([128, 128, 128])
    colour = tuple(int(v) for v in (255 - mean + np.array([rng.randint(-40, 40) for _ in range(3)])).clip(0, 255))

    mask = Image.new("L", (W, H), 0)
    ImageDraw.Draw(mask).polygon(pts, fill=255)
    mask = mask.filter(ImageFilter.GaussianBlur(1.5))
    layer = Image.new("RGB", (W, H), colour)
    return Image.composite(layer, img.convert("RGB"), mask)
