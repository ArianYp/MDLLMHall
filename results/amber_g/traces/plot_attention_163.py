"""Small multiples: where selected caption words look in the image (layers 19-31, head mean).

Drawn with PIL only (matplotlib is not installed in the project virtualenv).
"""

import json

import numpy as np
from PIL import Image, ImageDraw, ImageFont

BASE = "results/amber_g/traces/"
d = np.load(BASE + "image_attention_amber_163.npz")
meta = json.load(open(BASE + "image_attention_amber_163.json"))
abl = json.load(open(BASE + "image_ablation_amber_163.json"))
dl = {x["pos"]: x["image"]["logprob"] - x["no_image"]["logprob"] for s in abl["steps"] for x in s["commits"]}
index = {t["pos"]: i for i, t in enumerate(meta["tokens"])}
tokens = {t["pos"]: t["token"].strip() for t in meta["tokens"]}

WORDS = [(4, False), (7, False), (23, False), (57, False),
         (19, True), (20, True), (28, True), (39, False),
         (42, True), (44, True), (49, False), (50, True)]
LAYERS = slice(19, 32)
PANEL = 320
COLS, ROWS = 4, 3
GAP, HEAD, MARGIN, TITLE, LEGEND = 16, 52, 24, 78, 70
INK, MUTED = (31, 31, 31), (92, 92, 92)
LOW, MID, HIGH = np.array([251, 227, 216]), np.array([235, 104, 52]), np.array([138, 52, 19])


def font(size):
    try:
        return ImageFont.load_default(size=size)
    except TypeError:
        return ImageFont.load_default()


def ramp(t):
    """One-hue sequential ramp: light orange -> #eb6834 -> dark orange."""
    t = np.clip(t, 0, 1)[..., None]
    lower = LOW + (MID - LOW) * np.clip(t * 2, 0, 1)
    return np.where(t < 0.5, lower, MID + (HIGH - MID) * np.clip(t * 2 - 1, 0, 1))


# What the model sees: shortest side resized to 512, then centre crop.
photo = Image.open("data/amber/images/AMBER_163.jpg").convert("L")
w, h = photo.size
s = 512 / min(w, h)
photo = photo.resize((round(w * s), round(h * s)))
left, top = (photo.size[0] - 512) // 2, (photo.size[1] - 512) // 2
grey = np.asarray(photo.crop((left, top, left + 512, top + 512)).resize((PANEL, PANEL)), dtype=float) / 255
grey = (0.35 + 0.5 * grey)[..., None] * 255  # lightened so the overlay carries the contrast

maps = {}
for pos, _ in WORDS:
    m = d["image_map"][index[pos], LAYERS].astype(float).mean(0)
    maps[pos] = (m / m.sum()).reshape(32, 32)
vmax = max(m.max() for m in maps.values())

width = MARGIN * 2 + COLS * PANEL + (COLS - 1) * GAP
height = TITLE + ROWS * (HEAD + PANEL) + (ROWS - 1) * GAP + LEGEND
canvas = Image.new("RGB", (width, height), "white")
draw = ImageDraw.Draw(canvas)
draw.text((MARGIN, 14), "AMBER 163: where each word attends in the image when it is committed",
          fill=INK, font=font(22))
draw.text((MARGIN, 46), "MMaDA-8B, layers 19-31, mean over 32 heads, renormalised over the 32x32 image patches (model view: 512 centre crop). "
          "Shared colour scale across panels.", fill=MUTED, font=font(14))

for k, (pos, hallucinated) in enumerate(WORDS):
    r, c = divmod(k, COLS)
    x0 = MARGIN + c * (PANEL + GAP)
    y0 = TITLE + r * (HEAD + PANEL + GAP)
    m = maps[pos]
    t = np.kron(m / vmax, np.ones((PANEL // 32, PANEL // 32)))  # nearest: one block per patch
    alpha = (np.clip(t, 0, 1) ** 0.6 * 0.9)[..., None]
    rgb = grey * (1 - alpha) + ramp(t) * alpha
    canvas.paste(Image.fromarray(rgb.astype(np.uint8)), (x0, y0 + HEAD))
    pr, pc = divmod(int(m.argmax()), 32)
    cell = PANEL // 32
    bx, by = x0 + pc * cell, y0 + HEAD + pr * cell
    draw.rectangle((bx - 3, by - 3, bx + cell + 2, by + cell + 2), outline=INK, width=2)
    label = f"“{tokens[pos]}”" + ("   hallucinated" if hallucinated else "")
    draw.text((x0, y0 + 4), label, fill=INK, font=font(18))
    top5 = np.sort(m.ravel())[-5:].sum()
    draw.text((x0, y0 + 28), f"top-5 patches {top5:.0%} · image dlogp {dl[pos]:+.1f}", fill=MUTED, font=font(13))

# Colour legend.
ly = height - LEGEND + 22
lw = 360
bar = ramp(np.linspace(0, 1, lw))[None, :, :].repeat(14, axis=0)
canvas.paste(Image.fromarray(bar.astype(np.uint8)), (MARGIN, ly))
draw.text((MARGIN, ly + 18), "0", fill=MUTED, font=font(13))
draw.text((MARGIN + lw - 40, ly + 18), f"{vmax:.1%}", fill=MUTED, font=font(13))
draw.text((MARGIN + lw + 16, ly - 1), "share of image attention per patch.  Black box = most-attended patch.", fill=MUTED, font=font(14))

out = BASE + "image_attention_amber_163.png"
canvas.save(out)
print(out)
