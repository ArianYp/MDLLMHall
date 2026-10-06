"""Slide figure for the attention-sink problem: attention map, crop with the sink, crop with the sink removed, and both refills.

Uses the paired first trigger of a caption in the word-first runs with and without sink removal (oracle + zoom, A100). Until that trigger
both runs are in the baseline state, so the word's attention map is the one in the baseline trace (hallu_study/traces*/<id>.npz, late_map).
Sink = image patch that is the top late-layer patch for >= half of the caption's 128 answer positions (as in slot_refill_study.py).

Output: presentation/sink_<id>.png. CPU only. Example (from MDLLM/): python presentation/make_sink_example.py 342 248
"""

import json
import os
import sys

import numpy as np
from PIL import Image, ImageDraw, ImageFont

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))
sys.path.insert(0, ROOT)

from trace_amber_steps import IMAGE_DIR, QUERY_FILE, read_json

STUDY = os.path.join(ROOT, "results", "amber_g", "hallu_study")
SR = os.path.join(STUDY, "rag", "slot_refill")
KEPT, REMOVED = os.path.join(SR, "oracle_zoom_wordfirst"), os.path.join(SR, "oracle_zoom_wordfirst_nosink")
GREY, BLACK = (90, 90, 90), (20, 20, 20)
S = 360  # panel size


def font(size, bold=False):
    name = "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf"
    for d in ("/usr/share/fonts/dejavu-sans-fonts", "/usr/share/fonts/dejavu", "/usr/share/fonts/truetype/dejavu"):
        if os.path.exists(os.path.join(d, name)):
            return ImageFont.truetype(os.path.join(d, name), size)
    return ImageFont.load_default()


def patch_box(photo, index, grid=32, resolution=512):
    """Box of image patch `index` in original-photo pixels (the model sees the photo resized to short side 512, centre-cropped)."""
    r, c = divmod(index, grid)
    w, h = photo.size
    scale = resolution / min(w, h)
    left, top = (w * scale - resolution) / 2, (h * scale - resolution) / 2
    p = resolution / grid
    return [(c * p + left) / scale, (r * p + top) / scale, ((c + 1) * p + left) / scale, ((r + 1) * p + top) / scale]


def heat_overlay(photo, amap, grid=32, resolution=512):
    """Photo with the attention map (model view) as a red overlay."""
    w, h = photo.size
    scale = resolution / min(w, h)
    left, top = (w * scale - resolution) / 2, (h * scale - resolution) / 2
    m = (amap / amap.max()).reshape(grid, grid)
    m = np.sqrt(m)  # show weaker patches too
    heat = Image.fromarray((m * 255).astype(np.uint8)).resize((resolution, resolution), Image.NEAREST)
    canvas = Image.new("L", (round(w * scale), round(h * scale)), 0)
    canvas.paste(heat, (round(left), round(top)))
    canvas = canvas.resize((w, h))
    red = Image.new("RGB", (w, h), (255, 0, 0))
    dim = Image.blend(photo, Image.new("RGB", (w, h), (0, 0, 0)), 0.45)
    return Image.composite(red, dim, canvas.point(lambda v: int(v * 0.85)))


def fit(img, size=S):
    s = size / max(img.size)
    return img.resize((max(1, round(img.width * s)), max(1, round(img.height * s))))


def main():
    queries = {int(r["id"]): r for r in read_json(QUERY_FILE)}
    kept = {c["id"]: c for c in json.load(open(os.path.join(KEPT, "predictions_events.json")))["captions"]}
    removed = {c["id"]: c for c in json.load(open(os.path.join(REMOVED, "predictions_events.json")))["captions"]}
    for item_id in [int(a) for a in sys.argv[1:]]:
        a, b = kept[item_id]["events"][0], removed[item_id]["events"][0]
        assert (a["pos"], a["old_text"], a["step"]) == (b["pos"], b["old_text"], b["step"]), "first triggers differ"
        trace = next(os.path.join(STUDY, d, f"{item_id}.npz") for d in ("traces", "traces_clean")
                     if os.path.exists(os.path.join(STUDY, d, f"{item_id}.npz")))
        late = np.load(trace)["late_map"].astype(np.float32)  # answer positions x image patches
        counts = np.bincount(late.argmax(1), minlength=late.shape[1])
        sinks = np.flatnonzero(counts >= 0.5 * late.shape[0])
        amap = late[a["pos"]]
        word = a["old_text"][a["remasked"].index(a["pos"])].strip()
        photo = Image.open(os.path.join(IMAGE_DIR, queries[item_id]["image"])).convert("RGB")

        heat = heat_overlay(photo, amap)
        dh = ImageDraw.Draw(heat)
        lw = max(3, photo.width // 150)
        for sk in sinks:
            x0, y0, x1, y1 = patch_box(photo, int(sk))
            pad = (x1 - x0) * 1.2
            dh.ellipse([x0 - pad, y0 - pad, x1 + pad, y1 + pad], outline=(0, 230, 255), width=lw)
        panels = [fit(heat)]
        for e in (a, b):
            panels.append(fit(photo.crop(tuple(e["box"])).resize((S, S))))
        # full photo with both boxes, small
        both = photo.copy()
        db = ImageDraw.Draw(both)
        db.rectangle(a["box"], outline=(255, 140, 0), width=lw)
        db.rectangle(b["box"], outline=(0, 200, 60), width=lw)
        panels.insert(1, fit(both))

        slot = [max(p.width, S) for p in panels]  # every panel gets at least S px, so titles never overlap
        width = max(20 + sum(w + 30 for w in slot), 1420)
        sheet = Image.new("RGB", (width, S + 300), "white")
        d = ImageDraw.Draw(sheet)
        x = 20
        sink_share = [float(amap[sk] / amap.sum()) for sk in sinks]
        titles = [
            f"where \"{word}\" looks (red)",
            "where the two crops are",
            "crop WITH the sink",
            "crop with the sink REMOVED",
        ]
        for p, t, w in zip(panels, titles, slot):
            sheet.paste(p, (x, 50))
            d.text((x, 18), t, fill=BLACK, font=font(16, True))
            x += w + 30
        tops = lambda e: ", ".join(f"{t.strip() or repr(t)} {pr:.2f}" for t, pr in e["word_first"]["top5"][:3])
        d.text((20, 50 + panels[0].height + 10), "light-blue circle = attention sink", fill=(0, 150, 190), font=font(15, True))
        d.text((20, 50 + panels[0].height + 32), "(strongest patch for almost every word)", fill=(0, 150, 190), font=font(15))
        x_boxes = 20 + slot[0] + 30
        x_crop1 = x_boxes + slot[1] + 30
        x_crop2 = x_crop1 + slot[2] + 30
        d.text((x_boxes, 50 + panels[1].height + 10), "orange = with sink", fill=(230, 120, 0), font=font(15, True))
        d.text((x_boxes, 50 + panels[1].height + 32), "green = sink removed", fill=(0, 160, 50), font=font(15, True))
        for xx, e in ((x_crop1, a), (x_crop2, b)):
            d.text((xx, S + 60), f"top words: {tops(e)}", fill=GREY, font=font(15))
            d.text((xx, S + 85), f"\"{''.join(e['old_text']).strip()}\" → \"{''.join(e['new_text']).strip()}\"", fill=BLACK, font=font(17, True))
        y = S + 130
        lines = [
            f"Sink: an image patch that is the most-attended patch for {int(counts[sinks].max()) if len(sinks) else 0} of the caption's 128 tokens "
            f"(function words included) and holds {100 * max(sink_share, default=0):.0f}% of \"{word}\"'s image attention.",
            "The zoom crop is built around the single strongest patch, so a sink pulls the crop off the object.",
            "Removing sink patches (top patch for at least half of the tokens) before cropping moves the crop back onto the object.",
        ]
        for line in lines:
            d.text((20, y), line, fill=BLACK, font=font(17))
            y += 30
        path = os.path.join(HERE, f"sink_{item_id}.png")
        sheet.crop((0, 0, width, y + 20)).save(path)
        print("saved", path, "| sinks", sinks.tolist(), "counts", counts[sinks].tolist(), "| share", [round(s, 3) for s in sink_share],
              "| with sink:", "".join(a["new_text"]), "| removed:", "".join(b["new_text"]))


if __name__ == "__main__":
    main()
