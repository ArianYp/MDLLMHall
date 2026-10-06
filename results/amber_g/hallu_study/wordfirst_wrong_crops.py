"""Draw the zoom crops of selected wrong word-first refills: full photo with the crop box (red) next to the crop itself.

One figure per kind of failure (see NOTES.md, word-first): misreading of a real object, scene words, absence phrases,
labelling artefacts. Output: hallu_study/figures/wordfirst_wrong_<kind>.png

CPU only. Example (from MDLLM/): python results/amber_g/hallu_study/wordfirst_wrong_crops.py
"""

import json
import os
import sys

from PIL import Image, ImageDraw, ImageFont

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
sys.path.insert(0, ROOT)

from amber_noun_labels import AmberLabeler
from trace_amber_steps import IMAGE_DIR, QUERY_FILE, read_json

EVENTS = os.path.join(HERE, "rag", "slot_refill", "oracle_zoom_wordfirst", "predictions_events.json")
FIGURES = os.path.join(HERE, "figures")

# (caption id, flagged span text as logged); the first matching event is drawn.
KINDS = {
    "misread": [(517, " a cow"), (428, " nine cows in"), (407, " the mouse"), (587, " mouse"),
                (354, " a chair"), (541, " a table"), (690, " a bird's"), (312, " a desk,")],
    "scene": [(2, " the sun shining"), (304, " the sun shining"), (720, " the sun"), (634, " dirt road"),
              (130, " a beach or"), (526, " the ground and")],
    "absence": [(332, " no people"), (720, " no people"), (144, " other people or"), (28, " or vehicles visible"),
                (430, " for people to"), (670, " or people visible")],
    "labels": [(289, " two street lamps"), (252, " a backpack"), (84, " the soap"), (32, " dog collar,"),
               (67, " his phone."), (371, " A cup")],
}


H = 300  # tile height in pixels


def font(size):
    for path in ("/usr/share/fonts/dejavu-sans-fonts/DejaVuSans.ttf", "/usr/share/fonts/dejavu/DejaVuSans.ttf",
                 "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"):
        if os.path.exists(path):
            return ImageFont.truetype(path, size)
    try:
        return ImageFont.load_default(size=size)
    except TypeError:
        return ImageFont.load_default()


def tile(photo, e, item_id, span, truth):
    """Full photo with the red crop box, the crop, and three lines of text under them."""
    x0, y0, x1, y1 = e["box"]
    s = H / photo.height
    full = photo.resize((max(1, round(photo.width * s)), H))
    ImageDraw.Draw(full).rectangle([x0 * s, y0 * s, x1 * s, y1 * s], outline=(255, 0, 0), width=4)
    crop = photo.crop((x0, y0, x1, y1)).resize((H, H))
    fp = e["first_pass"]
    lines = [
        f"[{item_id}] {span.strip()!r} -> {''.join(e['new_text']).strip()!r}",
        "crop: " + ", ".join(f"{t.strip() or repr(t)} {p:.2f}" for t, p in e["word_first"]["top5"][:4]),
        f"mass correct {fp['p_correct']:.2f}  wrong {fp['p_wrong']:.2f}  neutral {fp['p_neutral']:.2f}",
        "truth: " + ", ".join(dict.fromkeys(truth)),
    ]
    width = max(full.width + 8 + H, 720)
    out = Image.new("RGB", (width, H + 4 + 22 * len(lines)), "white")
    out.paste(full, (0, 0))
    out.paste(crop, (full.width + 8, 0))
    d = ImageDraw.Draw(out)
    for k, line in enumerate(lines):
        d.text((4, H + 4 + 22 * k), line, fill="black", font=font(15 if k == 0 else 13))
    return out


def main():
    labeler = AmberLabeler()
    queries = {int(row["id"]): row for row in read_json(QUERY_FILE)}
    events = {c["id"]: c["events"] for c in json.load(open(EVENTS))["captions"]}
    os.makedirs(FIGURES, exist_ok=True)

    for kind, picks in KINDS.items():
        tiles = []
        for item_id, span in picks:
            e = next(e for e in events[item_id] if "".join(e["old_text"]) == span)
            photo = Image.open(os.path.join(IMAGE_DIR, queries[item_id]["image"])).convert("RGB")
            tiles.append(tile(photo, e, item_id, span, labeler.word_lists(item_id)[0]["truth"]))
        cols, gap = 2, 16
        w, h = max(t.width for t in tiles), max(t.height for t in tiles)
        rows = (len(tiles) + cols - 1) // cols
        sheet = Image.new("RGB", (cols * w + (cols - 1) * gap, rows * h + (rows - 1) * gap), "white")
        for k, t in enumerate(tiles):
            sheet.paste(t, ((k % cols) * (w + gap), (k // cols) * (h + gap)))
        path = os.path.join(FIGURES, f"wordfirst_wrong_{kind}.png")
        sheet.save(path)
        print("saved", path)


if __name__ == "__main__":
    main()
