"""Slide figure for the problem itself: the photo and MMaDA's baseline caption, with AMBER's labels.

Hallucinated object nouns in red, grounded ones in green (AMBER labeler on the saved baseline caption), plus the AMBER truth list.
Output: presentation/hallucination_<id>.png. CPU only. Example (from MDLLM/): python presentation/make_hallucination_example.py 163 537
"""

import os
import sys

from PIL import Image, ImageDraw

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from make_examples import BLACK, GREY, RED, Examples, font
from trace_amber_steps import IMAGE_DIR

GREEN = (20, 140, 40)
PH = 520  # photo height
TW = 760  # caption column width


def main():
    ex = Examples()
    for item_id in [int(a) for a in sys.argv[1:]]:
        text = " ".join(ex.base[item_id].split())
        rows = ex.labeler.label(item_id, text)
        marks = [(*r["span"], RED if r["label"] == "hallucinated" else GREEN) for r in rows
                 if r["span"] and r["label"] in ("hallucinated", "grounded")]
        n_h = sum(r["label"] == "hallucinated" for r in rows)
        n_g = sum(r["label"] == "grounded" for r in rows)

        photo = Image.open(os.path.join(IMAGE_DIR, ex.queries[item_id]["image"])).convert("RGB")
        photo = photo.resize((round(photo.width * PH / photo.height), PH))
        width = 20 + photo.width + 40 + TW + 20
        sheet = Image.new("RGB", (width, PH + 200), "white")
        d = ImageDraw.Draw(sheet)
        sheet.paste(photo, (20, 20))
        x0 = 20 + photo.width + 40
        d.text((x0, 20), "MMaDA's caption (\"Describe this image\"):", fill=BLACK, font=font(22, True))

        pieces, cur = [], 0
        for a, b, col in sorted(marks):
            pieces += [(w, BLACK, False) for w in text[cur:a].split()] + [(text[a:b], col, True)]
            cur = b
        pieces += [(w, BLACK, False) for w in text[cur:].split()]
        x, y = x0, 62
        for w, col, bold in pieces:
            f = font(21, bold)
            if not w[0].isalnum() and len(w) <= 2 and x > x0:
                x -= d.textlength(" ", font=f)
            wl = d.textlength(w + " ", font=f)
            if x + wl > x0 + TW:
                x, y = x0, y + 31
            d.text((x, y), w, fill=col, font=f)
            x += wl
        y += 50
        d.text((x0, y), f"red = not in the image ({n_h})", fill=RED, font=font(19, True))
        d.text((x0 + 300, y), f"green = in the image ({n_g})", fill=GREEN, font=font(19, True))
        truth = ", ".join(dict.fromkeys(ex.labeler.word_lists(item_id)[0]["truth"]))
        y += 34
        words = f"AMBER ground truth: {truth}".split()
        line = ""
        for w in words:
            if d.textlength(line + w + " ", font=font(17)) > TW:
                d.text((x0, y), line, fill=GREY, font=font(17))
                y, line = y + 26, ""
            line += w + " "
        d.text((x0, y), line, fill=GREY, font=font(17))
        height = max(PH + 40, y + 50)
        path = os.path.join(HERE, f"hallucination_{item_id}.png")
        sheet.crop((0, 0, width, height)).save(path)
        print("saved", path, "| hallucinated", [r["word"] for r in rows if r["label"] == "hallucinated"])


if __name__ == "__main__":
    main()
