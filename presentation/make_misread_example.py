"""Slide figure for a failure: a hallucination that the zoom does NOT fix (the model misreads the object even in the crop).

Shows the photo with the crop box, the crop with its top words, the baseline sentence (hallucinated noun in red) and the refilled sentence.
Uses the caption's first trigger in the oracle + zoom + word-first run (A100). Output: presentation/misread_<id>.png

CPU only. Example (from MDLLM/): python presentation/make_misread_example.py 517
"""

import json
import os
import sys

from PIL import Image, ImageDraw

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from make_examples import BLACK, GREY, H, RED, W, Examples, font, sentence_with
from trace_amber_steps import IMAGE_DIR

RUN = os.path.join(os.path.dirname(HERE), "results", "amber_g", "hallu_study", "rag", "slot_refill", "oracle_zoom_wordfirst")


def main():
    ex = Examples()
    final = {int(r["id"]): r["response"] for r in json.load(open(os.path.join(RUN, "predictions.json")))}
    caps = {c["id"]: c for c in json.load(open(os.path.join(RUN, "predictions_events.json")))["captions"]}
    for item_id in [int(a) for a in sys.argv[1:]]:
        e = caps[item_id]["events"][0]
        flagged = e["old_text"][e["remasked"].index(e["pos"])].strip()
        b_sent = sentence_with(ex.base[item_id], "".join(e["old_text"]))
        f_sent = sentence_with(final[item_id], "".join(e["new_text"]))
        photo = Image.open(os.path.join(IMAGE_DIR, ex.queries[item_id]["image"])).convert("RGB")
        x0, y0, x1, y1 = e["box"]
        s = H / photo.height
        full = photo.resize((round(photo.width * s), H))
        ImageDraw.Draw(full).rectangle([x0 * s, y0 * s, x1 * s, y1 * s], outline=(255, 0, 0), width=5)
        crop = photo.crop((x0, y0, x1, y1)).resize((H, H))

        sheet = Image.new("RGB", (W, 800), "white")
        d = ImageDraw.Draw(sheet)
        left_label = f"where MMaDA looked when writing \"{flagged}\""
        sheet.paste(full, (20, 20))
        cx = 20 + max(full.width, int(d.textlength(left_label, font=font(18)))) + 70
        d.text((cx - 55, 20 + H // 2 - 25), "→", fill=GREY, font=font(44, True))
        sheet.paste(crop, (cx, 20))
        d.text((20, H + 28), left_label, fill=GREY, font=font(18))
        top = ", ".join(f"{t.strip() or repr(t)} {p:.2f}" for t, p in e["word_first"]["top5"][:3])
        d.text((cx, H + 28), f"zoomed crop → top words: {top}", fill=GREY, font=font(18))

        def write(y, label, text):
            d.text((20, y), label, fill=BLACK, font=font(22, True))
            y += 36
            spans = [r["span"] for r in ex.labeler.label(item_id, text) if r["label"] == "hallucinated" and r["span"]]
            cur, xx = 0, 20
            pieces = []
            for a, b in sorted(spans):
                pieces += [(w, BLACK, False) for w in text[cur:a].split()] + [(text[a:b], RED, True)]
                cur = b
            pieces += [(w, BLACK, False) for w in text[cur:].split()]
            for w, col, bold in pieces:
                f = font(23, bold)
                if not w[0].isalnum() and len(w) <= 2 and xx > 20:
                    xx -= d.textlength(" ", font=f)
                wl = d.textlength(w + " ", font=f)
                if xx + wl > W - 20:
                    xx, y = 20, y + 34
                d.text((xx, y), w, fill=col, font=f)
                xx += wl
            return y + 52

        y = write(H + 80, "Baseline:", b_sent)
        y = write(y, "After zooming in (still wrong):", f_sent)
        truth = ", ".join(dict.fromkeys(ex.labeler.word_lists(item_id)[0]["truth"]))
        d.text((20, y), f"AMBER ground-truth objects: {truth}", fill=GREY, font=font(18))
        path = os.path.join(HERE, f"misread_{item_id}.png")
        sheet.crop((0, 0, W, y + 36)).save(path)
        print("saved", path, "|", b_sent, "|", f_sent)


if __name__ == "__main__":
    main()
