"""Slide figure: a baseline hallucination fixed by the zoom refill (oracle + zoom + word-first + sinks removed, A100).

Per caption id: the photo with the attended crop box (red), the crop the refill saw with its top words, the baseline caption
(hallucinated nouns in red) and the final caption (the refilled correct object in green, remaining hallucinations in red),
plus AMBER's truth objects. Output: presentation/example_<id>.png

CPU only. Example (from MDLLM/): python presentation/make_example.py 537 102 263
"""

import json
import os
import sys

from PIL import Image, ImageDraw, ImageFont

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))
sys.path.insert(0, ROOT)

from amber_noun_labels import AmberLabeler
from trace_amber_steps import IMAGE_DIR, QUERY_FILE, read_json

RUN = os.path.join(ROOT, "results", "amber_g", "hallu_study", "rag", "slot_refill", "oracle_zoom_wordfirst_nosink")
BASELINE = os.path.join(ROOT, "results", "amber_g", "mmada_predictions.json")
RED, GREEN, BLACK, GREY = (200, 30, 30), (20, 140, 40), (20, 20, 20), (90, 90, 90)
W, H = 1500, 460  # sheet width, image row height


def font(size, bold=False):
    name = "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf"
    for d in ("/usr/share/fonts/dejavu-sans-fonts", "/usr/share/fonts/dejavu", "/usr/share/fonts/truetype/dejavu"):
        if os.path.exists(os.path.join(d, name)):
            return ImageFont.truetype(os.path.join(d, name), size)
    return ImageFont.load_default()


def colored_words(text, marks):
    """Split text into (word, colour, bold) with the character spans in `marks` [(start, end, colour)] coloured."""
    out, cursor = [], 0
    for start, end, colour in sorted(marks):
        out += [(w, BLACK, False) for w in text[cursor:start].split()]
        out.append((text[start:end], colour, True))
        cursor = end
    out += [(w, BLACK, False) for w in text[cursor:].split()]
    # glue punctuation that followed a coloured word back onto it
    merged = []
    for w, c, b in out:
        if merged and w and not w[0].isalnum() and len(w) <= 2:
            pw, pc, pb = merged[-1]
            merged[-1] = (pw + w, pc, pb)
        else:
            merged.append((w, c, b))
    return merged


def draw_paragraph(draw, x, y, width, words, size=21):
    regular, bold = font(size), font(size, True)
    cx, line = x, int(size * 1.45)
    for w, colour, is_bold in words:
        f = bold if is_bold else regular
        wl = draw.textlength(w + " ", font=f)
        if cx + wl > x + width:
            cx, y = x, y + line
        draw.text((cx, y), w, fill=colour, font=f)
        cx += wl
    return y + line


def main():
    labeler = AmberLabeler()
    queries = {int(r["id"]): r for r in read_json(QUERY_FILE)}
    base = {int(r["id"]): r["response"] for r in json.load(open(BASELINE))}
    final = {int(r["id"]): r["response"] for r in json.load(open(os.path.join(RUN, "predictions.json")))}
    events = {c["id"]: c["events"] for c in json.load(open(os.path.join(RUN, "predictions_events.json")))["captions"]}
    lem = lambda t: labeler.lemmatizer.lemmatize(t.strip().lower())

    for item_id in [int(a) for a in sys.argv[1:]]:
        _, safe, _ = labeler.word_lists(item_id)
        truth = list(dict.fromkeys(labeler.word_lists(item_id)[0]["truth"]))
        # the refill to show: the first one that put a correct object in place of the flagged word
        event = None
        for e in events[item_id]:
            new = [lem(t) for t in e["new_text"] if t.strip().isalpha()]
            if e["old_text"] != e["new_text"] and any(w in safe or any(labeler.similar(w, s) for s in safe) for w in new
                                                       if w in labeler.object_words):
                event = e
                break
        new_lemmas = {lem(t) for t in event["new_text"] if t.strip().isalpha()}
        flagged = "".join(event["old_text"][event["remasked"].index(event["pos"]):][:1]).strip()

        photo = Image.open(os.path.join(IMAGE_DIR, queries[item_id]["image"])).convert("RGB")
        x0, y0, x1, y1 = event["box"]
        s = H / photo.height
        full = photo.resize((round(photo.width * s), H))
        ImageDraw.Draw(full).rectangle([x0 * s, y0 * s, x1 * s, y1 * s], outline=(255, 0, 0), width=5)
        crop = photo.crop((x0, y0, x1, y1)).resize((H, H))

        b_text, f_text = " ".join(base[item_id].split()), " ".join(final[item_id].split())
        # where the refilled phrase sits in the final caption (green only there)
        phrase = " ".join("".join(event["new_text"]).split())
        at = f_text.find(phrase)
        refill_span = (at, at + len(phrase)) if at != -1 else None

        def marks(text, final_caption):
            out = []
            for r in labeler.label(item_id, text):
                if r["span"] is None:
                    continue
                if r["label"] == "hallucinated":
                    out.append((*r["span"], RED))
                elif (final_caption and r["label"] == "grounded" and r["lemma"] in new_lemmas and refill_span
                      and refill_span[0] <= r["span"][0] < refill_span[1]):
                    out.append((*r["span"], GREEN))
            return out

        n_b = sum(r["label"] == "hallucinated" for r in labeler.label(item_id, b_text))
        n_f = sum(r["label"] == "hallucinated" for r in labeler.label(item_id, f_text))

        sheet = Image.new("RGB", (W, 1400), "white")
        d = ImageDraw.Draw(sheet)
        sheet.paste(full, (20, 20))
        cx = 20 + full.width + 60
        d.text((cx - 48, 20 + H // 2 - 20), "→", fill=GREY, font=font(40, True))
        sheet.paste(crop, (cx, 20))
        top = ", ".join(f"{t.strip() or repr(t)} {p:.2f}" for t, p in event["word_first"]["top5"][:3])
        d.text((20, H + 30), f"red box: where MMaDA looked when writing \"{flagged}\"", fill=GREY, font=font(18))
        d.text((cx, H + 30), f"zoomed crop → top words: {top}", fill=GREY, font=font(18))

        y = H + 80
        d.text((20, y), f"Baseline caption ({n_b} hallucinated nouns in red):", fill=BLACK, font=font(22, True))
        y = draw_paragraph(d, 20, y + 36, W - 40, colored_words(b_text, marks(b_text, False))) + 20
        d.text((20, y), f"After the zoom refill ({n_f} hallucinated nouns; the refilled object in green):", fill=BLACK, font=font(22, True))
        y = draw_paragraph(d, 20, y + 36, W - 40, colored_words(f_text, marks(f_text, True))) + 20
        d.text((20, y), f"Refill: \"{''.join(event['old_text']).strip()}\" → \"{''.join(event['new_text']).strip()}\"    "
               f"AMBER ground-truth objects: {', '.join(truth)}", fill=GREY, font=font(18))
        sheet = sheet.crop((0, 0, W, y + 40))
        path = os.path.join(HERE, f"example_{item_id}.png")
        sheet.save(path)
        print("saved", path, "| baseline", n_b, "-> final", n_f, "|", "".join(event["old_text"]), "->", "".join(event["new_text"]))


if __name__ == "__main__":
    main()
