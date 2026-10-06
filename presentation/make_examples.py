"""Slide figures: one hallucination, one zoom. The baseline sentence with the hallucinated word, the crop, and the same sentence after the refill.

Only the FIRST trigger of a caption is used: until then decoding is identical to the baseline, so the baseline sentence and the refilled
sentence are directly comparable. A candidate qualifies when the baseline sentence holds exactly one hallucinated noun (the flagged one),
the refill puts in a correct object, and the refilled sentence has no hallucinated noun.

  python presentation/make_examples.py --list            # print all candidates over the oracle + zoom runs
  python presentation/make_examples.py RUN:ID [RUN:ID …]  # draw presentation/zoom_<ID>.png

RUN is one of the keys of RUNS. CPU only, from MDLLM/.
"""

import json
import os
import re
import sys

from PIL import Image, ImageDraw, ImageFont

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))
sys.path.insert(0, ROOT)

from amber_noun_labels import AmberLabeler
from trace_amber_steps import IMAGE_DIR, QUERY_FILE, read_json

SR = os.path.join(ROOT, "results", "amber_g", "hallu_study", "rag", "slot_refill")
RUNS = {  # name: (events file, predictions file, description)
    "wf_nosink": (os.path.join(SR, "oracle_zoom_wordfirst_nosink"), "zoom, word-first, sinks removed"),
    "wf": (os.path.join(SR, "oracle_zoom_wordfirst"), "zoom, word-first"),
    "nosink": (os.path.join(SR, "oracle_zoom_nosink_a100"), "zoom, sinks removed"),
    "zoom": (os.path.join(SR, "oracle_zoom"), "zoom"),
    "full": (os.path.join(ROOT, "results", "amber_g", "oracle_zoom_full", "missing"), "zoom (full-set run)"),
}
BASELINE = os.path.join(ROOT, "results", "amber_g", "mmada_predictions.json")
RED, GREEN, BLACK, GREY = (200, 30, 30), (20, 140, 40), (20, 20, 20), (90, 90, 90)
W, H = 1400, 420


def font(size, bold=False):
    name = "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf"
    for d in ("/usr/share/fonts/dejavu-sans-fonts", "/usr/share/fonts/dejavu", "/usr/share/fonts/truetype/dejavu"):
        if os.path.exists(os.path.join(d, name)):
            return ImageFont.truetype(os.path.join(d, name), size)
    return ImageFont.load_default()


def sentences(text):
    return [s for s in re.split(r"(?<=[.!?])\s+", " ".join(text.split())) if s]


def sentence_with(text, phrase):
    phrase = " ".join(phrase.split())
    for s in sentences(text):
        if phrase and phrase in s:
            return s
    return None


class Examples:
    def __init__(self):
        self.labeler = AmberLabeler()
        self.objects = self.labeler.object_words - self.labeler.global_safe
        self.base = {int(r["id"]): r["response"] for r in json.load(open(BASELINE))}
        self.queries = {int(r["id"]): r for r in read_json(QUERY_FILE)}

    def lem(self, t):
        return self.labeler.lemmatizer.lemmatize(t.strip().lower())

    def correct(self, item_id, lemma):
        _, safe, _ = self.labeler.word_lists(item_id)
        return lemma in safe or any(self.labeler.similar(lemma, s) for s in safe)

    def load(self, run):
        path, _ = RUNS[run]
        final = {int(r["id"]): r["response"] for r in json.load(open(os.path.join(path, "predictions.json")))}
        caps = {c["id"]: c for c in json.load(open(os.path.join(path, "predictions_events.json")))["captions"]}
        return final, caps

    def candidate(self, run, item_id, final, cap):
        """The caption's first trigger, if it is a clean one-hallucination / one-zoom fix; else None."""
        if not cap["events"]:
            return None
        e = cap["events"][0]
        flagged = e["old_text"][e["remasked"].index(e["pos"])]
        new_objects = [self.lem(t) for t in e["new_text"] if t.strip().isalpha() and self.lem(t) in self.objects]
        if not new_objects or not all(self.correct(item_id, w) for w in new_objects) or self.lem(flagged) in new_objects:
            return None
        b_sent = sentence_with(self.base[item_id], "".join(e["old_text"]))
        f_sent = sentence_with(final[item_id], "".join(e["new_text"]))
        if not b_sent or not f_sent:
            return None
        b_h = [r for r in self.labeler.label(item_id, b_sent) if r["label"] == "hallucinated"]
        f_h = [r for r in self.labeler.label(item_id, f_sent) if r["label"] == "hallucinated"]
        if len(b_h) != 1 or self.lem(b_h[0]["word"]) != self.lem(flagged) or f_h:
            return None
        return dict(run=run, id=item_id, event=e, flagged=flagged.strip(), b_sent=b_sent, f_sent=f_sent, new_objects=new_objects)

    def draw(self, c):
        e, item_id = c["event"], c["id"]
        photo = Image.open(os.path.join(IMAGE_DIR, self.queries[item_id]["image"])).convert("RGB")
        x0, y0, x1, y1 = e["box"]
        s = H / photo.height
        full = photo.resize((round(photo.width * s), H))
        ImageDraw.Draw(full).rectangle([x0 * s, y0 * s, x1 * s, y1 * s], outline=(255, 0, 0), width=5)
        crop = photo.crop((x0, y0, x1, y1)).resize((H, H))
        sheet = Image.new("RGB", (W, 900), "white")
        d = ImageDraw.Draw(sheet)
        sheet.paste(full, (20, 20))
        left_label = f"where MMaDA looked when writing \"{c['flagged']}\""
        cx = 20 + max(full.width, int(d.textlength(left_label, font=font(18)))) + 70
        d.text((cx - 55, 20 + H // 2 - 25), "→", fill=GREY, font=font(44, True))
        sheet.paste(crop, (cx, 20))
        d.text((20, H + 28), left_label, fill=GREY, font=font(18))
        sub = "zoomed crop (replaces the image in the refill)"
        if e.get("word_first"):
            sub = "zoomed crop → top words: " + ", ".join(f"{t.strip() or repr(t)} {p:.2f}" for t, p in e["word_first"]["top5"][:3])
        d.text((cx, H + 28), sub, fill=GREY, font=font(18))

        def line(y, label, text, marks):
            d.text((20, y), label, fill=BLACK, font=font(22, True))
            y += 36
            cxw, size = 20, 23
            words, cursor = [], 0
            for a, b, col in sorted(marks):
                words += [(w, BLACK, False) for w in text[cursor:a].split()]
                words.append((text[a:b], col, True))
                cursor = b
            words += [(w, BLACK, False) for w in text[cursor:].split()]
            for w, col, bold in words:
                f = font(size, bold)
                if words and not w[0].isalnum() and len(w) <= 2 and cxw > 20:
                    cxw -= d.textlength(" ", font=f)
                wl = d.textlength(w + " ", font=f)
                if cxw + wl > W - 20:
                    cxw, y = 20, y + int(size * 1.5)
                d.text((cxw, y), w, fill=col, font=f)
                cxw += wl
            return y + int(size * 1.5) + 18

        # red: the hallucinated noun of the baseline sentence; green: the object word(s) the refill wrote, inside the refilled phrase
        red = [(*r["span"], RED) for r in self.labeler.label(item_id, c["b_sent"]) if r["label"] == "hallucinated" and r["span"]]
        phrase = " ".join("".join(e["new_text"]).split())
        at = c["f_sent"].find(phrase)
        green, cursor = [], max(at, 0)
        for t in e["new_text"]:
            w = t.strip()
            if w.isalpha() and self.lem(w) in c["new_objects"]:
                k = c["f_sent"].find(w, cursor)
                if k != -1:
                    green.append((k, k + len(w), GREEN))
                    cursor = k + len(w)
        y = H + 80
        y = line(y, "Baseline:", c["b_sent"], red)
        y = line(y, "After zooming in:", c["f_sent"], green)
        truth = ", ".join(dict.fromkeys(self.labeler.word_lists(item_id)[0]["truth"]))
        d.text((20, y), f"AMBER ground-truth objects: {truth}", fill=GREY, font=font(18))
        path = os.path.join(HERE, f"zoom_{item_id}.png")
        sheet.crop((0, 0, W, y + 36)).save(path)
        return path


def main():
    ex = Examples()
    if sys.argv[1:] == ["--list"]:
        seen = set()
        for run in RUNS:
            final, caps = ex.load(run)
            for item_id, cap in caps.items():
                c = ex.candidate(run, item_id, final, cap)
                if c:
                    e = c["event"]
                    top = e["word_first"]["top5"][0] if e.get("word_first") else None
                    tag = "" if item_id not in seen else " (also in an earlier run)"
                    seen.add(item_id)
                    print(f"{run}:{item_id}{tag} | {''.join(e['old_text'])!r} -> {''.join(e['new_text'])!r} | crop top {top}")
                    print(f"      B: {c['b_sent']}\n      F: {c['f_sent']}")
        return
    for spec in sys.argv[1:]:
        run, item_id = spec.split(":")
        final, caps = ex.load(run)
        c = ex.candidate(run, int(item_id), final, caps[int(item_id)])
        print(ex.draw(c) if c else f"{spec}: not a clean one-zoom fix")


if __name__ == "__main__":
    main()
