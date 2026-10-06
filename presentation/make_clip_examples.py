"""Slide figures for the CLIP detector (caption-aware full-image rank < 0.99, veto if the noun is CLIP's top word on the sink-free crop).

Signals are recomputed offline from rag/clip_relation/clip_vocab_embed.npz, exactly as in caption_aware_veto.py:
current 418-word vocabulary, "a photo of a {w}.", the noun's near-synonyms and the caption's other object nouns (with theirs) removed.

  python presentation/make_clip_examples.py --list          # candidates for both figures
  python presentation/make_clip_examples.py idea ID          # figure 1: every noun of one caption, how many words CLIP prefers over it
  python presentation/make_clip_examples.py veto ID:POS      # figure 2: a noun flagged on the full image and vetoed by the crop

CPU only, from MDLLM/. Output: presentation/clip_idea_<id>.png, presentation/clip_veto_<id>_<pos>.png
"""

import csv
import json
import os
import sys
from collections import defaultdict

import numpy as np
from nltk.corpus import wordnet as wn
from PIL import Image, ImageDraw

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))
sys.path.insert(0, ROOT)
sys.path.insert(0, HERE)

from amber_noun_labels import AmberLabeler
from make_examples import BLACK, GREY, RED, font
from trace_amber_steps import IMAGE_DIR, QUERY_FILE, read_json

CR = os.path.join(ROOT, "results", "amber_g", "hallu_study", "rag", "clip_relation")
GREEN = (20, 140, 40)
T_FULL = 0.99


class Clip:
    def __init__(self):
        data = np.load(os.path.join(CR, "clip_vocab_embed.npz"))
        self.words = [str(w) for w in data["words"]]
        self.widx = {w: i for i, w in enumerate(self.words)}
        self.rows = list(csv.DictReader(open(os.path.join(CR, "crop_veto_nosink.csv"), encoding="utf-8")))
        img_row = {int(i): k for k, i in enumerate(data["image_ids"])}
        self.S_full = data["full"][[img_row[int(r["id"])] for r in self.rows]] @ data["text"].T
        self.S_crop = data["nosink"] @ data["text"].T
        self.caption = defaultdict(set)
        for r in self.rows:
            self.caption[r["id"]].add(r["lemma"])
        self.labeler = AmberLabeler()
        vec = np.stack([self.labeler.nlp.vocab[w].vector for w in self.words])
        norm = np.linalg.norm(vec, axis=1)
        has = norm > 0
        vec = vec / np.where(has, norm, 1)[:, None]
        self.syn = {}
        for lemma in {r["lemma"] for r in self.rows}:
            close = {lemma}
            if has[self.widx[lemma]]:
                close |= {self.words[j] for j in np.flatnonzero((vec @ vec[self.widx[lemma]] > 0.8) & has)}
            for s in wn.synsets(lemma, wn.NOUN):
                for t in [s] + s.hypernyms() + s.hyponyms():
                    close |= {l.name().lower() for l in t.lemmas()}
            self.syn[lemma] = {self.widx[w] for w in close if w in self.widx}
        self.base = np.zeros(len(self.words), bool)
        self.base[[self.widx[str(w)] for w in data["current"]]] = True

    def signal(self, k, S):
        """(rank, words that beat the noun, best first) after removing synonyms and the caption's other nouns."""
        r = self.rows[k]
        lemma, s = r["lemma"], S[k]
        m = self.base.copy()
        drop = (self.syn[lemma] - {self.widx[lemma]}) | set().union(*(self.syn[o] for o in self.caption[r["id"]] if o != lemma))
        m[list(drop)] = False
        m[self.widx[lemma]] = True
        own = s[self.widx[lemma]]
        better = [self.words[j] for j in np.flatnonzero(m & (s > own))]
        better.sort(key=lambda w: -s[self.widx[w]])
        return float((s[m] < own).mean()), better

    def nouns(self, item_id):
        return [k for k, r in enumerate(self.rows) if int(r["id"]) == item_id]


def photo_of(item_id, queries):
    return Image.open(os.path.join(IMAGE_DIR, queries[item_id]["image"])).convert("RGB")


def draw_idea(c, item_id, queries):
    """Photo + one bar per distinct caption noun: how many vocabulary words CLIP prefers over it (full image, caption-aware)."""
    seen, items = set(), []
    for k in c.nouns(item_id):
        r = c.rows[k]
        if r["lemma"] in seen:
            continue
        seen.add(r["lemma"])
        rank, better = c.signal(k, c.S_full)
        items.append((r["lemma"], r["label"] == "hallucinated", rank, better))
    items.sort(key=lambda t: len(t[3]))
    PH = 460
    photo = photo_of(item_id, queries)
    photo = photo.resize((round(photo.width * PH / photo.height), PH))
    bar_w, row_h = 420, 52
    width = 20 + photo.width + 40 + 170 + bar_w + 360
    height = max(PH + 140, 110 + row_h * len(items) + 120)
    sheet = Image.new("RGB", (width, height), "white")
    d = ImageDraw.Draw(sheet)
    sheet.paste(photo, (20, 60))
    x0 = 20 + photo.width + 40
    d.text((20, 18), "CLIP check on the full image: how many words describe the photo better than the noun?", fill=BLACK, font=font(22, True))
    n_vocab = int(c.base.sum())
    limit = round((1 - T_FULL) * n_vocab)
    cap = max(max(len(t[3]) for t in items), limit + 2)
    y = 70
    for lemma, hallu, rank, better in items:
        col = RED if hallu else GREEN
        d.text((x0, y + 8), lemma, fill=col, font=font(21, True))
        bx = x0 + 170
        n = len(better)
        d.rectangle([bx, y + 6, bx + bar_w, y + 36], outline=(220, 220, 220), width=1)
        if n:
            d.rectangle([bx, y + 6, bx + max(3, int(bar_w * n / cap)), y + 36], fill=col)
        flagged = rank < T_FULL
        txt = f"{n} better" + (f": {', '.join(better[:3])}" if n else " (CLIP's top word)")
        d.text((bx + bar_w + 12, y + 2), txt, fill=BLACK, font=font(16))
        d.text((bx + bar_w + 12, y + 22), "→ flagged" if flagged else "→ kept", fill=RED if flagged else GREEN, font=font(16, True))
        y += row_h
    lx = x0 + 170 + int(bar_w * limit / cap)
    d.line([lx, 62, lx, y], fill=(60, 60, 60), width=2)
    d.text((lx - 40, y + 4), f"cutoff ≈ {limit} words", fill=(60, 60, 60), font=font(15))
    y += 40
    d.text((x0, y), "green = in the image   red = hallucinated (AMBER labels)", fill=GREY, font=font(16))
    d.text((x0, y + 24), "Synonyms of the noun and objects the caption already names are not counted.", fill=GREY, font=font(16))
    path = os.path.join(HERE, f"clip_idea_{item_id}.png")
    sheet.crop((0, 0, width, max(PH + 80, y + 60))).save(path)
    return path


def draw_veto(c, k, queries):
    """A grounded noun flagged on the full image (many words beat it) but CLIP's top word on its sink-free crop → flag dropped."""
    r = c.rows[k]
    item_id, lemma = int(r["id"]), r["lemma"]
    rank_f, better_f = c.signal(k, c.S_full)
    rank_c, better_c = c.signal(k, c.S_crop)
    photo = photo_of(item_id, queries)
    box = json.loads(r["box_nosink"])
    PH = 420
    s = PH / photo.height
    full = photo.resize((round(photo.width * s), PH))
    ImageDraw.Draw(full).rectangle([box[0] * s, box[1] * s, box[2] * s, box[3] * s], outline=(0, 200, 60), width=5)
    crop = photo.crop(tuple(box)).resize((PH, PH))
    width = max(20 + full.width + 70 + PH + 20, 1300)
    sheet = Image.new("RGB", (width, PH + 330), "white")
    d = ImageDraw.Draw(sheet)
    d.text((20, 16), f"Second check on the zoomed crop: \"{lemma}\" (really in the image)", fill=BLACK, font=font(22, True))
    sheet.paste(full, (20, 56))
    cx = 20 + full.width + 70
    d.text((cx - 55, 56 + PH // 2 - 25), "→", fill=GREY, font=font(44, True))
    sheet.paste(crop, (cx, 56))
    y = PH + 76
    full_words = ", ".join(better_f[:5]) + (" …" if len(better_f) > 5 else "")
    d.text((20, y), f"Full image: {len(better_f)} words beat \"{lemma}\" ({full_words})", fill=BLACK, font=font(19))
    d.text((20, y + 30), "→ flagged as a possible hallucination", fill=RED, font=font(19, True))
    crop_txt = "CLIP's top word" if not better_c else f"{len(better_c)} words beat it ({', '.join(better_c[:3])})"
    d.text((20, y + 74), f"Zoomed crop (sinks removed): \"{lemma}\" is {crop_txt}", fill=BLACK, font=font(19))
    d.text((20, y + 104), "→ the crop confirms the object, so the flag is dropped (veto)" if not better_c else "→ flag kept",
           fill=GREEN if not better_c else RED, font=font(19, True))
    d.text((20, y + 150), "On the full image, small or off-centre objects lose to the scene's main objects. In the crop they are the main object.",
           fill=GREY, font=font(16))
    path = os.path.join(HERE, f"clip_veto_{item_id}_{r['pos']}.png")
    sheet.crop((0, 0, width, y + 190)).save(path)
    return path


def main():
    c = Clip()
    queries = {int(q["id"]): q for q in read_json(QUERY_FILE)}
    if sys.argv[1] == "--list":
        print("FIGURE 1 candidates: captions where every hallucinated noun is flagged and every grounded noun kept")
        for item_id in sorted({int(r["id"]) for r in c.rows}):
            ks = c.nouns(item_id)
            sig = [(c.rows[k]["lemma"], c.rows[k]["label"], len(c.signal(k, c.S_full)[1]), c.signal(k, c.S_full)[0] < T_FULL) for k in ks]
            h = [x for x in sig if x[1] == "hallucinated"]
            g = [x for x in sig if x[1] == "grounded"]
            if h and len({x[0] for x in g}) >= 3 and all(x[3] for x in h) and not any(x[3] for x in g):
                print(f"  {item_id}: " + ", ".join(f"{l}{'*' if lab == 'hallucinated' else ''}:{n}" for l, lab, n, _ in sig))
        print("FIGURE 2 candidates: grounded nouns flagged on the full image, top word on the sink-free crop (vetoed)")
        for k, r in enumerate(c.rows):
            if r["label"] != "grounded":
                continue
            rf, bf = c.signal(k, c.S_full)
            rc, bc = c.signal(k, c.S_crop)
            if rf < T_FULL and not bc and len(bf) >= 8:
                print(f"  {r['id']}:{r['pos']} {r['lemma']}: full-image words above it {len(bf)} ({', '.join(bf[:5])})")
        return
    mode, spec = sys.argv[1], sys.argv[2]
    if mode == "idea":
        print(draw_idea(c, int(spec), queries))
    else:
        item_id, pos = spec.split(":")
        k = next(i for i, r in enumerate(c.rows) if r["id"] == item_id and r["pos"] == pos)
        print(draw_veto(c, k, queries))


if __name__ == "__main__":
    main()
