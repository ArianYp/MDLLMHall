"""How often does the zoom refill write a wrong word again (oracle trigger, so every flag is a real hallucination)?

For every trigger of a slot-refill run, the refilled span (flagged word + neighbours) is labelled with AmberLabeler:
  wrong    an object word in the span is not in the image (split: the same word again / a different wrong word)
  correct  an object word in the span is in the image, none is wrong
  neutral  no object word in the span
"can" (AMBER object, nearly always the verb) and negated mentions ("no people") are counted separately, as labeller artefacts.
Then --sample wrong refills (one per caption, no "can") are drawn: full photo with the crop box, and the crop.

CPU only (spaCy). Example (from MDLLM/):
    python results/amber_g/hallu_study/refill_errors.py --run oracle_zoom --sample 10
"""

import argparse
import json
import os
import random
import sys
from collections import Counter

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
sys.path.insert(0, ROOT)

from PIL import Image, ImageDraw

from amber_noun_labels import AmberLabeler

IMAGES = os.path.join(ROOT, "data", "amber", "images")
NEGATIONS = {"no", "not", "without", "nor"}


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--run", default="oracle_zoom")
    parser.add_argument("--sample", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    labeler = AmberLabeler()
    objects = labeler.object_words - labeler.global_safe
    annotations = labeler.annotations

    def label(item_id, token):
        word = token.strip().lower()
        if not word.isalpha():
            return None
        lemma = labeler.lemmatizer.lemmatize(word)
        if lemma not in objects:
            return None
        _, safe_words, _ = labeler.word_lists(item_id)
        grounded = lemma in safe_words or any(labeler.similar(lemma, w) for w in safe_words)
        return lemma, ("correct" if grounded else "wrong")

    run = json.load(open(os.path.join(HERE, "rag", "slot_refill", args.run, "predictions_events.json")))
    clean = set(json.load(open(os.path.join(HERE, "ids_clean.json")))["ids"])
    rows = []
    for c in run["captions"]:
        for e in c["events"]:
            k = e["remasked"].index(e["pos"])
            flagged = labeler.lemmatizer.lemmatize(e["old_text"][k].strip().lower())
            labels = [label(c["id"], t) for t in e["new_text"]]
            wrong = [l for l, kind in filter(None, labels) if kind == "wrong"]
            right = [l for l, kind in filter(None, labels) if kind == "correct"]
            outcome = "wrong" if wrong else "correct" if right else "neutral"
            text = [t.strip().lower() for t in e["new_text"]]
            negated = any(t in NEGATIONS for t in text)
            rows.append(dict(id=c["id"], subset="clean" if c["id"] in clean else "hallucinating", flagged=flagged,
                             old="".join(e["old_text"]).strip(), new="".join(e["new_text"]).strip(), outcome=outcome,
                             same_word=flagged in wrong, wrong=wrong, box=e["box"], verb_can=flagged == "can",
                             negated=negated, changed=e["old_text"][k] != e["new_text"][k], p=e["p_original_refill"]))

    print(f"run {args.run}: {len(rows)} refills ({sum(r['subset'] == 'hallucinating' for r in rows)} in hallucinating captions, "
          f"{sum(r['subset'] == 'clean' for r in rows)} in clean)")
    for name, S in (("all", rows), ("excluding verb 'can' and negations", [r for r in rows if not r["verb_can"] and not r["negated"]])):
        c = Counter(r["outcome"] for r in S)
        same = sum(r["outcome"] == "wrong" and r["same_word"] for r in S)
        print(f"\n{name}: {len(S)} refills")
        print(f"  wrong   {c['wrong']:>3} ({c['wrong'] / len(S):.0%})  = same wrong word again {same}, a different wrong word {c['wrong'] - same}")
        print(f"  neutral {c['neutral']:>3} ({c['neutral'] / len(S):.0%})")
        print(f"  correct {c['correct']:>3} ({c['correct'] / len(S):.0%})")
    print(f"\nartefacts: verb 'can' {sum(r['verb_can'] for r in rows)}, negated span {sum(r['negated'] for r in rows)}")
    # p(original word) in the first refill pass (whole span masked, crop as image): the moment a "word first" refill would commit.
    print("\np(original word) at the first refill pass (crop, span masked), by final outcome:")
    groups = (("wrong, same word again", lambda r: r["outcome"] == "wrong" and r["same_word"]),
              ("wrong, different word", lambda r: r["outcome"] == "wrong" and not r["same_word"]),
              ("neutral", lambda r: r["outcome"] == "neutral"), ("correct", lambda r: r["outcome"] == "correct"))
    for name, f in groups:
        ps = sorted(r["p"] for r in rows if f(r) and not r["verb_can"])
        if ps:
            print(f"  {name:<24} n={len(ps):>3}  median {ps[len(ps) // 2]:.2f}  p<0.5: {sum(p < 0.5 for p in ps):>3}  "
                  f"p<0.2: {sum(p < 0.2 for p in ps):>3}  p<0.05: {sum(p < 0.05 for p in ps):>3}")
    print("most common same-word repeats:", Counter(r["flagged"] for r in rows if r["outcome"] == "wrong" and r["same_word"]).most_common(10))

    pool = [r for r in rows if r["outcome"] == "wrong" and not r["verb_can"] and not r["negated"]]
    rng = random.Random(args.seed)
    rng.shuffle(pool)
    picked, seen = [], set()
    for r in pool:
        if r["id"] not in seen:
            picked.append(r)
            seen.add(r["id"])
        if len(picked) == args.sample:
            break
    out = os.path.join(HERE, "figures", f"refill_wrong_{args.run}")
    os.makedirs(out, exist_ok=True)
    print(f"\nsample of {len(picked)} wrong refills (seed {args.seed}) -> {out}")
    for r in picked:
        truth = annotations[r["id"] - 1]["truth"]
        print(f"  {r['id']:>4} flagged {r['flagged']!r:<10} {r['old']!r:<24} -> {r['new']!r:<24} wrong {r['wrong']}  "
              f"p(orig) {r['p']:.2f} | truth: {', '.join(truth)}")
        photo = Image.open(os.path.join(IMAGES, f"AMBER_{r['id']}.jpg")).convert("RGB")
        im = photo.copy()
        lw = max(2, min(photo.size) // 120)
        ImageDraw.Draw(im).rectangle(r["box"], outline=(255, 0, 0), width=lw)
        h = 340
        full = im.resize((int(im.width * h / im.height), h))
        sheet = Image.new("RGB", (full.width + h + 10, h + 46), "white")
        sheet.paste(full, (0, 46))
        sheet.paste(photo.crop(tuple(r["box"])).resize((h, h)), (full.width + 10, 46))
        g = ImageDraw.Draw(sheet)
        g.text((5, 4), f"{r['id']} flagged '{r['flagged']}': {r['old']!r} -> refill {r['new']!r}  (wrong: {r['wrong']}, p(orig) {r['p']:.2f})", fill="black")
        g.text((5, 20), f"AMBER truth: {', '.join(truth)}", fill=(0, 100, 0))
        sheet.save(os.path.join(out, f"{r['id']}_{r['flagged']}.jpg"), quality=85)


if __name__ == "__main__":
    main()
