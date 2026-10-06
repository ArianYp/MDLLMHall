"""Word-first refill with and without attention-sink removal (oracle + zoom, both A100).

1) Refill outcomes per run (wrong same word / wrong other word / neutral / correct; verb "can" excluded).
2) Blank crops: grey-level std of the crop (true sinks sit on featureless patches, std ~1-3). Calibrated on the
   blank crops seen by eye in figures/wordfirst_wrong_*.png. Outcome by blank / not blank, per run.
3) Paired first triggers (identical state in both runs until the first refill): did sink removal move the crop,
   was the old crop blank, and how did the outcome change.

CPU only (spaCy + PIL). Example (from MDLLM/): python results/amber_g/hallu_study/wordfirst_sink_analysis.py
"""

import json
import os
import statistics as st
import sys
from collections import Counter

import numpy as np
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
sys.path.insert(0, ROOT)

from amber_noun_labels import AmberLabeler
from trace_amber_steps import IMAGE_DIR, QUERY_FILE, read_json

SR = os.path.join(HERE, "rag", "slot_refill")
RUNS = ("oracle_zoom_wordfirst", "oracle_zoom_wordfirst_nosink")
ORDER = ("wrong, same word", "wrong, other word", "neutral", "correct")
BLANK_STD = 12.0  # grey-level std below which a crop counts as blank (checked against the hand-picked examples below)
# Blank crops identified by eye (caption id, flagged span) in the word-first run.
SEEN_BLANK = [(2, " the sun shining"), (720, " the sun"), (67, " his phone."), (371, " A cup"), (541, " a table"),
              (587, " mouse"), (312, " a desk,"), (690, " a bird's"), (526, " the ground and")]
SEEN_OBJECT = [(517, " a cow"), (428, " nine cows in"), (407, " the mouse"), (289, " two street lamps"),
               (252, " a backpack"), (84, " the soap"), (32, " dog collar,"), (304, " the sun shining")]


def main():
    labeler = AmberLabeler()
    objects = labeler.object_words - labeler.global_safe
    lem = lambda t: labeler.lemmatizer.lemmatize(t.strip().lower())
    queries = {int(row["id"]): row for row in read_json(QUERY_FILE)}
    photos = {}

    def crop_std(item_id, box):
        if item_id not in photos:
            photos[item_id] = Image.open(os.path.join(IMAGE_DIR, queries[item_id]["image"])).convert("L")
        return float(np.asarray(photos[item_id].crop(tuple(box)), dtype=np.float32).std())

    def outcome(item_id, e):
        flagged = lem(e["old_text"][e["remasked"].index(e["pos"])])
        _, safe, _ = labeler.word_lists(item_id)
        kinds = []
        for t in e["new_text"]:
            w = t.strip().lower()
            if w.isalpha() and lem(w) in objects:
                l = lem(w)
                kinds.append((l, "correct" if l in safe or any(labeler.similar(l, s) for s in safe) else "wrong"))
        wrong = [l for l, k in kinds if k == "wrong"]
        if wrong:
            return ("wrong, same word" if flagged in wrong else "wrong, other word"), flagged
        return ("correct" if kinds else "neutral"), flagged

    runs = {}
    for name in RUNS:
        caps = json.load(open(os.path.join(SR, name, "predictions_events.json")))["captions"]
        runs[name] = {c["id"]: [dict(e=e, out=o, flagged=f, std=crop_std(c["id"], e["box"]) if e["box"] else None)
                                for e in c["events"] for o, f in [outcome(c["id"], e)]] for c in caps}

    print("1) refill outcomes (verb 'can' excluded)")
    for name, caps in runs.items():
        rows = [r for v in caps.values() for r in v if r["flagged"] != "can"]
        c = Counter(r["out"] for r in rows)
        print(f"  {name:<30} refills {len(rows):>3} | " + " | ".join(f"{k} {c[k]:>3} ({c[k] / len(rows):.0%})" for k in ORDER))
        sinks = [r for r in rows if r["e"].get("sinks")]
        if name.endswith("nosink"):
            print(f"  {'':<30} refills with a sink in the commit pass: {len(sinks)} / {len(rows)}")

    print(f"\n2) blank crops (grey std < {BLANK_STD})")
    wf = runs[RUNS[0]]
    find = lambda i, s: next(r for r in wf[i] if "".join(r["e"]["old_text"]) == s)
    print("  calibration, seen blank: " + ", ".join(f"{i} {find(i, s)['std']:.1f}" for i, s in SEEN_BLANK))
    print("  calibration, seen object: " + ", ".join(f"{i} {find(i, s)['std']:.1f}" for i, s in SEEN_OBJECT))
    for name, caps in runs.items():
        rows = [r for v in caps.values() for r in v if r["flagged"] != "can" and r["std"] is not None]
        print(f"  {name}: crop std median {st.median(r['std'] for r in rows):.1f}")
        for label, group in (("blank", [r for r in rows if r["std"] < BLANK_STD]), ("not blank", [r for r in rows if r["std"] >= BLANK_STD])):
            c = Counter(r["out"] for r in group)
            n = max(len(group), 1)
            print(f"    {label:<10} {len(group):>3} | " + " | ".join(f"{k} {c[k]:>3} ({c[k] / n:.0%})" for k in ORDER))

    print("\n3) paired first triggers (same state in both runs)")
    A, B = runs[RUNS[0]], runs[RUNS[1]]
    pairs = []
    for i in A:
        if not A[i] or not B.get(i):
            continue
        a, b = A[i][0], B[i][0]
        if (a["e"]["pos"], a["e"]["old_text"], a["e"]["step"]) == (b["e"]["pos"], b["e"]["old_text"], b["e"]["step"]) and a["flagged"] != "can":
            pairs.append((i, a, b))
    moved = [p for p in pairs if p[1]["e"]["box"] != p[2]["e"]["box"]]
    print(f"  pairs {len(pairs)}; sink present {sum(bool(b['e'].get('sinks')) for _, _, b in pairs)}; crop moved {len(moved)}")
    for label, group in (("crop moved", moved), ("crop unchanged", [p for p in pairs if p not in moved])):
        changed = [p for p in group if p[1]["e"]["new_text"] != p[2]["e"]["new_text"]]
        print(f"  {label}: {len(group)}, refill text differs in {len(changed)}")
        for (x, y), n in sorted(Counter((a["out"], b["out"]) for _, a, b in changed).items(), key=lambda t: -t[1]):
            print(f"    word-first: {x:<18} -> + sink removal: {y:<18} {n}")
    print("  moved crops (std before -> after):")
    for i, a, b in moved:
        ta, tb = a["e"]["word_first"]["top5"][0], b["e"]["word_first"]["top5"][0]
        print(f"    {i:>4} {''.join(a['e']['old_text'])!r:<22} std {a['std']:5.1f} -> {b['std']:5.1f} | "
              f"crop word {ta[0].strip()!r} {ta[1]:.2f} -> {tb[0].strip()!r} {tb[1]:.2f} | "
              f"{''.join(a['e']['new_text'])!r} [{a['out']}] -> {''.join(b['e']['new_text'])!r} [{b['out']}]")


if __name__ == "__main__":
    main()
