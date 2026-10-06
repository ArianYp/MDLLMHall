"""Did removing the attention sink change the refilled word, and was the new word correct?

Paired first triggers of the 84 hallucinating study captions in the two L40S oracle+zoom runs
(oracle_zoom_l40s = crop with sink, oracle_zoom_nosink = sink removed): until the first refill both runs are in
the same state, so the only difference is the crop. The word refilled at the flagged position is labelled with
AmberLabeler for that image:
  correct     object word that is in the image (grounded)
  wrong       object word that is not in the image (hallucinated)
  neutral     not an AMBER object word (or a global safe word): the hallucination is gone, nothing correct replaced it
Optionally restricted to sinks that also pass the uniformity check (--max-rel-std).

CPU only (spaCy). Example (from MDLLM/): python results/amber_g/hallu_study/sink_effect_words.py
"""

import argparse
import json
import os
import sys
from collections import Counter

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
sys.path.insert(0, ROOT)
sys.path.insert(0, HERE)

from amber_noun_labels import AmberLabeler
from sink_uniformity import model_view


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--max-rel-std", type=float, default=None)
    args = parser.parse_args()
    labeler = AmberLabeler()
    objects = labeler.object_words - labeler.global_safe

    def judge(item_id, token):
        word = token.strip().lower()
        if not word.isalpha():
            return "neutral", word
        lemma = labeler.lemmatizer.lemmatize(word)
        if lemma not in objects:
            return "neutral", lemma
        _, safe_words, _ = labeler.word_lists(item_id)
        grounded = lemma in safe_words or any(labeler.similar(lemma, w) for w in safe_words)
        return ("correct" if grounded else "wrong"), lemma

    def judge_span(item_id, tokens):
        """The refilled span (word and its neighbours): wrong if any object word is hallucinated, correct if any is grounded."""
        labels = [judge(item_id, t) for t in tokens]
        for kind in ("wrong", "correct"):
            hits = [lemma for k, lemma in labels if k == kind]
            if hits:
                return kind, hits
        return "neutral", []

    hall = set(json.load(open(os.path.join(HERE, "ids.json")))["ids"])
    sr = os.path.join(HERE, "rag", "slot_refill")
    def first(name):
        return {c["id"]: c["events"][0] for c in json.load(open(os.path.join(sr, name, "predictions_events.json")))["captions"]
                if c["id"] in hall and c["events"]}
    A, B = first("oracle_zoom_l40s"), first("oracle_zoom_nosink")
    pairs = [(i, A[i], B[i]) for i in sorted(set(A) & set(B))
             if A[i]["pos"] == B[i]["pos"] and A[i]["old_text"] == B[i]["old_text"] and A[i]["step"] == B[i]["step"]]

    def rel_std(view, p):
        r, c = divmod(p, 32)
        return float(view[r * 16:(r + 1) * 16, c * 16:(c + 1) * 16].std() / view.std())

    rows = []
    for i, a, b in pairs:
        k = a["remasked"].index(a["pos"])
        orig, wa, wb = a["old_text"][k], a["new_text"][k], b["new_text"][k]
        moved = a["box"] != b["box"]
        if moved and args.max_rel_std is not None:
            view = model_view(i)
            if not any(rel_std(view, p) < args.max_rel_std for p in b["sinks"]):
                moved, wb = False, wa          # under the uniform rule this sink is kept: same crop, same word
        span_b_tokens = b["new_text"] if moved else a["new_text"]
        rows.append(dict(id=i, flagged=orig.strip(), moved=moved, with_sink="".join(a["new_text"]).strip(),
                         no_sink="".join(span_b_tokens).strip(),
                         judge_a=judge_span(i, a["new_text"]), judge_b=judge_span(i, span_b_tokens)))

    rule = "count rule" if args.max_rel_std is None else f"count rule + rel_std < {args.max_rel_std}"
    moved = [r for r in rows if r["moved"]]
    diff = [r for r in moved if r["with_sink"] != r["no_sink"]]
    print(f"paired first triggers: {len(rows)} | sink removal moved the crop: {len(moved)} ({rule}) | "
          f"refilled text different: {len(diff)}")
    print(f"\nall {len(rows)} triggers, outcome of the refilled word:")
    for side in ("judge_a", "judge_b"):
        c = Counter(r[side][0] for r in rows)
        print(f"  {'with sink' if side == 'judge_a' else 'sink removed':<13} correct {c['correct']:>2}  neutral {c['neutral']:>2}  wrong {c['wrong']:>2}")
    print(f"\nthe {len(moved)} triggers where the crop moved:")
    for side in ("judge_a", "judge_b"):
        c = Counter(r[side][0] for r in moved)
        print(f"  {'with sink' if side == 'judge_a' else 'sink removed':<13} correct {c['correct']:>2}  neutral {c['neutral']:>2}  wrong {c['wrong']:>2}")
    trans = Counter((r["judge_a"][0], r["judge_b"][0]) for r in diff)
    print(f"\nthe {len(diff)} triggers where sink removal changed the refilled word (with sink -> sink removed):")
    for (x, y), n in sorted(trans.items()):
        print(f"  {x:>8} -> {y:<8} {n}")
    print("\nper trigger (crop moved):")
    for r in moved:
        tag = "" if r["with_sink"] != r["no_sink"] else "  (same text)"
        print(f"  {r['id']:>4} flagged {r['flagged']!r:<10} with sink {r['with_sink']!r:<22} {r['judge_a'][0]:<7} | "
              f"sink removed {r['no_sink']!r:<22} {r['judge_b'][0]:<7}{tag}")


if __name__ == "__main__":
    main()
