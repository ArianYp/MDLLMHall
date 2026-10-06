"""Uniformity (rel_std) of the sink patches removed online in the oracle_zoom_nosink run, vs the offline candidates."""
import csv, json, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from sink_uniformity import HERE, model_view, late_maps
ev = json.load(open(os.path.join(HERE, "rag", "slot_refill", "oracle_zoom_nosink", "predictions_events.json")))["captions"]
offline = {(int(r["id"]), int(r["patch"])) for r in csv.DictReader(open(os.path.join(HERE, "figures", "sink_candidates.csv")))}
rows, views = [], {}
for c in ev:
    for e in c["events"]:
        for p in e.get("sinks", []):
            if c["id"] not in views:
                views[c["id"]] = model_view(c["id"])
            v = views[c["id"]]; r, col = divmod(p, 32)
            maps = late_maps(c["id"]); tokens = int((maps.argmax(1) == p).sum())
            rows.append((c["id"], e["lemma"], p, round(float(v[r*16:(r+1)*16, col*16:(col+1)*16].std() / v.std()), 2), tokens, (c["id"], p) in offline))
print(f"online sink removals: {len(rows)} (triggers x sink patches); also an offline candidate: {sum(r[5] for r in rows)}")
print("not offline candidates (id, word, patch, rel_std, offline top-patch tokens):")
for r in rows:
    if not r[5]: print("  ", r[:5])
v = np.array([r[3] for r in rows]); print("rel_std of online sinks: p25 %.2f p50 %.2f p75 %.2f p90 %.2f" % tuple(np.percentile(v, [25, 50, 75, 90])))
