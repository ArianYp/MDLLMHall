"""How big a WordNet-physical-object vocabulary do COCO train captions support? (size estimate only)"""
import json, re
from collections import Counter
from nltk.corpus import wordnet as wn

caps = json.load(open("/data/gpfs/datasets/COCO/annotations/captions_train2017.json"))["annotations"]
count = Counter()
for a in caps:
    for t in re.findall(r"[a-z]+", a["caption"].lower()):
        count[t] += 1
physical = wn.synset("physical_entity.n.01")
cache = {}
def is_physical(lemma):
    if lemma not in cache:
        cache[lemma] = any(physical in {h for p in s.hypernym_paths() for h in p} for s in wn.synsets(lemma, "n")[:3])
    return cache[lemma]
lemmas = Counter()
for tok, c in count.items():
    l = wn.morphy(tok, wn.NOUN)
    if l and is_physical(l):
        lemmas[l] += c
amber = set(json.load(open("third_party/AMBER/data/relation.json")))
print("captions", len(caps), "distinct tokens", len(count), "physical-noun lemmas", len(lemmas))
for k in (1, 5, 20, 50, 100):
    v = {w for w, c in lemmas.items() if c >= k}
    print(f"  >= {k:>3} mentions: {len(v):>6} words, covers {len(amber & v)}/{len(amber)} AMBER main words")
v = {w for w, c in lemmas.items() if c >= 20}
print("key words (count):", {w: lemmas.get(w, 0) for w in ["lime","goat","kayak","shutter","melon","collar","horse","cow","sheep","canoe","lemon","bench","stool","window","door","road","ground","dirt","desk"]})
print("random sample >=20:", sorted(v)[::max(1, len(v)//40)])
