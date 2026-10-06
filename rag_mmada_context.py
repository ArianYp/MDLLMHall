"""Give MMaDA the COCO captions retrieved by rag_retrieve.py and see how its answer changes.

Two probes on one AMBER-g caption, both with the image present:

  regenerate  the whole caption is decoded again (baseline schedule) with
              retrieved captions placed before the AMBER query. Contexts: none,
              the full image's top-5, the top-2 of every noun's region (all
              nouns, per crop size), the top-3 of the hallucinated nouns' regions
              only (an oracle detector), and 5 random COCO captions (control).
              Each output is labelled with AmberLabeler.
  slot        the finished baseline caption is kept, one noun is masked and
              re-predicted in a single forward pass, with that noun's region
              top-5 captions (per crop size), the full image's top-5, random
              captions or no context.

Example:
    python rag_mmada_context.py --id 163
"""

import argparse
import json
import os
import random

import torch
from PIL import Image

from amber_noun_labels import AmberLabeler
from mmada_infer import build_input_ids, generate_answer, load_mmada
from rag_retrieve import COCO, CROPS
from remask_decoding import STUDY, clean_response, write_json
from trace_amber_steps import IMAGE_DIR, QUERY_FILE, read_json
from trace_mmada_steps import MASK_ID

TEMPLATE = ("Captions of similar images retrieved from a database (they may not exactly match this image):\n"
            "{lines}\n\n{query}")


def with_context(query, captions):
    if not captions:
        return query
    return TEMPLATE.format(lines="\n".join(f"- {c}" for c in captions), query=query)


def first_captions(entry, k):
    return [h["captions"][0] for h in entry["hits"][:k] if h["captions"]]


def region_captions(entries, k):
    """Top-k first captions of each region, in caption order, without repeating a COCO image."""
    seen, out = set(), []
    for entry in entries:
        for h in entry["hits"][:k]:
            if h["coco_id"] not in seen and h["captions"]:
                seen.add(h["coco_id"])
                out.append(h["captions"][0])
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--id", type=int, default=163)
    parser.add_argument("--traces", default=os.path.join(STUDY, "traces"))
    parser.add_argument("--model", default="Gen-Verse/MMaDA-8B-MixCoT")
    parser.add_argument("--vq-model", default="showlab/magvitv2")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    rag_dir = os.path.join(STUDY, "rag", str(args.id))
    run = dict(max_new_tokens=128, steps=64, block_length=32, temperature=0.0, resolution=512)

    item = {int(r["id"]): r for r in read_json(QUERY_FILE)}[args.id]
    trace = read_json(os.path.join(args.traces, f"{args.id}.json"))
    answer_ids = torch.tensor([r["token_id"] for r in trace["records"]])
    summary = read_json(os.path.join(rag_dir, "summary.json"))
    regions = {}
    for q in summary["queries"]:
        if "/" in q["query"]:
            folder, crop = q["query"].split("/")
            regions.setdefault(folder, {})[crop] = read_json(os.path.join(rag_dir, folder, crop, "retrieved.json"))
    full = read_json(os.path.join(rag_dir, "full_image", "retrieved.json"))

    rng = random.Random(args.seed)
    with open(os.path.join(COCO, "annotations", "captions_train2017.json"), encoding="utf-8") as handle:
        annotations = json.load(handle)["annotations"]
    random_captions = [a["caption"].strip() for a in rng.sample(annotations, 5)]
    del annotations

    device = torch.device("cuda")
    torch.manual_seed(args.seed)
    model, vq_model, uni_prompting, _ = load_mmada(args.model, args.vq_model, device)
    tokenizer = uni_prompting.text_tokenizer
    labeler = AmberLabeler()
    image = Image.open(os.path.join(IMAGE_DIR, item["image"]))
    query = item["query"]

    contexts = dict(none=[], full_top5=first_captions(full, 5), random5=random_captions)
    for crop in CROPS:
        contexts[f"regions_all_{crop}"] = region_captions([regions[f][crop] for f in sorted(regions)], 2)
        contexts[f"regions_hallucinated_{crop}_oracle"] = region_captions(
            [regions[f][crop] for f in sorted(regions) if f.startswith("hallucinated")], 3)

    results = dict(id=args.id, baseline=trace["answer"], template=TEMPLATE, contexts=contexts, regenerate={}, slot={})
    for name, captions in contexts.items():
        text = clean_response(generate_answer(model, vq_model, uni_prompting, with_context(query, captions),
                                              image, device, **run))
        nouns = labeler.label(args.id, text)
        hallucinated = [n["word"] for n in nouns if n["label"] == "hallucinated"]
        grounded = [n["word"] for n in nouns if n["label"] == "grounded"]
        results["regenerate"][name] = dict(caption=text, hallucinated=hallucinated, grounded=grounded,
                                           same_as_baseline=text == trace["answer"])
        print(f"\n[regenerate:{name}] ({len(captions)} captions) hallucinated={hallucinated} grounded={grounded}\n  {text}",
              flush=True)

    @torch.no_grad()
    def slot(captions, pos):
        prefix = build_input_ids(vq_model, uni_prompting, with_context(query, captions), image, device)
        x = torch.cat([prefix, answer_ids[None].to(device)], dim=1)
        x[0, prefix.shape[1] + pos] = MASK_ID
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits = model(x).logits[0, prefix.shape[1] + pos].float()
        probs = torch.softmax(logits, -1)
        top = torch.topk(probs, 5)
        return dict(p_original=float(probs[answer_ids[pos]]),
                    top5=[(tokenizer.decode([int(i)]), round(float(p), 4)) for p, i in zip(top.values, top.indices)])

    print("\n[slot] p(original token) and top-3 with each context", flush=True)
    for folder in sorted(regions, key=lambda f: int(f.split("_")[1])):
        label, pos, word = folder.split("_", 2)
        pos = int(pos)
        slot_contexts = dict(none=[], full_top5=contexts["full_top5"], random5=random_captions)
        for crop in CROPS:
            slot_contexts[crop] = first_captions(regions[folder][crop], 5)
        rows = {name: slot(caps, pos) for name, caps in slot_contexts.items()}
        results["slot"][folder] = dict(label=label, pos=pos, word=word, contexts=slot_contexts, results=rows)
        print(f"{label:12s} {pos:3d} {word:9s} " + " | ".join(
            f"{n}: {r['p_original']:.2f} {[t for t, _ in r['top5'][:3]]}" for n, r in rows.items()), flush=True)

    write_json(os.path.join(rag_dir, "mmada_context.json"), results, indent=1)
    print(f"\nwrote {os.path.join(rag_dir, 'mmada_context.json')}")


if __name__ == "__main__":
    main()
