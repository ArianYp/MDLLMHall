"""Print each unmasking step of baseline MMaDA for chosen AMBER-g images.

Same approach as trace_mmada_steps.py: the step callback only reads the answer
tokens and the logits the step already computed. It does not change which
tokens are chosen. Watched words (by default the objects AMBER annotates as
absent from the image) are flagged at the step they are committed.
"""

import argparse
import json
import os

import torch
from PIL import Image

from mmada_infer import generate_answer, load_mmada
from trace_mmada_steps import MASK_ID, render

ROOT = os.path.dirname(os.path.abspath(__file__))
AMBER_DATA = os.path.join(ROOT, "third_party", "AMBER", "data")
QUERY_FILE = os.path.join(ROOT, "data", "amber", "query_generative.json")
IMAGE_DIR = os.path.join(ROOT, "data", "amber", "images")
PREDICTIONS = os.path.join(ROOT, "results", "amber_g", "mmada_predictions.json")
OUTPUT_DIR = os.path.join(ROOT, "results", "amber_g", "traces")


def read_json(path):
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def normalise(word):
    word = word.strip().lower()
    return word[:-1] if word.endswith("s") and len(word) > 3 else word


def main():
    parser = argparse.ArgumentParser(description="Trace MMaDA unmasking on AMBER-g images.")
    parser.add_argument("--ids", type=int, nargs="+", default=[163])
    parser.add_argument("--watch", nargs="*", default=None,
                        help="Words to flag. Defaults to the AMBER 'hallu' (absent) objects of each image.")
    parser.add_argument("--model", default="Gen-Verse/MMaDA-8B-MixCoT")
    parser.add_argument("--vq-model", default="showlab/magvitv2")
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--steps", type=int, default=64)
    parser.add_argument("--block-length", type=int, default=32)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    queries = {int(item["id"]): item for item in read_json(QUERY_FILE)}
    annotations = read_json(os.path.join(AMBER_DATA, "annotations.json"))
    saved = {}
    if os.path.isfile(PREDICTIONS):
        saved = {int(row["id"]): row["response"] for row in read_json(PREDICTIONS)}

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)
    model, vq_model, uni_prompting, _ = load_mmada(args.model, args.vq_model, device)
    tokenizer = uni_prompting.text_tokenizer
    num_blocks = args.max_new_tokens // args.block_length
    steps_per_block = args.steps // num_blocks
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    for item_id in args.ids:
        item = queries[item_id]
        truth = annotations[item_id - 1]
        watch = args.watch if args.watch is not None else truth["hallu"]
        watch = {normalise(word) for word in watch}
        image_path = os.path.join(IMAGE_DIR, item["image"])
        print("=" * 100)
        print(f"AMBER id {item_id}  image {image_path}")
        print(f"prompt: {item['query']}")
        print(f"present objects (truth): {truth['truth']}")
        print(f"absent objects (hallu):  {truth['hallu']}")
        print(f"watching: {sorted(watch)}")
        print(f"saved prediction: {saved.get(item_id)}")
        print(f"schedule: {args.max_new_tokens} tokens, {num_blocks} blocks x {steps_per_block} steps")
        print("-" * 100)

        steps = []

        def on_step(num_block, step_in_block, token_ids, committed, logits):
            ids = token_ids[0].cpu()
            just = committed[0].cpu()
            probs = torch.softmax(logits[0].float(), dim=-1).cpu()
            step_number = num_block * steps_per_block + step_in_block + 1
            commits = []
            for pos in just.nonzero().flatten().tolist():
                token_id = int(ids[pos])
                text = tokenizer.decode([token_id], skip_special_tokens=False)
                commits.append(dict(
                    pos=pos,
                    token=text,
                    prob=float(probs[pos, token_id]),
                    watched=normalise(text) in watch,
                ))
            masked_left = int((ids == MASK_ID).sum())
            shown = ", ".join(f"[{c['pos']:3d}] {c['token']!r} p={c['prob']:.3f}" for c in commits)
            flag = "  <<< WATCHED OBJECT" if any(c["watched"] for c in commits) else ""
            print(f"step {step_number:02d} (block {num_block}) masked left {masked_left:3d}  committed: {shown}{flag}")
            print(f"        answer: {render(tokenizer, ids)}")
            steps.append(dict(step=step_number, block=num_block, commits=commits,
                              answer=render(tokenizer, ids)))

        answer = generate_answer(
            model,
            vq_model,
            uni_prompting,
            item["query"],
            Image.open(image_path),
            device,
            max_new_tokens=args.max_new_tokens,
            steps=args.steps,
            block_length=args.block_length,
            temperature=args.temperature,
            resolution=512,
            step_callback=on_step,
        )
        print("-" * 100)
        print(f"final decoded answer: {answer.strip()}")
        print(f"matches saved prediction: {saved.get(item_id) == answer.strip()}")
        watched = [(s["step"], c) for s in steps for c in s["commits"] if c["watched"]]
        for step_number, commit in watched:
            print(f"watched word {commit['token']!r} committed at step {step_number} "
                  f"position {commit['pos']} with p={commit['prob']:.3f}")
        out = os.path.join(OUTPUT_DIR, f"trace_amber_{item_id}.json")
        with open(out, "w", encoding="utf-8") as handle:
            json.dump(dict(id=item_id, image=image_path, truth=truth, watch=sorted(watch),
                           saved=saved.get(item_id), final=answer.strip(), steps=steps),
                      handle, ensure_ascii=False, indent=2)
        print(f"saved trace to {out}")


if __name__ == "__main__":
    main()
