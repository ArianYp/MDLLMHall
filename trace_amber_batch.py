"""Image-removal logits and image attention for every answer token, over many AMBER-g captions.

One with-image decoding pass per caption (MMadaModelLM.mmu_generate, unchanged
choices). At each step, for the tokens committed in that step, it keeps:
  - attention from the forward pass that produced them (trace_amber_image_attention.AttentionRecorder)
  - probabilities with the image and from a read-only forward pass without the image tokens
    on the same pre-commit answer state (as in trace_amber_image_ablation.py)
The recorder is switched off during the no-image pass so it cannot overwrite the
with-image attention. Labels (grounded / hallucinated nouns) are added later by
analyze_hallu_study.py so they can change without regenerating.
"""

import argparse
import json
import math
import os

import numpy as np
import torch
from PIL import Image

from mmada_infer import generate_answer, image_transform, load_mmada
from trace_amber_image_ablation import prompt_ids
from trace_amber_image_attention import IMAGE_START, AttentionRecorder
from trace_amber_steps import IMAGE_DIR, PREDICTIONS, QUERY_FILE, read_json
from trace_mmada_steps import MASK_ID

ROOT = os.path.dirname(os.path.abspath(__file__))
DEFAULT_IDS = os.path.join(ROOT, "results", "amber_g", "hallu_study", "ids.json")
DEFAULT_OUT = os.path.join(ROOT, "results", "amber_g", "hallu_study", "traces")
LATE_LAYERS = slice(19, 32)


def token_offsets(tokenizer, ids):
    """Character span of each answer token in the decoded answer (skip_special_tokens=True)."""
    offsets = []
    previous = 0
    for i in range(len(ids)):
        text = tokenizer.decode(ids[: i + 1], skip_special_tokens=True, clean_up_tokenization_spaces=False)
        offsets.append((previous, len(text)))
        previous = len(text)
    return offsets


def main():
    parser = argparse.ArgumentParser(description="Batch image-removal + attention traces on AMBER-g.")
    parser.add_argument("--ids-file", default=DEFAULT_IDS)
    parser.add_argument("--output-dir", default=DEFAULT_OUT)
    parser.add_argument("--model", default="Gen-Verse/MMaDA-8B-MixCoT")
    parser.add_argument("--vq-model", default="showlab/magvitv2")
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--steps", type=int, default=64)
    parser.add_argument("--block-length", type=int, default=32)
    parser.add_argument("--resolution", type=int, default=512)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    ids = read_json(args.ids_file)["ids"]
    queries = {int(row["id"]): row for row in read_json(QUERY_FILE)}
    saved = {int(row["id"]): row["response"] for row in read_json(PREDICTIONS)}
    os.makedirs(args.output_dir, exist_ok=True)
    pending = [i for i in ids if args.overwrite or not os.path.isfile(os.path.join(args.output_dir, f"{i}.json"))]
    print(f"{len(ids)} ids, {len(pending)} to run", flush=True)
    if not pending:
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)
    model, vq_model, uni_prompting, _ = load_mmada(args.model, args.vq_model, device)
    tokenizer = uni_prompting.text_tokenizer
    steps_per_block = args.steps // (args.max_new_tokens // args.block_length)
    n_image = (args.resolution // 16) ** 2
    recorder = AttentionRecorder(args.max_new_tokens, slice(IMAGE_START, IMAGE_START + n_image))
    for block in [m for m in model.modules() if hasattr(m, "_scaled_dot_product_attention") and hasattr(m, "layer_id")]:
        recorder.wrap(block)

    for done, item_id in enumerate(pending, start=1):
        item = queries[item_id]
        image = Image.open(os.path.join(IMAGE_DIR, item["image"]))
        with torch.no_grad():
            pixels = image_transform(image.convert("RGB"), resolution=args.resolution).unsqueeze(0).to(device)
            if int(vq_model.get_code(pixels).shape[1]) != n_image:
                raise RuntimeError("Unexpected number of image tokens.")
        text_ids = prompt_ids(uni_prompting, item["query"], device)
        records = {}
        maps = np.zeros((args.max_new_tokens, n_image), dtype=np.float16)

        def on_step(num_block, step_in_block, token_ids, committed, logits):
            step_number = num_block * steps_per_block + step_in_block + 1
            positions = committed[0].nonzero().flatten().tolist()
            attention = {pos: recorder.token(pos) for pos in positions}
            before = token_ids.clone()
            before[committed] = MASK_ID
            recorder.enabled = False
            no_logits = model(torch.cat([text_ids, before], dim=1)).logits[0, text_ids.shape[1]:].float()
            recorder.enabled = True
            img_logits = logits[0].float()
            img_lp = torch.log_softmax(img_logits, dim=-1)
            no_lp = torch.log_softmax(no_logits, dim=-1)
            for pos in positions:
                token_id = int(token_ids[0, pos])
                a = attention[pos]
                late = a["image_map"][LATE_LAYERS].mean(0).float()
                late = late / late.sum()
                maps[pos] = late.numpy().astype(np.float16)
                no_top = int(no_logits[pos].argmax())
                records[pos] = dict(
                    pos=pos,
                    step=step_number,
                    block=num_block,
                    token_id=token_id,
                    logp_image=float(img_lp[pos, token_id]),
                    logp_no_image=float(no_lp[pos, token_id]),
                    rank_no_image=int((no_logits[pos] > no_logits[pos, token_id]).sum()) + 1,
                    no_image_argmax=no_top,
                    kl_image_vs_no_image=float((img_lp[pos].exp() * (img_lp[pos] - no_lp[pos])).sum()),
                    entropy_image=float(-(img_lp[pos].exp() * img_lp[pos]).sum()),
                    image_mass_by_layer=a["image_mass"].mean(1).tolist(),
                    prompt_mass_by_layer=a["prompt_mass"].mean(1).tolist(),
                    answer_mass_by_layer=a["answer_mass"].mean(1).tolist(),
                    image_entropy_by_layer=a["image_entropy"].mean(1).tolist(),
                    image_entropy_min_head_by_layer=a["image_entropy"].min(1).values.tolist(),
                    late_top5_share=float(late.sort().values[-5:].sum()),
                    late_top_patch=int(late.argmax()),
                )

        answer_raw = generate_answer(model, vq_model, uni_prompting, item["query"], image, device,
                                     max_new_tokens=args.max_new_tokens, steps=args.steps,
                                     block_length=args.block_length, temperature=0.0,
                                     resolution=args.resolution, step_callback=on_step)
        token_ids = [records[p]["token_id"] for p in range(args.max_new_tokens)]
        payload = dict(
            id=item_id,
            image=item["image"],
            answer_raw=answer_raw,
            answer=answer_raw.strip(),
            saved=saved.get(item_id),
            matches_saved=answer_raw.strip() == saved.get(item_id),
            tokens=[tokenizer.decode([t], skip_special_tokens=False) for t in token_ids],
            offsets=token_offsets(tokenizer, token_ids),
            records=[records[p] for p in range(args.max_new_tokens)],
            settings=dict(max_new_tokens=args.max_new_tokens, steps=args.steps, block_length=args.block_length,
                          resolution=args.resolution, late_layers=[LATE_LAYERS.start, LATE_LAYERS.stop - 1]),
        )
        np.savez_compressed(os.path.join(args.output_dir, f"{item_id}.npz"), late_map=maps)
        with open(os.path.join(args.output_dir, f"{item_id}.json"), "w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False)
        print(f"[{done}/{len(pending)}] id {item_id} matches_saved={payload['matches_saved']}", flush=True)


if __name__ == "__main__":
    main()
