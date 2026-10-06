"""Compare MMaDA logits with and without the image along the with-image trajectory.

Decoding follows the normal with-image path (MMadaModelLM.mmu_generate). At
every step the callback takes the partial answer that step saw (before its
commits) and runs one extra forward pass with the image tokens removed, the
same text-only input generate_answer builds when image is None. That pass is
read only; it never changes which token is committed.

Pass 1 generates the with-image answer to learn the final token at every
position. Pass 2 repeats the same trajectory and records, at every step and
position, the logit / probability of that final token under both conditions.
"""

import argparse
import json
import os

import torch
from PIL import Image

from mmada_infer import generate_answer, load_mmada
from trace_amber_steps import (
    AMBER_DATA, IMAGE_DIR, OUTPUT_DIR, PREDICTIONS, QUERY_FILE, normalise, read_json,
)
from trace_mmada_steps import MASK_ID


def prompt_ids(uni_prompting, prompt, device):
    return uni_prompting.text_tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt",
    ).to(device)


def stats(logit_row, logprob_row, token_id):
    value = logit_row[token_id]
    return dict(
        logit=float(value),
        logprob=float(logprob_row[token_id]),
        prob=float(logprob_row[token_id].exp()),
        rank=int((logit_row > value).sum()) + 1,
    )


def main():
    parser = argparse.ArgumentParser(description="With-image vs without-image logits on the with-image trajectory.")
    parser.add_argument("--id", type=int, default=163)
    parser.add_argument("--watch", nargs="*", default=None,
                        help="Hallucinated words. Defaults to the AMBER 'hallu' objects of the image.")
    parser.add_argument("--contrast", nargs="*", default=None,
                        help="Grounded words to compare against. Defaults to the AMBER 'truth' objects.")
    parser.add_argument("--model", default="Gen-Verse/MMaDA-8B-MixCoT")
    parser.add_argument("--vq-model", default="showlab/magvitv2")
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--steps", type=int, default=64)
    parser.add_argument("--block-length", type=int, default=32)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    item = {int(row["id"]): row for row in read_json(QUERY_FILE)}[args.id]
    truth = read_json(os.path.join(AMBER_DATA, "annotations.json"))[args.id - 1]
    saved = {int(row["id"]): row["response"] for row in read_json(PREDICTIONS)}.get(args.id)
    watch = {normalise(w) for w in (args.watch if args.watch is not None else truth["hallu"])}
    contrast = {normalise(w) for w in (args.contrast if args.contrast is not None else truth["truth"])}

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)
    model, vq_model, uni_prompting, _ = load_mmada(args.model, args.vq_model, device)
    tokenizer = uni_prompting.text_tokenizer
    steps_per_block = args.steps // (args.max_new_tokens // args.block_length)
    image = Image.open(os.path.join(IMAGE_DIR, item["image"]))
    text_ids = prompt_ids(uni_prompting, item["query"], device)
    run = dict(max_new_tokens=args.max_new_tokens, steps=args.steps,
               block_length=args.block_length, temperature=0.0, resolution=512)

    # Pass 1: final with-image tokens.
    final = {}

    def keep_final(num_block, step_in_block, token_ids, committed):
        final["ids"] = token_ids[0].clone()

    generate_answer(model, vq_model, uni_prompting, item["query"], image, device,
                    step_callback=keep_final, **run)
    final_ids = final["ids"]
    tokens = [tokenizer.decode([int(t)], skip_special_tokens=False) for t in final_ids.tolist()]
    positions = range(len(tokens))

    # Pass 2: same trajectory, plus a read-only forward pass without the image.
    steps = []

    def on_step(num_block, step_in_block, token_ids, committed, logits):
        before = token_ids.clone()
        before[committed] = MASK_ID
        no_image = model(torch.cat([text_ids, before], dim=1)).logits[:, text_ids.shape[1]:]
        img_logits = logits[0].float()
        no_logits = no_image[0].float()
        img_lp = torch.log_softmax(img_logits, dim=-1)
        no_lp = torch.log_softmax(no_logits, dim=-1)
        step_number = num_block * steps_per_block + step_in_block + 1
        masked = (before[0] == MASK_ID)

        commits = []
        for pos in committed[0].nonzero().flatten().tolist():
            token_id = int(token_ids[0, pos])
            no_top = int(no_logits[pos].argmax())
            kl = float((img_lp[pos].exp() * (img_lp[pos] - no_lp[pos])).sum())
            commits.append(dict(
                pos=pos,
                token=tokens[pos],
                image=stats(img_logits[pos], img_lp[pos], token_id),
                no_image=stats(no_logits[pos], no_lp[pos], token_id),
                no_image_argmax=tokenizer.decode([no_top], skip_special_tokens=False),
                no_image_argmax_prob=float(no_lp[pos, no_top].exp()),
                kl_image_vs_no_image=kl,
            ))

        final_gather = final_ids.to(img_lp.device).unsqueeze(-1)
        steps.append(dict(
            step=step_number,
            block=num_block,
            masked=masked.cpu().tolist(),
            final_prob_image=img_lp.gather(-1, final_gather).squeeze(-1).exp().cpu().tolist(),
            final_prob_no_image=no_lp.gather(-1, final_gather).squeeze(-1).exp().cpu().tolist(),
            final_logit_image=img_logits.gather(-1, final_gather).squeeze(-1).cpu().tolist(),
            final_logit_no_image=no_logits.gather(-1, final_gather).squeeze(-1).cpu().tolist(),
            argmax_agree=(img_logits.argmax(-1) == no_logits.argmax(-1)).cpu().tolist(),
            commits=commits,
        ))

    answer = generate_answer(model, vq_model, uni_prompting, item["query"], image, device,
                             step_callback=on_step, **run).strip()

    print("=" * 110)
    print(f"AMBER id {args.id}  truth {truth['truth']}  absent {truth['hallu']}")
    print(f"answer: {answer}")
    print(f"matches saved prediction: {answer == saved}")
    print("-" * 110)
    print(f"{'step':>4} {'pos':>4} {'token':<16} {'p_img':>7} {'p_noimg':>8} {'rank_no':>7} "
          f"{'dlogp':>7} {'KL':>6}  no-image argmax")
    for step in steps:
        for c in step["commits"]:
            if c["token"] == "<|endoftext|>":
                continue
            word = normalise(c["token"])
            tag = " HALLU" if word in watch else (" TRUE" if word in contrast else "")
            dlogp = c["image"]["logprob"] - c["no_image"]["logprob"]
            print(f"{step['step']:>4} {c['pos']:>4} {c['token']!r:<16} {c['image']['prob']:7.3f} "
                  f"{c['no_image']['prob']:8.3f} {c['no_image']['rank']:7d} {dlogp:7.2f} "
                  f"{c['kl_image_vs_no_image']:6.2f}  {c['no_image_argmax']!r} ({c['no_image_argmax_prob']:.2f}){tag}")

    print("-" * 110)
    print("Probability of the final token at its position, step by step until it is committed (image / no image)")
    commit_step = {c["pos"]: s["step"] for s in steps for c in s["commits"]}
    for pos in positions:
        word = normalise(tokens[pos])
        if word not in watch and word not in contrast:
            continue
        tag = "HALLU" if word in watch else "TRUE"
        trail = [
            f"s{s['step']:02d} {s['final_prob_image'][pos]:.2f}/{s['final_prob_no_image'][pos]:.2f}"
            for s in steps if s["step"] <= commit_step[pos]
        ]
        print(f"[{pos:3d}] {tokens[pos]!r} {tag} committed step {commit_step[pos]}:")
        print("   " + "  ".join(trail))

    out = os.path.join(OUTPUT_DIR, f"image_ablation_amber_{args.id}.json")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    with open(out, "w", encoding="utf-8") as handle:
        json.dump(dict(id=args.id, truth=truth, watch=sorted(watch), contrast=sorted(contrast),
                       answer=answer, saved=saved, tokens=tokens, steps=steps),
                  handle, ensure_ascii=False)
    print(f"saved to {out}")


if __name__ == "__main__":
    main()
