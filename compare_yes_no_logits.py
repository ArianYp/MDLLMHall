"""Compare Yes/No logits for the same question with and without the image.

The decoding loop is still MMadaModelLM.mmu_generate. The callback only reads
the logits that the existing step already computed; it does not change which
token is committed.
"""

import argparse
import math
import os

import torch
from PIL import Image

from eval_hallusionbench import DEFAULT_IMAGE_ROOT, read_json, resolve_image
from mmada_infer import generate_answer, load_mmada

MASK_ID = 126336
PREDICTIONS = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "results",
    "hallusionbench",
    "mmada_predictions_32.json",
)
# Longer wrong answers from the 128-step trace, not bare yes/no strings.
DEFAULT_INDICES = (406, 585)
POLARITY_STRINGS = ("Yes", "No", "yes", "no", " Yes", " No", " yes", " no")


def single_token_ids(tokenizer, strings):
    found = {}
    for text in strings:
        ids = tokenizer.encode(text, add_special_tokens=False)
        if len(ids) == 1:
            found[ids[0]] = text
    return found


def best_logit(row, token_ids):
    best_id = token_ids[0]
    best_value = float(row[best_id])
    for token_id in token_ids[1:]:
        value = float(row[token_id])
        if value > best_value:
            best_id = token_id
            best_value = value
    return best_value, best_id


def two_way_yes(yes_logit, no_logit):
    # Probability of Yes if the choice were only between these two logits.
    gap = yes_logit - no_logit
    if gap >= 0:
        return 1.0 / (1.0 + math.exp(-gap))
    return math.exp(gap) / (1.0 + math.exp(gap))


def main():
    parser = argparse.ArgumentParser(description="Compare Yes/No logits with and without the image.")
    parser.add_argument("--predictions", default=PREDICTIONS)
    parser.add_argument("--image-root", default=DEFAULT_IMAGE_ROOT)
    parser.add_argument("--indices", type=int, nargs="+", default=list(DEFAULT_INDICES))
    parser.add_argument("--model", default="Gen-Verse/MMaDA-8B-MixCoT")
    parser.add_argument("--vq-model", default="showlab/magvitv2")
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--steps", type=int, default=128)
    parser.add_argument("--block-length", type=int, default=32)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    samples = read_json(args.predictions)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)
    model, vq_model, uni_prompting, _ = load_mmada(args.model, args.vq_model, device)
    tokenizer = uni_prompting.text_tokenizer
    polarity = single_token_ids(tokenizer, POLARITY_STRINGS)
    yes_ids = [token_id for token_id, text in polarity.items() if text.strip().lower() == "yes"]
    no_ids = [token_id for token_id, text in polarity.items() if text.strip().lower() == "no"]
    if not yes_ids or not no_ids:
        raise RuntimeError(f"Could not find single-token Yes/No ids in {polarity}")
    num_blocks = args.max_new_tokens // args.block_length
    steps_per_block = args.steps // num_blocks
    print("Yes/No token ids:")
    for token_id, text in sorted(polarity.items()):
        print(f"  {token_id:6d}  {text!r}  decoded {tokenizer.decode([token_id])!r}")

    for index in args.indices:
        sample = samples[index]
        image_path = resolve_image(sample, args.image_root)
        image = Image.open(image_path) if image_path is not None else None
        print("=" * 80)
        print(f"index {index}  {sample['category']} / {sample['subcategory']}")
        print(f"image: {image_path}  visual_input={sample['visual_input']}")
        print(f"question: {sample['question']}")
        print(f"ground truth ({sample['gt_answer']}): {sample['gt_answer_details']}")

        summaries = {}
        for condition, used_image in (("with image", image), ("without image", None)):
            trace = []

            def on_step(num_block, step_in_block, token_ids, committed, logits, trace=trace):
                step_number = num_block * steps_per_block + step_in_block + 1
                start = num_block * args.block_length
                end = start + args.block_length
                answer = token_ids[0]
                just = committed[0]
                candidate = (answer[start:end] == MASK_ID) | just[start:end]
                if not bool(candidate.any()):
                    return
                block_logits = logits[0, start:end].float()
                yes_logit = torch.stack([block_logits[:, token_id] for token_id in yes_ids]).max(dim=0).values
                no_logit = torch.stack([block_logits[:, token_id] for token_id in no_ids]).max(dim=0).values
                strength = torch.maximum(yes_logit, no_logit)
                strength = torch.where(candidate, strength, torch.full_like(strength, float("-inf")))
                local_pos = int(torch.argmax(strength).item())
                pos = start + local_pos
                row = block_logits[local_pos]
                yes_value, yes_id = best_logit(row, yes_ids)
                no_value, no_id = best_logit(row, no_ids)
                chosen_id = int(torch.argmax(row).item())
                committed_ids = answer[just].tolist()
                committed_text = tokenizer.decode(committed_ids, skip_special_tokens=False).replace("\n", " ")
                polarity_commit = None
                just_positions = just.nonzero().flatten().tolist()
                for commit_pos in just_positions:
                    token_id = int(answer[commit_pos].item())
                    if token_id not in polarity:
                        continue
                    commit_row = logits[0, commit_pos].float()
                    c_yes, _ = best_logit(commit_row, yes_ids)
                    c_no, _ = best_logit(commit_row, no_ids)
                    polarity_commit = (token_id, polarity[token_id], commit_pos, c_yes, c_no)
                    break
                trace.append(
                    dict(
                        step=step_number,
                        pos=pos,
                        yes=yes_value,
                        no=no_value,
                        yes_token=polarity[yes_id],
                        no_token=polarity[no_id],
                        argmax=tokenizer.decode([chosen_id], skip_special_tokens=False).replace("\n", " "),
                        committed=committed_text,
                        polarity_commit=polarity_commit,
                    )
                )

            print("-" * 80)
            print(condition)
            answer = generate_answer(
                model,
                vq_model,
                uni_prompting,
                sample["question"],
                used_image,
                device,
                max_new_tokens=args.max_new_tokens,
                steps=args.steps,
                block_length=args.block_length,
                temperature=args.temperature,
                resolution=512,
                step_callback=on_step,
            )
            commits = [row for row in trace if row["polarity_commit"] is not None]
            first = trace[0]
            print(
                f"step {first['step']:03d} strongest Yes/No position {first['pos']:3d}: "
                f"Yes {first['yes']:8.3f} ({first['yes_token']!r})  "
                f"No {first['no']:8.3f} ({first['no_token']!r})  "
                f"Yes-No {first['yes'] - first['no']:8.3f}  "
                f"P(Yes|Yes,No) {two_way_yes(first['yes'], first['no']):.4f}  "
                f"argmax {first['argmax']!r}"
            )
            if commits:
                row = commits[0]
                token_id, text, pos, yes_value, no_value = row["polarity_commit"]
                print(
                    f"first Yes/No commit at step {row['step']:03d} position {pos}: {text!r}  "
                    f"Yes {yes_value:8.3f}  No {no_value:8.3f}  "
                    f"Yes-No {yes_value - no_value:8.3f}  "
                    f"P(Yes|Yes,No) {two_way_yes(yes_value, no_value):.4f}"
                )
            else:
                print("no Yes or No token was committed")
                yes_value = no_value = None
            print(f"final: {answer.strip()}")
            summaries[condition] = dict(
                first=first,
                commit=commits[0] if commits else None,
                yes=yes_value,
                no=no_value,
                answer=answer.strip(),
            )

        print("-" * 80)
        with_image = summaries["with image"]
        without = summaries["without image"]
        for label, key in (("first step, before any answer token is fixed", "first"),):
            gap_with = with_image[key]["yes"] - with_image[key]["no"]
            gap_without = without[key]["yes"] - without[key]["no"]
            print(label)
            print(f"  with image    Yes-No {gap_with:8.3f}  P(Yes) {two_way_yes(with_image[key]['yes'], with_image[key]['no']):.4f}")
            print(f"  without image Yes-No {gap_without:8.3f}  P(Yes) {two_way_yes(without[key]['yes'], without[key]['no']):.4f}")
            print(f"  change in Yes-No gap (with minus without): {gap_with - gap_without:8.3f}")
        if with_image["commit"] and without["commit"]:
            yes_shift = with_image["yes"] - without["yes"]
            no_shift = with_image["no"] - without["no"]
            print("at the step each run commits its Yes or No token")
            print(f"  Yes logit change (with minus without): {yes_shift:8.3f}")
            print(f"  No logit change (with minus without):  {no_shift:8.3f}")
        print(f"with image answer:    {with_image['answer']}")
        print(f"without image answer: {without['answer']}")


if __name__ == "__main__":
    main()
