"""Print each unmasking step of baseline MMaDA for chosen HallusionBench samples.

This uses MMadaModelLM.mmu_generate unchanged, except for an optional callback
that reads the answer tokens after each commitment. It does not change which
tokens are chosen.
"""

import argparse
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
# Keyword-mismatch misses from the 32-token run, both with an image.
# VS: model says China/Hong Kong leads gold imports; the reference says Switzerland.
# VD: model says the orange circles differ in size; the reference says they match.
DEFAULT_INDICES = (4, 538)


def render(tokenizer, token_ids):
    parts = []
    for token_id in token_ids.tolist():
        if int(token_id) == MASK_ID:
            parts.append("_")
        else:
            parts.append(tokenizer.decode([int(token_id)], skip_special_tokens=False))
    return "".join(parts).replace("\n", " ")


def main():
    parser = argparse.ArgumentParser(description="Trace MMaDA unmasking on saved HallusionBench rows.")
    parser.add_argument("--predictions", default=PREDICTIONS)
    parser.add_argument("--image-root", default=DEFAULT_IMAGE_ROOT)
    parser.add_argument("--indices", type=int, nargs="+", default=list(DEFAULT_INDICES))
    parser.add_argument("--model", default="Gen-Verse/MMaDA-8B-MixCoT")
    parser.add_argument("--vq-model", default="showlab/magvitv2")
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--steps", type=int, default=32)
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
    num_blocks = args.max_new_tokens // args.block_length
    steps_per_block = args.steps // num_blocks

    for index in args.indices:
        sample = samples[index]
        image_path = resolve_image(sample, args.image_root)
        image = Image.open(image_path) if image_path is not None else None
        print("=" * 80)
        print(f"index {index}  {sample['category']} / {sample['subcategory']}")
        print(f"image: {image_path or 'none'}  visual_input={sample['visual_input']}")
        print(f"question: {sample['question']}")
        print(f"ground truth ({sample['gt_answer']}): {sample['gt_answer_details']}")
        print(f"saved prediction: {sample.get('model_prediction')}")
        print("-" * 80)

        def on_step(num_block, step_in_block, token_ids, committed, tokenizer=tokenizer):
            ids = token_ids[0].cpu()
            new_ids = ids[committed[0].cpu()]
            new_text = tokenizer.decode(new_ids.tolist(), skip_special_tokens=False).replace("\n", " ")
            step_number = num_block * steps_per_block + step_in_block + 1
            print(f"step {step_number:02d}  committed: {new_text!r}")
            print(f"         answer: {render(tokenizer, ids)}")

        answer = generate_answer(
            model,
            vq_model,
            uni_prompting,
            sample["question"],
            image,
            device,
            max_new_tokens=args.max_new_tokens,
            steps=args.steps,
            block_length=args.block_length,
            temperature=args.temperature,
            resolution=512,
            step_callback=on_step,
        )
        print("-" * 80)
        print(f"final decoded answer: {answer}")


if __name__ == "__main__":
    main()
