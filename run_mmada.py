"""Ask MMaDA about an image and print the model's reply.

Example:
    python run_mmada.py --image photo.jpg --prompt "What is in this image?"
"""

import argparse

import torch

from mmada_infer import generate_answer, load_mmada, validate_schedule


def main():
    parser = argparse.ArgumentParser(description="Run MMaDA on one prompt and one image.")
    parser.add_argument("--image", required=True, help="Path to an RGB image.")
    parser.add_argument("--prompt", required=True, help="Question or instruction about the image.")
    parser.add_argument("--model", default="Gen-Verse/MMaDA-8B-MixCoT")
    parser.add_argument("--vq-model", default="showlab/magvitv2")
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--block-length", type=int, default=None)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--resolution", type=int, default=512)
    args = parser.parse_args()

    max_new_tokens = args.max_new_tokens
    block_length = args.block_length if args.block_length is not None else max(1, max_new_tokens // 4)
    steps = args.steps if args.steps is not None else max(1, max_new_tokens // 2)
    validate_schedule(max_new_tokens, steps, block_length)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, vq_model, uni_prompting, _ = load_mmada(args.model, args.vq_model, device)
    from PIL import Image

    response = generate_answer(
        model,
        vq_model,
        uni_prompting,
        args.prompt,
        Image.open(args.image),
        device,
        max_new_tokens=max_new_tokens,
        steps=steps,
        block_length=block_length,
        temperature=args.temperature,
        resolution=args.resolution,
    )
    print(f"Prompt: {args.prompt}")
    print(f"Image: {args.image}")
    print("Response:")
    print(response)


if __name__ == "__main__":
    main()
