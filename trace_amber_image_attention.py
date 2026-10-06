"""Measure how much each answer token attends to the image, and how focused it is.

Decoding is the normal with-image MMadaModelLM.mmu_generate. Each attention
layer's _scaled_dot_product_attention is wrapped: it still returns the original
SDPA output, and on the side recomputes softmax(q k^T / sqrt(d)) for the answer
rows (queries) from the same post-RoPE q and k. Nothing that decoding uses is
changed.

For every answer token we keep the attention from the forward pass that
produced its committed prediction (the position was still [MASK] in that pass):
  image_mass     share of attention on the image tokens
  prompt_mass    share on the text prompt (chat template + question)
  answer_mass    share on the answer region (other generated tokens and masks)
  image_entropy  entropy of attention renormalised over image tokens / log(n_image)
                 (0 = one patch, 1 = uniform over the image)
  image_map      head-averaged attention over the image grid, per layer
"""

import argparse
import json
import math
import os

import numpy as np
import torch
from PIL import Image

from mmada_infer import generate_answer, image_transform, load_mmada
from trace_amber_steps import AMBER_DATA, IMAGE_DIR, OUTPUT_DIR, PREDICTIONS, QUERY_FILE, normalise, read_json

IMAGE_START = 2  # <|mmu|> <|soi|> come first


class AttentionRecorder:
    def __init__(self, n_answer, image_slice):
        self.n_answer = n_answer
        self.image_slice = image_slice
        self.layers = {}
        self.enabled = True

    def wrap(self, block):
        original = block._scaled_dot_product_attention
        layer_id = block.layer_id

        def wrapped(q, k, v, attn_mask=None, dropout_p=0.0, is_causal=False):
            out = original(q, k, v, attn_mask=attn_mask, dropout_p=dropout_p, is_causal=is_causal)
            if self.enabled:
                self.record(layer_id, q, k, attn_mask)
            return out

        block._scaled_dot_product_attention = wrapped

    @torch.no_grad()
    def record(self, layer_id, q, k, attn_mask):
        if k.size(1) != q.size(1):
            k = k.repeat_interleave(q.size(1) // k.size(1), dim=1)
        q_answer = q[0, :, -self.n_answer:, :].float()           # heads, answer, d
        scores = q_answer @ k[0].float().transpose(-1, -2) / math.sqrt(q.size(-1))
        if attn_mask is not None:
            scores = scores + attn_mask[0, :, -self.n_answer:, :].float()
        attn = torch.softmax(scores, dim=-1)                     # heads, answer, keys
        image = attn[..., self.image_slice]
        image_mass = image.sum(-1)                               # heads, answer
        p = image / image_mass.unsqueeze(-1).clamp_min(1e-12)
        entropy = -(p * p.clamp_min(1e-12).log()).sum(-1) / math.log(image.size(-1))
        self.layers[layer_id] = dict(
            image_mass=image_mass,
            prompt_mass=attn[..., self.image_slice.stop:-self.n_answer].sum(-1),
            answer_mass=attn[..., -self.n_answer:].sum(-1),
            image_entropy=entropy,
            image_map=image.mean(0),                             # answer, n_image
        )

    def token(self, pos):
        layer_ids = sorted(self.layers)
        stack = lambda key: torch.stack([self.layers[l][key][:, pos] for l in layer_ids]).cpu()  # layers, heads
        return dict(
            image_mass=stack("image_mass"),
            prompt_mass=stack("prompt_mass"),
            answer_mass=stack("answer_mass"),
            image_entropy=stack("image_entropy"),
            image_map=torch.stack([self.layers[l]["image_map"][pos] for l in layer_ids]).cpu(),
        )


def main():
    parser = argparse.ArgumentParser(description="Per-token attention to the image for an AMBER-g caption.")
    parser.add_argument("--id", type=int, default=163)
    parser.add_argument("--model", default="Gen-Verse/MMaDA-8B-MixCoT")
    parser.add_argument("--vq-model", default="showlab/magvitv2")
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--steps", type=int, default=64)
    parser.add_argument("--block-length", type=int, default=32)
    parser.add_argument("--resolution", type=int, default=512)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    item = {int(row["id"]): row for row in read_json(QUERY_FILE)}[args.id]
    truth = read_json(os.path.join(AMBER_DATA, "annotations.json"))[args.id - 1]
    saved = {int(row["id"]): row["response"] for row in read_json(PREDICTIONS)}.get(args.id)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)
    model, vq_model, uni_prompting, _ = load_mmada(args.model, args.vq_model, device)
    tokenizer = uni_prompting.text_tokenizer
    steps_per_block = args.steps // (args.max_new_tokens // args.block_length)
    image = Image.open(os.path.join(IMAGE_DIR, item["image"]))

    with torch.no_grad():
        pixels = image_transform(image.convert("RGB"), resolution=args.resolution).unsqueeze(0).to(device)
        n_image = int(vq_model.get_code(pixels).shape[1])
    grid = int(round(math.sqrt(n_image)))
    image_slice = slice(IMAGE_START, IMAGE_START + n_image)

    recorder = AttentionRecorder(args.max_new_tokens, image_slice)
    blocks = [m for m in model.modules() if hasattr(m, "_scaled_dot_product_attention") and hasattr(m, "layer_id")]
    for block in blocks:
        recorder.wrap(block)

    records = {}

    def on_step(num_block, step_in_block, token_ids, committed):
        step_number = num_block * steps_per_block + step_in_block + 1
        for pos in committed[0].nonzero().flatten().tolist():
            token_id = int(token_ids[0, pos])
            records[pos] = dict(step=step_number, token_id=token_id,
                                token=tokenizer.decode([token_id], skip_special_tokens=False),
                                **recorder.token(pos))

    answer = generate_answer(model, vq_model, uni_prompting, item["query"], image, device,
                             max_new_tokens=args.max_new_tokens, steps=args.steps,
                             block_length=args.block_length, temperature=0.0,
                             resolution=args.resolution, step_callback=on_step).strip()

    positions = [pos for pos in sorted(records) if records[pos]["token"] != "<|endoftext|>"]
    hallu = {normalise(w) for w in truth["hallu"]}
    print("=" * 112)
    print(f"AMBER id {args.id}  layers {len(blocks)}  image tokens {n_image} ({grid}x{grid})")
    print(f"answer: {answer}")
    print(f"matches saved prediction: {answer == saved}")
    print("Values are means over all layers and heads, at the step the token was committed.")
    print("img_ent: 0 = one patch, 1 = uniform over the image.  eff_patches = exp(entropy).")
    print("-" * 112)
    print(f"{'pos':>4} {'token':<14} {'step':>4} {'img_mass':>8} {'prompt':>7} {'answer':>7} "
          f"{'img_ent':>7} {'eff_patches':>11} {'top patch (row,col) / share':>28}")
    summary = []
    for pos in positions:
        r = records[pos]
        image_mass = r["image_mass"].mean().item()
        entropy = r["image_entropy"].mean().item()
        head_map = r["image_map"].mean(0)
        head_map = head_map / head_map.sum()
        top = int(head_map.argmax())
        summary.append(dict(pos=pos, token=r["token"], step=r["step"], image_mass=image_mass,
                            prompt_mass=r["prompt_mass"].mean().item(),
                            answer_mass=r["answer_mass"].mean().item(),
                            image_entropy=entropy, top_patch=[top // grid, top % grid],
                            top_patch_share=float(head_map[top]),
                            image_mass_by_layer=r["image_mass"].mean(1).tolist(),
                            image_entropy_by_layer=r["image_entropy"].mean(1).tolist()))
        tag = " HALLU" if normalise(r["token"]) in hallu else ""
        print(f"{pos:>4} {r['token']!r:<14} {r['step']:>4} {image_mass:8.3f} {summary[-1]['prompt_mass']:7.3f} "
              f"{summary[-1]['answer_mass']:7.3f} {entropy:7.3f} {math.exp(entropy * math.log(n_image)):11.1f} "
              f"{str((top // grid, top % grid)):>14} {float(head_map[top]):.3f}{tag}")

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    out_json = os.path.join(OUTPUT_DIR, f"image_attention_amber_{args.id}.json")
    with open(out_json, "w", encoding="utf-8") as handle:
        json.dump(dict(id=args.id, answer=answer, n_image=n_image, grid=grid, tokens=summary), handle,
                  ensure_ascii=False)
    out_npz = os.path.join(OUTPUT_DIR, f"image_attention_amber_{args.id}.npz")
    np.savez_compressed(
        out_npz,
        positions=np.array(positions),
        image_mass=np.stack([records[p]["image_mass"].numpy() for p in positions]),        # tok, layer, head
        image_entropy=np.stack([records[p]["image_entropy"].numpy() for p in positions]),  # tok, layer, head
        image_map=np.stack([records[p]["image_map"].numpy() for p in positions]).astype(np.float16),  # tok, layer, n_image
    )
    print(f"saved {out_json}\nsaved {out_npz}")


if __name__ == "__main__":
    main()
