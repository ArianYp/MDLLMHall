"""MMaDA decoding with image-contrast remasking, for AMBER-g captions.

The loop follows MMadaModelLM.mmu_generate (semi-autoregressive blocks,
low-confidence commits, greedy). With --mode baseline it reproduces the saved
captions. The other modes act on every token at the step it is committed,
from global step --start-step on:

  rule    if p(token | no image) > p(token | image) on the same pre-commit state
          and the token's late-layer (19-31) attention mass on the image is at
          least --attention-threshold, the token and its neighbours (pos-1, pos+1)
          are masked again and decoded again.
  neighbour  if both neighbours (pos-1, pos+1) were committed at an earlier step
          (the token is filled in between them) and its late-layer attention
          mass on the image is at least --attention-threshold, the token and
          its neighbours are masked again and decoded again. No image ablation.
  zoom    if the committed token's confidence is below --confidence-threshold,
          crop the region it attends to (late-layer map, connected cluster of
          its --zoom-top-k patches around the peak, square, --zoom times the
          cluster) from the original photo, upscale it to the model resolution,
          and re-predict the token and its committed neighbours (pos-1, pos+1)
          with the crop in place of the image. Decoding then continues with
          the full image. Nothing is remasked.
  random  control: each committed token is remasked (with its neighbours) with
          probability --random-rate, regardless of image evidence.

Each position can trigger at most --max-triggers times. Commits per step follow
mmu_generate's block schedule (2 per step for 32 tokens / 16 steps). Masks
added by remasking are filled in extra steps at the same rate, so no step ever
commits a large batch at once. Remasking is allowed for the block's regular
steps plus --max-extra-steps; after that the block finishes without it.
"""

import argparse
import json
import math
import os
import random
from collections import deque

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from mmada_infer import SPECIAL_TOKENS, build_input_ids, image_transform, load_mmada
from trace_amber_image_ablation import prompt_ids
from trace_amber_image_attention import IMAGE_START, AttentionRecorder
from trace_amber_steps import IMAGE_DIR, QUERY_FILE, read_json
from trace_mmada_steps import MASK_ID

ROOT = os.path.dirname(os.path.abspath(__file__))
STUDY = os.path.join(ROOT, "results", "amber_g", "hallu_study")
LATE_LAYERS = range(19, 32)


def write_json(path, payload, indent=None):
    """Write atomically, so a reader never sees a half-written file."""
    temporary = path + ".tmp"
    with open(temporary, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=indent)
    os.replace(temporary, path)


def clean_response(text):
    """Same cleanup as benchmarks/amber/run_mmada_amber_g.py."""
    for token in SPECIAL_TOKENS:
        text = text.replace(token, "")
    return text.strip()


def late_image_mass(recorder, pos):
    return float(torch.stack([recorder.layers[l]["image_mass"][:, pos].mean() for l in LATE_LAYERS]).mean())


def zoom_crop(photo, image_map, resolution, zoom, top_k, min_patches):
    """Square crop of `photo` around the connected cluster of the top_k attended patches that holds the peak.

    The side is at least min_patches patches, so a one-patch cluster is not blown up into a blur.

    image_map is over the model's view: the photo resized (short side = resolution) and centre-cropped.
    The crop may extend outside that view, up to the photo border.
    """
    grid = math.isqrt(image_map.numel())
    m = image_map.float().cpu().numpy()
    top = set(np.argsort(-m)[:top_k].tolist())
    seed = int(np.argmax(m))
    seen, queue = {seed}, deque([seed])
    while queue:
        r, c = divmod(queue.popleft(), grid)
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                n = (r + dr) * grid + (c + dc)
                if 0 <= r + dr < grid and 0 <= c + dc < grid and n in top and n not in seen:
                    seen.add(n)
                    queue.append(n)
    rows = [p // grid for p in seen]
    cols = [p % grid for p in seen]
    patch = resolution / grid
    width, height = photo.size
    scale = resolution / min(width, height)
    left, top_offset = (width * scale - resolution) / 2, (height * scale - resolution) / 2
    x0, x1 = [(v * patch + left) / scale for v in (min(cols), max(cols) + 1)]
    y0, y1 = [(v * patch + top_offset) / scale for v in (min(rows), max(rows) + 1)]
    side = min(max(max(x1 - x0, y1 - y0) * zoom, min_patches * patch / scale), width, height)
    cx = min(max((x0 + x1) / 2, side / 2), width - side / 2)
    cy = min(max((y0 + y1) / 2, side / 2), height - side / 2)
    box = (int(cx - side / 2), int(cy - side / 2), int(cx + side / 2), int(cy + side / 2))
    return photo.crop(box), box


@torch.no_grad()
def decode(model, input_ids, text_ids, recorder, args, rng, eot_id, photo=None, encode_image=None):
    prompt_len = input_ids.shape[1]
    n_answer = args.max_new_tokens
    x = torch.full((1, prompt_len + n_answer), MASK_ID, dtype=torch.long, device=input_ids.device)
    x[:, :prompt_len] = input_ids
    num_blocks = n_answer // args.block_length
    steps_per_block = args.steps // num_blocks
    triggers = np.zeros(n_answer, dtype=int)
    events = []
    n_eligible = 0
    global_step = 0
    for num_block in range(num_blocks):
        end = prompt_len + (num_block + 1) * args.block_length
        # mmu_generate's per-block schedule (get_num_transfer_tokens); extra steps keep the same rate.
        block_masks = int((x[0, end - args.block_length:end] == MASK_ID).sum())
        base, remainder = divmod(block_masks, steps_per_block)
        schedule = [base + (1 if s < remainder else 0) for s in range(steps_per_block)]
        rate = max(1, -(-args.block_length // steps_per_block))
        i = 0
        while True:
            n_masked = int((x[0, prompt_len:end] == MASK_ID).sum())
            if n_masked == 0:
                break
            k = min(n_masked, max(1, schedule[i] if i < steps_per_block else rate))
            global_step += 1
            active = (args.mode != "baseline" and global_step >= args.start_step
                      and i < steps_per_block + args.max_extra_steps)
            recorder.enabled = active and args.mode in ("rule", "neighbour", "zoom")
            logits = model(x).logits
            x0 = torch.argmax(logits, dim=-1)
            p = F.softmax(logits.to(torch.float64), dim=-1)
            x0_p = torch.gather(p, dim=-1, index=x0.unsqueeze(-1)).squeeze(-1)
            x0_p[:, end:] = -np.inf
            mask_index = x == MASK_ID
            x0 = torch.where(mask_index, x0, x)
            confidence = torch.where(mask_index, x0_p, torch.tensor(-np.inf, device=x.device, dtype=x0_p.dtype))
            _, select = torch.topk(confidence[0], k=k)
            before = x.clone()
            x[0, select] = x0[0, select]
            if not active:
                i += 1
                continue

            committed = sorted(int(s) - prompt_len for s in select.tolist())
            eligible = [pos for pos in committed
                        if int(x[0, prompt_len + pos]) != eot_id and triggers[pos] < args.max_triggers]
            n_eligible += len(eligible)
            fire = []
            if args.mode == "rule" and eligible:
                recorder.enabled = False
                no_logits = model(torch.cat([text_ids, before[:, prompt_len:]], dim=1)).logits
                img_lp = torch.log_softmax(logits[0, prompt_len:].float(), dim=-1)
                no_lp = torch.log_softmax(no_logits[0, text_ids.shape[1]:].float(), dim=-1)
                for pos in eligible:
                    token_id = int(x[0, prompt_len + pos])
                    mass = late_image_mass(recorder, pos)
                    lp_img, lp_no = float(img_lp[pos, token_id]), float(no_lp[pos, token_id])
                    if lp_no > lp_img and mass >= args.attention_threshold:
                        fire.append(dict(pos=pos, token_id=token_id, p_image=math.exp(lp_img),
                                         p_no_image=math.exp(lp_no), image_mass=mass))
            elif args.mode == "neighbour" and eligible:
                recorder.enabled = False
                for pos in eligible:
                    # Committed before this step: not a mask in `before`. pos+1 in the next block is still a mask.
                    surrounded = pos >= 1 and pos + 1 < end - prompt_len and all(
                        int(before[0, prompt_len + q]) != MASK_ID for q in (pos - 1, pos + 1))
                    if not surrounded:
                        continue
                    mass = late_image_mass(recorder, pos)
                    if mass >= args.attention_threshold:
                        fire.append(dict(pos=pos, token_id=int(x[0, prompt_len + pos]), image_mass=mass))
            elif args.mode == "zoom" and eligible:
                recorder.enabled = False
                for pos in eligible:
                    confidence = float(x0_p[0, prompt_len + pos])
                    if confidence >= args.confidence_threshold or triggers[pos] >= args.max_triggers:
                        continue
                    image_map = torch.stack([recorder.layers[l]["image_map"][pos] for l in LATE_LAYERS]).mean(0)
                    crop, box = zoom_crop(photo, image_map, args.resolution, args.zoom, args.zoom_top_k,
                                          args.zoom_min_patches)
                    # Second pass: the token and its committed neighbours, refilled one at a time with the crop.
                    span = [q for q in (pos - 1, pos, pos + 1)
                            if 0 <= q < end - prompt_len and int(x[0, prompt_len + q]) != MASK_ID]
                    old = [int(x[0, prompt_len + q]) for q in span]
                    zx = x.clone()
                    zx[0, recorder.image_slice] = encode_image(crop)
                    zx[0, [prompt_len + q for q in span]] = MASK_ID
                    left = list(span)
                    while left:
                        zlogits = model(zx).logits[0, [prompt_len + q for q in left]].float()
                        zlogits[:, MASK_ID] = -np.inf
                        zconf, zarg = F.softmax(zlogits, dim=-1).max(dim=-1)
                        j = int(zconf.argmax())
                        zx[0, prompt_len + left.pop(j)] = zarg[j]
                    new = [int(zx[0, prompt_len + q]) for q in span]
                    for q, token_id in zip(span, new):
                        x[0, prompt_len + q] = token_id
                        triggers[q] += 1
                    events.append(dict(pos=pos, token_id=old[span.index(pos)], confidence=confidence, box=box,
                                       step=global_step, block=num_block, remasked=span,
                                       remasked_tokens=old, new_tokens=new))
            elif args.mode == "random":
                for pos in eligible:
                    if rng.random() < args.random_rate:
                        fire.append(dict(pos=pos, token_id=int(x[0, prompt_len + pos])))
            remask = set()
            for event in fire:
                triggers[event["pos"]] += 1
                span = [q for q in (event["pos"] - 1, event["pos"], event["pos"] + 1)
                        if 0 <= q < end - prompt_len]
                remask.update(span)
                event.update(step=global_step, block=num_block, remasked=span,
                             remasked_tokens=[int(x[0, prompt_len + q]) for q in span])
                events.append(event)
            if remask:
                x[0, [prompt_len + q for q in sorted(remask)]] = MASK_ID
            i += 1
    recorder.enabled = False
    return x[:, prompt_len:], events, n_eligible, global_step


def main():
    parser = argparse.ArgumentParser(description="AMBER-g decoding with image-contrast remasking.")
    parser.add_argument("--mode", choices=["baseline", "rule", "neighbour", "zoom", "random"], required=True)
    parser.add_argument("--ids-files", nargs="+",
                        default=[os.path.join(STUDY, "ids.json"), os.path.join(STUDY, "ids_clean.json")])
    parser.add_argument("--limit", type=int, default=None, help="Only the first N ids (for checks).")
    parser.add_argument("--output", required=True, help="Predictions JSON in the official AMBER format.")
    parser.add_argument("--start-step", type=int, default=15)
    parser.add_argument("--attention-threshold", type=float, default=None,
                        help="Late-layer image attention mass. Defaults to the median in attention_threshold.json.")
    parser.add_argument("--max-triggers", type=int, default=2)
    parser.add_argument("--max-extra-steps", type=int, default=16)
    parser.add_argument("--random-rate", type=float, default=0.0)
    parser.add_argument("--confidence-threshold", type=float, default=0.3, help="zoom: second pass below this.")
    parser.add_argument("--zoom", type=float, default=1.5, help="zoom: crop side as a multiple of the cluster.")
    parser.add_argument("--zoom-top-k", type=int, default=30, help="zoom: attended patches the cluster grows in.")
    parser.add_argument("--zoom-min-patches", type=int, default=8, help="zoom: minimum crop side, in patches.")
    parser.add_argument("--model", default="Gen-Verse/MMaDA-8B-MixCoT")
    parser.add_argument("--vq-model", default="showlab/magvitv2")
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--steps", type=int, default=64)
    parser.add_argument("--block-length", type=int, default=32)
    parser.add_argument("--resolution", type=int, default=512)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    if args.attention_threshold is None:
        args.attention_threshold = read_json(os.path.join(STUDY, "attention_threshold.json"))["late_image_mass_median"]

    ids = [i for path in args.ids_files for i in read_json(path)["ids"]]
    ids = ids[: args.limit] if args.limit else ids
    queries = {int(row["id"]): row for row in read_json(QUERY_FILE)}

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)
    model, vq_model, uni_prompting, _ = load_mmada(args.model, args.vq_model, device)
    tokenizer = uni_prompting.text_tokenizer
    eot_id = tokenizer.convert_tokens_to_ids("<|endoftext|>")
    n_image = (args.resolution // 16) ** 2
    recorder = AttentionRecorder(args.max_new_tokens, slice(IMAGE_START, IMAGE_START + n_image))
    recorder.enabled = False
    for block in [m for m in model.modules() if hasattr(m, "_scaled_dot_product_attention") and hasattr(m, "layer_id")]:
        recorder.wrap(block)

    def encode_image(image):
        """Image tokens exactly as build_input_ids makes them (fp32 VQ, outside the bf16 autocast)."""
        with torch.autocast("cuda", enabled=False):
            pixels = image_transform(image.convert("RGB"), resolution=args.resolution).unsqueeze(0).to(device)
            return vq_model.get_code(pixels)[0] + len(tokenizer)

    events_path = os.path.splitext(args.output)[0] + "_events.json"
    predictions, log = [], []
    if os.path.isfile(args.output) and os.path.isfile(events_path):
        predictions = read_json(args.output)
        log = read_json(events_path)["captions"]
        print(f"resuming: {len(predictions)} captions already in {args.output}", flush=True)
    finished = {int(row["id"]) for row in predictions}
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)

    for done, item_id in enumerate(ids, start=1):
        if item_id in finished:
            continue
        item = queries[item_id]
        image = Image.open(os.path.join(IMAGE_DIR, item["image"]))
        with torch.no_grad():
            input_ids = build_input_ids(vq_model, uni_prompting, item["query"], image, device, args.resolution)
            text_ids = prompt_ids(uni_prompting, item["query"], device)
        rng = random.Random(args.seed * 1_000_003 + item_id)  # per caption, so resuming gives the same draws
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
            answer_ids, events, n_eligible, n_steps = decode(model, input_ids, text_ids, recorder, args, rng, eot_id,
                                                             photo=image, encode_image=encode_image)
        text = tokenizer.batch_decode(answer_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        predictions.append(dict(id=item_id, response=clean_response(text)))
        for event in events:
            event["token"] = tokenizer.decode([event["token_id"]])
            event["remasked_text"] = [tokenizer.decode([t]) for t in event["remasked_tokens"]]
            if "new_tokens" in event:
                event["new_text"] = [tokenizer.decode([t]) for t in event["new_tokens"]]
        log.append(dict(id=item_id, n_triggers=len(events), n_eligible=n_eligible, n_steps=n_steps, events=events))
        write_json(args.output, predictions, indent=2)
        write_json(events_path, dict(settings=vars(args), captions=log))
        print(f"[{done}/{len(ids)}] id {item_id} triggers {len(events)} steps {n_steps}", flush=True)

    total_triggers = sum(c["n_triggers"] for c in log)
    total_eligible = sum(c["n_eligible"] for c in log)
    print(f"triggers {total_triggers} / eligible committed tokens {total_eligible} "
          f"(rate {total_triggers / max(total_eligible, 1):.4f}); mean steps {np.mean([c['n_steps'] for c in log]):.1f}")


if __name__ == "__main__":
    main()
