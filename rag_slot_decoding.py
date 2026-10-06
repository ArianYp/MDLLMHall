"""Retrieval-context second pass inside MMaDA decoding, for chosen words of one AMBER-g caption.

Decoding follows the baseline schedule (as remask_decoding.decode). At the step a
watched token is committed (by default the baseline caption's hallucinated nouns),
its tight crop is taken from the live late-layer attention (rag_retrieve.CROPS["tight"]),
embedded with CLIP and matched against COCO train. The token and its committed
neighbours (pos-1, pos+1) are masked again and refilled one at a time, most confident
first, with the top-k retrieved captions placed before the query
(rag_mmada_context.TEMPLATE). Decoding then continues with the normal prompt.
Each position triggers at most once. --context random uses k random COCO captions
instead, as a control. --context zoom puts the tight crop (upscaled to the model
resolution) in place of the photo for the refill, with the normal prompt, as
remask_decoding --mode zoom does.

Example:
    python rag_slot_decoding.py --id 163 --contexts tight random
"""

import argparse
import json
import os
import random

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from transformers import CLIPModel, CLIPProcessor

from amber_noun_labels import AmberLabeler
from mmada_infer import build_input_ids, image_transform, load_mmada
from rag_mmada_context import with_context
from rag_retrieve import CLIP_ID, COCO, CROPS, EMBEDDINGS, embed
from remask_decoding import LATE_LAYERS, STUDY, clean_response, write_json, zoom_crop
from trace_amber_image_ablation import prompt_ids
from trace_amber_image_attention import IMAGE_START, AttentionRecorder
from trace_amber_steps import IMAGE_DIR, QUERY_FILE, read_json
from trace_mmada_steps import MASK_ID


@torch.no_grad()
def decode(model, input_ids, recorder, args, fire, eot_id, second_pass, record_attention=True, state=None):
    """Baseline schedule. `fire(pos, token_id, step, image_map)` picks committed tokens to act on;
    `second_pass(x, prompt_len, pos, span, image_map)` refills `span` in x and returns the event.
    If `state` (a dict) is given, it holds the live sequence as state["x"] and state["prompt_len"], so `fire` can read the caption so far."""
    prompt_len = input_ids.shape[1]
    n_answer = args.max_new_tokens
    x = torch.full((1, prompt_len + n_answer), MASK_ID, dtype=torch.long, device=input_ids.device)
    x[:, :prompt_len] = input_ids
    if state is not None:
        state.update(x=x, prompt_len=prompt_len)
    num_blocks = n_answer // args.block_length
    steps_per_block = args.steps // num_blocks
    triggered = set()
    events = []
    global_step = 0
    for num_block in range(num_blocks):
        end = prompt_len + (num_block + 1) * args.block_length
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
            recorder.enabled = record_attention
            logits = model(x).logits
            recorder.enabled = False
            x0 = torch.argmax(logits, dim=-1)
            p = F.softmax(logits.to(torch.float64), dim=-1)
            x0_p = torch.gather(p, dim=-1, index=x0.unsqueeze(-1)).squeeze(-1)
            x0_p[:, end:] = -np.inf
            mask_index = x == MASK_ID
            x0 = torch.where(mask_index, x0, x)
            confidence = torch.where(mask_index, x0_p, torch.tensor(-np.inf, device=x.device, dtype=x0_p.dtype))
            _, select = torch.topk(confidence[0], k=k)
            x[0, select] = x0[0, select]
            i += 1
            for pos in sorted(int(s) - prompt_len for s in select.tolist()):
                token_id = int(x[0, prompt_len + pos])
                if token_id == eot_id or pos in triggered:
                    continue
                image_map = (torch.stack([recorder.layers[l]["image_map"][pos] for l in LATE_LAYERS]).mean(0)
                             if record_attention else None)
                if not fire(pos, token_id, global_step, image_map):
                    continue
                triggered.add(pos)
                radius = getattr(args, "span_radius", 1)
                span = [q for q in range(pos - radius, pos + radius + 1)
                        if 0 <= q < end - prompt_len and int(x[0, prompt_len + q]) != MASK_ID]
                event = second_pass(x, prompt_len, pos, span, image_map)
                event.update(step=global_step, block=num_block, p_commit=float(x0_p[0, prompt_len + pos]))
                events.append(event)
    return x[:, prompt_len:], events


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--id", type=int, default=163)
    parser.add_argument("--traces", default=os.path.join(STUDY, "traces"))
    parser.add_argument("--nouns", default=os.path.join(STUDY, "with_clean", "nouns.csv"))
    parser.add_argument("--watch", nargs="*", default=None, help="Words to act on. Defaults to the baseline's hallucinated nouns.")
    parser.add_argument("--contexts", nargs="+", default=["tight", "random"], choices=["tight", "random", "zoom"])
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--zoom-top-k", type=int, default=30)
    parser.add_argument("--model", default="Gen-Verse/MMaDA-8B-MixCoT")
    parser.add_argument("--vq-model", default="showlab/magvitv2")
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--steps", type=int, default=64)
    parser.add_argument("--block-length", type=int, default=32)
    parser.add_argument("--resolution", type=int, default=512)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args()
    out_dir = args.output_dir or os.path.join(STUDY, "rag", str(args.id), "rag_slot")
    os.makedirs(out_dir, exist_ok=True)
    device = torch.device("cuda")
    torch.manual_seed(args.seed)

    item = {int(r["id"]): r for r in read_json(QUERY_FILE)}[args.id]
    trace = read_json(os.path.join(args.traces, f"{args.id}.json"))
    if args.watch is None:
        import csv
        with open(args.nouns, encoding="utf-8") as handle:
            rows = [r for r in csv.DictReader(handle) if int(r["id"]) == args.id and r["label"] == "hallucinated"]
        watch = {trace["records"][int(r["pos"])]["token_id"] for r in rows}
    photo = Image.open(os.path.join(IMAGE_DIR, item["image"]))
    photo_rgb = photo.convert("RGB")

    bank = torch.load(EMBEDDINGS, map_location="cpu")
    coco_ids = torch.as_tensor(bank["indices"]).tolist()
    bank = F.normalize(bank["clip_embeds"].float(), dim=-1).to(device)
    with open(os.path.join(COCO, "annotations", "captions_train2017.json"), encoding="utf-8") as handle:
        annotations = json.load(handle)["annotations"]
    captions = {}
    for ann in annotations:
        captions.setdefault(ann["image_id"], []).append(ann["caption"].strip())
    clip = CLIPModel.from_pretrained(CLIP_ID, torch_dtype=torch.float16).to(device).eval()
    processor = CLIPProcessor.from_pretrained(CLIP_ID)

    model, vq_model, uni_prompting, _ = load_mmada(args.model, args.vq_model, device)
    tokenizer = uni_prompting.text_tokenizer
    if args.watch is not None:
        watch = {tokenizer.encode(" " + w, add_special_tokens=False)[0] for w in args.watch}
    print(f"watching {sorted(tokenizer.decode([t]) for t in watch)}", flush=True)
    eot_id = tokenizer.convert_tokens_to_ids("<|endoftext|>")
    recorder = AttentionRecorder(args.max_new_tokens, slice(IMAGE_START, IMAGE_START + (args.resolution // 16) ** 2))
    recorder.enabled = False
    for block in [m for m in model.modules() if hasattr(m, "_scaled_dot_product_attention") and hasattr(m, "layer_id")]:
        recorder.wrap(block)
    labeler = AmberLabeler()
    zoom, min_patches = CROPS["tight"]
    with torch.no_grad():
        input_ids = build_input_ids(vq_model, uni_prompting, item["query"], photo, device, args.resolution)
    image_end = IMAGE_START + (args.resolution // 16) ** 2 + 1  # through <|eoi|>

    def encode_image(image):
        """Image tokens exactly as build_input_ids makes them (fp32 VQ, outside the bf16 autocast)."""
        with torch.autocast("cuda", enabled=False):
            pixels = image_transform(image.convert("RGB"), resolution=args.resolution).unsqueeze(0).to(device)
            return vq_model.get_code(pixels)[0] + len(tokenizer)

    results = dict(id=args.id, baseline=trace["answer"], watch=sorted(tokenizer.decode([t]) for t in watch),
                   settings=vars(args), runs={})
    for context in args.contexts:
        rng = random.Random(args.seed)
        n_event = [0]

        @torch.no_grad()
        def second_pass(x, prompt_len, pos, span, image_map):
            n_event[0] += 1
            crop, box = zoom_crop(photo_rgb, image_map, args.resolution, zoom, args.zoom_top_k, min_patches)
            crop.save(os.path.join(out_dir, f"{context}_event{n_event[0]}_pos{pos:03d}_crop.jpg"), quality=95)
            if context == "tight":
                with torch.autocast("cuda", enabled=False):
                    sims = bank @ embed([crop], clip, processor, device)[0]
                top = torch.topk(sims, args.top_k)
                hits = [dict(coco_id=coco_ids[i], similarity=float(s)) for s, i in zip(top.values.tolist(), top.indices.tolist())]
            elif context == "random":
                hits = [dict(coco_id=a["image_id"], similarity=None) for a in rng.sample(annotations, args.top_k)]
            else:
                hits = []
            context_captions = [captions[h["coco_id"]][0] for h in hits]
            if context == "zoom":
                # The crop, upscaled to the model resolution, in place of the photo; the prompt is unchanged.
                prefix = input_ids.clone()
                prefix[0, IMAGE_START:image_end - 1] = encode_image(crop)
            else:
                # Same image tokens as the main pass; only the text after <|eoi|> changes.
                prefix = torch.cat([input_ids[:, :image_end],
                                    prompt_ids(uni_prompting, with_context(item["query"], context_captions), device)], dim=1)
            cx = torch.cat([prefix, x[:, prompt_len:]], dim=1)
            offset = prefix.shape[1]
            old = [int(x[0, prompt_len + q]) for q in span]
            cx[0, [offset + q for q in span]] = MASK_ID
            left, first = list(span), None
            while left:
                clogits = model(cx).logits[0, [offset + q for q in left]].float()
                clogits[:, MASK_ID] = -np.inf
                probs = F.softmax(clogits, dim=-1)
                if first is None:
                    row = probs[left.index(pos)]
                    top5 = torch.topk(row, 5)
                    first = dict(p_original=float(row[old[span.index(pos)]]),
                                 top5=[(tokenizer.decode([int(t)]), round(float(v), 4)) for v, t in zip(top5.values, top5.indices)])
                conf, arg = probs.max(dim=-1)
                j = int(conf.argmax())
                cx[0, offset + left.pop(j)] = arg[j]
            new = [int(cx[0, offset + q]) for q in span]
            for q, token_id in zip(span, new):
                x[0, prompt_len + q] = token_id
            return dict(pos=pos, token=tokenizer.decode([old[span.index(pos)]]), box=box, remasked=span,
                        old_text=[tokenizer.decode([t]) for t in old], new_text=[tokenizer.decode([t]) for t in new],
                        context_pass=first, retrieved=[dict(h, caption=c) for h, c in zip(hits, context_captions)])

        with torch.autocast("cuda", dtype=torch.bfloat16):
            answer_ids, events = decode(model, input_ids, recorder, args, lambda pos, t, step, image_map: t in watch, eot_id, second_pass)
        text = clean_response(tokenizer.batch_decode(answer_ids, skip_special_tokens=True,
                                                     clean_up_tokenization_spaces=False)[0])
        nouns = labeler.label(args.id, text)
        run = dict(caption=text, hallucinated=[n["word"] for n in nouns if n["label"] == "hallucinated"],
                   grounded=[n["word"] for n in nouns if n["label"] == "grounded"], events=events)
        results["runs"][context] = run
        print(f"\n===== context: {context} =====", flush=True)
        for e in events:
            cp = e["context_pass"]
            print(f"step {e['step']:2d} pos {e['pos']:3d} {e['token']!r:10s} p_commit={e['p_commit']:.2f} -> "
                  f"p with context={cp['p_original']:.2f} top5={cp['top5']}\n"
                  f"    remasked {e['old_text']} -> {e['new_text']}", flush=True)
            for r in e["retrieved"]:
                sim = f"{r['similarity']:.3f}" if r["similarity"] is not None else "random"
                print(f"      [{sim}] coco {r['coco_id']}: {r['caption']}", flush=True)
        print(f"caption: {text}\nhallucinated={run['hallucinated']} grounded={run['grounded']}", flush=True)

    write_json(os.path.join(out_dir, "rag_slot.json"), results, indent=1)
    print(f"\nwrote {out_dir}")


if __name__ == "__main__":
    main()
