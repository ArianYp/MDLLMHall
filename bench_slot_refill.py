"""Cost of slot_refill_study.py relative to baseline decoding, measured on one GPU, plus detection efficiency from its event logs.

Timings (median over --n captions of the study set, after a warm-up caption):
  baseline       the decode loop with no trigger and no attention recording (64 forward passes)
  recorder       the same loop with the attention recorder on (needed by the zoom refill)
  forward        one MMaDA forward pass on a full sequence (each refilled token costs one)
  vq_crop        zoom_crop + VQ encoding of the crop (once per zoom trigger)
  clip_image     CLIP ViT-H-14 embedding of the full photo + rank lookup (once per caption, clip trigger)

Then, for every finished condition in rag/slot_refill, extra cost per caption =
refill forwards x forward + triggers x vq_crop (zoom) + recorder overhead (zoom) + clip_image (clip),
and hallucinated nouns removed per 100 extra forward-pass equivalents (from eval_report.json if present).

Example:
    python bench_slot_refill.py --n 5
"""

import argparse
import json
import os
import time
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from transformers import CLIPModel, CLIPProcessor

from mmada_infer import build_input_ids, image_transform, load_mmada
from rag_retrieve import CLIP_ID, CROPS, embed
from rag_slot_decoding import decode
from remask_decoding import STUDY, write_json, zoom_crop
from trace_amber_image_attention import IMAGE_START, AttentionRecorder
from trace_amber_steps import IMAGE_DIR, QUERY_FILE, read_json

RUNS = os.path.join(STUDY, "rag", "slot_refill")


def timed(fn):
    torch.cuda.synchronize()
    start = time.perf_counter()
    out = fn()
    torch.cuda.synchronize()
    return time.perf_counter() - start, out


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--n", type=int, default=5)
    args = parser.parse_args()
    device = torch.device("cuda")
    run = SimpleNamespace(max_new_tokens=128, steps=64, block_length=32)
    ids = read_json(os.path.join(STUDY, "ids.json"))["ids"][: args.n + 1]
    queries = {int(r["id"]): r for r in read_json(QUERY_FILE)}

    model, vq_model, uni_prompting, _ = load_mmada("Gen-Verse/MMaDA-8B-MixCoT", "showlab/magvitv2", device)
    tokenizer = uni_prompting.text_tokenizer
    eot_id = tokenizer.convert_tokens_to_ids("<|endoftext|>")
    n_image = 1024
    recorder = AttentionRecorder(128, slice(IMAGE_START, IMAGE_START + n_image))
    recorder.enabled = False
    for block in [m for m in model.modules() if hasattr(m, "_scaled_dot_product_attention") and hasattr(m, "layer_id")]:
        recorder.wrap(block)
    clip = CLIPModel.from_pretrained(CLIP_ID, torch_dtype=torch.float16).to(device).eval()
    processor = CLIPProcessor.from_pretrained(CLIP_ID)
    text = torch.randn(418, 1024, device=device)  # rank lookup cost only; the vocabulary is embedded once per run

    times = {k: [] for k in ("baseline", "recorder", "forward", "vq_crop", "clip_image")}
    never = lambda pos, token_id, step, image_map: False
    for n, item_id in enumerate(ids):
        item = queries[item_id]
        photo = Image.open(os.path.join(IMAGE_DIR, item["image"]))
        rgb = photo.convert("RGB")
        with torch.no_grad():
            input_ids = build_input_ids(vq_model, uni_prompting, item["query"], photo, device)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                t_base, (answer, _) = timed(lambda: decode(model, input_ids, recorder, run, never, eot_id, None,
                                                           record_attention=False))
                t_rec, _ = timed(lambda: decode(model, input_ids, recorder, run, never, eot_id, None,
                                                record_attention=True))
                x = torch.cat([input_ids, answer], dim=1)
                t_fwd, _ = timed(lambda: model(x).logits)
            image_map = torch.rand(n_image, device=device)
            zoom, min_patches = CROPS["tight"]

            def vq():
                crop, _ = zoom_crop(rgb, image_map, 512, zoom, 30, min_patches)
                with torch.autocast("cuda", enabled=False):
                    return vq_model.get_code(image_transform(crop, resolution=512).unsqueeze(0).to(device))
            t_vq, _ = timed(vq)

            def clip_rank():
                sims = (embed([rgb], clip, processor, device) @ text.T)[0]
                return (sims[None] < sims[:, None]).float().mean(1)
            t_clip, _ = timed(clip_rank)
        if n == 0:
            continue  # warm-up
        for key, value in zip(times, (t_base, t_rec, t_fwd, t_vq, t_clip)):
            times[key].append(value)
        print(f"id {item_id}: baseline {t_base:.2f}s recorder {t_rec:.2f}s forward {t_fwd * 1000:.0f}ms "
              f"vq_crop {t_vq * 1000:.0f}ms clip_image {t_clip * 1000:.0f}ms", flush=True)
    t = {k: float(np.median(v)) for k, v in times.items()}
    t["gpu"] = torch.cuda.get_device_name()
    print(f"\nmedian: {t}", flush=True)

    report = {}
    for name in sorted(os.listdir(RUNS)):
        events_path = os.path.join(RUNS, name, "predictions_events.json")
        if not os.path.isfile(events_path):
            continue
        log = read_json(events_path)
        captions = log["captions"]
        trigger, refill = name.split("_")
        n_cap = len(captions)
        triggers = sum(len(c["events"]) for c in captions)
        refill_forwards = sum(len(e["remasked"]) for c in captions for e in c["events"])
        true_hits = sum(e["oracle_hallucinated"] for c in captions for e in c["events"])
        changed = sum(e["old_text"][e["remasked"].index(e["pos"])] != e["new_text"][e["remasked"].index(e["pos"])]
                      for c in captions for e in c["events"])
        extra = (refill_forwards * t["forward"] + (triggers * t["vq_crop"] if refill == "zoom" else 0)) / n_cap
        extra += (t["recorder"] - t["baseline"]) if refill == "zoom" else 0
        extra += t["clip_image"] if trigger == "clip" else 0
        report[name] = dict(captions=n_cap, eligible=sum(c["eligible"] for c in captions), triggers=triggers,
                            triggers_per_caption=triggers / n_cap, refill_forwards_per_caption=refill_forwards / n_cap,
                            precision=true_hits / max(triggers, 1), word_changed=changed / max(triggers, 1),
                            extra_seconds_per_caption=extra, overhead_vs_baseline=extra / t["baseline"])
    eval_path = next((p for p in (os.path.join(RUNS, "eval", "eval_report.json"),
                                  os.path.join(RUNS, "eval_partial", "eval_report.json")) if os.path.isfile(p)), None)
    if eval_path:
        scores = read_json(eval_path)
        base = scores["hallucinating/baseline"]["hallucinated_nouns"]
        for name, r in report.items():
            key = f"hallucinating/{name}"
            if key in scores and r["captions"] == 144:
                removed = base - scores[key]["hallucinated_nouns"]
                r["hallucinated_removed"] = removed
                forwards = r["extra_seconds_per_caption"] * r["captions"] / t["forward"]
                r["removed_per_100_forward_equivalents"] = 100 * removed / forwards
    print(f"\n{'condition':14s} {'capt':>4s} {'trig/capt':>9s} {'precision':>9s} {'changed':>7s} {'extra s/capt':>12s} "
          f"{'overhead':>8s} {'removed':>7s} {'per 100 fwd':>11s}")
    for name, r in report.items():
        print(f"{name:14s} {r['captions']:4d} {r['triggers_per_caption']:9.2f} {r['precision']:9.2f} {r['word_changed']:7.2f} "
              f"{r['extra_seconds_per_caption']:12.2f} {r['overhead_vs_baseline']:8.1%} "
              f"{r.get('hallucinated_removed', '-')!s:>7s} {r.get('removed_per_100_forward_equivalents', float('nan')):11.1f}")
    write_json(os.path.join(RUNS, "efficiency.json"), dict(timings=t, conditions=report, eval=eval_path), indent=1)
    print(f"\nwrote {os.path.join(RUNS, 'efficiency.json')}")


if __name__ == "__main__":
    main()
