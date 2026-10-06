"""Re-prediction confidence of every study noun in its finished baseline caption, under several perturbations.

For each noun of with_clean/nouns.csv, the finished baseline caption (trace token ids) is kept,
the noun's tokens are masked, and one forward pass re-predicts them. Conditions:

  plain       the image and the AMBER query
  plain_nb    as plain, but the neighbours (pos-1 and the token after the word) are masked too
  random5     5 random COCO train captions before the query (seeded per caption), image present
  random5b    a second, independent set of 5 random captions
  noimage     the query alone, no image tokens
  zoom        the tight crop around the noun's baseline late-layer attention (rag_retrieve.CROPS)
              in place of the image

Per condition: p of the original first token, mean log p over the word's tokens, top-1 token,
entropy at the first token. All nouns of a caption are scored in one batch per condition.
Output: with_clean/repredict.csv (one row per noun). Resumable per caption.

Example:
    python repredict_confidence.py
"""

import argparse
import csv
import json
import os
import random

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from mmada_infer import build_input_ids, image_transform, load_mmada
from rag_mmada_context import with_context
from rag_retrieve import COCO, CROPS
from remask_decoding import STUDY, zoom_crop
from trace_amber_image_attention import IMAGE_START
from trace_amber_steps import IMAGE_DIR, QUERY_FILE, read_json
from trace_mmada_steps import MASK_ID

CONDITIONS = ["plain", "plain_nb", "random5", "random5b", "noimage", "zoom"]


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--nouns", default=os.path.join(STUDY, "with_clean", "nouns.csv"))
    parser.add_argument("--output", default=os.path.join(STUDY, "with_clean", "repredict.csv"))
    parser.add_argument("--zoom-top-k", type=int, default=30)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--ids", type=int, nargs="*", default=None, help="Only these captions (smoke test).")
    parser.add_argument("--model", default="Gen-Verse/MMaDA-8B-MixCoT")
    parser.add_argument("--vq-model", default="showlab/magvitv2")
    parser.add_argument("--resolution", type=int, default=512)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    with open(args.nouns, encoding="utf-8") as handle:
        nouns = list(csv.DictReader(handle))
    by_caption = {}
    for row in nouns:
        by_caption.setdefault(int(row["id"]), []).append(row)
    if args.ids:
        by_caption = {i: by_caption[i] for i in args.ids}
    queries = {int(r["id"]): r for r in read_json(QUERY_FILE)}

    done_rows = []
    if os.path.isfile(args.output):
        with open(args.output, encoding="utf-8") as handle:
            done_rows = list(csv.DictReader(handle))
    finished = {int(r["id"]) for r in done_rows}
    print(f"{len(by_caption)} captions, {len(nouns)} nouns, {len(finished)} captions already done", flush=True)

    with open(os.path.join(COCO, "annotations", "captions_train2017.json"), encoding="utf-8") as handle:
        all_captions = [a["caption"].strip() for a in json.load(handle)["annotations"]]

    device = torch.device("cuda")
    torch.manual_seed(args.seed)
    model, vq_model, uni_prompting, _ = load_mmada(args.model, args.vq_model, device)
    tokenizer = uni_prompting.text_tokenizer
    n_image = (args.resolution // 16) ** 2
    zoom, min_patches = CROPS["tight"]

    def encode_image(image):
        with torch.autocast("cuda", enabled=False):
            pixels = image_transform(image.convert("RGB"), resolution=args.resolution).unsqueeze(0).to(device)
            return vq_model.get_code(pixels)[0] + len(tokenizer)

    def trace_dir(item_id):
        for name in ("traces", "traces_clean"):
            if os.path.isfile(os.path.join(STUDY, name, f"{item_id}.json")):
                return os.path.join(STUDY, name)
        raise FileNotFoundError(item_id)

    @torch.no_grad()
    def score(rows_x, prompt_len, spans, targets):
        """rows_x: list of 1-D id tensors (same length). spans: masked positions per row. targets: word positions per row."""
        out = []
        for i in range(0, len(rows_x), args.batch):
            x = torch.stack(rows_x[i:i + args.batch])
            for r, span in enumerate(spans[i:i + args.batch]):
                x[r, [prompt_len + q for q in span]] = MASK_ID
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logits = model(x).logits.float()
            for r, word_pos in enumerate(targets[i:i + args.batch]):
                orig = rows_x[i + r]
                lp = F.log_softmax(logits[r, [prompt_len + q for q in word_pos]], dim=-1)
                ids = orig[[prompt_len + q for q in word_pos]]
                tok_lp = lp.gather(-1, ids[:, None])[:, 0]
                first = lp[0]
                out.append(dict(p=float(tok_lp[0].exp()), logp_mean=float(tok_lp.mean()),
                                top1=tokenizer.decode([int(first.argmax())]),
                                entropy=float(-(first.exp() * first).sum())))
        return out

    fields = list(nouns[0].keys()) + [f"{c}_{k}" for c in CONDITIONS for k in ("p", "logp_mean", "top1", "entropy")]
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)

    for n_done, (item_id, rows) in enumerate(by_caption.items(), start=1):
        if item_id in finished:
            continue
        item = queries[item_id]
        tdir = trace_dir(item_id)
        trace = read_json(os.path.join(tdir, f"{item_id}.json"))
        late_map = torch.from_numpy(np.load(os.path.join(tdir, f"{item_id}.npz"))["late_map"].astype(np.float32))
        answer = torch.tensor([r["token_id"] for r in trace["records"]], device=device)
        photo = Image.open(os.path.join(IMAGE_DIR, item["image"]))
        photo_rgb = photo.convert("RGB")
        rng = random.Random(args.seed * 1_000_003 + item_id)
        rand_a, rand_b = rng.sample(all_captions, 5), rng.sample(all_captions, 5)

        word_pos = [list(range(int(r["pos"]), int(r["pos"]) + int(r["n_tokens"]))) for r in rows]
        nb_span = [[q for q in [w[0] - 1] + w + [w[-1] + 1] if 0 <= q < len(answer)] for w in word_pos]
        for r, w in zip(rows, word_pos):
            assert tokenizer.decode(answer[w[0]:w[0] + 1].tolist()) == r["token"], (item_id, r["word"])

        results = {}
        with torch.no_grad():
            prefixes = dict(plain=build_input_ids(vq_model, uni_prompting, item["query"], photo, device, args.resolution),
                            random5=build_input_ids(vq_model, uni_prompting, with_context(item["query"], rand_a), photo,
                                                    device, args.resolution),
                            random5b=build_input_ids(vq_model, uni_prompting, with_context(item["query"], rand_b), photo,
                                                     device, args.resolution),
                            noimage=build_input_ids(vq_model, uni_prompting, item["query"], None, device))
        for cond in ["plain", "random5", "random5b", "noimage"]:
            x = torch.cat([prefixes[cond][0].long(), answer])
            results[cond] = score([x] * len(rows), prefixes[cond].shape[1], word_pos, word_pos)
        x = torch.cat([prefixes["plain"][0].long(), answer])
        results["plain_nb"] = score([x] * len(rows), prefixes["plain"].shape[1], nb_span, word_pos)
        zoom_rows = []
        for w in word_pos:
            crop, _ = zoom_crop(photo_rgb, late_map[w[0]], args.resolution, zoom, args.zoom_top_k, min_patches)
            zx = x.clone()
            with torch.no_grad():
                zx[IMAGE_START:IMAGE_START + n_image] = encode_image(crop)
            zoom_rows.append(zx)
        results["zoom"] = score(zoom_rows, prefixes["plain"].shape[1], word_pos, word_pos)

        for i, r in enumerate(rows):
            out = dict(r)
            for cond in CONDITIONS:
                for k, v in results[cond][i].items():
                    out[f"{cond}_{k}"] = v
            done_rows.append(out)
        with open(args.output, "w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(done_rows)
        summary = " ".join(f"{r['word']}[{r['label'][0]}]{results['plain'][i]['p']:.2f}/{results['zoom'][i]['p']:.2f}"
                           for i, r in enumerate(rows))
        print(f"[{n_done}/{len(by_caption)}] id {item_id}: {summary}", flush=True)


if __name__ == "__main__":
    main()
