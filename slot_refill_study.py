"""Remask-and-refill at commit time on the study captions, triggered by the AMBER oracle, the CLIP detector or at random.

Decoding follows the baseline schedule (rag_slot_decoding.decode). A committed token is
eligible if it decodes to a whole word whose lemma is an AMBER object word (not a global
safe word). The trigger decides which eligible tokens are acted on:

  oracle  the word is hallucinated by the AMBER rule (AmberLabeler: not in the image's
          truth words or their associations, spaCy similarity included)
  clip    the word's CLIP rank against the full image is below --clip-threshold
          (rank = share of the object vocabulary that matches the image worse,
          as rank_full in rag_clip_relation.py)
          With --crop-veto, a flag is dropped if the word's CLIP rank on its tight
          crop (the zoom crop) is at least --crop-veto: the crop sees the object.
          With --no-background, background ("stuff") words (BACKGROUND) are never flagged.
  random  with probability --random-rate (or the fired / eligible rate of another run,
          --random-rate-from), the matched control for clip

  With --oracle-keep < 1 the oracle keeps each of its flags with that probability (lower recall),
  and with --oracle-false-rate > 0 it also flags non-hallucinated eligible words with that
  probability (lower precision). Both are drawn from the per-caption seeded generator.

On a trigger the token and its committed neighbours (pos-1, pos+1) are masked and
refilled one at a time, most confident first, in a second diffusion pass
(--refill-order word-first: the refill image first decides the flagged token alone, then the
neighbours are refilled around it with the full image):

  plain   same full image and prompt
  zoom    the tight crop around the token's late-layer attention (rag_retrieve.CROPS)
          in place of the image. With --remove-sinks, attention-sink patches (the top patch
          for >= --sink-frac of the answer positions in that forward pass) are ignored first.

--noun-trigger only acts on words tagged NN* in context (the scorer's test), and --refill-vocab noun / object
restricts the token the refill puts at the flagged position.

Decoding then continues normally. Each position triggers at most once. The oracle
label is logged for every trigger, so detection precision can be read from any run.
Outputs: predictions in the official AMBER format and <output>_events.json. Resumable.

Example:
    python slot_refill_study.py --trigger clip --refill zoom --output .../clip_zoom/predictions.json
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

from amber_noun_labels import AmberLabeler
from mmada_infer import build_input_ids, image_transform, load_mmada
from rag_retrieve import CLIP_ID, CROPS, embed
from rag_slot_decoding import decode
from remask_decoding import LATE_LAYERS, STUDY, clean_response, write_json, zoom_crop
from trace_amber_image_attention import IMAGE_START, AttentionRecorder
from trace_amber_steps import AMBER_DATA, IMAGE_DIR, QUERY_FILE, read_json
from trace_mmada_steps import MASK_ID

# COCO-Stuff-style scene words, where CLIP rank is a weak detector (inspect_false_flags.py, full_clip/analyze_full_clip.py).
BACKGROUND = {"sky", "cloud", "wall", "floor", "ground", "grass", "tree", "bush", "plant", "road", "street", "path", "sidewalk",
              "water", "sea", "ocean", "river", "lake", "beach", "sand", "snow", "mountain", "hill", "field", "forest", "dirt",
              "court", "building", "fence", "rock", "leaf", "sun"}


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--trigger", choices=["oracle", "clip", "random"], required=True)
    parser.add_argument("--refill", choices=["plain", "zoom"], required=True)
    parser.add_argument("--ids-files", nargs="+",
                        default=[os.path.join(STUDY, "ids.json"), os.path.join(STUDY, "ids_clean.json")])
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--output", required=True)
    parser.add_argument("--clip-threshold", type=float, default=0.9)
    parser.add_argument("--crop-veto", type=float, default=None,
                        help="clip: drop a flag if the word's CLIP rank on its tight crop is at least this (second check).")
    parser.add_argument("--no-background", action="store_true", help="clip: never flag BACKGROUND words.")
    parser.add_argument("--caption-aware", action="store_true",
                        help="clip: before ranking, drop from the vocabulary the word's near-synonyms (spaCy > 0.8, WordNet synonyms "
                             "and direct hypernyms / hyponyms) and the object words already committed in the caption (with theirs). "
                             "Applies to the full-image rank and to the crop check.")
    parser.add_argument("--crop-not-top1", action="store_true",
                        help="clip: crop check = drop a flag only if the word is CLIP's best (remaining) word on its crop "
                             "(instead of --crop-veto's rank cutoff).")
    parser.add_argument("--nouns", default=os.path.join(STUDY, "with_clean", "nouns.csv"),
                        help="Only its lemma column is used, to build the same vocabulary as rag_clip_relation.py.")
    parser.add_argument("--oracle-keep", type=float, default=1.0, help="oracle: keep each flag with this probability.")
    parser.add_argument("--oracle-false-rate", type=float, default=0.0,
                        help="oracle: also flag non-hallucinated eligible words with this probability.")
    parser.add_argument("--random-rate", type=float, default=None)
    parser.add_argument("--random-rate-from", default=None, help="An _events.json whose fired / eligible rate to match.")
    parser.add_argument("--zoom-top-k", type=int, default=30)
    parser.add_argument("--remove-sinks", action="store_true",
                        help="crops: ignore attention-sink patches, i.e. patches that are the top late-layer patch for "
                             ">= --sink-frac of the answer positions in the same forward pass")
    parser.add_argument("--sink-frac", type=float, default=0.5)
    parser.add_argument("--refill-order", choices=["confidence", "word-first"], default="confidence",
                        help="confidence: refill the span most confident first, all with the refill image. "
                             "word-first: the refill image (crop for zoom) alone decides the flagged word, with its "
                             "neighbours masked; then, with the full image, the neighbours are refilled around it.")
    parser.add_argument("--noun-trigger", action="store_true",
                        help="Only act on a word that NLTK tags NN* (as the AMBER scorer does) in the committed text: "
                             "everything to its left plus the committed tokens right after it, up to the first mask.")
    parser.add_argument("--refill-vocab", choices=["any", "noun", "object"], default="any",
                        help="Token allowed at the flagged position in the refill. noun: whole-word tokens NLTK tags NN* "
                             "on their own, plus every eligible object word; object: eligible object words only.")
    parser.add_argument("--span-radius", type=int, default=1,
                        help="on a trigger remask pos-r .. pos+r (committed tokens of the current block or earlier); 1 = the usual 3 tokens")
    parser.add_argument("--model", default="Gen-Verse/MMaDA-8B-MixCoT")
    parser.add_argument("--vq-model", default="showlab/magvitv2")
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--steps", type=int, default=64)
    parser.add_argument("--block-length", type=int, default=32)
    parser.add_argument("--resolution", type=int, default=512)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    if args.trigger == "random" and args.random_rate is None:
        log = read_json(args.random_rate_from)
        args.random_rate = log["totals"]["fired"] / max(log["totals"]["eligible"], 1)
    print(f"trigger={args.trigger} refill={args.refill} clip_threshold={args.clip_threshold} "
          f"random_rate={args.random_rate}", flush=True)

    ids = [i for path in args.ids_files for i in read_json(path)["ids"]]
    ids = ids[: args.limit] if args.limit else ids
    queries = {int(r["id"]): r for r in read_json(QUERY_FILE)}
    device = torch.device("cuda")
    torch.manual_seed(args.seed)

    labeler = AmberLabeler()
    eligible_words = labeler.object_words - labeler.global_safe
    model, vq_model, uni_prompting, _ = load_mmada(args.model, args.vq_model, device)
    tokenizer = uni_prompting.text_tokenizer
    eot_id = tokenizer.convert_tokens_to_ids("<|endoftext|>")
    # Vocabulary tokens that are a whole eligible object word (" horse"), to split first-pass probability mass into
    # correct (object in the image) / wrong (object not in the image) / neutral (everything else) per caption.
    object_token = {}
    for t in range(len(tokenizer)):
        w = tokenizer.decode([t])
        if w.startswith(" ") and w.strip().isalpha():
            lemma = labeler.lemmatizer.lemmatize(w.strip().lower())
            if lemma in eligible_words:
                object_token[t] = lemma
    object_ids = torch.tensor(sorted(object_token), device=device)
    print(f"{len(object_token)} vocabulary tokens are eligible object words", flush=True)
    allowed_ids = None  # tokens the refill may put at the flagged position (None = any)
    if args.refill_vocab != "any":
        allowed = set(object_token)
        if args.refill_vocab == "noun":
            from nltk.tag import PerceptronTagger
            tagger = PerceptronTagger()
            for t in range(len(tokenizer)):
                w = tokenizer.decode([t])
                if w.startswith(" ") and w.strip().isalpha() and tagger.tag([w.strip()])[0][1].startswith("NN"):
                    allowed.add(t)
        allowed_ids = torch.tensor(sorted(allowed), device=device)
        print(f"refill vocabulary ({args.refill_vocab}): {len(allowed)} tokens", flush=True)
    n_image = (args.resolution // 16) ** 2
    recorder = AttentionRecorder(args.max_new_tokens, slice(IMAGE_START, IMAGE_START + n_image))
    recorder.enabled = False
    for block in [m for m in model.modules() if hasattr(m, "_scaled_dot_product_attention") and hasattr(m, "layer_id")]:
        recorder.wrap(block)
    zoom, min_patches = CROPS["tight"]

    if args.trigger == "clip":
        from transformers import CLIPModel, CLIPProcessor
        clip = CLIPModel.from_pretrained(CLIP_ID, torch_dtype=torch.float16).to(device).eval()
        processor = CLIPProcessor.from_pretrained(CLIP_ID)
        relation = json.load(open(os.path.join(AMBER_DATA, "relation.json"), encoding="utf-8"))
        with open(args.nouns, encoding="utf-8") as handle:
            lemmas = {r["lemma"] for r in csv.DictReader(handle)}
        vocabulary = sorted(set(relation) | {w for v in relation.values() for w in v} | lemmas | eligible_words)
        with torch.no_grad():
            text = []
            for i in range(0, len(vocabulary), 256):
                inputs = processor(text=[f"a photo of a {w}." for w in vocabulary[i:i + 256]], return_tensors="pt",
                                   padding=True).to(device)
                text.append(F.normalize(clip.get_text_features(**inputs).float(), dim=-1))
        text = torch.cat(text)
        word_index = {w: i for i, w in enumerate(vocabulary)}
        if args.caption_aware:
            from nltk.corpus import wordnet as wn
            vec = np.stack([labeler.nlp.vocab[w].vector for w in vocabulary])
            norm = np.linalg.norm(vec, axis=1)
            has_vec = norm > 0
            vec = vec / np.where(has_vec, norm, 1)[:, None]
            synonym_cache = {}

            def synonyms(lemma):
                """Vocabulary indices of the lemma and its near-synonyms (clip_caption_exclude.py)."""
                if lemma not in synonym_cache:
                    close = {lemma}
                    i = word_index.get(lemma)
                    if i is not None and has_vec[i]:
                        close |= {vocabulary[j] for j in np.flatnonzero((vec @ vec[i] > 0.8) & has_vec)}
                    for s in wn.synsets(lemma, wn.NOUN):
                        for t in [s] + s.hypernyms() + s.hyponyms():
                            close |= {l.name().lower() for l in t.lemmas()}
                    synonym_cache[lemma] = [word_index[w] for w in close if w in word_index]
                return synonym_cache[lemma]

    def clip_rank(sims, lemma, others):
        """(rank, is top-1) of lemma among the vocabulary, minus its synonyms and the `others` lemmas with theirs
        (all of the vocabulary when --caption-aware is off)."""
        own = sims[word_index[lemma]]
        if not args.caption_aware:
            return float((sims < own).float().mean()), bool((sims <= own).all())
        keep = torch.ones(len(vocabulary), dtype=torch.bool, device=sims.device)
        keep[synonyms(lemma)] = False
        for other in others:
            keep[synonyms(other)] = False
        keep[word_index[lemma]] = True
        rest = sims[keep]
        return float((rest < own).float().mean()), bool((rest <= own).all())

    def crop_map(image_map):
        """The token's late-layer map, with the pass's sink patches zeroed when --remove-sinks is set.

        The recorder still holds the commit pass (it is disabled during refills), with maps for all answer positions."""
        if not args.remove_sinks:
            return image_map, []
        maps = torch.stack([recorder.layers[l]["image_map"] for l in LATE_LAYERS]).mean(0)  # answer, n_image
        counts = torch.bincount(maps.argmax(1), minlength=maps.shape[1])
        sinks = torch.nonzero(counts >= args.sink_frac * maps.shape[0]).flatten().tolist()
        image_map = image_map.clone()
        image_map[sinks] = 0
        return image_map, sinks

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
    finished = {int(r["id"]) for r in predictions}
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)

    for done, item_id in enumerate(ids, start=1):
        if item_id in finished:
            continue
        item = queries[item_id]
        photo = Image.open(os.path.join(IMAGE_DIR, item["image"]))
        photo_rgb = photo.convert("RGB")
        _, safe_words, _ = labeler.word_lists(item_id)
        rng = random.Random(args.seed * 1_000_003 + item_id)
        with torch.no_grad():
            input_ids = build_input_ids(vq_model, uni_prompting, item["query"], photo, device, args.resolution)
            if args.trigger == "clip":
                with torch.autocast("cuda", enabled=False):
                    sims = (embed([photo_rgb], clip, processor, device) @ text.T)[0]
                rank = {w: float((sims < sims[i]).float().mean()) for w, i in word_index.items()}
        counts = dict(eligible=0, fired=0, vetoed=0, vetoed_oracle_hallucinated=0, not_noun=0, not_noun_oracle_hallucinated=0)
        pending = {}
        grounded_lemmas = {l for l in set(object_token.values())
                           if l in safe_words or any(labeler.similar(l, w) for w in safe_words)}
        grounded_mask = torch.tensor([object_token[int(t)] in grounded_lemmas for t in object_ids], device=device)

        def first_pass_mass(probs):
            """Probability mass of the first refill pass at the flagged position: correct / wrong object words, neutral."""
            p_obj = probs[object_ids]
            best = lambda m: (tokenizer.decode([int(object_ids[m][p_obj[m].argmax()])]), float(p_obj[m].max())) if m.any() else None
            p_correct, p_wrong = float(p_obj[grounded_mask].sum()), float(p_obj[~grounded_mask].sum())
            return dict(p_correct=p_correct, p_wrong=p_wrong, p_neutral=1.0 - p_correct - p_wrong,
                        best_correct=best(grounded_mask), best_wrong=best(~grounded_mask))

        def is_noun(pos):
            """NLTK NN* tag (the scorer's test) for the word at pos, in the committed text: everything to its left plus
            the committed tokens right after it, up to the first mask."""
            answer = state["x"][0, state["prompt_len"]:].tolist()
            text, start = "", None
            for q, t in enumerate(answer):
                if t in (MASK_ID, eot_id):
                    if q > pos:
                        break
                    continue
                if q == pos:
                    start = len(text) + 1  # the word token starts with a space
                text += tokenizer.decode([t])
            return any(n["span"] is not None and n["span"][0] == start for n in labeler.nouns(text))

        def fire(pos, token_id, step, image_map):
            word = tokenizer.decode([token_id])
            if not word.startswith(" ") or not word.strip().isalpha():
                return False
            lemma = labeler.lemmatizer.lemmatize(word.strip().lower())
            if lemma not in eligible_words:
                return False
            counts["eligible"] += 1
            hallucinated = not (lemma in safe_words or any(labeler.similar(lemma, w) for w in safe_words))
            if args.noun_trigger and not is_noun(pos):
                counts["not_noun"] += 1
                counts["not_noun_oracle_hallucinated"] += int(hallucinated)
                return False
            if args.trigger == "oracle":
                fired = rng.random() < args.oracle_keep if hallucinated else rng.random() < args.oracle_false_rate
            elif args.trigger == "clip":
                others = set()
                if args.caption_aware:
                    # object words already committed elsewhere in the caption
                    answer = state["x"][0, state["prompt_len"]:].tolist()
                    for q, t in enumerate(answer):
                        if q == pos or t == MASK_ID:
                            continue
                        w = tokenizer.decode([t])
                        if w.startswith(" ") and w.strip().isalpha():
                            other = labeler.lemmatizer.lemmatize(w.strip().lower())
                            if other in eligible_words and other != lemma:
                                others.add(other)
                full_rank, _ = clip_rank(sims, lemma, others) if args.caption_aware else (rank[lemma], None)
                fired = full_rank < args.clip_threshold and not (args.no_background and lemma in BACKGROUND)
                crop_rank, crop_top1 = None, None
                if fired and (args.crop_veto is not None or args.crop_not_top1):
                    crop, _ = zoom_crop(photo_rgb, crop_map(image_map)[0], args.resolution, zoom, args.zoom_top_k, min_patches)
                    with torch.no_grad(), torch.autocast("cuda", enabled=False):
                        crop_sims = (embed([crop], clip, processor, device) @ text.T)[0]
                    crop_rank, crop_top1 = clip_rank(crop_sims, lemma, others)
                    if (crop_top1 if args.crop_not_top1 else crop_rank >= args.crop_veto):
                        fired = False
                        counts["vetoed"] += 1
                        counts["vetoed_oracle_hallucinated"] += int(hallucinated)
            else:
                fired = rng.random() < args.random_rate
            if fired:
                counts["fired"] += 1
                pending[pos] = dict(lemma=lemma, oracle_hallucinated=hallucinated,
                                    clip_rank=full_rank if args.trigger == "clip" else None,
                                    crop_rank=crop_rank if args.trigger == "clip" else None,
                                    crop_top1=crop_top1 if args.trigger == "clip" else None,
                                    caption_objects=sorted(others) if args.trigger == "clip" else None)
            return fired

        @torch.no_grad()
        def second_pass(x, prompt_len, pos, span, image_map):
            cx = x.clone()
            box, sinks = None, []
            if args.refill == "zoom":
                image_map, sinks = crop_map(image_map)
                crop, box = zoom_crop(photo_rgb, image_map, args.resolution, zoom, args.zoom_top_k, min_patches)
                cx[0, IMAGE_START:IMAGE_START + n_image] = encode_image(crop)
            old = [int(x[0, prompt_len + q]) for q in span]
            cx[0, [prompt_len + q for q in span]] = MASK_ID
            left, first, word_first, mass = list(span), None, None, None
            if args.refill_order == "word-first":
                # 1) the refill image (crop for zoom) decides the flagged word alone, neighbours still masked
                logits = model(cx).logits[0, prompt_len + pos].float()
                logits[MASK_ID] = -np.inf
                probs = F.softmax(logits, dim=-1)
                first = float(probs[old[span.index(pos)]])
                mass = first_pass_mass(probs)
                free = int(probs.argmax())
                word = free if allowed_ids is None else int(allowed_ids[probs[allowed_ids].argmax()])
                word_first = dict(token=tokenizer.decode([word]), p=float(probs[word]), free_token=tokenizer.decode([free]),
                                  top5=[(tokenizer.decode([int(t)]), round(float(p), 3)) for p, t in zip(*probs.topk(5))])
                # 2) back to the full image: keep that word, let the neighbours adapt to it
                cx = x.clone()
                cx[0, prompt_len + pos] = word
                cx[0, [prompt_len + q for q in span if q != pos]] = MASK_ID
                left = [q for q in span if q != pos]
            while left:
                logits = model(cx).logits[0, [prompt_len + q for q in left]].float()
                logits[:, MASK_ID] = -np.inf
                probs = F.softmax(logits, dim=-1)
                if first is None:
                    first = float(probs[left.index(pos), old[span.index(pos)]])
                    mass = first_pass_mass(probs[left.index(pos)])
                if allowed_ids is not None and pos in left:
                    r = left.index(pos)
                    kept = torch.zeros_like(probs[r])
                    kept[allowed_ids] = probs[r, allowed_ids]
                    probs[r] = kept
                conf, arg = probs.max(dim=-1)
                j = int(conf.argmax())
                cx[0, prompt_len + left.pop(j)] = arg[j]
            new = [int(cx[0, prompt_len + q]) for q in span]
            for q, token_id in zip(span, new):
                x[0, prompt_len + q] = token_id
            return dict(pos=pos, box=box, sinks=sinks, remasked=span, p_original_refill=first, word_first=word_first,
                        first_pass=mass,
                        old_text=[tokenizer.decode([t]) for t in old], new_text=[tokenizer.decode([t]) for t in new],
                        **pending.pop(pos))

        state = {}  # live sequence for fire(), filled by decode()
        with torch.autocast("cuda", dtype=torch.bfloat16):
            answer_ids, events = decode(model, input_ids, recorder, args, fire, eot_id, second_pass,
                                        record_attention=args.refill == "zoom" or args.crop_veto is not None
                                        or args.crop_not_top1, state=state)
        text_out = clean_response(tokenizer.batch_decode(answer_ids, skip_special_tokens=True,
                                                         clean_up_tokenization_spaces=False)[0])
        predictions.append(dict(id=item_id, response=text_out))
        log.append(dict(id=item_id, events=events, **counts))
        totals = dict(eligible=sum(c["eligible"] for c in log), fired=sum(c["fired"] for c in log),
                      fired_oracle_hallucinated=sum(e["oracle_hallucinated"] for c in log for e in c["events"]),
                      vetoed=sum(c.get("vetoed", 0) for c in log),
                      vetoed_oracle_hallucinated=sum(c.get("vetoed_oracle_hallucinated", 0) for c in log),
                      not_noun=sum(c.get("not_noun", 0) for c in log),
                      not_noun_oracle_hallucinated=sum(c.get("not_noun_oracle_hallucinated", 0) for c in log))
        write_json(args.output, predictions, indent=2)
        write_json(events_path, dict(settings=vars(args), totals=totals, captions=log))
        print(f"[{done}/{len(ids)}] id {item_id} eligible {counts['eligible']} fired {counts['fired']} "
              f"changed {[(e['old_text'], e['new_text']) for e in events if e['old_text'] != e['new_text']]}", flush=True)

    totals = read_json(events_path)["totals"]
    print(f"totals: {totals} | fire rate {totals['fired'] / max(totals['eligible'], 1):.4f} | "
          f"precision {totals['fired_oracle_hallucinated'] / max(totals['fired'], 1):.3f}")


if __name__ == "__main__":
    main()
