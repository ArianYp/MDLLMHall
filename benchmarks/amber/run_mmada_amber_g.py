"""Generate AMBER-g captions with the existing MMaDA image+text path.

This script does not score predictions. Official CHAIR, Cover, Hal, and Cog
come from third_party/AMBER/inference.py with --evaluation_type g.
"""

import argparse
import datetime as datetime_module
import json
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

DEFAULT_QUERY = os.path.join(ROOT, "data", "amber", "query_generative.json")
DEFAULT_IMAGE_DIR = os.path.join(ROOT, "data", "amber", "images")
DEFAULT_OUTPUT = os.path.join(ROOT, "results", "amber_g", "mmada_predictions.json")
DEFAULT_AMBER_REPO = os.path.join(ROOT, "third_party", "AMBER")
GENERATIVE_MIN_ID = 1
GENERATIVE_MAX_ID = 1004

# AMBER generation settings reported for MMaDA: semi-autoregressive decoding
# with generation length 128, block size 32, and 64 denoising steps.
# mmu_generate treats `steps` as the total budget and splits it across blocks.
DEFAULT_MODEL = "Gen-Verse/MMaDA-8B-MixCoT"
DEFAULT_VQ_MODEL = "showlab/magvitv2"
DEFAULT_MAX_NEW_TOKENS = 128
DEFAULT_STEPS = 64
DEFAULT_BLOCK_LENGTH = 32
DEFAULT_TEMPERATURE = 0.0
DEFAULT_CFG_SCALE = 0.0
DEFAULT_REMASKING = "low_confidence"
DEFAULT_SEED = 0
DEFAULT_RESOLUTION = 512


def read_json(path):
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path, payload):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    temporary = path + ".tmp"
    with open(temporary, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    os.replace(temporary, path)


def sibling(path, filename):
    return os.path.join(os.path.dirname(os.path.abspath(path)), filename)


def git_sha(path):
    try:
        return subprocess.check_output(
            ["git", "-C", path, "rev-parse", "HEAD"],
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def config_value(conf, dotted, default=None):
    current = conf
    for part in dotted.split("."):
        if current is None:
            return default
        try:
            current = current[part]
        except Exception:
            return default
    if current is None:
        return default
    return current


def apply_optional_config(args):
    """Fill checkpoint paths from an OmegaConf YAML when the CLI left them unset.

    Generation length and sampling stay on the evaluation defaults unless the
    command line sets them. Training YAML sequence lengths are not generation
    lengths for this benchmark.
    """
    if not args.config:
        if args.model is None:
            args.model = DEFAULT_MODEL
        if args.vq_model is None:
            args.vq_model = DEFAULT_VQ_MODEL
        return
    try:
        from omegaconf import OmegaConf
    except ImportError as exc:
        raise SystemExit(
            "Passing --config requires OmegaConf, which the MMaDA training code already uses. "
            "Install omegaconf or omit --config and pass --model / --vq-model."
        ) from exc
    conf = OmegaConf.load(args.config)
    if args.model is None:
        args.model = config_value(conf, "mmu_model_path")
        if args.model is None:
            args.model = config_value(conf, "model.mmada.pretrained_model_path", DEFAULT_MODEL)
    if args.vq_model is None:
        args.vq_model = config_value(conf, "model.vq_model.vq_model_name", DEFAULT_VQ_MODEL)


def select_queries(queries, start_id, end_id):
    if not isinstance(queries, list):
        raise SystemExit("AMBER query file must be a JSON list.")
    by_id = {}
    for index, item in enumerate(queries):
        if not isinstance(item, dict) or "id" not in item or "image" not in item or "query" not in item:
            raise SystemExit(f"Query item {index} must contain id, image, and query.")
        item_id = int(item["id"])
        if item_id in by_id:
            raise SystemExit(f"Duplicate query id {item_id}.")
        by_id[item_id] = item
    if start_id > end_id:
        raise SystemExit(f"--start-id ({start_id}) is greater than --end-id ({end_id}).")
    missing = [item_id for item_id in range(start_id, end_id + 1) if item_id not in by_id]
    if missing:
        preview = ", ".join(str(item_id) for item_id in missing[:8])
        raise SystemExit(
            f"Query file is missing {len(missing)} id(s) in {start_id}..{end_id}: {preview}"
        )
    return [by_id[item_id] for item_id in range(start_id, end_id + 1)]


def is_full_generative_range(start_id, end_id):
    return start_id == GENERATIVE_MIN_ID and end_id == GENERATIVE_MAX_ID


def clean_response(text):
    from mmada_infer import SPECIAL_TOKENS

    for token in SPECIAL_TOKENS:
        text = text.replace(token, "")
    return text.strip()


def record_from_item(item_id, response):
    return {"id": int(item_id), "response": response}


def completed_responses(path):
    if not os.path.isfile(path):
        return {}
    payload = read_json(path)
    if not isinstance(payload, list):
        raise SystemExit(f"Existing predictions file {path} must be a JSON list.")
    finished = {}
    for item in payload:
        if not isinstance(item, dict) or "id" not in item:
            continue
        response = item.get("response")
        if isinstance(response, str) and response.strip():
            finished[int(item["id"])] = response
    return finished


def validate_records(records, start_id, end_id):
    expected = list(range(start_id, end_id + 1))
    if not isinstance(records, list):
        raise SystemExit("Predictions must be a JSON list.")
    seen = []
    for index, item in enumerate(records):
        if not isinstance(item, dict):
            raise SystemExit(f"Prediction {index} is not an object.")
        if "id" not in item or "response" not in item:
            raise SystemExit(f"Prediction {index} must contain id and response.")
        if type(item["id"]) is not int:
            raise SystemExit(f"Prediction {index} id must be an int, got {item['id']!r}.")
        if not isinstance(item["response"], str) or not item["response"].strip():
            raise SystemExit(f"Prediction id {item['id']} has an empty response.")
        extra = set(item) - {"id", "response"}
        if extra:
            raise SystemExit(
                f"Prediction id {item['id']} has extra fields {sorted(extra)}. "
                "The official file must contain only id and response."
            )
        seen.append(item["id"])
    if len(seen) != len(set(seen)):
        raise SystemExit("Prediction ids are not unique.")
    if seen != expected and sorted(seen) != expected:
        raise SystemExit(
            f"Prediction ids {sorted(seen)[:5]}... do not match requested ids {start_id}..{end_id}."
        )
    if set(seen) != set(expected):
        raise SystemExit(f"Prediction ids are not exactly {start_id}..{end_id}.")
    if is_full_generative_range(start_id, end_id):
        if len(records) != GENERATIVE_MAX_ID or set(seen) != set(range(1, GENERATIVE_MAX_ID + 1)):
            raise SystemExit("A full AMBER-g file must contain unique ids 1 through 1004.")
    return True


def scope_label(start_id, end_id):
    return "full" if is_full_generative_range(start_id, end_id) else "subset"


def seed_everything(seed):
    import random

    import torch

    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(args):
    import torch

    if args.cpu or args.device == "cpu":
        return torch.device("cpu")
    if args.device:
        device = torch.device(args.device)
        if device.type == "cuda" and not torch.cuda.is_available():
            raise SystemExit(f"Requested device {args.device}, but CUDA is not available.")
        return device
    if torch.cuda.is_available():
        return torch.device("cuda")
    raise SystemExit("CUDA is not available. Pass --cpu to run on CPU.")


def write_metadata(args, device, count, status, failed_id=None):
    metadata = {
        "timestamp": datetime_module.datetime.now(datetime_module.timezone.utc).isoformat(),
        "status": status,
        "git_commit": git_sha(ROOT),
        "checkpoint": args.model,
        "vq_model": args.vq_model,
        "config": os.path.abspath(args.config) if args.config else None,
        "seed": args.seed,
        "generation_hyperparameters": {
            "decoding_strategy": "semi-autoregressive",
            "max_new_tokens": args.max_new_tokens,
            "steps": args.steps,
            "block_length": args.block_length,
            "temperature": args.temperature,
            "cfg_scale": args.cfg_scale,
            "remasking": args.remasking,
            "resolution": args.resolution,
        },
        "number_of_examples": count,
        "start_id": args.start_id,
        "end_id": args.end_id,
        "scope": scope_label(args.start_id, args.end_id),
        "amber_repo": os.path.abspath(args.amber_repo),
        "amber_repo_commit": git_sha(args.amber_repo) if os.path.isdir(args.amber_repo) else None,
        "query_file": os.path.abspath(args.query_file),
        "image_dir": os.path.abspath(args.image_dir),
        "output": os.path.abspath(args.output),
        "prompt_policy": "Each response uses query_generative.json field 'query' verbatim.",
        "generator": "mmada_infer.generate_answer -> MMadaModelLM.mmu_generate",
        "failed_id": failed_id,
        "device": str(device) if device is not None else None,
    }
    metadata_path = sibling(args.output, "amber_g_metadata.json")
    write_json(metadata_path, metadata)
    return metadata_path


def generate(args):
    if not os.path.isfile(args.query_file):
        raise SystemExit(
            f"Query file not found: {args.query_file}. Run benchmarks/amber/download_amber.sh first."
        )
    if not os.path.isdir(args.image_dir):
        raise SystemExit(
            f"Image directory not found: {args.image_dir}. Run benchmarks/amber/download_amber.sh first."
        )

    selected = select_queries(read_json(args.query_file), args.start_id, args.end_id)
    for item in selected:
        image_path = os.path.join(args.image_dir, item["image"])
        if not os.path.isfile(image_path):
            raise SystemExit(f"Missing image for id {item['id']}: {image_path}")

    if os.path.isfile(args.output) and args.overwrite and args.resume:
        raise SystemExit("Pass only one of --overwrite and --resume.")
    if os.path.isfile(args.output) and not args.overwrite and not args.resume:
        raise SystemExit(
            f"{args.output} already exists. Pass --resume to skip finished ids or --overwrite to replace it."
        )

    finished = {}
    if args.resume:
        finished = completed_responses(args.output)

    records = [
        record_from_item(item_id, response)
        for item_id, response in sorted(finished.items())
        if args.start_id <= item_id <= args.end_id
    ]
    pending = [item for item in selected if int(item["id"]) not in finished]

    device = None
    if pending:
        import torch
        from PIL import Image
        from tqdm import tqdm

        from mmada_infer import generate_answer, load_mmada, validate_schedule

        validate_schedule(args.max_new_tokens, args.steps, args.block_length)
        device = resolve_device(args)
        seed_everything(args.seed)
        model, vq_model, uni_prompting, _ = load_mmada(args.model, args.vq_model, device)
        for offset, item in enumerate(tqdm(pending, desc="AMBER-g"), start=1):
            item_id = int(item["id"])
            image_path = os.path.join(args.image_dir, item["image"])
            prompt = item["query"]
            try:
                with torch.inference_mode():
                    text = generate_answer(
                        model,
                        vq_model,
                        uni_prompting,
                        prompt,
                        Image.open(image_path),
                        device,
                        max_new_tokens=args.max_new_tokens,
                        steps=args.steps,
                        block_length=args.block_length,
                        temperature=args.temperature,
                        cfg_scale=args.cfg_scale,
                        remasking=args.remasking,
                        resolution=args.resolution,
                    )
                text = clean_response(text)
                if not text:
                    raise RuntimeError("MMaDA returned an empty response after special-token cleanup.")
            except Exception as exc:
                records = sorted(records, key=lambda row: row["id"])
                write_json(args.output, records)
                write_metadata(args, device, len(records), status="failed", failed_id=item_id)
                raise SystemExit(f"Generation failed on id {item_id}: {exc}") from exc
            records.append(record_from_item(item_id, text))
            if offset % args.checkpoint_every == 0:
                write_json(args.output, sorted(records, key=lambda row: row["id"]))
    else:
        print("Every requested id already has a response. Nothing to generate.")

    records = sorted(records, key=lambda row: row["id"])
    write_json(args.output, records)
    validate_records(records, args.start_id, args.end_id)
    metadata_path = sibling(args.output, "amber_g_metadata.json")
    if pending or not os.path.isfile(metadata_path):
        metadata_path = write_metadata(args, device, len(records), status="completed")
        print(f"Saved run metadata to {metadata_path}")
    else:
        print(f"Kept existing run metadata at {metadata_path}")
    print(f"Saved {len(records)} predictions to {args.output}")
    print(f"AMBER_SCOPE {scope_label(args.start_id, args.end_id)}")


def validate_only(args):
    if not os.path.isfile(args.output):
        raise SystemExit(f"Predictions file not found: {args.output}")
    records = read_json(args.output)
    validate_records(records, args.start_id, args.end_id)
    print(f"Validated {len(records)} predictions in {args.output}")
    print(f"AMBER_SCOPE {scope_label(args.start_id, args.end_id)}")


def parse_args():
    parser = argparse.ArgumentParser(description="Run MMaDA on AMBER-g and write official response JSON.")
    parser.add_argument("--query-file", default=DEFAULT_QUERY)
    parser.add_argument("--image-dir", default=DEFAULT_IMAGE_DIR)
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    parser.add_argument("--config", default=None, help="Optional MMaDA OmegaConf YAML. Supplies model paths only.")
    parser.add_argument(
        "--model",
        "--checkpoint",
        dest="model",
        default=None,
        help=f"MMaDA checkpoint. Defaults to {DEFAULT_MODEL}, or the path stored in --config.",
    )
    parser.add_argument("--vq-model", default=None, help=f"VQ tokenizer checkpoint. Defaults to {DEFAULT_VQ_MODEL}.")
    parser.add_argument("--device", default=None, help="cuda, cpu, or a specific CUDA device such as cuda:0.")
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--start-id", type=int, default=GENERATIVE_MIN_ID)
    parser.add_argument("--end-id", type=int, default=GENERATIVE_MAX_ID)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--max-new-tokens", type=int, default=DEFAULT_MAX_NEW_TOKENS)
    parser.add_argument("--steps", type=int, default=DEFAULT_STEPS)
    parser.add_argument("--block-length", type=int, default=DEFAULT_BLOCK_LENGTH)
    parser.add_argument("--temperature", type=float, default=DEFAULT_TEMPERATURE)
    parser.add_argument("--cfg-scale", type=float, default=DEFAULT_CFG_SCALE)
    parser.add_argument("--remasking", default=DEFAULT_REMASKING, choices=["low_confidence", "random"])
    parser.add_argument("--resolution", type=int, default=DEFAULT_RESOLUTION)
    parser.add_argument("--checkpoint-every", type=int, default=1)
    parser.add_argument("--amber-repo", default=DEFAULT_AMBER_REPO, help="Used only to record the official scorer commit.")
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()
    if args.checkpoint_every < 1:
        parser.error("--checkpoint-every must be positive.")
    if args.validate_only:
        return args
    apply_optional_config(args)
    return args


def main():
    args = parse_args()
    if args.validate_only:
        validate_only(args)
        return
    generate(args)


if __name__ == "__main__":
    main()
