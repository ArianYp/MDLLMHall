"""Baseline MMaDA evaluation on the official HallusionBench release.

Generation and scoring are separate. Scoring imports the metric functions from
the official HallusionBench repository and judges each answer with the official
correctness prompt. The default judge is local Qwen3-8B, which is the judge
named by the VISAGE paper. The published VISAGE prompt is the official
per-answer prompt; the paper does not state whether its single accuracy number
is aAcc, qAcc, or fAcc.
"""

import argparse
import importlib.util
import json
import os
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))
HALLUSION_UTILS = os.path.join(ROOT, "third_party", "HallusionBench", "utils.py")
DEFAULT_ANNOTATIONS = os.path.join(ROOT, "third_party", "HallusionBench", "HallusionBench.json")
DEFAULT_IMAGE_ROOT = os.path.join(ROOT, "data", "hallusionbench", "data")
DEFAULT_PREDICTIONS = os.path.join(ROOT, "results", "hallusionbench", "mmada_predictions.json")
JUDGE_DEPS = os.path.join(ROOT, "judge_pydeps")

CORRECTNESS_ENTRY = "gpt4v_output_gpt_check"
PREDICTION_ENTRY = "model_prediction"

# Official prompts from HallusionBench utils.py. VISAGE Appendix D reprints the
# first of these as prompt P1 and does not publish a different judge prompt.
CORRECTNESS_PREFIX = (
    "Imagine you are an intelligent teacher. Thoroughly read the question, reference answer "
    "and the prediction answer to ensure a clear understanding of the information provided. "
    "Assess the correctness of the predictions. "
    "If the prediction answer does not conflict with the reference answer, please generate “correct”. "
    "If the prediction answer conflict with the reference answer, please generate “incorrect”. "
    "If the prediction answer is unclear about the answer, please generate \"unclear\". \n\n Question:"
)
CONSISTENCY_PREFIX = (
    "Imagine you are an intelligent teacher. Thoroughly read the two responses to two different questions. "
    "Assess the consistency of the information provided within those two responses. "
    "You do not know the specific questions, but you can asssess the consistency among the two responses "
    "by checking for logical conflicts if both responses are correct. "
    "If response1 does not conflict with response2, please generate “same”. Otherwise, generate \"different\". \n\n response1:"
)


def sample_key(sample):
    return "|".join(
        str(sample[field])
        for field in ("category", "subcategory", "set_id", "figure_id", "question_id")
    )


def needs_image(sample):
    return str(sample.get("visual_input")) != "0" and bool(sample.get("filename"))


def resolve_image(sample, image_root):
    if not needs_image(sample):
        return None
    relative = str(sample["filename"])
    if relative.startswith("./"):
        relative = relative[2:]
    path = os.path.join(image_root, relative)
    if os.path.isfile(path):
        return path
    directory = os.path.dirname(path)
    wanted = os.path.basename(path).lower()
    if os.path.isdir(directory):
        for name in os.listdir(directory):
            if name.lower() == wanted:
                return os.path.join(directory, name)
    raise FileNotFoundError(path)


def read_json(path):
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path, payload):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)


def sibling(path, filename):
    return os.path.join(os.path.dirname(os.path.abspath(path)), filename)


def correctness_prompt(sample):
    return (
        CORRECTNESS_PREFIX
        + sample["question"]
        + "\nReference answer: "
        + sample["gt_answer_details"]
        + "\nPrediction answer:"
        + sample[PREDICTION_ENTRY]
        + "\nOutput:"
    )


def consistency_prompt(response_1, response_2):
    return (
        CONSISTENCY_PREFIX
        + response_1
        + "\nresponse2: "
        + response_2
        + "\nOutput:"
    )


def parse_correctness(text):
    lowered = text.lower()
    if "incorrect" in lowered:
        return "0"
    if "correct" in lowered:
        return "1"
    return "2"


def parse_same(text):
    lowered = text.lower()
    if "same" in lowered:
        return "1"
    if "different" in lowered:
        return "0"
    return "0"


def answer_text_for_judge(text):
    """Drop a Qwen3 thinking trace so the official substring parser sees the label."""
    end = text.rfind("</think>")
    if end != -1:
        return text[end + len("</think>") :].strip()
    return text.strip()


def hide_deepspeed_from_transformers():
    """Keep the judge's Transformers from importing the environment's DeepSpeed.

    DeepSpeed imports Transformers while Transformers is still initializing, which
    raises a circular import. Scoring does not use DeepSpeed.
    """
    import importlib.metadata as importlib_metadata

    original = importlib_metadata.metadata

    def metadata(distribution_name):
        if distribution_name == "deepspeed":
            raise importlib_metadata.PackageNotFoundError(distribution_name)
        return original(distribution_name)

    importlib_metadata.metadata = metadata


class Qwen3Judge:
    name = "Qwen/Qwen3-8B"

    def __init__(self, model_name):
        self.name = model_name
        if "transformers" in sys.modules:
            raise SystemExit(
                "Qwen3 scoring needs a fresh Python process so it can load transformers 4.51 from judge_pydeps. "
                "Run --score separately from --generate."
            )
        if os.path.isdir(JUDGE_DEPS) and JUDGE_DEPS not in sys.path:
            sys.path.insert(0, JUDGE_DEPS)
        hide_deepspeed_from_transformers()
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=dtype,
            trust_remote_code=True,
            device_map="auto" if torch.cuda.is_available() else None,
        )
        self.model.eval()
        self.torch = torch

    def complete(self, prompt):
        messages = [{"role": "user", "content": prompt}]
        try:
            rendered = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
        except TypeError:
            rendered = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        inputs = self.tokenizer(rendered, return_tensors="pt")
        inputs = {key: value.to(self.model.device) for key, value in inputs.items()}
        with self.torch.no_grad():
            output = self.model.generate(
                **inputs,
                max_new_tokens=32,
                do_sample=False,
            )
        new_tokens = output[0, inputs["input_ids"].shape[1] :]
        raw = self.tokenizer.decode(new_tokens, skip_special_tokens=True)
        return answer_text_for_judge(raw)


class OpenAIJudge:
    def __init__(self, model_name):
        self.name = model_name
        from openai import OpenAI

        if not os.environ.get("OPENAI_API_KEY"):
            raise SystemExit("Set OPENAI_API_KEY to use --judge openai.")
        self.client = OpenAI()
        self.model_name = model_name

    def complete(self, prompt):
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            max_tokens=16,
        )
        return response.choices[0].message.content or ""


def build_judge(kind, model_name):
    if kind == "qwen3":
        return Qwen3Judge(model_name)
    if kind == "openai":
        return OpenAIJudge(model_name)
    raise SystemExit(f"Unknown judge: {kind}")


def needs_judge_call(sample):
    if sample.get("generation_error") or not sample.get(PREDICTION_ENTRY):
        return False
    if CORRECTNESS_ENTRY not in sample:
        return True
    if str(sample.get("figure_id")) != "0" and "same" not in sample and "same_error" not in sample:
        return True
    return False


def load_hallusion_metrics():
    if not os.path.isfile(HALLUSION_UTILS):
        raise SystemExit(
            f"Official HallusionBench utils.py not found at {HALLUSION_UTILS}. "
            "Clone https://github.com/tianyi-lab/HallusionBench into third_party/HallusionBench."
        )
    spec = importlib.util.spec_from_file_location("hallusionbench_utils", HALLUSION_UTILS)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def percent(numerator, denominator):
    if not denominator:
        return None
    return round(100 * numerator / denominator, 4)


def split_category(samples, category):
    return [sample for sample in samples if sample["category"] == category]


def generate(args):
    import torch
    from PIL import Image
    from tqdm import tqdm

    from mmada_infer import generate_answer, load_mmada, validate_schedule

    validate_schedule(args.max_new_tokens, args.steps, args.block_length)
    annotations = read_json(args.annotations)
    if not isinstance(annotations, list):
        raise SystemExit(f"{args.annotations} must be a JSON list.")
    selected = annotations[: args.limit] if args.limit else annotations

    existing = {}
    if os.path.isfile(args.output) and not args.overwrite:
        for sample in read_json(args.output):
            if PREDICTION_ENTRY in sample and sample[PREDICTION_ENTRY] is not None and "generation_error" not in sample:
                existing[sample_key(sample)] = sample

    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    torch.manual_seed(args.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)

    model, vq_model, uni_prompting, dtype = load_mmada(args.model, args.vq_model, device)
    results = []
    failures = 0
    for sample in tqdm(selected, desc="HallusionBench"):
        key = sample_key(sample)
        if key in existing:
            results.append(existing[key])
            continue
        saved = dict(sample)
        try:
            image_path = resolve_image(sample, args.image_root)
            image = Image.open(image_path) if image_path is not None else None
            saved[PREDICTION_ENTRY] = generate_answer(
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
                cfg_scale=args.cfg_scale,
                remasking=args.remasking,
                resolution=args.resolution,
            )
        except Exception as exc:
            failures += 1
            saved[PREDICTION_ENTRY] = None
            saved["generation_error"] = str(exc)
            print(f"[Error] {key}: {exc}")
        results.append(saved)
        write_json(args.output, results)

    config = {
        "benchmark": "HallusionBench",
        "annotations": os.path.abspath(args.annotations),
        "image_root": os.path.abspath(args.image_root),
        "image_source": "rayguan/HallusionBench (author Hugging Face dataset; the official Google Drive zip was not publicly retrievable). Annotations match HallusionBench.json on the official main branch.",
        "n_annotations_in_file": len(annotations),
        "n_selected": len(selected),
        "n_written": len(results),
        "n_generation_failures": failures,
        "prompt_policy": "Raw HallusionBench question inside the existing MMaDA chat template. No added yes/no instruction.",
        "text_only_policy": "visual_input=0 skips image tokens and still calls mmu_generate.",
        "checkpoint": args.model,
        "vq_model": args.vq_model,
        "dtype": str(dtype),
        "device": str(device),
        "seed": args.seed,
        "max_new_tokens": args.max_new_tokens,
        "steps": args.steps,
        "block_length": args.block_length,
        "temperature": args.temperature,
        "cfg_scale": args.cfg_scale,
        "remasking": args.remasking,
        "resolution": args.resolution,
        "generator": "MMadaModelLM.mmu_generate",
        "annotation_note": (
            "README still lists the October 2023 release of 254 questions and 69 figures. "
            "The HallusionBench.json currently on the official main branch contains the later expanded set."
        ),
    }
    config_path = sibling(args.output, "mmada_run_config.json")
    write_json(config_path, config)
    print(f"Saved {len(results)} predictions to {args.output}")
    print(f"Saved run config to {config_path}")
    if failures:
        print(f"Generation failures: {failures}")


def judge_samples(samples, judge, judged_path=None):
    from tqdm import tqdm

    by_split = {
        "VD": split_category(samples, "VD"),
        "VS": split_category(samples, "VS"),
    }
    judged = []
    for split_name, split in by_split.items():
        originals = {}
        for sample in split:
            if str(sample["figure_id"]) == "0" and sample.get(PREDICTION_ENTRY):
                originals[sample_key(sample)] = sample[PREDICTION_ENTRY]
        for sample in tqdm(split, desc=f"Judge {split_name}"):
            saved = dict(sample)
            prediction = saved.get(PREDICTION_ENTRY)
            if not prediction or saved.get("generation_error"):
                saved["score_status"] = "skipped"
                judged.append(saved)
                continue
            if CORRECTNESS_ENTRY not in saved:
                raw = judge.complete(correctness_prompt(saved))
                saved["judge_correctness_text"] = raw
                saved[CORRECTNESS_ENTRY] = parse_correctness(raw)
            if "same" not in saved and "same_error" not in saved:
                if str(saved["figure_id"]) == "0":
                    saved["same"] = "1"
                else:
                    original_key = sample_key({**saved, "figure_id": "0"})
                    if original_key not in originals:
                        saved["same_error"] = "No figure_id=0 answer is available for this question pair."
                    else:
                        raw_same = judge.complete(consistency_prompt(prediction, originals[original_key]))
                        saved["judge_same_text"] = raw_same
                        saved["same"] = parse_same(raw_same)
            saved["score_status"] = "judged"
            saved["judge_model"] = judge.name
            judged.append(saved)
            if judged_path is not None:
                write_json(judged_path, judged)
    return judged


def score(args):
    samples = read_json(args.score)
    judged_path = sibling(args.score, "mmada_judged.json")
    if os.path.isfile(judged_path) and not args.rejudge:
        previous = read_json(judged_path)
        previous_by_key = {sample_key(sample): sample for sample in previous}
        merged = []
        for sample in samples:
            prior = previous_by_key.get(sample_key(sample))
            if prior and prior.get("score_status") == "judged" and prior.get("judge_model") == args.judge_model:
                merged.append(prior)
            else:
                merged.append(sample)
        samples = merged

    pending = [sample for sample in samples if needs_judge_call(sample)]
    if pending:
        judge = build_judge(args.judge, args.judge_model)
        samples = judge_samples(samples, judge, judged_path)

    scoreable = [sample for sample in samples if sample.get("score_status") == "judged"]
    skipped = [sample for sample in samples if sample.get("score_status") != "judged"]
    metrics = load_hallusion_metrics()
    metrics.assign_correctness(scoreable, CORRECTNESS_ENTRY)

    def pack(data):
        question = metrics.get_eval_all(data, CORRECTNESS_ENTRY) if data else None
        figure = metrics.get_eval_fig(data) if data else None
        pair = None
        pair_error = None
        try:
            if data:
                pair = metrics.get_eval_pair_all(data, CORRECTNESS_ENTRY)
        except Exception as exc:
            pair_error = str(exc)
        easy = metrics.get_eval_pair_easy(data) if data else None
        hard = metrics.get_eval_pair_hard(data) if data else None

        def acc_from(stats):
            if not stats:
                return None
            return percent(stats["correct"], stats["total"])

        def count_from(stats, key):
            if not stats or key not in stats:
                return None
            return int(stats[key])

        return {
            "aAcc": acc_from(question),
            "qAcc": acc_from(pair) if pair_error is None else None,
            "qAcc_error": pair_error,
            "fAcc": acc_from(figure),
            "easy_question_pair_acc": acc_from(easy),
            "hard_question_pair_acc": acc_from(hard),
            "easy_aAcc_as_printed_by_official_leaderboard": acc_from(easy),
            "hard_aAcc_as_printed_by_official_leaderboard": acc_from(hard),
            "n_questions": count_from(question, "total"),
            "n_question_pairs": count_from(pair, "total"),
            "n_figures": count_from(figure, "total"),
            "figure_inconsistent": count_from(figure, "inconsistent"),
            "figure_wrong": count_from(figure, "wrong"),
            "language_hallucination": count_from(pair, "LH"),
            "visual_illusion": count_from(pair, "VI"),
            "mixed": count_from(pair, "Mix"),
        }

    vd = split_category(scoreable, "VD")
    vs = split_category(scoreable, "VS")
    official_order = vd + vs
    report = {
        "judge": args.judge,
        "judge_model": args.judge_model,
        "judge_prompt": "Official HallusionBench evaluate_by_chatgpt prompt (VISAGE Appendix D prompt P1).",
        "consistency_prompt": "Official HallusionBench check_same_by_chatgpt prompt.",
        "metric_source": "third_party/HallusionBench/utils.py",
        "comparability_note": (
            "VISAGE Table 1 reports one HallusionBench Acc (%): MMaDA 34.18, VISAGE 36.83. "
            "The paper says the Qwen3-8B judge decides whether a generated answer aligns with the reference, "
            "and Appendix D reprints the official per-answer prompt. It does not name aAcc, qAcc, or fAcc, "
            "and it does not say whether unclear answers on image-free Visual Supplement questions count as correct. "
            "Per-question aAcc is the closest official number, not a confirmed match."
        ),
        "n_predictions": len(samples),
        "n_scored": len(scoreable),
        "n_skipped": len(skipped),
        "skipped_keys": [sample_key(sample) for sample in skipped],
        "overall": pack(official_order),
        "VD": pack(vd),
        "VS": pack(vs),
    }
    scores_path = sibling(args.score, "mmada_scores.json")
    write_json(scores_path, report)
    write_json(judged_path, samples)
    print(json.dumps(report, indent=2))
    print(f"Saved judged samples to {judged_path}")
    print(f"Saved scores to {scores_path}")


def parse_args():
    parser = argparse.ArgumentParser(description="Run baseline MMaDA on HallusionBench and score it.")
    parser.add_argument("--generate", action="store_true")
    parser.add_argument("--score", metavar="PREDICTIONS_JSON")
    parser.add_argument("--output", default=DEFAULT_PREDICTIONS)
    parser.add_argument("--annotations", default=DEFAULT_ANNOTATIONS)
    parser.add_argument("--image-root", default=DEFAULT_IMAGE_ROOT)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--rejudge", action="store_true")
    parser.add_argument("--model", default="Gen-Verse/MMaDA-8B-MixCoT")
    parser.add_argument("--vq-model", default="showlab/magvitv2")
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--steps", type=int, default=256)
    parser.add_argument("--block-length", type=int, default=32)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--cfg-scale", type=float, default=0.0)
    parser.add_argument("--remasking", default="low_confidence", choices=["low_confidence", "random"])
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--resolution", type=int, default=512)
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--judge", default="qwen3", choices=["qwen3", "openai"])
    parser.add_argument("--judge-model", default=None)
    args = parser.parse_args()
    if not args.generate and not args.score:
        parser.error("Pass --generate, --score PREDICTIONS_JSON, or both.")
    if args.judge_model is None:
        args.judge_model = "Qwen/Qwen3-8B" if args.judge == "qwen3" else "gpt-4"
    return args


def main():
    args = parse_args()
    if args.generate:
        generate(args)
    if args.score:
        if args.generate and args.score == args.output:
            pass
        score(args)
        return
    if args.generate and args.score is None:
        return


if __name__ == "__main__":
    main()
