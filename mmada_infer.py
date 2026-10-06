"""Baseline MMaDA multimodal generation used by the single-image runner and HallusionBench.

The diffusion loop is MMadaModelLM.mmu_generate. This module only assembles the
existing image tokens, prompt tokens, and mask schedule.
"""

import os
import sys

import torch
from PIL import Image
from torchvision import transforms
from transformers import AutoTokenizer

MMADA_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "third_party", "MMaDA")
if MMADA_ROOT not in sys.path:
    sys.path.insert(0, MMADA_ROOT)

os.environ.setdefault("TOKENIZERS_PARALLELISM", "true")

from models import MAGVITv2, MMadaModelLM
from training.prompting_utils import UniversalPrompting

SPECIAL_TOKENS = (
    "<|soi|>", "<|eoi|>", "<|sov|>", "<|eov|>", "<|t2i|>",
    "<|mmu|>", "<|t2v|>", "<|v2v|>", "<|lvg|>",
)


def image_transform(image, resolution=512):
    image = transforms.Resize(resolution, interpolation=transforms.InterpolationMode.BICUBIC)(image)
    image = transforms.CenterCrop((resolution, resolution))(image)
    image = transforms.ToTensor()(image)
    return transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])(image)


def validate_schedule(max_new_tokens, steps, block_length):
    if block_length <= 0 or max_new_tokens % block_length != 0:
        raise ValueError(
            f"max_new_tokens ({max_new_tokens}) must be divisible by block_length ({block_length})."
        )
    num_blocks = max_new_tokens // block_length
    if steps <= 0 or steps % num_blocks != 0:
        raise ValueError(
            f"steps ({steps}) must be positive and divisible by the number of blocks ({num_blocks})."
        )


def load_mmada(model_path, vq_model_name, device, max_text_len=512):
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    tokenizer = AutoTokenizer.from_pretrained(model_path, padding_side="left", trust_remote_code=True)
    uni_prompting = UniversalPrompting(
        tokenizer,
        max_text_len=max_text_len,
        special_tokens=SPECIAL_TOKENS,
        ignore_id=-100,
        cond_dropout_prob=0.1,
        use_reserved_token=True,
    )
    vq_model = MAGVITv2.from_pretrained(vq_model_name).to(device)
    vq_model.eval()
    vq_model.requires_grad_(False)
    model = MMadaModelLM.from_pretrained(model_path, trust_remote_code=True, torch_dtype=dtype).to(device)
    model.eval()
    return model, vq_model, uni_prompting, dtype


def generate_answer(
    model,
    vq_model,
    uni_prompting,
    prompt,
    image,
    device,
    max_new_tokens,
    steps,
    block_length,
    temperature=0.0,
    cfg_scale=0.0,
    remasking="low_confidence",
    resolution=512,
    step_callback=None,
):
    """Generate one answer. `image` is a PIL image, or None for text-only questions."""
    validate_schedule(max_new_tokens, steps, block_length)
    input_ids = build_input_ids(vq_model, uni_prompting, prompt, image, device, resolution)

    generate_kwargs = dict(
        max_new_tokens=max_new_tokens,
        steps=steps,
        block_length=block_length,
        temperature=temperature,
        cfg_scale=cfg_scale,
        remasking=remasking,
        step_callback=step_callback,
    )
    with torch.no_grad():
        if device.type == "cuda":
            with torch.autocast("cuda", dtype=torch.bfloat16):
                output_ids = model.mmu_generate(input_ids, **generate_kwargs)
        else:
            output_ids = model.mmu_generate(input_ids, **generate_kwargs)

    generated_ids = output_ids[:, input_ids.shape[1] :]
    return uni_prompting.text_tokenizer.batch_decode(
        generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )[0]


def build_input_ids(vq_model, uni_prompting, prompt, image, device, resolution=512):
    """<|mmu|><|soi|> image tokens <|eoi|> chat-template prompt, or the prompt alone when image is None."""
    text_token_ids = uni_prompting.text_tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt",
    ).to(device)

    if image is None:
        input_ids = text_token_ids.long()
    else:
        pixels = image_transform(image.convert("RGB"), resolution=resolution).unsqueeze(0).to(device)
        image_tokens = vq_model.get_code(pixels) + len(uni_prompting.text_tokenizer)
        batch_size = image_tokens.shape[0]
        mmu_id = uni_prompting.sptids_dict["<|mmu|>"].to(device)
        soi_id = uni_prompting.sptids_dict["<|soi|>"].to(device)
        eoi_id = uni_prompting.sptids_dict["<|eoi|>"].to(device)
        input_ids = torch.cat(
            [
                torch.ones(batch_size, 1, device=device) * mmu_id,
                torch.ones(batch_size, 1, device=device) * soi_id,
                image_tokens,
                torch.ones(batch_size, 1, device=device) * eoi_id,
                text_token_ids,
            ],
            dim=1,
        ).long()
    return input_ids
