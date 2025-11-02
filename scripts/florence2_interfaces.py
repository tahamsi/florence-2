from __future__ import annotations

from pathlib import Path

import torch
from PIL import Image

from models.florence2_runtime import (
    DEFAULT_CAPTION_PROMPT,
    DEFAULT_DETAILED_CAPTION_PROMPT,
    DEFAULT_MODEL_ID,
    DEFAULT_OCR_PROMPT,
    DEFAULT_VQA_PROMPT_PREFIX,
    clear_cuda_cache,
    get_model,
    get_processor,
    get_runtime_device,
    move_tensors,
    prepare_inputs,
    reset_runtime_device,
)


def _encode_inputs(path: Path, prompt: str, *, model_source: str) -> dict[str, object]:
    processor = get_processor(model_source)
    with Image.open(path) as image:
        image = image.convert("RGB")
        return prepare_inputs(image, processor, prompt=prompt)


def _generate_response(
    image_path: str,
    prompt: str,
    *,
    model_source: str = DEFAULT_MODEL_ID,
    max_new_tokens: int = 100,
    num_beams: int = 3,
    repetition_penalty: float = 1.0,
) -> str:
    path = Path(image_path).expanduser()
    if not path.is_file():
        raise FileNotFoundError(f"Image not found: {path}")

    processor = get_processor(model_source)
    raw_inputs = _encode_inputs(path, prompt, model_source=model_source)
    model = get_model(model_source)
    inputs = move_tensors(raw_inputs)

    def _generate() -> torch.Tensor:
        return model.generate(
            input_ids=inputs.get("input_ids"),
            pixel_values=inputs.get("pixel_values"),
            attention_mask=inputs.get("attention_mask"),
            max_new_tokens=max_new_tokens,
            num_beams=num_beams,
            repetition_penalty=repetition_penalty,
        )

    device = get_runtime_device()
    try:
        output_ids = _generate()
    except RuntimeError:
        if device.type == "cuda":
            print("Encountered a CUDA error with Florence-2; retrying on CPU.", flush=True)
            reset_runtime_device(torch.device("cpu"))
            processor = get_processor(model_source)
            raw_inputs = _encode_inputs(path, prompt, model_source=model_source)
            inputs = move_tensors(raw_inputs)
            model = get_model(model_source)
            clear_cuda_cache()
            output_ids = _generate()
        else:
            raise

    return processor.batch_decode(output_ids, skip_special_tokens=True)[0].strip()


def florence2_caption(
    image_path: str,
    *,
    prompt_token: str = DEFAULT_CAPTION_PROMPT,
    model_source: str = DEFAULT_MODEL_ID,
    max_new_tokens: int = 100,
    num_beams: int = 3,
    repetition_penalty: float = 1.0,
) -> str:
    return _generate_response(
        image_path,
        prompt=prompt_token,
        model_source=model_source,
        max_new_tokens=max_new_tokens,
        num_beams=num_beams,
        repetition_penalty=repetition_penalty,
    )


def florence2_detailed_caption(
    image_path: str,
    *,
    model_source: str = DEFAULT_MODEL_ID,
    max_new_tokens: int = 150,
    num_beams: int = 3,
    repetition_penalty: float = 1.0,
) -> str:
    return _generate_response(
        image_path,
        prompt=DEFAULT_DETAILED_CAPTION_PROMPT,
        model_source=model_source,
        max_new_tokens=max_new_tokens,
        num_beams=num_beams,
        repetition_penalty=repetition_penalty,
    )


def florence2_vqa(
    image_path: str,
    question: str,
    *,
    model_source: str = DEFAULT_MODEL_ID,
    max_new_tokens: int = 60,
    num_beams: int = 1,
    repetition_penalty: float = 1.0,
) -> str:
    cleaned_question = question.strip()
    if not cleaned_question:
        raise ValueError("Question must be a non-empty string.")
    prompt = f"{DEFAULT_VQA_PROMPT_PREFIX}{cleaned_question}"
    return _generate_response(
        image_path,
        prompt=prompt,
        model_source=model_source,
        max_new_tokens=max_new_tokens,
        num_beams=num_beams,
        repetition_penalty=repetition_penalty,
    )


def florence2_ocr(
    image_path: str,
    *,
    model_source: str = DEFAULT_MODEL_ID,
    max_new_tokens: int = 128,
    num_beams: int = 1,
    repetition_penalty: float = 1.0,
) -> str:
    return _generate_response(
        image_path,
        prompt=DEFAULT_OCR_PROMPT,
        model_source=model_source,
        max_new_tokens=max_new_tokens,
        num_beams=num_beams,
        repetition_penalty=repetition_penalty,
    )
