from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch
from PIL import Image
from torch.utils.data import Dataset
from transformers import AutoConfig, AutoModelForCausalLM, AutoProcessor, PreTrainedModel, Trainer, TrainingArguments

DEFAULT_MODEL_ID = "microsoft/Florence-2-base"
DEFAULT_CAPTION_PROMPT = "<CAPTION>"
DEFAULT_DETAILED_CAPTION_PROMPT = "<DETAILED_CAPTION>"
DEFAULT_VQA_PROMPT_PREFIX = "<QUESTION>"
DEFAULT_OCR_PROMPT = "<OCR>"
DEFAULT_IGNORE_PROMPT_TOKENS = 1

_device: torch.device
_dtype: torch.dtype
_cache_dir: Path | None = None


if not hasattr(PreTrainedModel, "_supports_sdpa"):
    PreTrainedModel._supports_sdpa = False


def _from_pretrained_with_dtype(
    loader: type[PreTrainedModel],
    model_source: str | Path,
    *,
    dtype: torch.dtype,
    **kwargs,
) -> PreTrainedModel:
    call_kwargs = dict(kwargs)
    call_kwargs.setdefault("trust_remote_code", True)
    call_kwargs.setdefault("cache_dir", str(_cache_dir) if _cache_dir else None)
    call_kwargs["dtype"] = dtype
    identifier = str(model_source)
    try:
        return loader.from_pretrained(identifier, **call_kwargs)
    except TypeError as exc:
        if "dtype" not in str(exc).lower():
            raise
        call_kwargs.pop("dtype", None)
        call_kwargs["torch_dtype"] = dtype
        return loader.from_pretrained(identifier, **call_kwargs)


def _set_device(device: torch.device) -> None:
    global _device, _dtype
    _device = device
    _dtype = torch.float16 if device.type == "cuda" else torch.float32


_set_device(torch.device("cuda" if torch.cuda.is_available() else "cpu"))

if _device.type != "cuda":
    print("CUDA not available; running on CPU.", flush=True)


def get_runtime_device() -> torch.device:
    return _device


def get_runtime_dtype() -> torch.dtype:
    return _dtype


def set_runtime_device(device: torch.device) -> None:
    _set_device(device)
    get_processor.cache_clear()
    get_model.cache_clear()


def reset_runtime_device(device: torch.device) -> None:
    set_runtime_device(device)


def set_cache_dir(path: str | Path | None) -> None:
    global _cache_dir
    if path is None:
        _cache_dir = None
        return
    _cache_dir = Path(path).expanduser()
    _cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ["HF_HOME"] = str(_cache_dir)
    get_processor.cache_clear()
    get_model.cache_clear()


def clear_cuda_cache() -> None:
    if torch.cuda.is_available():
        try:
            torch.cuda.empty_cache()
        except RuntimeError as exc:
            print(f"Warning: unable to clear CUDA cache ({exc}); continuing on CPU.", flush=True)


def _is_offline() -> bool:
    for key in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE"):
        value = os.environ.get(key)
        if value and value.lower() not in ("0", "false", "no"):
            return True
    return False


def _ensure_extra_dependencies() -> None:
    missing: list[str] = []
    for package in ("einops", "timm"):
        try:
            __import__(package)
        except ModuleNotFoundError:
            missing.append(package)
    if missing:
        joined = ", ".join(missing)
        raise ModuleNotFoundError(
            f"Florence-2 requires additional packages: {joined}. "
            "Install them with `pip install einops timm`."
        )


def _load_processor(model_source: str) -> AutoProcessor:
    source_path = Path(model_source).expanduser()
    kwargs = {
        "trust_remote_code": True,
        "cache_dir": str(_cache_dir) if _cache_dir else None,
    }
    if source_path.exists():
        return AutoProcessor.from_pretrained(str(source_path), **kwargs)

    try:
        return AutoProcessor.from_pretrained(model_source, local_files_only=True, **kwargs)
    except OSError as exc:
        if _is_offline():
            raise RuntimeError(
                f"Florence-2 processor files for '{model_source}' are not available locally. "
                "Download the model with internet access or pass a local snapshot path."
            ) from exc
        print(f"Local processor files for '{model_source}' not found; attempting download...", flush=True)
        return AutoProcessor.from_pretrained(model_source, **kwargs)


def _load_config(model_source: str) -> Any:
    source_path = Path(model_source).expanduser()
    kwargs = {
        "trust_remote_code": True,
        "cache_dir": str(_cache_dir) if _cache_dir else None,
    }
    if source_path.exists():
        return AutoConfig.from_pretrained(str(source_path), **kwargs)

    try:
        return AutoConfig.from_pretrained(model_source, local_files_only=True, **kwargs)
    except OSError as exc:
        if _is_offline():
            raise RuntimeError(
                f"Florence-2 config for '{model_source}' is not available locally. "
                "Download it with internet access or point to a local snapshot."
            ) from exc
        print(f"Local config for '{model_source}' not found; attempting download...", flush=True)
        return AutoConfig.from_pretrained(model_source, **kwargs)


def _load_model(model_source: str, *, dtype: torch.dtype) -> PreTrainedModel:
    _ensure_extra_dependencies()
    config = _load_config(model_source)
    source_path = Path(model_source).expanduser()
    load_kwargs = {}
    if source_path.exists():
        model = _from_pretrained_with_dtype(AutoModelForCausalLM, str(source_path), dtype=dtype, **load_kwargs)
    else:
        try:
            model = _from_pretrained_with_dtype(
                AutoModelForCausalLM,
                model_source,
                dtype=dtype,
                local_files_only=True,
                **load_kwargs,
            )
        except OSError as exc:
            if _is_offline():
                raise RuntimeError(
                    f"Florence-2 weights for '{model_source}' are not available locally. "
                    "Download them with internet access or point to a local snapshot."
                ) from exc
            print(f"Local model weights for '{model_source}' not found; attempting download...", flush=True)
            model = _from_pretrained_with_dtype(AutoModelForCausalLM, model_source, dtype=dtype, **load_kwargs)

    if not hasattr(model, "_supports_sdpa"):
        setattr(model, "_supports_sdpa", False)
    model.config = config  # ensure patched config is attached, including attention implementation
    if not getattr(model.config, "_attn_implementation", None):
        setattr(model.config, "_attn_implementation", "eager")
    setattr(model.config, "_attn_implementation_internal", getattr(model.config, "_attn_implementation", "eager"))
    return model


def _move_model_to_runtime(model: PreTrainedModel) -> PreTrainedModel:
    try:
        return model.to(device=_device, dtype=_dtype).eval()
    except RuntimeError as exc:
        if _device.type == "cuda" and "out of memory" in str(exc).lower():
            print("CUDA ran out of memory; falling back to CPU.", flush=True)
            clear_cuda_cache()
            set_runtime_device(torch.device("cpu"))
            return model.to(device=_device, dtype=_dtype).eval()
        raise


@lru_cache()
def get_processor(model_source: str = DEFAULT_MODEL_ID) -> AutoProcessor:
    return _load_processor(model_source)


@lru_cache()
def get_model(model_source: str = DEFAULT_MODEL_ID) -> PreTrainedModel:
    model = _load_model(model_source, dtype=_dtype)
    return _move_model_to_runtime(model)


def move_tensors(inputs: Mapping[str, object]) -> dict[str, object]:
    def _move_all(*, non_blocking: bool) -> dict[str, object]:
        moved: dict[str, object] = {}
        for key, value in inputs.items():
            if not isinstance(value, torch.Tensor):
                moved[key] = value
                continue
            if value.is_floating_point():
                moved[key] = value.to(device=_device, dtype=_dtype, non_blocking=non_blocking)
            else:
                moved[key] = value.to(device=_device, non_blocking=non_blocking)
        return moved

    try:
        return _move_all(non_blocking=_device.type == "cuda")
    except RuntimeError as exc:
        if _device.type == "cuda" and "out of memory" in str(exc).lower():
            print("CUDA ran out of memory while preparing inputs; falling back to CPU.", flush=True)
            clear_cuda_cache()
            set_runtime_device(torch.device("cpu"))
            return _move_all(non_blocking=False)
        raise


def prepare_inputs(
    image: Image.Image,
    processor: AutoProcessor,
    *,
    prompt: str,
    padding: str | bool | None = None,
    max_length: int | None = None,
) -> dict[str, object]:
    processor_kwargs: dict[str, object] = {
        "images": image,
        "text": prompt,
        "return_tensors": "pt",
    }
    if padding is not None:
        processor_kwargs["padding"] = padding
    if max_length is not None:
        processor_kwargs["max_length"] = max_length
        processor_kwargs["truncation"] = True
    tensors = processor(**processor_kwargs)
    return {key: value for key, value in tensors.items()}


def _load_caption_manifest(path: str | Path) -> list[tuple[str, str]]:
    manifest_path = Path(path).expanduser()
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")

    rows: list[tuple[str, str]] = []
    with manifest_path.open("r", encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            parts = [part.strip() for part in line.split("\t")]
            if len(parts) != 2:
                raise ValueError(
                    f"Expected 2 tab-delimited fields on line {line_number} of {manifest_path}; found {len(parts)}."
                )
            rows.append((parts[0], parts[1]))

    if not rows:
        raise ValueError(f"No training samples found in manifest: {manifest_path}")
    return rows


class FlorenceCaptionDataset(Dataset):
    """Dataset wrapper for Florence-2 caption fine-tuning."""

    def __init__(
        self,
        annotations: Sequence[tuple[str, str]],
        processor: AutoProcessor,
        *,
        prompt_token: str = DEFAULT_CAPTION_PROMPT,
        max_length: int = 128,
        ignore_prompt_tokens: int = DEFAULT_IGNORE_PROMPT_TOKENS,
    ) -> None:
        self._processor = processor
        self._prompt = prompt_token
        self._max_length = max_length
        self._ignore_prompt_tokens = ignore_prompt_tokens
        self._entries = [(Path(image_path).expanduser(), str(caption)) for image_path, caption in annotations]

    def __len__(self) -> int:
        return len(self._entries)

    def __getitem__(self, idx: int) -> Mapping[str, torch.Tensor]:
        image_path, caption = self._entries[idx]
        if not image_path.is_file():
            raise FileNotFoundError(f"Image not found: {image_path}")

        with Image.open(image_path) as image:
            image = image.convert("RGB")
            text = f"{self._prompt}{caption}"
            encoding = self._processor(
                images=image,
                text=text,
                return_tensors="pt",
                padding="max_length",
                truncation=True,
                max_length=self._max_length,
            )

        squeezed = {key: value.squeeze(0) for key, value in encoding.items()}
        labels = squeezed["input_ids"].clone()
        if self._ignore_prompt_tokens > 0:
            labels[: self._ignore_prompt_tokens] = -100
        squeezed["labels"] = labels
        return squeezed


def train_florence2_caption(
    train_manifest: str,
    *,
    val_manifest: str | None,
    output_dir: str,
    model_source: str = DEFAULT_MODEL_ID,
    prompt_token: str = DEFAULT_CAPTION_PROMPT,
    max_length: int = 128,
    learning_rate: float = 1e-4,
    weight_decay: float = 0.0,
    num_train_epochs: int = 1,
    per_device_train_batch_size: int = 4,
    per_device_eval_batch_size: int | None = None,
    gradient_accumulation_steps: int = 1,
    warmup_ratio: float = 0.0,
    logging_steps: int = 50,
    save_strategy: str = "epoch",
    evaluation_strategy: str | None = "epoch",
    fp16: bool | None = None,
    ignore_prompt_tokens: int = DEFAULT_IGNORE_PROMPT_TOKENS,
) -> dict[str, float | None]:
    annotations = _load_caption_manifest(train_manifest)
    processor = get_processor(model_source)
    train_dataset = FlorenceCaptionDataset(
        annotations,
        processor,
        prompt_token=prompt_token,
        max_length=max_length,
        ignore_prompt_tokens=ignore_prompt_tokens,
    )

    eval_dataset: FlorenceCaptionDataset | None = None
    if val_manifest:
        val_annotations = _load_caption_manifest(val_manifest)
        eval_dataset = FlorenceCaptionDataset(
            val_annotations,
            processor,
            prompt_token=prompt_token,
            max_length=max_length,
            ignore_prompt_tokens=ignore_prompt_tokens,
        )

    device = get_runtime_device()
    train_dtype = torch.float16 if device.type == "cuda" else torch.float32
    model = _load_model(model_source, dtype=train_dtype)
    model.train()

    effective_fp16 = fp16 if fp16 is not None else device.type == "cuda"
    per_device_eval_batch_size = per_device_eval_batch_size or per_device_train_batch_size

    training_args = TrainingArguments(
        output_dir=output_dir,
        learning_rate=learning_rate,
        per_device_train_batch_size=per_device_train_batch_size,
        per_device_eval_batch_size=per_device_eval_batch_size,
        num_train_epochs=num_train_epochs,
        weight_decay=weight_decay,
        warmup_ratio=warmup_ratio,
        gradient_accumulation_steps=gradient_accumulation_steps,
        logging_steps=logging_steps,
        save_strategy=save_strategy,
        evaluation_strategy="no" if eval_dataset is None else evaluation_strategy or "epoch",
        bf16=False,
        fp16=effective_fp16,
        remove_unused_columns=False,
        report_to=[],
        load_best_model_at_end=eval_dataset is not None,
        metric_for_best_model="eval_loss" if eval_dataset is not None else None,
        greater_is_better=False if eval_dataset is not None else None,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
    )

    train_output = trainer.train()
    metrics: dict[str, float | None] = {
        "train_loss": train_output.training_loss,
    }

    if eval_dataset is not None:
        eval_metrics = trainer.evaluate()
        metrics["eval_loss"] = float(eval_metrics.get("eval_loss", float("nan")))
    else:
        metrics["eval_loss"] = None

    trainer.save_state()
    trainer.save_model(output_dir)
    processor.save_pretrained(output_dir)
    return metrics
