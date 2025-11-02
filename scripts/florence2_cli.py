from __future__ import annotations

import argparse
import torch

from models.florence2_runtime import (
    DEFAULT_CAPTION_PROMPT,
    DEFAULT_MODEL_ID,
    set_cache_dir,
    set_runtime_device,
    train_florence2_caption,
)
from scripts.florence2_interfaces import (
    florence2_caption,
    florence2_detailed_caption,
    florence2_ocr,
    florence2_vqa,
)


def _add_generation_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--model-source", default=DEFAULT_MODEL_ID, help="Model ID or local snapshot path.")
    parser.add_argument("--max-new-tokens", type=int, default=100, help="Maximum tokens to generate.")
    parser.add_argument("--num-beams", type=int, default=3, help="Beam width for beam search.")
    parser.add_argument(
        "--repetition-penalty",
        type=float,
        default=1.0,
        help="Repetition penalty (>1 discourages repeats).",
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Florence-2 utilities: captioning, VQA, OCR, and fine-tuning.",
    )
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda"),
        default="auto",
        help="Force execution device (default: auto detect).",
    )
    parser.add_argument(
        "--cache-dir",
        help="Optional Hugging Face cache directory override.",
    )

    subparsers = parser.add_subparsers(dest="command", required=True)

    caption_parser = subparsers.add_parser("caption", help="Generate a caption for an image.")
    caption_parser.add_argument("image_path", help="Path to the image file.")
    caption_parser.add_argument(
        "--prompt-token",
        default=DEFAULT_CAPTION_PROMPT,
        help="Prompt token to control caption style (default: <CAPTION>).",
    )
    _add_generation_args(caption_parser)

    detailed_parser = subparsers.add_parser("detailed-caption", help="Generate a detailed caption.")
    detailed_parser.add_argument("image_path", help="Path to the image file.")
    detailed_parser.add_argument(
        "--model-source",
        default=DEFAULT_MODEL_ID,
        help="Model ID or local snapshot path.",
    )
    detailed_parser.add_argument("--max-new-tokens", type=int, default=150, help="Maximum tokens to generate.")
    detailed_parser.add_argument("--num-beams", type=int, default=3, help="Beam width for beam search.")
    detailed_parser.add_argument(
        "--repetition-penalty",
        type=float,
        default=1.0,
        help="Repetition penalty (>1 discourages repeats).",
    )

    vqa_parser = subparsers.add_parser("vqa", help="Answer a visual question about an image.")
    vqa_parser.add_argument("image_path", help="Path to the image file.")
    vqa_parser.add_argument("question", help="Question to ask about the image.")
    vqa_parser.add_argument(
        "--model-source",
        default=DEFAULT_MODEL_ID,
        help="Model ID or local snapshot path.",
    )
    vqa_parser.add_argument("--max-new-tokens", type=int, default=60, help="Maximum tokens to generate.")
    vqa_parser.add_argument("--num-beams", type=int, default=1, help="Beam width for beam search.")
    vqa_parser.add_argument(
        "--repetition-penalty",
        type=float,
        default=1.0,
        help="Repetition penalty (>1 discourages repeats).",
    )

    ocr_parser = subparsers.add_parser("ocr", help="Extract text from an image using Florence-2.")
    ocr_parser.add_argument("image_path", help="Path to the image file.")
    ocr_parser.add_argument(
        "--model-source",
        default=DEFAULT_MODEL_ID,
        help="Model ID or local snapshot path.",
    )
    ocr_parser.add_argument("--max-new-tokens", type=int, default=128, help="Maximum tokens to generate.")
    ocr_parser.add_argument("--num-beams", type=int, default=1, help="Beam width for beam search.")
    ocr_parser.add_argument(
        "--repetition-penalty",
        type=float,
        default=1.0,
        help="Repetition penalty (>1 discourages repeats).",
    )

    train_parser = subparsers.add_parser("train", help="Fine-tune Florence-2 for image captioning.")
    train_parser.add_argument("train_manifest", help="Path to training manifest (tab-delimited image\\tcaption).")
    train_parser.add_argument(
        "--val-manifest",
        help="Optional validation manifest for evaluation.",
    )
    train_parser.add_argument(
        "--output-dir",
        required=True,
        help="Directory to store the fine-tuned checkpoint and processor.",
    )
    train_parser.add_argument(
        "--model-source",
        default=DEFAULT_MODEL_ID,
        help="Model ID or local snapshot path to fine-tune.",
    )
    train_parser.add_argument(
        "--prompt-token",
        default=DEFAULT_CAPTION_PROMPT,
        help="Prompt token to prepend during training (default: <CAPTION>).",
    )
    train_parser.add_argument(
        "--max-length",
        type=int,
        default=128,
        help="Maximum sequence length for tokenized captions (default: 128).",
    )
    train_parser.add_argument(
        "--learning-rate",
        type=float,
        default=1e-4,
        help="Learning rate for the optimizer (default: 1e-4).",
    )
    train_parser.add_argument(
        "--weight-decay",
        type=float,
        default=0.0,
        help="Weight decay (default: 0.0).",
    )
    train_parser.add_argument(
        "--epochs",
        type=int,
        default=1,
        help="Number of fine-tuning epochs (default: 1).",
    )
    train_parser.add_argument(
        "--batch-size",
        type=int,
        default=4,
        help="Per-device batch size (default: 4).",
    )
    train_parser.add_argument(
        "--eval-batch-size",
        type=int,
        help="Optional per-device evaluation batch size (defaults to train batch size).",
    )
    train_parser.add_argument(
        "--gradient-accumulation",
        type=int,
        default=1,
        help="Gradient accumulation steps (default: 1).",
    )
    train_parser.add_argument(
        "--warmup-ratio",
        type=float,
        default=0.0,
        help="Warmup ratio for learning rate scheduling (default: 0.0).",
    )
    train_parser.add_argument(
        "--logging-steps",
        type=int,
        default=50,
        help="Frequency of logging steps (default: 50).",
    )
    train_parser.add_argument(
        "--save-strategy",
        choices=("no", "steps", "epoch"),
        default="epoch",
        help="Checkpoint save strategy (default: epoch).",
    )
    train_parser.add_argument(
        "--evaluation-strategy",
        choices=("no", "steps", "epoch"),
        default="epoch",
        help="Evaluation strategy when validation data is provided (default: epoch).",
    )
    train_parser.add_argument(
        "--fp16",
        action="store_true",
        help="Enable FP16 training even on CPU (use only when supported).",
    )
    train_parser.add_argument(
        "--no-fp16",
        action="store_true",
        help="Disable FP16 training even if a CUDA device is available.",
    )

    return parser.parse_args()


def _apply_global_options(args: argparse.Namespace) -> None:
    if args.cache_dir:
        set_cache_dir(args.cache_dir)

    if args.device != "auto":
        requested = torch.device(args.device)
        if requested.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but no compatible GPU is available.")
        set_runtime_device(requested)


def _run_caption(args: argparse.Namespace) -> None:
    caption = florence2_caption(
        args.image_path,
        prompt_token=args.prompt_token,
        model_source=args.model_source,
        max_new_tokens=args.max_new_tokens,
        num_beams=args.num_beams,
        repetition_penalty=args.repetition_penalty,
    )
    print(f"Caption: {caption}")


def _run_detailed_caption(args: argparse.Namespace) -> None:
    caption = florence2_detailed_caption(
        args.image_path,
        model_source=args.model_source,
        max_new_tokens=args.max_new_tokens,
        num_beams=args.num_beams,
        repetition_penalty=args.repetition_penalty,
    )
    print(f"Detailed Caption: {caption}")


def _run_vqa(args: argparse.Namespace) -> None:
    answer = florence2_vqa(
        args.image_path,
        args.question,
        model_source=args.model_source,
        max_new_tokens=args.max_new_tokens,
        num_beams=args.num_beams,
        repetition_penalty=args.repetition_penalty,
    )
    print(f"Answer: {answer}")


def _run_ocr(args: argparse.Namespace) -> None:
    transcription = florence2_ocr(
        args.image_path,
        model_source=args.model_source,
        max_new_tokens=args.max_new_tokens,
        num_beams=args.num_beams,
        repetition_penalty=args.repetition_penalty,
    )
    print(f"OCR: {transcription}")


def _run_train(args: argparse.Namespace) -> None:
    fp16: bool | None
    if args.fp16 and args.no_fp16:
        raise ValueError("Cannot pass both --fp16 and --no-fp16.")
    if args.fp16:
        fp16 = True
    elif args.no_fp16:
        fp16 = False
    else:
        fp16 = None

    metrics = train_florence2_caption(
        args.train_manifest,
        val_manifest=args.val_manifest,
        output_dir=args.output_dir,
        model_source=args.model_source,
        prompt_token=args.prompt_token,
        max_length=args.max_length,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation,
        warmup_ratio=args.warmup_ratio,
        logging_steps=args.logging_steps,
        save_strategy=args.save_strategy,
        evaluation_strategy=args.evaluation_strategy,
        fp16=fp16,
    )

    train_loss = metrics["train_loss"]
    eval_loss = metrics["eval_loss"]
    message = f"Training complete. Final train loss: {train_loss:.4f}"
    if isinstance(eval_loss, float):
        message += f" | Final eval loss: {eval_loss:.4f}"
    print(message, flush=True)


def main() -> None:
    args = _parse_args()
    _apply_global_options(args)

    command = args.command
    if command == "caption":
        _run_caption(args)
    elif command == "detailed-caption":
        _run_detailed_caption(args)
    elif command == "vqa":
        _run_vqa(args)
    elif command == "ocr":
        _run_ocr(args)
    elif command == "train":
        _run_train(args)
    else:
        raise ValueError(f"Unknown command: {command}")


if __name__ == "__main__":
    main()
