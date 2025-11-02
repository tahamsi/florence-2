# Florence-2 Utilities & Fine-Tuning Toolkit

[Florence-2](https://github.com/microsoft/Florence-2) is Microsoft’s latest unified vision-language foundation model, described in the paper [Florence-2: Advancing a Unified Representation for Vision-Language Tasks](https://arxiv.org/abs/2403.00849). The model pairs a ViT-based visual encoder with a large causal language model and a collection of task-specific prompt tokens. This repository wraps the Florence-2 **base** checkpoint with a CLI and Python helpers for downstream inference (captioning, VQA, OCR) and lightweight fine-tuning.

---

## Project Layout

- `scripts/florence2_cli.py` — command-line entry point for inference and fine-tuning.
- `scripts/florence2_interfaces.py` — task-oriented helpers used by the CLI (captioning, detailed captioning, VQA, OCR).
- `models/florence2_runtime.py` — runtime utilities: model/processor loading, dataset wrappers, training routines.
- `data/` — place your manifests and sample assets here.
- `models/` — suggested output directory for fine-tuned checkpoints.
- `florence2.py` — thin shim that delegates to the CLI (kept for backwards compatibility).

---

## Environment Setup

1. Create and activate a Python environment (Python 3.10+ recommended):

   ```bash
    python3 -m venv .venv
    source .venv/bin/activate
   ```

2. Install dependencies:

   ```bash
   pip install -r requirements.txt
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121  # adjust for your CUDA
   pip install einops timm
   ```

3. (Optional) point Hugging Face downloads to a specific cache directory:

   ```bash
   export HF_HOME=$(pwd)/.hf-cache
   ```

---

## Data Preparation

Caption fine-tuning expects a tab-separated manifest where each line maps an image to its target caption:

```
image_path<TAB>caption_text
```

Example (`data/caption/train_manifest.tsv`):

```
/data/coco/train/000000000009.jpg	A man riding a bike down a city street.
/data/coco/train/000000000025.jpg	A child is flying a kite in a park.
```

Guidelines:

- Paths can be absolute or relative to the working directory when you run the CLI.
- Comment lines (starting with `#`) and blank lines are ignored.
- For validation, provide a second manifest with the same format.
- Prompts such as `<CAPTION>` are automatically prepended during training; you only supply the natural-language caption text.

---

## Command-Line Usage

All commands are accessible through the CLI:

```bash
python -m scripts.florence2_cli <command> [options]
```

### Quick Reference

```bash
python -m scripts.florence2_cli --help
python -m scripts.florence2_cli caption --help
python -m scripts.florence2_cli train --help
```

### Caption Generation

```bash
python -m scripts.florence2_cli caption data/samples/image.jpg \
    --prompt-token "<CAPTION>" \
    --max-new-tokens 120 \
    --model-source microsoft/Florence-2-base
```

Use different prompt tokens (e.g. `<DETAILED_CAPTION>`, `<MORE_DETAILED_CAPTION>`) to control style.

### Detailed Captioning

```bash
python -m scripts.florence2_cli detailed-caption data/samples/image.jpg \
    --max-new-tokens 160 \
    --model-source microsoft/Florence-2-base
```

### Visual Question Answering

```bash
python -m scripts.florence2_cli vqa data/samples/image.jpg \
    "What object is on the table?" \
    --max-new-tokens 40 \
    --model-source microsoft/Florence-2-base
```

### OCR / Reading Text

```bash
python -m scripts.florence2_cli ocr data/samples/menu.jpg \
    --max-new-tokens 256 \
    --model-source microsoft/Florence-2-base
```

> ℹ️ The CLI automatically falls back to CPU if CUDA kernels trigger an error (for example, when GPU memory is exhausted). Cached models and processors are reloaded on the new device so commands continue without restarting the process.

---

## Fine-Tuning Florence-2 (Captioning)

Fine-tuning uses the same manifest format described above. The CLI wraps Hugging Face `Trainer` to fine-tune the causal LM head while keeping Florence’s multimodal prompt strategy intact.

```bash
python -m scripts.florence2_cli train data/caption/train_manifest.tsv \
    --val-manifest data/caption/val_manifest.tsv \
    --output-dir models/florence2-caption-finetuned \
    --model-source microsoft/Florence-2-base \
    --prompt-token "<CAPTION>" \
    --epochs 3 \
    --batch-size 8 \
    --learning-rate 1e-4 \
    --warmup-ratio 0.05 \
    --max-length 128 \
    --gradient-accumulation 2 \
    --logging-steps 25
```

After training, the CLI saves:

- The fine-tuned model weights under `output-dir`.
- The processor (`AutoProcessor`) so the checkpoint can be reloaded offline.

You can immediately re-use the checkpoint:

```bash
python -m scripts.florence2_cli caption data/samples/image.jpg \
    --model-source models/florence2-caption-finetuned
```

---

## Notes & Recommendations

- Install `einops` and `timm` before running; Florence-2’s remote code depends on them.
- The default device is CUDA when available; override with `--device cpu` to stay on CPU.
- Set `--cache-dir` if the default Hugging Face cache location is read-only.
- FP16 training is enabled automatically when a CUDA device is detected. Force on/off with `--fp16` / `--no-fp16`.
- Prompt masking assumes Florence’s task token (`<CAPTION>`) occupies a single token. If you use a custom multi-token prompt, adjust `--prompt-token` during training and update `DEFAULT_IGNORE_PROMPT_TOKENS` in `models/florence2_runtime.py`.

With these tools you can explore Florence-2 for quick experimentation, downstream task prototyping, and caption fine-tuning without leaving the command line.
