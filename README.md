# Gemma 3n Impact Challenge — Multimodal Fine-tuning & Inference

My submission for [Google's Gemma 3n Impact Challenge](https://kaggle.com/competitions/gemma-impact-challenge).

This notebook demonstrates how to fine-tune **Google's Gemma 3N multimodal model** on a custom dataset using [Unsloth](https://github.com/unslothai/unsloth) for memory-efficient training, and then run inference on the fine-tuned model — all on a free **Tesla T4 GPU** in Google Colab.

---

## 📋 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Requirements](#requirements)
- [Getting Started](#getting-started)
- [Notebook Walkthrough](#notebook-walkthrough)
- [Training Configuration](#training-configuration)
- [Results](#results)
- [Model Saving Options](#model-saving-options)
- [License](#license)

---

## Overview

**Gemma 3N** is Google's efficient multimodal model family capable of processing text, images, and audio. This project fine-tunes the `unsloth/gemma-3n-E2B-it` (2B parameter, instruction-tuned) variant using **LoRA (Low-Rank Adaptation)** with 4-bit quantization, making the entire process accessible on consumer-grade hardware.

The fine-tuning is performed on the [FineTome-100k](https://huggingface.co/datasets/mlabonne/FineTome-100k) dataset (first 3,000 samples) formatted in ShareGPT-style multi-turn conversations.

---

## Features

- 🚀 **2× faster fine-tuning** via Unsloth optimizations
- 🧠 **4-bit quantization** (BNB) to minimize VRAM usage
- 🔧 **LoRA adapters** — only 0.53% of parameters trained (~10.5M / 2B)
- 🖼️ **Multimodal inference** — text, image, and audio inputs
- 💬 **Response-only training** using `train_on_responses_only` to focus loss on model outputs
- 💾 **Multiple export formats** — LoRA adapters, merged model, GGUF

---

## Requirements

| Requirement | Details |
|---|---|
| **GPU** | NVIDIA Tesla T4 (≥14 GB VRAM) or better |
| **CUDA** | 12.4+ |
| **Python** | 3.10+ |
| **Platform** | Linux (Google Colab recommended) |

### Key Dependencies

```
torch==2.6.0+cu124
transformers==4.54.0.dev0
unsloth==2025.6.12
trl
datasets
timm
xformers==0.0.29.post3
```

> All dependencies are installed automatically by the first cell of the notebook.

---

## Getting Started

1. **Open in Google Colab** — upload or open `gemma-3n-4b-multimodal-finetuning-inference.ipynb`.
2. **Set runtime** to *T4 GPU* (Runtime → Change runtime type → T4 GPU).
3. **Run all cells** sequentially.

No local setup is required.

---

## Notebook Walkthrough

| Section | Description |
|---|---|
| **1. Installation** | Installs PyTorch, Unsloth, Transformers, and other dependencies |
| **2. Model Loading** | Loads `gemma-3n-E2B-it` with 4-bit quantization via Unsloth |
| **3. Multimodal Inference Demo** | Demonstrates vision, text, and audio capabilities before fine-tuning |
| **4. Data Preparation** | Loads FineTome-100k, applies Gemma-3 chat template, standardizes format |
| **5. Training** | Fine-tunes with LoRA using HuggingFace TRL's `SFTTrainer` |
| **6. Post-training Inference** | Runs generation with the fine-tuned model |
| **7. Model Saving** | Saves LoRA adapters, merged model, or GGUF quantized format |

---

## Training Configuration

| Hyperparameter | Value |
|---|---|
| **Base Model** | `unsloth/gemma-3n-E2B-it` |
| **Quantization** | 4-bit (BNB) |
| **Max Sequence Length** | 1024 |
| **LoRA Trainable Parameters** | 10,567,680 (0.53% of 2B) |
| **Batch Size** | 1 (device) × 4 (grad accum) = effective 4 |
| **Learning Rate** | 2e-4 |
| **Optimizer** | AdamW 8-bit |
| **LR Scheduler** | Linear |
| **Warmup Steps** | 5 |
| **Max Steps** | 60 (demo; set `num_train_epochs=1` for full run) |
| **Weight Decay** | 0.01 |
| **Seed** | 3407 |
| **Dataset** | mlabonne/FineTome-100k (first 3,000 samples) |

**Inference settings** (Gemma-3 recommended):
- `temperature = 1.0`, `top_p = 0.95`, `top_k = 64`

---

## Results

- ⏱️ **Training time**: ~6.5 minutes (60 steps on T4)
- 🎯 **Peak GPU memory**: ~10.99 GB (74.5% of 14.74 GB T4 VRAM)
- ✅ Model produces coherent multi-turn responses and handles multimodal inputs after fine-tuning

---

## Model Saving Options

After training, the notebook provides three export options:

```python
# Option 1: Save LoRA adapters only (smallest)
model.save_pretrained("gemma-3n-lora-adapters")

# Option 2: Save merged 16-bit model
model.save_pretrained_merged("gemma-3n-merged", tokenizer, save_method="merged_16bit")

# Option 3: Save as GGUF (for llama.cpp / Ollama)
model.save_pretrained_gguf("gemma-3n-gguf", tokenizer, quantization_method="q4_k_m")
```

---

## License

This project is released under the [Apache 2.0 License](LICENSE).
