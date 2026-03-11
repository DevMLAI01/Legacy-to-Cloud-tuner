# Legacy-to-Cloud Tuner

Fine-tune a quantized LLaMA 3.1-8B model to automatically translate legacy **Netezza SQL** queries into optimized **PySpark DataFrame** code — running entirely on Google Colab's free T4 GPU.

---

## Overview

Migrating from IBM Netezza to a modern cloud data platform (Databricks, EMR, Synapse) requires rewriting thousands of SQL queries into PySpark. This project fine-tunes a large language model specifically for that transformation, producing runnable PySpark code as output.

The notebook is designed to be:
- **Zero-cost** — runs on Google Colab Free Tier (no paid GPU needed)
- **Memory-safe** — 4-bit quantization + LoRA keeps the 8B model within 15 GB VRAM
- **Production-ready** — exports a GGUF model deployable with Ollama, LM Studio, or llama.cpp

---

## Architecture

```
Netezza SQL (input)
        │
        ▼
┌───────────────────────────────┐
│   Meta-Llama-3.1-8B (4-bit)  │  ← Base model (frozen weights)
│   + LoRA Adapters (r=16)      │  ← Fine-tuned delta weights (~100 MB)
└───────────────────────────────┘
        │
        ▼
PySpark DataFrame code (output)
```

**Method:** QLoRA — 4-bit NF4 quantization of the base model + full-precision LoRA adapter training. Only ~0.8% of parameters are updated during training.

---

## Tech Stack

| Component | Library / Tool |
|---|---|
| Base Model | `unsloth/Meta-Llama-3.1-8B-bnb-4bit` |
| Fine-tuning Framework | [Unsloth](https://github.com/unslothai/unsloth) |
| Trainer | `trl` — `SFTTrainer` + `SFTConfig` |
| Adapter Method | `peft` — LoRA |
| Quantization | `bitsandbytes` — NF4 4-bit |
| Dataset Format | Alpaca (instruction / input / output) |
| Export Format | GGUF `q4_k_m` |
| Runtime | Google Colab Free Tier — Tesla T4, 15 GB VRAM |

---

## Notebook Structure

The project is a single Colab notebook: `netezza_to_pyspark_finetune.ipynb`

### Cell 1 — Environment Setup
Installs Unsloth using the official Colab build command (pre-built CUDA wheel) along with `trl`, `peft`, `bitsandbytes`, `transformers`, `xformers`, and `triton`. Verifies GPU availability on startup.

### Cell 2 — Model & LoRA Setup
Loads `unsloth/Meta-Llama-3.1-8B-bnb-4bit` with `load_in_4bit=True`, reducing the model footprint from ~16 GB to ~5 GB. Attaches LoRA adapters to all 7 projection layers (attention + FFN) with `r=16`. Uses Unsloth's custom gradient checkpointing to save an additional ~30% VRAM.

**LoRA target modules:** `q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj`

### Cell 3 — Data Pipeline
Builds a 3-item dummy dataset in Alpaca JSONL format, applies the prompt template, and **explicitly pre-tokenizes** the dataset before training. All raw string columns (`instruction`, `input`, `output`, `text`) are stripped via `.remove_columns()` so the trainer receives only numeric tensors.

> This explicit tokenization step is the critical bug prevention measure against `ValueError: too many dimensions 'str'` which occurs when SFTTrainer tries to auto-process string columns in recent versions of `trl`.

### Cell 4 — Training Loop
Configures `SFTConfig` (the correct modern class — not the deprecated `TrainingArguments`) and runs `trainer.train()` for 60 steps.

| Hyperparameter | Value | Reason |
|---|---|---|
| `max_steps` | 60 | Demo run; increase for production |
| `per_device_train_batch_size` | 2 | T4 VRAM constraint |
| `gradient_accumulation_steps` | 4 | Effective batch size = 8 |
| `learning_rate` | 2e-4 | Standard LoRA LR |
| `optim` | `adamw_8bit` | Halves optimizer state memory |
| `fp16` | `True` | T4 does not support bf16 |
| `warmup_steps` | 5 | Stabilizes early training |

### Cell 5 — Export
Saves the LoRA adapter (~100 MB) then merges it into the base model and exports to GGUF format using Unsloth's built-in llama.cpp pipeline. No manual llama.cpp compilation required.

> **Important:** `save_pretrained_gguf` automatically appends `_gguf` to the directory argument. Pass `"model"` → actual output lands in `model_gguf/`.

| Output | Path | Size |
|---|---|---|
| LoRA adapter | `./lora_model/` | ~100 MB |
| GGUF (q4_k_m) | `./model_gguf/` | ~4–5 GB |

### Cell 6 — Inference & Validation
Switches the model to Unsloth's fast inference mode and runs 3 unseen Netezza SQL test queries through the model, printing the generated PySpark output for manual inspection.

Includes a runtime-restart fallback that reloads the model from the saved LoRA adapter on disk if the Colab session was interrupted.

---

## Bug Prevention Reference

A summary of breaking API changes that are guarded against in this notebook:

| Bug | Symptom | Fix Applied |
|---|---|---|
| Deprecated `tokenizer=` arg | `TypeError` in SFTTrainer init | Use `processing_class=tokenizer` |
| `TrainingArguments` missing SFT fields | `TypeError` on unknown kwargs | Import and use `SFTConfig` from `trl` |
| String columns passed to trainer | `ValueError: too many dimensions 'str'` | Pre-tokenize + `.remove_columns()` all string fields |
| bf16 on T4 GPU | Silent precision errors / crashes | `fp16=True`, `bf16=False` explicitly |
| `save_pretrained_gguf` dir naming | `WARNING: No .gguf file found` (false alarm) | Pass `"model"` not `"model_gguf"` as the dir argument |

---

## Running in Google Colab

1. Open `netezza_to_pyspark_finetune.ipynb` in Google Colab
2. Set runtime: **Runtime → Change runtime type → T4 GPU**
3. Run cells **in order** (Cell 1 → 6)
4. Cell 1 may prompt a kernel restart after install — do so, then continue from Cell 2

> Estimated total runtime: ~25–35 minutes (install 5 min + training 10 min + GGUF export 15 min)

---

## Local Deployment (after export)

**Ollama:**
```bash
ollama create netezza-pyspark -f ./model_gguf/Meta-Llama-3.1-8B.Q4_K_M.gguf
ollama run netezza-pyspark
```

**llama.cpp:**
```bash
./llama.cpp/llama-cli \
  --model model_gguf/Meta-Llama-3.1-8B.Q4_K_M.gguf \
  -p "Translate this Netezza SQL to PySpark: SELECT ..."
```

---

## Scaling to Production

This notebook uses 3 dummy training examples — sufficient to validate the pipeline end-to-end. For a production-grade translator:

| What to scale | Recommendation |
|---|---|
| Dataset size | 500–5,000 real Netezza → PySpark pairs |
| Training steps | `max_steps=500` or use `num_train_epochs=3` |
| LoRA rank | Increase `r` to 32 or 64 for harder patterns |
| Evaluation | Add a validation split; track `eval_loss` |
| Hardware | Colab Pro (A100) for larger batch sizes |

---

## Project Structure

```
Legacy-to-Cloud-tuner/
├── netezza_to_pyspark_finetune.ipynb   # Main notebook (all 6 cells)
├── train.jsonl                         # Generated during Cell 3 run
├── lora_model/                         # Saved after Cell 5 run
│   ├── adapter_config.json
│   └── adapter_model.safetensors
├── model_gguf/                         # Saved after Cell 5 run
│   └── Meta-Llama-3.1-8B.Q4_K_M.gguf
└── outputs/                            # Trainer checkpoints
```

---

## License

This project is provided for research and internal tooling purposes.
The base model (`Meta-Llama-3.1-8B`) is subject to the [Meta Llama 3.1 Community License](https://llama.meta.com/llama3_1/license/).
