# 📂 `train_and_evaluate` — Training & Evaluation Guide

This folder hosts **self‑contained scripts** for fine‑tuning small language models on the Mental‑Emotion dataset with 4‑bit **QLoRA** and evaluating their performance.

> **Scope** · One‑shot experiments for research validation, *not* a production pipeline.

---

## 1. Folder Layout

```text
train_and_evaluate/
├── QLoRA_Qwen_2.5_3B_train.py   # Fine‑tunes Qwen‑2.5‑3B
├── QLoRA_Llama_3.2_3B_train.py  # Fine‑tunes Llama‑3.2‑3B
└── outputs/                     # <script‑generated> checkpoints & logs
```

Each script internally downloads (or re‑uses) the base checkpoint to `<ROOT_PATH>/pretrained_models/<MODEL_NAME>`.

---

## 2. Prerequisites

| Requirement | Minimum                                                                                        | Notes                          |
| ----------- | ---------------------------------------------------------------------------------------------- | ------------------------------ |
| **Python**  | 3.10+                                                                                          | Tested on 3.11                 |
| **CUDA**    | 12.1                                                                                           | NVIDIA Ampere↑ GPU (bfloat16)  |
| **VRAM**    | ≥ 24 GB                                                                                        | Effective batch = 16 (1×16 GA) |
| **PyPI**    | `transformers>=4.40.0`, `datasets`, `bitsandbytes`, `peft`, `accelerate`, `wandb` *(optional)* |                                |

Install once:

```bash
pip install -r requirements.txt  # or copy the list above
```

> **Tip**  If compilation of **bitsandbytes** fails, use the official pre‑built wheels:
> `pip install bitsandbytes-cuda121` (or your CUDA version).

---

## 3. Data Preparation

1. Run the prompting script **once** to create training/test CSVs:

   ```bash
   python ../Tiny-Mental-LLM/Mental_Emotion_Dataset/raw_dataset/DepSeverity/add_prompt.py \
          --seed 42 --train_ratio 0.9
   ```

2. Place or symlink the resulting folder
   `Tiny-Mental-LLM/Mental_Emotion_Dataset/prompted_dataset/DepSeverity/`
   next to this `train_and_evaluate` directory **or** pass its path via `--train_file` / `--eval_file`.

Dataset CSVs must contain exactly two columns:

```csv
text,label
"I feel empty inside...",2
...
```

---

## 4. Quick‑Start Commands

### Qwen‑2.5‑3B (4‑bit QLoRA)

```bash
python QLoRA_Qwen_2.5_3B_train.py \
  --root_path /workspace/iclab_slm \
  --train_file ../Tiny-Mental-LLM/.../train.csv \
  --eval_file  ../Tiny-Mental-LLM/.../test.csv \
  --batch_size 1 --gradient_accumulation_steps 16 \
  --num_epochs 3 --seed 42
```

### Llama‑3.2‑3B (4‑bit QLoRA)

```bash
python QLoRA_Llama_3.2_3B_train.py \
  --root_path /workspace/iclab_slm \
  --train_file ... --eval_file ...
```

Both scripts share a common argument schema:

| Flag                            | Default           | Meaning                                                |
| ------------------------------- | ----------------- | ------------------------------------------------------ |
| `--root_path`                   | **REQUIRED**      | Project root where `pretrained_models/` lives          |
| `--train_file` / `--eval_file`  | Relative CSV path | Custom dataset location                                |
| `--batch_size`                  | 1                 | GPU micro‑batch size                                   |
| `--gradient_accumulation_steps` | 16                | Accumulate steps to reach effective batch *(batch×GA)* |
| `--max_length`                  | 512               | Context length (tokens)                                |
| `--num_epochs`                  | 3                 | Fine‑tuning epochs                                     |
| `--seed`                        | 42                | RNG seed for reproducibility                           |

> **bfloat16 note** ‑ Scripts set `bf16=True`; disable with `--no_bf16` when using older GPUs.

---

## 5. Outputs

After each epoch the *best* checkpoint (lowest loss) is saved to

```
train_and_evaluate/outputs/<model>-qlora-<timestamp>/
├── adapter_config.json   # LoRA adapter
├── adapter_model.bin     # Trained Δ‑weights (few MB)
├── trainer_state.json    # Log history
└── ...
```

Keep the base model in `pretrained_models/` and load the adapter via PEFT:

```python
from peft import PeftModel
from transformers import AutoModelForCausalLM

base = AutoModelForCausalLM.from_pretrained("qwen/Qwen1.5-4B")
model = PeftModel.from_pretrained(base, "outputs/qwen-qlora-2025-05-11")
```

---

## 6. Evaluation & Perplexity

Each script runs an **epoch‑level eval** pass (`Trainer.evaluate`) and prints
perplexity (if `compute_metrics` is enabled) to stdout and `trainer_state.json`.
For custom metrics add a `compute_metrics()` function near the bottom of the script.

Example:

```python
def compute_metrics(eval_preds):
    logits, labels = eval_preds
    preds = logits.argmax(-1)
    acc = (preds == labels).mean()
    return {"accuracy": acc}
```

---

## 7. Reproducibility Checklist

* `--seed` passed → seeds Python, NumPy, PyTorch, and `random`.
* `deterministic=True` is set internally when the seed flag is on.
* Split files are version‑controlled; re‑generate only with explicit seed.

---

## 8. Troubleshooting

| Symptom                                   | Likely Cause                    | Fix                                                           |
| ----------------------------------------- | ------------------------------- | ------------------------------------------------------------- |
| **CUDA out of memory**                    | batch × GA too high             | Lower `--gradient_accumulation_steps` or `--max_length`       |
| `RuntimeError: bf16 not supported`        | GPU < Ampere                    | Add `--no_bf16` flag (casts to fp16)                          |
| `AttributeError: tril` during ONNX export | PyTorch / transformers mismatch | Upgrade PyTorch ≥ 2.2 or apply monkey‑patch (see `sLM_SNPE/`) |

If issues persist, raise an [Issue](https://github.com/your‑org/your‑repo/issues) with full logs.

---

## 10. Practical QLoRA & LoRA Tips

Below is a distilled checklist of actionable advice drawn from Sebastian Raschka’s *“Practical Tips for Fine‑Tuning LMs Using LoRA”* article on Continuum Labs¹. These apply equally to our QLoRA scripts in this folder.

| Area                       | Tip                                                                                                                           |
| -------------------------- | ----------------------------------------------------------------------------------------------------------------------------- |
| **Layer Coverage**         | Apply LoRA to **all linear layers** (Q, K, V, O *and* MLP projections) for best accuracy—KV‑only adapters often underperform. |
| **Rank (r) & Alpha (α)**   | Start with **α ≈ 2 × r**. Scaling factor α/r should neither explode (>32) nor vanish (<1).                                    |
| **QLoRA Trade‑offs**       | 4‑bit QLoRA cuts **\~⅓ VRAM** but slows training by **≈ 40 %**; use when memory‑bound.                                        |
| **Optimizer Choice**       | AdamW, AdamW + scheduler, or SGD + scheduler perform similarly; plain SGD lags.                                               |
| **Epoch Count**            | More epochs ≠ better—multi‑epoch passes on the *same* small dataset quickly overfit. Monitor val loss & stop early.           |
| **Reproducibility**        | Despite GPU nondeterminism, LoRA runs are **remarkably consistent**. Fix seeds and log configs to compare fairly.             |
| **Single‑GPU Feasibility** | With QLoRA (r=256, α=512) you can fine‑tune a **7 B** model in ≈ 3 h on one A100 (14 GB).                                     |
| **Memory Hotspots**        | Most memory goes to activations, not Adam states; Adam’s 2× param overhead is negligible for LLMs.                            |
| **Combining Adapters**     | You can merge or stack multiple LoRA adapters for multi‑domain expertise—experiment!                                          |

¹ [https://training.continuumlabs.ai/training/the-fine-tuning-process/parameter-efficient-fine-tuning/practical-tips-for-fine-tuning-lms-using-lora-low-rank-adaptation](https://training.continuumlabs.ai/training/the-fine-tuning-process/parameter-efficient-fine-tuning/practical-tips-for-fine-tuning-lms-using-lora-low-rank-adaptation)

---

## 9. Citation & Credits

* **Qwen‑2.5‑3B** — *Qwen Team, 2024*
* **Llama‑3.2‑3B** — *Meta AI, 2025*
* **QLoRA** — *Dettmers et al., ICML 2023*

---

Happy fine‑tuning! 🚀
