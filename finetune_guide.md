# Fine‑tuning a Small LLM with 🤗 TRL (SFT) – Step‑by‑step guide

**Target**: Fine‑tune a ~1 B‑parameter causal language model on a custom instruction‑following dataset (e.g. `tatsu‑lab/alpaca`).

---

## 1. Why use `SFTTrainer`?
* Implements the classic supervised‑fine‑tuning (SFT) recipe used in most instruction‑tuned LLMs.
* Handles **standard LM**, **prompt‑completion**, and **conversational** dataset formats automatically.
* Built on top of `transformers.Trainer` → inherits all battle‑tested features (mixed‑precision, gradient checkpointing, logging, checkpointing, etc.).
* Seamless integration with **PEFT** (LoRA, adapters) and **TrackIO** for experiment tracking.

---

## 2. Prerequisites
| Item | Command |
|------|---------|
| Python (≥3.9) | `python -V` |
| `pip` | `python -m pip install --upgrade pip` |
| Core libraries | `pip install transformers datasets trl accelerate trackio` |
| Optional PEFT | `pip install peft` |
| Optional GPU kernels | `pip install flash-attn` |
| (If using HF Jobs) | No local install needed – just a script and a `requirements.txt`.

*Make sure you have a CUDA‑enabled GPU (≥12 GB VRAM) for reasonable speed. If you only have CPU, add `--no_cuda` to the script.

---

## 3. Inspect the dataset
We’ll use the public **Alpaca** dataset (`tatsu‑lab/alpaca`).
```bash
python - <<'PY'
from datasets import load_dataset

ds = load_dataset("tatsu-lab/alpaca", split="train")
print(ds.column_names)
print(ds[0])
PY
```
Typical columns:
```text
['instruction', 'input', 'output', 'text']
```
* `instruction` + optional `input` → **prompt**
* `output` → **completion**
* `text` already contains a fully formatted prompt‑completion string that can be used directly.

### 3.1 Convert to the format expected by `SFTTrainer`
`SFTTrainer` accepts any of the following canonical formats:
```json
# Standard LM (single "text" field)
{"text": "..."}

# Prompt‑completion
{"prompt": "...", "completion": "..."}

# Conversational
{"messages": [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]}
```
The Alpaca dataset already matches the **prompt‑completion** style, but we must rename the columns:
```python
from datasets import load_dataset

ds = load_dataset("tatsu-lab/alpaca", split="train")

def rename(example):
    return {
        "prompt": example["instruction"] + ("\n" + example["input"] if example["input"] else ""),
        "completion": example["output"],
    }

train_ds = ds.map(rename, remove_columns=[c for c in ds.column_names if c not in ["instruction", "input", "output"]])
```
*If you prefer to keep the ready‑made `text` column, you can simply pass `train_dataset=load_dataset(..., split="train")` – the trainer will treat it as a **standard LM**.

---

## 4. Training script (minimal, works locally & on HF Jobs)
Save the following as `train_sft.py` in the repo root.
```python
# ------------------------------------------------------------
# train_sft.py – Supervised Fine‑Tuning with 🤗 TRL
# ------------------------------------------------------------
import argparse
from pathlib import Path

from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTTrainer, SFTConfig, ScriptArguments, ModelArguments, get_peft_config

# ------------------------------------------------------------------
# Helper: parse CLI arguments (mirrors the example in the TRL repo)
# ------------------------------------------------------------------
parser = argparse.ArgumentParser(description="SFT training script")
# 1️⃣ Script‑level arguments (dataset configuration)
parser.add_argument("--dataset_name", type=str, default="tatsu-lab/alpaca",
                    help="HF hub identifier of the training dataset")
parser.add_argument("--dataset_config", type=str, default=None,
                    help="Optional config name for the dataset (if it has multiple configs)")
parser.add_argument("--train_split", type=str, default="train",
                    help="Dataset split to use for training")
# 2️⃣ Model arguments (model to fine‑tune)
parser.add_argument("--model_name_or_path", type=str, default="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
                    help="Model identifier on the hub or a local checkpoint")
parser.add_argument("--dtype", type=str, default="bfloat16",
                    help="Model dtype – bfloat16 works well on recent GPUs")
# 3️⃣ Training arguments (SFTConfig – a thin wrapper around Transformers TrainingArguments)
parser.add_argument("--output_dir", type=str, default="./sft‑output",
                    help="Where to store checkpoints & final model")
parser.add_argument("--per_device_train_batch_size", type=int, default=4,
                    help="Batch size per GPU. Adjust for your GPU memory.")
parser.add_argument("--gradient_accumulation_steps", type=int, default=8,
                    help="Accumulate gradients to simulate larger batch size.")
parser.add_argument("--learning_rate", type=float, default=2e-5,
                    help="Base learning rate.")
parser.add_argument("--num_train_epochs", type=float, default=3,
                    help="Number of epochs.")
parser.add_argument("--logging_steps", type=int, default=10,
                    help="Log every N steps (TrackIO will ingest these)")
parser.add_argument("--push_to_hub", action="store_true",
                    help="Push final checkpoint to the Hub (required for HF Jobs)")
parser.add_argument("--hub_model_id", type=str, default=None,
                    help="Repository name for the pushed model – e.g. username/your‑model")
# Optional PEFT (LoRA) – uncomment to enable lightweight fine‑tuning
# parser.add_argument("--use_lora", action="store_true", help="Enable LoRA adapters")

args = parser.parse_args()

# ------------------------------------------------------------------
# 1️⃣ Load tokenizer & model (respect dtype & device)
# ------------------------------------------------------------------
tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
# Some chat models ship without a pad token – we set it to eos if needed
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model_kwargs = {
    "torch_dtype": getattr(torch, args.dtype),
    "low_cpu_mem_usage": True,
}
model = AutoModelForCausalLM.from_pretrained(args.model_name_or_path, **model_kwargs)

# ------------------------------------------------------------------
# 2️⃣ (Optional) Wrap with LoRA – very memory‑efficient for 1 B‑scale models
# ------------------------------------------------------------------
# if args.use_lora:
#     from peft import LoraConfig
#     peft_cfg = LoraConfig(r=8, lora_alpha=32, target_modules=["q_proj", "v_proj"], 
#                         bias="none", task_type="CAUSAL_LM")
# else:
#     peft_cfg = None
peft_cfg = None  # keep it simple for the base guide

# ------------------------------------------------------------------
# 3️⃣ Load & (optionally) preprocess the dataset
# ------------------------------------------------------------------
raw_ds = load_dataset(args.dataset_name, name=args.dataset_config, split=args.train_split)

# Convert Alpaca’s columns to the canonical prompt/completion format
def to_prompt_completion(example):
    prompt = example["instruction"]
    if example["input"]:
        prompt += "\n" + example["input"]
    return {"prompt": prompt, "completion": example["output"]}

train_dataset = raw_ds.map(to_prompt_completion, remove_columns=[c for c in raw_ds.column_names if c not in ["instruction", "input", "output"]])

# ------------------------------------------------------------------
# 4️⃣ Assemble the SFT configuration
# ------------------------------------------------------------------
training_args = SFTConfig(
    output_dir=args.output_dir,
    per_device_train_batch_size=args.per_device_train_batch_size,
    gradient_accumulation_steps=args.gradient_accumulation_steps,
    learning_rate=args.learning_rate,
    num_train_epochs=args.num_train_epochs,
    logging_steps=args.logging_steps,
    fp16=False,  # we use bfloat16 instead
    bf16=args.dtype == "bfloat16",
    gradient_checkpointing=True,
    report_to=["trackio"] if args.push_to_hub else [],
    push_to_hub=args.push_to_hub,
    hub_model_id=args.hub_model_id,
    # For instruction‑tuning we usually want `assistant_only_loss=True`
    assistant_only_loss=True,
)

# ------------------------------------------------------------------
# 5️⃣ Instantiate the trainer and start training
# ------------------------------------------------------------------
trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    args=training_args,
    train_dataset=train_dataset,
    peft_config=peft_cfg,
)

trainer.train()
trainer.save_model(args.output_dir)
if args.push_to_hub:
    trainer.push_to_hub()

print("✅ Fine‑tuning complete. Model saved to", args.output_dir)
```
**Explanation of key flags**
* `assistant_only_loss=True` – loss is computed **only** on assistant messages; the user prompt is ignored (standard for instruction‑tuning).
* `gradient_checkpointing=True` – halves VRAM usage at a minor speed cost.
* `bf16` (or `fp16`) – mixed‑precision dramatically speeds up training on recent GPUs.
* `per_device_train_batch_size` + `gradient_accumulation_steps` – adjust to fit your GPU. Example: batch 4 + accumulate 8 → effective batch 32.

---

## 5. Running locally with **Accelerate** (optional but recommended)
```bash
# 1️⃣ Install accelerate and create a config (single‑GPU example)
pip install accelerate
accelerate config
# Follow the prompts – select "CUDA” and “single GPU”.

# 2️⃣ Launch the script
accelerate launch train_sft.py \
    --model_name_or_path TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
    --dataset_name tatsu-lab/alpaca \
    --output_dir ./sft‑tinyllama \
    --push_to_hub \
    --hub_model_id <your-username>/tinyllama‑alpaca
```
*`accelerate launch`* handles distributed‐training flags for you; you can later scale to multi‑GPU or multi‑node by adjusting the config.

---

## 6. Submitting the job to **HF Jobs** (cloud, no local GPU needed)
1. **Create a repo** (e.g. `username/llm‑sft‑script`).
2. **Upload the script** (`train_sft.py`) and a minimal `requirements.txt`:
   ```text
   transformers
   datasets
   trl
   accelerate
   trackio
   torch
   ```
3. **Run the job** (replace placeholders):
```bash
hf jobs run \
  --script ./train_sft.py \
  --dependencies "transformers,datasets,trl,accelerate,trackio,torch" \
  --hardware-flavor a10g-large \
  --timeout 8h \
  --args "--model_name_or_path TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
          --dataset_name tatsu-lab/alpaca \
          --output_dir ./sft-output \
          --push_to_hub \
          --hub_model_id <your-username>/tinyllama‑alpaca"
```
*The job will automatically stream logs; copy the displayed **TrackIO URL** to monitor loss curves in real time.*

---

## 7. Verifying the fine‑tuned model
```python
from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer

model_id = "<your-username>/tinyllama-alpaca"
pipe = pipeline("text-generation", model=model_id, tokenizer=model_id, torch_dtype="bfloat16", device=0)

prompt = [{"role": "user", "content": "Write a short poem about the moon."}]
print(pipe(prompt))
```
You should see a coherent, instruction‑following response.

---

## 8. Hyper‑parameter sweep (basic grid search)
Create `sweep.yaml` for **HF Jobs**:
```yaml
method: grid
metric: loss
goal: minimize
parameters:
  learning_rate:
    values: [1e-5, 2e-5, 5e-5]
  per_device_train_batch_size:
    values: [2, 4]
  num_train_epochs:
    values: [2, 3]
```
Submit the sweep (replace `<repo>` and `<script>`):
```bash
hf jobs sweep \
  --script ./train_sft.py \
  --sweep-config sweep.yaml \
  --hardware-flavor a10g-large \
  --dependencies "transformers,datasets,trl,accelerate,trackio,torch"
```
Each trial will push its checkpoint to a sub‑folder under your hub repo, making comparative analysis easy.

---

## 9. Checklist before you start
- [ ] **Dataset columns** match one of the accepted formats (`text`, `prompt/completion`, or `messages`).
- [ ] **Model & tokenizer** are compatible (most HF models are).
- [ ] `assistant_only_loss=True` for instruction‑tuning (optional otherwise).
- [ ] `push_to_hub=True` **or** a backup path is set – otherwise the trained weights disappear after the job.
- [ ] `timeout` ≥ 2 h for any real training run.
- [ ] GPU memory fits the effective batch size – adjust `per_device_train_batch_size` / `gradient_accumulation_steps` accordingly.
- [ ] TrackIO dashboard URL is bookmarked for live monitoring.

---

## 10. References & further reading
* 🤗 TRL **SFTTrainer** docs – https://huggingface.co/docs/trl/sft_trainer
* Alpaca dataset – https://huggingface.co/datasets/tatsu-lab/alpaca
* TinyLlama‑Chat model – https://huggingface.co/TinyLlama/TinyLlama-1.1B-Chat-v1.0
* **LoRA** tutorial (optional) – https://huggingface.co/docs/peft/main/en/quickstart#lora
* **TrackIO** monitoring guide – https://huggingface.co/docs/trackio

---

*You now have a fully reproducible pipeline to fine‑tune a small LLM on any instruction dataset, with optional cloud‑scale execution and experiment tracking.*
