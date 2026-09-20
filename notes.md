# Fine-Tuning 

---

# PART 1: CORE CONCEPTS

## 1.1 What is fine-tuning?
- A pre-trained model has learned general language. Fine-tuning continues its training on a small, specialized dataset so it behaves the way you need (e.g., a customer-support agent).
- Analogy: hiring a smart graduate and giving them two weeks of company-specific training.
- Benefits: better accuracy, reduced bias, improved knowledge base.
- Success depends on:
  - Data quality
  - Model capacity (a tiny model can only learn so much)
  - A clearly defined task

## 1.2 The pipeline (big picture)
**model + tokenizer + training dataset + training arguments + fine-tuning class -> training -> fine-tuned model -> evaluation**
- **Tokenizer**: converts text into numeric tokens.
- **Training arguments**: settings (learning rate, batch size, etc.).
- **Fine-tuning class**: runs the training loop (`SFTTrainer` or a TorchTune recipe).
- **Evaluation**: test on unseen data using a metric or benchmark.

## 1.3 What happens when we train a model?
- Tokens form an input vector.
- Matrix (model) multiplication produces output vectors.
- Errors are used to update the model weights.
- Model size determines training difficulty.

---

# PART 2: THE TOOLS (Library Options)

## 2.1 Options for Llama fine-tuning
| Library | Strength | Ideal for |
|---|---|---|
| TorchTune | Configurable templates (recipes) | Scaling quickly |
| SFTTrainer (Hugging Face) | Access to other LLMs | Fine-tuning multiple models |
| Unsloth | Efficient memory usage | Limited hardware |
| Axolotl | Modular approach | No extensive reconfiguration |

## 2.2 Are Hugging Face and TorchTune different ways to fine-tune?
**Yes. They are two different toolchains for the same job (supervised fine-tuning).** You normally pick one per project, not both.

### What "Hugging Face" means here
An ecosystem of libraries plus a model hub, each covering one step:
| Piece | Role |
|---|---|
| Hub (huggingface.co) | Hosts models (e.g. `Maykeye/TinyLLama-v0`, `nvidia/Llama3-ChatQA-1.5-8B`) and datasets (e.g. Bitext) |
| transformers | Loads models/tokenizers (`AutoModel...`, `AutoTokenizer`), provides `TrainingArguments` |
| datasets | Loads, maps, splits, saves data |
| TRL | Provides `SFTTrainer` (the fine-tuning class) |
| PEFT | Provides LoRA (`LoraConfig`) |
| bitsandbytes | Provides quantization (`BitsAndBytesConfig`) |
| evaluate | Provides metrics such as ROUGE |
- Style: you write **Python code** and wire the pieces together.
- Strength: works with a huge range of models, not just Llama.

### What "TorchTune" means
- A PyTorch library focused on fine-tuning LLMs (Llama in these slides).
- Style: **command line + YAML config**. Pick a **recipe** and a **config**, then run `tune run`.
- Strength: reproducible, quick to scale, less code.
- Can still use Hugging Face `datasets` for data, which is why data prep looks the same in both routes.

### Side by side
| | Hugging Face (SFTTrainer) | TorchTune |
|---|---|---|
| Interface | Python code | CLI + YAML |
| You control | Every step in code | Settings in a config file |
| Model support | Many LLM families | Llama-focused |
| LoRA | `peft` + `LoraConfig` | LoRA recipes/configs |
| Quantization | `bitsandbytes` | Via configs (e.g. 8-bit optimizer in sample recipe) |
| Best for | Multiple models, custom experiments | Scaling fast, reproducible Llama runs |

### How to choose
- Flexibility and many model types -> Hugging Face SFTTrainer
- Reproducible, config-driven Llama runs that scale -> TorchTune
- Tight on GPU memory -> Unsloth, or LoRA + quantization in either route
- Modular setup with little rework -> Axolotl

### Shared across every route
- Same data prep (train/validation/test, one text field)
- Same core concepts (learning rate, batch size, epochs, loss)
- Same techniques available (LoRA, quantization)
- Same evaluation (held-out data + ROUGE)

---

# PART 3: DATA PREPARATION

## 3.1 Dataset splits
- **Train set**: the model learns from it (majority of data).
- **Validation set**: used during development to pick the best version (e.g., best checkpoint).
- **Test set**: used once at the end for an honest performance estimate.
- Why split? Models can memorize training data. Grading on practiced questions is misleading.
- **Quality of the data is key.**

## 3.2 Hugging Face `datasets` library
Handles preprocessing, splitting, loading, and memory management.

```python
from datasets import load_dataset, Dataset

ds = load_dataset(
    'bitext/Bitext-customer-support-llm-chatbot-training-dataset',
    split="train")
print(ds.column_names)   # ['flags','instruction','category','intent','response']
print(ds.shape)          # (26872, 5)

# Peek at a row
import pprint
pprint.pprint(ds[0])

# Filter: keep first 1000 rows
first_thousand_points = ds[:1000]
ds = Dataset.from_dict(first_thousand_points)

# Preprocess: merge instruction + response into one text field
def merge_example(row):
    row['conversation'] = f"Query: {row['instruction']}\nResponse: {row['response']}"
    return row
ds = ds.map(merge_example)

# Save / load
ds.save_to_disk("preprocessed_dataset")
from datasets import load_from_disk
ds_preprocessed = load_from_disk("preprocessed_dataset")
```
- `SFTTrainer` needs one text field, hence the merge into `conversation`.
- The model learns the pattern: after `Query: ...` comes `Response: ...`.
- `.map()` applies the function to every row.
- `save_to_disk()` / `load_from_disk()` avoids redoing preprocessing.

---

# PART 4: THE FLOW

## 4.1 Hugging Face route (SFTTrainer)
1. **Load the dataset**: `load_dataset(..., split="train")`
2. **Inspect / filter**: `ds.column_names`, `ds[0]`, `ds.shape`, optional subset
3. **Preprocess**: `ds.map(merge_example)` to create the `conversation` column; optionally save to disk
4. **Split the data**: train / validation / test (keep an evaluation set aside)
5. **Load model and tokenizer**
   - `AutoModelForCausalLM.from_pretrained(model_name)`
   - `AutoTokenizer.from_pretrained(model_name)`
   - `tokenizer.pad_token = tokenizer.eos_token`
6. **(Optional) Quantize the model** with `BitsAndBytesConfig`, passed as `quantization_config=`
7. **(Optional) Define the LoRA config** with `LoraConfig` (required if the model is quantized)
8. **Define training arguments**: `TrainingArguments(...)`
9. **Build the trainer**: `SFTTrainer(model, tokenizer, train_dataset, dataset_text_field, max_seq_length, args, peft_config)`
10. **Train**: `trainer.train()`, then check loss, runtime, epochs
11. **Evaluate**: generate predictions on the eval set, compute ROUGE, compare against the base model

**One-liner:** load dataset -> inspect/filter -> preprocess -> split -> load model + tokenizer -> (quantize) -> (LoRA config) -> training arguments -> SFTTrainer -> train -> generate on eval set -> ROUGE

## 4.2 TorchTune route
1. **Install**: `pip3 install torchtune`
2. **Pick a model config**: `tune ls`
3. **Prepare the dataset** with HF `datasets` -> `ds.save_to_disk("preprocessed_dataset")`
4. **Write or choose the recipe config (YAML)**: general settings, model, optimizer, dataset
5. **Launch training**: `tune run ...`
6. **Monitor**: logs, epoch/step progress, loss
7. **Evaluate** on held-out data (e.g. ROUGE), same idea as the HF route

**One-liner:** install -> `tune ls` -> prepare dataset -> write recipe YAML -> `tune run` -> monitor loss -> evaluate

---

# PART 5: TORCHTUNE IN DETAIL

## 5.1 Recipes vs configs
- **Recipe** = the training procedure (what code runs)
  - `full_finetune_single_device` (one GPU)
  - `full_finetune_distributed` (many GPUs)
- **Config** = YAML file of settings (what values it uses)
- Recipes are modular templates. They keep code organized and ensure reproducibility.
- The whole experiment lives in one YAML file, so anyone can rerun it.

## 5.2 Three components of TorchTune fine-tuning
- **Model**: architecture + pre-trained weights (many versions/sizes via `tune ls`)
- **Dataset**: the training data (e.g. `ds.save_to_disk("new_dataset")`)
- **Recipe**: central config (e.g. `custom_recipe.yaml`) combining model, dataset, training parameters

## 5.3 Commands
```bash
pip3 install torchtune                     # install
tune ls                                    # list recipes and configs (!tune ls in IPython)
tune run full_finetune_single_device --config llama3_1/8B_lora_single_device
tune run full_finetune_single_device --config llama3/8B_full_single_device \
    dataset=preprocessed_dataset dataset.split=train
tune run --config custom_recipe.yaml       # run from your own YAML
```
- Pattern: `tune run <recipe> --config <config>`
- Overridable parameters: `device=cpu` or `device=cuda`, `epochs=<int>`
- `tune ls` output has RECIPE and CONFIG columns (e.g. `llama3/8B_full_single_device`, `llama3_1/8B_full_single_device`, `llama3_2/1B_full_single_device`, `llama3_2/3B_full_single_device`, `llama3/8B_full`).

## 5.4 Anatomy of a recipe
- General settings and output directory (batch size, device, epochs)
- Model (architecture and configuration)
- Optimizer (includes learning rate)
- Dataset (preprocessing and dataset path)

```yaml
batch_size: 4
device: cuda
epochs: 20
output_dir: /tmp/full-llama3.2-finetune

model:
  _component_: torchtune.models.llama3_2.llama3_2_1b

optimizer:
  _component_: bitsandbytes.optim.PagedAdamW8bit
  lr: 2.0e-05

dataset:
  _component_: torchtune.datasets.alpaca_dataset
```
- The optimizer decides how weights are updated from errors. `PagedAdamW8bit` stores optimizer state in 8-bit to save memory. `lr` = learning rate.

## 5.5 Building a recipe in Python
```python
import yaml
config_dict = {
    "batch_size": 4,
    "device": "cuda",
    "model": {"_component_": "torchtune.models.llama3_2.llama3_2_1b"},
    # ...
}
yaml_file_path = "custom_recipe.yaml"
with open(yaml_file_path, "w") as yaml_file:
    yaml.dump(config_dict, yaml_file)
```

## 5.6 Reading the run output
- Saved logs (path to the log file)
- Successful initialization (model precision, tokenizer loaded)
- Epoch and step progress
- Loss metrics

---

# PART 6: HUGGING FACE TRAINING IN DETAIL

## 6.1 What you need
1. Language model + tokenizer (e.g. `Maykeye/TinyLLama-v0`)
2. Training dataset (Bitext customer service)
3. Training arguments
4. Fine-tuning class (`SFTTrainer` from TRL)
5. Evaluation benchmark or dataset

## 6.2 Load model and tokenizer (Auto classes)
```python
model_name = "Maykeye/TinyLLama-v0"
model = AutoModelForCausalLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token
```
- Batches need equal-length sequences. Llama tokenizers often lack a pad token, so reuse the end-of-sequence token.

## 6.3 Training arguments
```python
training_arguments = TrainingArguments(
    per_device_train_batch_size=1,
    learning_rate=2e-3,
    max_grad_norm=0.3,
    max_steps=200,
    gradient_accumulation_steps=2,
    save_steps=10,
)
```
| Argument | Meaning |
|---|---|
| `per_device_train_batch_size` | Examples processed before one weight update. Larger = more GPU memory |
| `gradient_accumulation_steps` | Sums gradients over N mini-batches before updating. Effective batch = batch size x N (memory trick) |
| `learning_rate` | Step size of each update. Too high = unstable, too low = barely learns |
| `max_grad_norm` | Gradient clipping. Caps update size so one odd example can't wreck the model |
| `max_steps` | Stop after this many updates |
| `save_steps` | Save a checkpoint every N steps |

## 6.4 SFTTrainer
```python
trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=dataset,
    dataset_text_field='conversation',
    max_seq_length=250,
    args=training_arguments
)
trainer.train()
```
- **SFT** = Supervised Fine-Tuning.
- `dataset_text_field`: which column to learn from.
- `max_seq_length=250`: truncates long examples to bound memory.

## 6.5 Reading the output
- `global_step=200`, `training_loss ~1.94` (average error, lower is better)
- `train_runtime ~142.6 s`, samples/steps per second, `total_flos`
- `epoch: 2.0`: the model saw the full dataset twice

---

# PART 7: EVALUATION WITH ROUGE

- Loss shows how training went. ROUGE judges the generated text.
- **ROUGE-1**: ratio of word overlap between generated text and a reference.
  - "hello there" vs "hello there" = 100%
  - "general kenobi" vs "master yoda" = 0%
  - Average = 0.5
- **ROUGE-2**: overlap of 2-word sequences (stricter).
- **ROUGE-L**: longest common subsequence of words.

```python
import evaluate
rouge = evaluate.load('rouge')
predictions = ["hello there", "general kenobi"]
references  = ["hello there", "master yoda"]
results = rouge.compute(predictions=predictions, references=references)
# {'rouge1': 0.5, 'rouge2': 0.5, 'rougeL': 0.5, 'rougeLsum': 0.5}
```

## Running it on an evaluation set
```python
def generate_predictions_and_reference(dataset):
    predictions = []
    references = []
    for row in dataset:
        inputs = tokenizer.encode(row["instruction"], return_tensors="pt")
        outputs = model.generate(inputs)
        decoded_outputs = tokenizer.decode(
            outputs[0, inputs.shape[1]:], skip_special_tokens=True)
        references += [row["response"]]
        predictions += [decoded_outputs]
    return references, predictions

references, predictions = generate_predictions_and_reference(evaluation_dataset)
rouge = evaluate.load('rouge')
results = rouge.compute(predictions=predictions, references=references)
print(results)
```
- `outputs[0, inputs.shape[1]:]` skips the prompt and decodes only the new tokens.

## Results: fine-tuned vs not
| Metric | Fine-tuned | No fine-tuning |
|---|---|---|
| rouge1 | 0.224 | 0.131 |
| rouge2 | 0.040 | 0.046 |
| rougeL | 0.150 | 0.084 |
| rougeLsum | 0.187 | 0.122 |
- Fine-tuning improved rouge1, rougeL, rougeLsum. rouge2 dipped slightly, so it's not a uniform win.
- Caveat: scores are low in absolute terms. Correct answers phrased differently score poorly. Use ROUGE as a signal, not the whole truth.

---

# PART 8: LoRA (Low-Rank Adaptation)

## 8.1 The idea
- **Problem**: full fine-tuning updates all weights, which needs gradients and optimizer state for billions of parameters (expensive).
- **Idea**: freeze the original weight matrix **W** and learn only a small change:
  - **DeltaW = B x A**, where B is (d x r), A is (r x k), and r is small.
  - One big matrix becomes two thin ones (low-rank decomposition).
- **Savings** (4096 x 4096 matrix):
  - Full update: ~16.8 million parameters
  - LoRA with r=12: 12 x (4096 + 4096) = ~98 thousand (~0.6%)
- Benefits: reduces trainable parameters, maintains performance, has a regularization effect (less overfitting on small data, less forgetting of general abilities).

## 8.2 Implementation with PEFT
```python
from peft import LoraConfig
lora_config = LoraConfig(
    r=12,
    lora_alpha=32,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=['q_proj', 'v_proj']
)
```
- `r=12`: rank = adapter capacity (higher = more expressive, larger)
- `lora_alpha=32`: scaling factor for the adapter's contribution (commonly alpha/r)
- `lora_dropout=0.05`: drops 5% of adapter activations to reduce overfitting
- `target_modules=['q_proj','v_proj']`: attention query and value projections (cheap, common choice)
- `task_type="CAUSAL_LM"`: next-token-prediction models like Llama
- `bias="none"`: don't train bias terms

## 8.3 Integrate into training
```python
trainer = SFTTrainer(
    model=model,
    train_dataset=ds,
    max_seq_length=250,
    dataset_text_field='conversation',
    tokenizer=tokenizer,
    args=training_arguments,
    peft_config=lora_config,
)
trainer.train()
```
- PEFT (Parameter-Efficient Fine-Tuning) wraps the model for you.

## 8.4 LoRA vs regular fine-tuning
| Model | Params | Samples | Time |
|---|---|---|---|
| TinyLlama/TinyLlama-1.1B-Chat-v1.0 (regular) | 1.1B | 11k | ~30 min |
| nvidia/Llama3-ChatQA-1.5-8B (LoRA) | 8B | 11k | ~30 min |
- LoRA lets you fine-tune a much bigger model in about the same time.

---

# PART 9: QUANTIZATION

## 9.1 The idea
- LoRA cuts *training* cost. Quantization cuts *storage/memory* cost by using fewer bits per number.
- Reduces precision: 32-bit float -> 8-bit or 4-bit integer.
- Memory for an 8B model: 32-bit ~32 GB, 16-bit ~16 GB, 4-bit ~4 GB.
- Analogy: saving a photo at lower quality. Much smaller file, slight loss of detail.

## 9.2 Types
- **Weight quantization**: reduce weight precision
- **Activation quantization**: reduce precision of activation values
- **Post-training quantization**: reduce precision after training
- **Quantization-aware training**: account for low precision during training

## 9.3 bitsandbytes config
```python
from transformers import BitsAndBytesConfig, AutoModelForCausalLM

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,                       # precision (4-bit or 8-bit)
    bnb_4bit_quant_type="nf4",               # 'fp4' or 'nf4' (normalized 4-bit float)
    bnb_4bit_compute_dtype=torch.bfloat16    # compute precision (32-bit float or 16-bit bfloat)
)

model = AutoModelForCausalLM.from_pretrained(
    "nvidia/Llama3-ChatQA-1.5-8B",
    quantization_config=bnb_config
)
```
- `load_in_4bit=True`: store weights in 4 bits
- `nf4` (NormalFloat4): grid tuned to the bell-curve distribution of weights, so less accuracy loss than `fp4`
- `bfloat16` compute: weights stored in 4-bit but converted for the math, keeping calculations accurate

## 9.4 Using a quantized model
```python
promptstr = """System: You are a helpful chatbot who answers questions about planets.
User: Explain the history of Mars
Assistant: """
inputs = tokenizer.encode(promptstr, return_tensors="pt")
outputs = model.generate(inputs, max_length=200)
decoded_outputs = tokenizer.decode(
    outputs[0, inputs.shape[1]:], skip_special_tokens=True)
print(decoded_outputs)
```

## 9.5 Fine-tuning a quantized model (QLoRA)
- **Full quantization does not support fine-tuning.** 4-bit values are too coarse for small, precise gradient updates.
- Solution: combine both techniques.
  1. Load the big model in 4-bit and keep it frozen.
  2. Attach LoRA adapters (small, higher precision).
  3. Train only the adapters.
- Commonly called **QLoRA**. It lets you fine-tune an 8B model on a single modest GPU.
```python
trainer = SFTTrainer(
    model=model,
    peft_config=peft_config,
    train_dataset=ds,
    max_seq_length=250,
    dataset_text_field='conversation',
    tokenizer=tokenizer,
    args=training_arguments
)
trainer.train()
```

---