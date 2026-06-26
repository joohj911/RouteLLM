# RouteLLM

RouteLLM is a framework for training and evaluating LLM routers that intelligently route queries between a strong model and a weak model based on query difficulty.

This fork focuses on **agent/tool-calling routing** using [Qwen3.5](https://huggingface.co/Qwen) as the model pair and [BFCL v4](https://github.com/ShishirPatil/gorilla/tree/main/berkeley-function-call-leaderboard) as the evaluation benchmark, with local embeddings via `intfloat/multilingual-e5-small`.

## Overview

- **Weak model**: `Qwen/Qwen3.5-2B` — fast, handles simple tool calls
- **Strong model**: `Qwen/Qwen3.5-9B` — higher accuracy on complex tool calls
- **Router**: Matrix Factorization (`mf`) trained on BFCL v4 pass/fail labels
- **Embeddings**: `intfloat/multilingual-e5-small` (384-dim, fully local, no API key)
- **Benchmark**: BFCL v4 (17 categories: non-live, live, multi-turn)

The router learns which queries require the 9B model and routes the rest to the 2B model, reducing inference cost while preserving tool-calling accuracy.

## Installation

```bash
git clone https://github.com/joohj911/RouteLLM.git
cd RouteLLM
pip install -e ".[eval]"
# Optional: 4-bit quantization for low-VRAM eval
pip install -e ".[eval,quant]"
```

## Full Pipeline

### Step 1: Generate BFCL Embeddings

Download BFCL v4 from GitHub and generate multilingual-e5-small embeddings:

```bash
python routellm/routers/matrix_factorization/prepare_bfcl_data.py embed \
  --output-dir ./bfcl_data
```

Output:
- `bfcl_data/embeddings.npy` — prompt embeddings (shape: N × 384)
- `bfcl_data/prompts.json` — prompt metadata with BFCL IDs and split names

### Step 2: Evaluate Models on BFCL

Pass any number of models to evaluate. Available GPUs are detected automatically and models are distributed across them — up to N models run concurrently on N GPUs, then the next batch, and so on:

```bash
python routellm/evals/eval_bfcl_models.py \
  --prompts-path ./bfcl_data/prompts.json \
  --output-path ./eval_results.json \
  --models Qwen/Qwen3.5-0.6B Qwen/Qwen3.5-2B Qwen/Qwen3.5-9B \
  --batch-size 16
```

Options:
- `--models` — one or more HuggingFace model IDs to evaluate
- `--batch-size` — number of samples per `model.generate()` call (default: 8; increase to 16-32 if VRAM allows)
- `--load-in-4bit` — 4-bit quantization for low-VRAM setups (requires `bitsandbytes`)
- `--max-new-tokens` — max generation length (default: 512)

GPU scheduling: with 2 GPUs and 3 models, model 1 runs on `cuda:0` and model 2 on `cuda:1` simultaneously, then model 3 runs on `cuda:0`. No flags needed — GPU count is detected via `torch.cuda.device_count()`.

Output: `eval_results.json` — per-sample pass/fail for each model.

The script prints a summary at the end:

```
============================================================
BFCL Evaluation Summary
============================================================
  Total samples :  1234
      Qwen/Qwen3.5-0.6B :  612/1234  (49.6%)
        Qwen/Qwen3.5-2B :  768/1234  (62.2%)
        Qwen/Qwen3.5-9B : 1003/1234  (81.3%)
============================================================
```

> **Note:** The script prints the full HuggingFace model IDs used in the next step, e.g. `--weak-model Qwen/Qwen3.5-2B --strong-model Qwen/Qwen3.5-9B`.

### Step 3: Convert to Train/Test Split

Split results into training data (80%) and test data (20%) using stratified split by BFCL category:

```bash
python routellm/routers/matrix_factorization/prepare_bfcl_data.py convert \
  --results-path ./eval_results.json \
  --prompts-path ./bfcl_data/prompts.json \
  --output-dir ./bfcl_data \
  --weak-model Qwen/Qwen3.5-2B \
  --strong-model Qwen/Qwen3.5-9B \
  --train-ratio 0.8
```

Output:
- `bfcl_data/train_data.json` — training labels for the MF router
- `bfcl_data/test_data.json` — held-out test set for evaluation

**Labeling rule:**
| Weak passes | Strong passes | Label | Reason |
|---|---|---|---|
| ✓ | ✓ | Route to weak | Weak is sufficient |
| ✗ | ✓ | Route to strong | Strong is needed |
| ✓ | ✗ | Route to weak | Weak succeeded; strong failed |
| ✗ | ✗ | Route to strong | Neither local model succeeded → send to frontier |

### Step 4: Train the MF Router

```bash
python routellm/routers/matrix_factorization/train_matrix_factorization.py \
  --train-data ./bfcl_data/train_data.json \
  --npy-path ./bfcl_data/embeddings.npy \
  --output-path ./bfcl_mf_model.pt \
  --num-epochs 100 \
  --dim 128 \
  --text-dim 384 \
  --batch-size 64
```

The checkpoint is saved without the prompt embedding matrix (not needed at inference time).

### Step 5: Evaluate the Router

Evaluate on the BFCL test set. Reports pass rate for weak-only, strong-only, and the router at each threshold:

```bash
python -m routellm.evals.evaluate \
  --routers mf \
  --mf-checkpoint ./bfcl_mf_model.pt \
  --test-data ./bfcl_data/test_data.json \
  --strong-model Qwen/Qwen3.5-9B \
  --weak-model Qwen/Qwen3.5-2B
```

Example output:

```
================================================================
BFCL Routing Summary
================================================================
  Weak model   (      Qwen/Qwen3.5-2B):   62.4%
  Strong model (      Qwen/Qwen3.5-9B):   81.3%

  Router             Threshold  Pass Rate    Weak%   Strong%
  ---------------------------------------------------------
  mf                    0.3000     79.8%    72.1%    27.9%
  mf                    0.5000     77.2%    85.3%    14.7%
  ...
================================================================
```

## Local Inference (Two-GPU Setup)

For production use, load both models locally with `LocalController`. Each model is pinned to its own GPU — no per-request model loading:

```python
from routellm.local_pipeline import LocalController

controller = LocalController(
    routers=["mf"],
    strong_model="Qwen/Qwen3.5-9B",
    weak_model="Qwen/Qwen3.5-2B",
    strong_device="cuda:1",
    weak_device="cuda:0",
    config={"mf": {"checkpoint_path": "./bfcl_mf_model.pt", "text_dim": 384}},
)

response = controller.completion(
    router="mf",
    threshold=0.3,   # tune based on desired strong model call %
    messages=[{"role": "user", "content": "What's the weather in Seoul?"}],
    tools=[...],     # OpenAI tool format
)
print(response["choices"][0]["message"])
```

## Using the Router via API

Use the router against any OpenAI-compatible API endpoint:

```python
from routellm.controller import Controller

controller = Controller(
    routers=["mf"],
    config={"mf": {"checkpoint_path": "./bfcl_mf_model.pt", "text_dim": 384}},
    strong_model="Qwen/Qwen3.5-9B",
    weak_model="Qwen/Qwen3.5-2B",
)

response = controller.chat.completions.create(
    model="router-mf-0.3",   # threshold controls strong model call rate
    messages=[{"role": "user", "content": "What's the weather in Seoul?"}],
)
```

The `model` field format is `router-[ROUTER_NAME]-[THRESHOLD]`.

## OpenAI-Compatible Server

```bash
python -m routellm.openai_server \
  --routers mf \
  --strong-model Qwen/Qwen3.5-9B \
  --weak-model Qwen/Qwen3.5-2B \
  --config config.example.yaml
```

## Configuration

For `evaluate.py` and the OpenAI server, router configuration is passed via `--config` YAML or (for the `mf` router) via `--mf-checkpoint` shortcut:

```yaml
# config.example.yaml
mf:
  checkpoint_path: ./bfcl_mf_model.pt
  text_dim: 384
```

## Routers

| Router | Description |
|--------|-------------|
| `mf` | Matrix factorization on prompt embeddings (recommended) |
| `bert` | BERT classifier trained on preference data |
| `causal_llm` | LLM-based classifier |
| `random` | Random baseline |

## Extending RouteLLM

### Adding a new router

Implement the abstract `Router` class in `routellm/routers/routers.py` and add it to `ROUTER_CLS`. The only required method is `calculate_strong_win_rate(prompt) -> float`. If the returned value exceeds the user-specified threshold, the request goes to the strong model.

### Adding a new benchmark

Implement the abstract `Benchmark` class in `routellm/evals/benchmarks.py` and update `routellm/evals/evaluate.py` to initialize it.

## Citation

```bibtex
@misc{ong2024routellmlearningroutellms,
      title={RouteLLM: Learning to Route LLMs with Preference Data},
      author={Isaac Ong and Amjad Almahairi and Vincent Wu and Wei-Lin Chiang and Tianhao Wu and Joseph E. Gonzalez and M Waleed Kadous and Ion Stoica},
      year={2024},
      eprint={2406.18665},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2406.18665},
}
```
