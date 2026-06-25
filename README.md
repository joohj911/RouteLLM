# RouteLLM

RouteLLM is a framework for training and evaluating LLM routers that intelligently route queries between a strong model and a weak model based on query difficulty.

This fork focuses on **agent/tool-calling routing** using [Qwen3.5](https://huggingface.co/Qwen) as the model pair and [BFCL](https://huggingface.co/datasets/gorilla-llm/Berkeley-Function-Calling-Leaderboard) as the evaluation benchmark, with local embeddings via `intfloat/multilingual-e5-small`.

## Overview

- **Weak model**: `Qwen/Qwen3.5-2B` — fast and cheap
- **Strong model**: `Qwen/Qwen3.5-9B` — higher accuracy on complex tool calls
- **Router**: Matrix Factorization (`mf`) trained on BFCL pass/fail labels
- **Embeddings**: `intfloat/multilingual-e5-small` (384-dim, fully local, no API key needed)
- **Benchmark**: BFCL v1 + v2 + v3 (~4,500 unique tool-calling samples)

The router learns which queries are hard enough to require the 9B model and routes the rest to the 2B model, reducing inference cost while preserving tool-calling accuracy.

## Installation

```bash
git clone https://github.com/joohj911/RouteLLM.git
cd RouteLLM
pip install -e ".[serve,eval]"
```

## Full Pipeline

### Step 1: Generate BFCL Embeddings

Download BFCL v1/v2/v3 from HuggingFace and generate multilingual-e5-small embeddings:

```bash
python routellm/routers/matrix_factorization/prepare_bfcl_data.py embed \
  --output-dir ./bfcl_data
```

Output:
- `bfcl_data/embeddings.npy` — prompt embeddings (shape: N × 384)
- `bfcl_data/prompts.json` — prompt metadata with BFCL IDs and split names

### Step 2: Evaluate Models on BFCL

Run both Qwen3.5-2B and Qwen3.5-9B on the BFCL prompts to get pass/fail results:

```bash
python routellm/routers/matrix_factorization/eval_bfcl_models.py \
  --prompts-path ./bfcl_data/prompts.json \
  --output-path ./eval_results.json \
  --weak-model Qwen/Qwen3.5-2B \
  --strong-model Qwen/Qwen3.5-9B
```

Options:
- `--load-in-4bit` — enable 4-bit quantization (requires `bitsandbytes`)
- `--max-new-tokens` — max generation length (default: 512)
- `--device` — `cuda` or `cpu` (default: `cuda`)

Output: `eval_results.json` with per-sample pass/fail for each model.

### Step 3: Convert to Train/Test Split

Split the evaluation results into training data (80%) and test data (20%) using stratified split by BFCL category:

```bash
python routellm/routers/matrix_factorization/prepare_bfcl_data.py convert \
  --results-path ./eval_results.json \
  --prompts-path ./bfcl_data/prompts.json \
  --output-dir ./bfcl_data \
  --weak-model qwen3.5-2b \
  --strong-model qwen3.5-9b \
  --train-ratio 0.8
```

Output:
- `bfcl_data/train_data.json` — training data for the MF router
- `bfcl_data/test_data.json` — held-out test data for evaluation

Training labels: samples where only the strong model passes are labeled "route to strong"; samples where both pass are labeled "route to weak". Samples where both fail or only the weak model passes are discarded (no routing signal).

### Step 4: Train the MF Router

```bash
python routellm/routers/matrix_factorization/train_matrix_factorization.py \
  --data-path ./bfcl_data/train_data.json \
  --embeddings-path ./bfcl_data/embeddings.npy \
  --output-path ./bfcl_mf_model.pt \
  --epochs 30 \
  --batch-size 64 \
  --text-dim 384
```

### Step 5: Evaluate the Router

Evaluate the trained router on the BFCL test set. The output shows:
- **Router pass rate** at each routing threshold
- **Weak model only** pass rate (baseline)
- **Strong model only** pass rate (ceiling)
- **% routed to weak model** at each threshold

```bash
python -m routellm.evals.evaluate \
  --routers mf \
  --benchmark bfcl \
  --test-data ./bfcl_data/test_data.json \
  --strong-model qwen3.5-9b \
  --weak-model qwen3.5-2b \
  --config config.example.yaml
```

Example output:

```
================================================================
BFCL Routing Summary
================================================================
  Weak model   (           qwen3.5-2b):   62.4%
  Strong model (           qwen3.5-9b):   81.3%

  Router             Threshold  Pass Rate    Weak%   Strong%
  ---------------------------------------------------------
  mf                    0.3000     79.8%    72.1%    27.9%
  mf                    0.5000     77.2%    85.3%    14.7%
  ...
================================================================
```

## Using the Router in Code

Once trained, use the MF router as a drop-in replacement via the Python SDK:

```python
import yaml
from routellm.controller import Controller

controller = Controller(
    routers=["mf"],
    config=yaml.safe_load(open("config.example.yaml")),
    strong_model="qwen3.5-9b",
    weak_model="qwen3.5-2b",
)

response = controller.chat.completions.create(
    model="router-mf-0.3",   # threshold controls strong model call rate
    messages=[{"role": "user", "content": "What's the weather in Seoul?"}],
)
```

The `model` field format is `router-[ROUTER_NAME]-[THRESHOLD]`. Higher threshold = fewer strong model calls.

## Threshold Calibration

Calibrate the threshold so that a specific percentage of queries go to the strong model:

```bash
python -m routellm.calibrate_threshold \
  --routers mf \
  --strong-model-pct 0.3 \
  --config config.example.yaml
```

## OpenAI-Compatible Server

Launch a server compatible with any OpenAI client:

```bash
python -m routellm.openai_server \
  --routers mf \
  --strong-model qwen3.5-9b \
  --weak-model qwen3.5-2b \
  --config config.example.yaml
```

## Configuration

Router configuration is specified in a YAML file and passed via `--config`. See `config.example.yaml` for the format. The `mf` router accepts:

```yaml
mf:
  checkpoint_path: ./bfcl_mf_model.pt
  text_dim: 384
  num_models: 66
```

## Routers

| Router | Description |
|--------|-------------|
| `mf` | Matrix factorization on prompt embeddings (recommended) |
| `sw_ranking` | Weighted Elo based on prompt similarity |
| `bert` | BERT classifier trained on preference data |
| `causal_llm` | LLM-based classifier |
| `random` | Random baseline |

## Extending RouteLLM

### Adding a new router

Implement the abstract `Router` class in `routellm/routers/routers.py` and add it to `ROUTER_CLS`. The only method to implement is `calculate_strong_win_rate(prompt)`, which returns a float in [0, 1] — if it exceeds the user-specified threshold, the request goes to the strong model.

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
