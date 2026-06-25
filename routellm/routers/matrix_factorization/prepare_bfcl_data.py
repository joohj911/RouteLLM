"""
BFCL (v1, v2, v3) 데이터로 MF 라우터 학습 데이터를 준비하는 스크립트.

두 단계로 동작:
  1. embed  : BFCL 프롬프트를 multilingual-e5-small로 임베딩 → .npy 저장
  2. convert: 모델 평가 결과(pass/fail JSON)를 학습용 pairwise JSON으로 변환

사용 예시:
  # Step 1 - 임베딩 생성
  python prepare_bfcl_data.py embed --output-dir ./bfcl_data

  # Step 2 - 모델 평가 결과를 학습 데이터로 변환
  python prepare_bfcl_data.py convert \
    --results-path ./eval_results.json \
    --prompts-path ./bfcl_data/prompts.json \
    --output-path ./bfcl_data/pairwise_data.json \
    --strong-model qwen3.5-9b \
    --weak-model qwen3.5-2b

eval_results.json 형식 (모델 실행 후 직접 생성):
  [
    {
      "id": "live_simple_0",
      "prompt": "...",
      "qwen3.5-2b_pass": true,
      "qwen3.5-9b_pass": true
    },
    ...
  ]
"""

import argparse
import json
import os

import numpy as np
from datasets import concatenate_datasets, load_dataset
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

BFCL_DATASET = "gorilla-llm/Berkeley-Function-Calling-Leaderboard"

# BFCL v1, v2, v3의 주요 split 이름
BFCL_SPLITS = [
    # V1 - expert curated, single-turn
    "gorilla_openfunctions_v1_test_simple",
    "gorilla_openfunctions_v1_test_multiple_function",
    "gorilla_openfunctions_v1_test_parallel_function",
    "gorilla_openfunctions_v1_test_parallel_multiple_function",
    "gorilla_openfunctions_v1_test_relevance",
    # V2 - community contributed (live)
    "live_simple",
    "live_multiple",
    "live_parallel",
    "live_parallel_multiple",
    "live_relevance",
    "live_irrelevance",
    # V3 - multi-turn
    "multi_turn_base",
    "multi_turn_miss_func",
    "multi_turn_miss_param",
    "multi_turn_long_context",
]


def extract_prompt(sample, split_name: str) -> str:
    """샘플에서 라우팅에 사용할 프롬프트(첫 번째 유저 메시지)를 추출한다."""
    question = sample.get("question", [])
    if not question:
        return ""

    # multi_turn: question = [[turn1_msgs], [turn2_msgs], ...]
    # single_turn: question = [[msgs]]
    first_turn = question[0] if isinstance(question[0], list) else question
    for msg in first_turn:
        if isinstance(msg, dict) and msg.get("role") == "user":
            return msg.get("content", "").strip()
    return ""


def load_bfcl_prompts() -> list[dict]:
    """BFCL v1/v2/v3 에서 유니크한 프롬프트를 수집한다."""
    prompts = []
    seen_ids = set()

    for split in BFCL_SPLITS:
        try:
            ds = load_dataset(BFCL_DATASET, split=split)
        except Exception as e:
            print(f"  [skip] {split}: {e}")
            continue

        for sample in ds:
            sample_id = sample.get("id", "")
            if sample_id in seen_ids:
                continue
            seen_ids.add(sample_id)

            prompt = extract_prompt(sample, split)
            if prompt:
                prompts.append(
                    {
                        "idx": len(prompts),
                        "id": sample_id,
                        "split": split,
                        "prompt": prompt,
                    }
                )

        print(f"  [ok] {split}: +{len(ds)} samples (total unique: {len(prompts)})")

    return prompts


def generate_embeddings(prompts: list[dict], output_dir: str):
    """multilingual-e5-small로 임베딩 생성 후 .npy와 prompts.json 저장."""
    os.makedirs(output_dir, exist_ok=True)

    print("\nLoading intfloat/multilingual-e5-small ...")
    model = SentenceTransformer("intfloat/multilingual-e5-small")

    texts = [f"query: {p['prompt']}" for p in prompts]
    print(f"Encoding {len(texts)} prompts ...")
    embeddings = model.encode(texts, batch_size=256, show_progress_bar=True)

    npy_path = os.path.join(output_dir, "embeddings.npy")
    np.save(npy_path, embeddings)
    print(f"Saved embeddings → {npy_path}  shape={embeddings.shape}")

    prompts_path = os.path.join(output_dir, "prompts.json")
    with open(prompts_path, "w") as f:
        json.dump(prompts, f, ensure_ascii=False, indent=2)
    print(f"Saved prompts    → {prompts_path}")


def convert_results_to_pairwise(
    results_path: str,
    prompts_path: str,
    output_path: str,
    strong_model: str,
    weak_model: str,
):
    """
    모델 평가 결과(pass/fail)를 MF 학습용 pairwise JSON으로 변환한다.

    입력 results_path JSON 형식:
      [{"id": "...", "qwen3.5-2b_pass": true, "qwen3.5-9b_pass": false}, ...]

    출력 형식 (train_matrix_factorization.py 호환):
      [{"model_a": "qwen3.5-2b", "model_b": "qwen3.5-9b",
        "winner": "model_b", "idx": 0}, ...]
    """
    with open(results_path) as f:
        results = json.load(f)
    with open(prompts_path) as f:
        prompts = json.load(f)

    id_to_idx = {p["id"]: p["idx"] for p in prompts}

    weak_key = f"{weak_model}_pass"
    strong_key = f"{strong_model}_pass"

    pairs = []
    skipped = 0
    for r in results:
        weak_pass = r.get(weak_key, False)
        strong_pass = r.get(strong_key, False)

        if weak_pass and strong_pass:
            winner = "model_a"  # weak 로 충분
        elif not weak_pass and strong_pass:
            winner = "model_b"  # strong 필요
        else:
            # 둘 다 실패하거나 weak만 성공하는 케이스는 라우팅 신호가 약해 제거
            skipped += 1
            continue

        idx = id_to_idx.get(r.get("id", ""), -1)
        if idx == -1:
            skipped += 1
            continue

        pairs.append(
            {
                "model_a": weak_model,
                "model_b": strong_model,
                "winner": winner,
                "idx": idx,
            }
        )

    with open(output_path, "w") as f:
        json.dump(pairs, f, ensure_ascii=False, indent=2)

    print(f"Pairwise pairs : {len(pairs)}")
    print(f"Skipped        : {skipped}")
    print(f"Saved          → {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command")

    # Step 1: embed
    embed_parser = subparsers.add_parser("embed", help="BFCL 프롬프트 임베딩 생성")
    embed_parser.add_argument("--output-dir", type=str, default="./bfcl_data")

    # Step 2: convert
    convert_parser = subparsers.add_parser(
        "convert", help="모델 평가 결과 → pairwise 학습 데이터 변환"
    )
    convert_parser.add_argument("--results-path", type=str, required=True)
    convert_parser.add_argument("--prompts-path", type=str, required=True)
    convert_parser.add_argument("--output-path", type=str, required=True)
    convert_parser.add_argument("--strong-model", type=str, default="qwen3.5-9b")
    convert_parser.add_argument("--weak-model", type=str, default="qwen3.5-2b")

    args = parser.parse_args()

    if args.command == "embed":
        print("Loading BFCL splits ...")
        prompts = load_bfcl_prompts()
        print(f"\nTotal unique prompts: {len(prompts)}")
        generate_embeddings(prompts, args.output_dir)

    elif args.command == "convert":
        convert_results_to_pairwise(
            args.results_path,
            args.prompts_path,
            args.output_path,
            args.strong_model,
            args.weak_model,
        )

    else:
        parser.print_help()
