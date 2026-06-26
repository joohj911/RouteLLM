"""
UniRoute K-Means 라우터 학습 스크립트.

사용법:
  python lm_routing/routers/uniroute/train_uniroute.py \\
    --train-data ./bfcl_data_2B/train_data.json \\
    --npy-path   ./bfcl_data/embeddings.npy \\
    --output-path ./bfcl_data_2B/uniroute_model.pt \\
    --weak-model  Qwen/Qwen3.5-2B \\
    --strong-model Qwen/Qwen3.5-9B

학습 절차:
  1. train_data.json → 80%(cluster_train) / 20%(val) stratified split
  2. cluster_train 임베딩으로 K-Means 학습  (K 후보 sweep)
  3. 각 K에 대해 val에서 deferral curve AUC 계산 → 최적 K 선택
  4. 최적 K로 Ψ_weak[k], Ψ_strong[k] (per-cluster error rate) 계산
  5. 체크포인트 저장
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split


# ─────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────

def compute_psi(cluster_labels: np.ndarray, pass_labels: np.ndarray, K: int) -> np.ndarray:
    """
    Per-cluster error rate: Ψ_k(model) = 1 − (pass rate in cluster k).

    cluster_labels : (N,) int  — cluster assignment for each val sample
    pass_labels    : (N,) bool — whether the model passed for each val sample
    """
    psi = np.zeros(K, dtype=np.float32)
    for k in range(K):
        mask = cluster_labels == k
        if mask.sum() == 0:
            psi[k] = 0.5  # uninformative prior for empty cluster
        else:
            psi[k] = 1.0 - float(pass_labels[mask].mean())
    return psi


def val_auc(
    psi_weak: np.ndarray,
    psi_strong: np.ndarray,
    cluster_labels: np.ndarray,
    weak_labels: np.ndarray,
    strong_labels: np.ndarray,
    n_bins: int = 10,
) -> float:
    """
    Area under the deferral curve on the validation set.
    Used internally to select the best K.
    """
    # Strong win rate for each val sample
    scores = np.array([
        (psi_weak[k] - psi_strong[k] + 1.0) / 2.0
        for k in cluster_labels
    ], dtype=np.float32)

    try:
        _, thresholds = pd.qcut(scores, n_bins, retbins=True, duplicates="drop")
    except ValueError:
        thresholds = np.linspace(scores.min(), scores.max(), n_bins + 1)

    accs, strong_pcts = [], []
    for j, thr in enumerate(thresholds):
        sel = scores >= thr if j < len(thresholds) - 1 else scores > thr
        results = np.where(sel, strong_labels, weak_labels)
        accs.append(results.mean())
        strong_pcts.append(sel.mean())

    # Sort by strong_pct ascending for trapz
    order = np.argsort(strong_pcts)
    return float(np.trapz(np.array(accs)[order], np.array(strong_pcts)[order]))


# ─────────────────────────────────────────────
# Main training function
# ─────────────────────────────────────────────

def train_uniroute(
    train_data_path: str,
    npy_path: str,
    output_path: str,
    weak_model: str,
    strong_model: str,
    train_ratio: float = 0.8,
    k_candidates: list[int] | None = None,
    seed: int = 42,
) -> dict:
    print(f"\nLoading train data from {train_data_path}")
    df = pd.read_json(train_data_path)
    print(f"  {len(df)} samples, columns: {list(df.columns)}")

    for col in [weak_model, strong_model]:
        if col not in df.columns:
            raise ValueError(
                f"Column '{col}' not found in train_data. "
                f"Available: {list(df.columns)}"
            )

    print(f"Loading embeddings from {npy_path}")
    all_embs = np.load(npy_path).astype(np.float32)  # (N_total, D)
    train_embs = all_embs[df["idx"].values]           # (N_train, D)
    D = train_embs.shape[1]
    print(f"  Embedding dim: {D}, total prompts: {all_embs.shape[0]}")

    # 80/20 stratified split
    indices = np.arange(len(df))
    try:
        cl_idx, val_idx = train_test_split(
            indices,
            train_size=train_ratio,
            stratify=df["bfcl_split"].values,
            random_state=seed,
        )
    except ValueError:
        cl_idx, val_idx = train_test_split(
            indices, train_size=train_ratio, random_state=seed
        )

    cl_embs = train_embs[cl_idx]
    val_embs = train_embs[val_idx]
    val_df = df.iloc[val_idx].reset_index(drop=True)
    val_weak = val_df[weak_model].astype(bool).values
    val_strong = val_df[strong_model].astype(bool).values

    print(f"  cluster_train: {len(cl_idx)} samples, val: {len(val_idx)} samples")

    # K candidate list: range 3 to len(val)//50, plus fixed checkpoints
    n_val = len(val_idx)
    if k_candidates is None:
        k_upper = max(5, n_val // 50)
        k_candidates = sorted(set([5, 10, 13, 20, 30, k_upper]))
    k_candidates = [k for k in k_candidates if 2 <= k <= n_val]
    print(f"\nK candidates: {k_candidates}")

    best = {"K": None, "auc": -np.inf, "centroids": None, "psi_weak": None, "psi_strong": None}

    for K in k_candidates:
        km = KMeans(n_clusters=K, random_state=seed, n_init=10, max_iter=300)
        km.fit(cl_embs)

        val_labels = km.predict(val_embs)
        psi_w = compute_psi(val_labels, val_weak, K)
        psi_s = compute_psi(val_labels, val_strong, K)
        auc = val_auc(psi_w, psi_s, val_labels, val_weak, val_strong)

        print(f"  K={K:3d} → val AUC = {auc:.5f}")

        if auc > best["auc"]:
            best.update({
                "K": K,
                "auc": auc,
                "centroids": km.cluster_centers_.astype(np.float32),
                "psi_weak": psi_w,
                "psi_strong": psi_s,
            })

    print(f"\nBest K={best['K']}  (val AUC={best['auc']:.5f})")
    print(f"  Ψ_weak  (mean={best['psi_weak'].mean():.3f}): {best['psi_weak'].round(3)}")
    print(f"  Ψ_strong(mean={best['psi_strong'].mean():.3f}): {best['psi_strong'].round(3)}")

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "K": best["K"],
            "centroids": best["centroids"],
            "psi_weak": best["psi_weak"],
            "psi_strong": best["psi_strong"],
            "weak_model": weak_model,
            "strong_model": strong_model,
            "val_auc": best["auc"],
            "embedding_model": "intfloat/multilingual-e5-small",
            "embedding_prefix": "query: ",
        },
        output_path,
    )
    print(f"Saved UniRoute checkpoint → {output_path}\n")
    return best


# ─────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train UniRoute K-Means router")
    parser.add_argument("--train-data", required=True, help="train_data.json from prepare_bfcl_data.py convert")
    parser.add_argument("--npy-path",   required=True, help="embeddings.npy from prepare_bfcl_data.py embed")
    parser.add_argument("--output-path", required=True, help="Output .pt checkpoint path")
    parser.add_argument("--weak-model",   required=True, help="HuggingFace weak model ID")
    parser.add_argument("--strong-model", required=True, help="HuggingFace strong model ID")
    parser.add_argument("--train-ratio",  type=float, default=0.8, help="Fraction for cluster training (default: 0.8)")
    parser.add_argument(
        "--k-candidates",
        type=int, nargs="+", default=None,
        help="K values to try (default: auto from val size)",
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    train_uniroute(
        train_data_path=args.train_data,
        npy_path=args.npy_path,
        output_path=args.output_path,
        weak_model=args.weak_model,
        strong_model=args.strong_model,
        train_ratio=args.train_ratio,
        k_candidates=args.k_candidates,
        seed=args.seed,
    )
