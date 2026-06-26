"""
UniRoute K-Means cluster-based routing model.

LLM representation: Ψ(h) ∈ [0,1]^K — per-cluster error rate on a validation set.
Prompt representation: Φ(x) ∈ {0,1}^K — one-hot cluster membership (nearest centroid).

Routing score: Ψ_k(weak) - Ψ_k(strong)  (> 0 means strong is better in this cluster)
Normalised to [0,1]: (diff + 1) / 2  so threshold 0.5 = "strong is better than weak"
"""

import numpy as np
import torch

EMBEDDING_MODEL_NAME = "intfloat/multilingual-e5-small"
_EMBEDDING_MODEL = None


def get_embedding_model():
    global _EMBEDDING_MODEL
    if _EMBEDDING_MODEL is None:
        from sentence_transformers import SentenceTransformer
        _EMBEDDING_MODEL = SentenceTransformer(EMBEDDING_MODEL_NAME)
    return _EMBEDDING_MODEL


class UniRouteModel:
    """
    Inference-only cluster-based UniRoute model.

    centroids : np.ndarray (K, D) — K-Means cluster centres (D=384 for e5-small)
    psi_weak  : np.ndarray (K,)  — per-cluster error rate of the weak model
    psi_strong: np.ndarray (K,)  — per-cluster error rate of the strong model
    """

    def __init__(
        self,
        centroids: np.ndarray,
        psi_weak: np.ndarray,
        psi_strong: np.ndarray,
    ):
        self.centroids = centroids          # (K, D)
        self.psi_weak = psi_weak            # (K,)
        self.psi_strong = psi_strong        # (K,)

    def predict(self, embedding: np.ndarray) -> float:
        """
        Return strong_win_rate ∈ [0, 1] for a single prompt embedding.

        Higher → more reason to route to strong model.
        Threshold 0.5 corresponds to λ=0 (pure accuracy, ignore cost).
        """
        dists = np.sum((self.centroids - embedding) ** 2, axis=1)  # (K,)
        k = int(np.argmin(dists))
        diff = float(self.psi_weak[k] - self.psi_strong[k])        # ∈ [-1, 1]
        return (diff + 1.0) / 2.0                                   # ∈ [0, 1]

    @classmethod
    def load(cls, checkpoint_path: str) -> "UniRouteModel":
        ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        return cls(
            centroids=np.asarray(ckpt["centroids"], dtype=np.float32),
            psi_weak=np.asarray(ckpt["psi_weak"], dtype=np.float32),
            psi_strong=np.asarray(ckpt["psi_strong"], dtype=np.float32),
        )
