"""
ml/evaluation/metrics.py
─────────────────────────
Standard LTR evaluation metrics: NDCG@K, Precision@K, MRR, MAP, Spearman.
"""
from __future__ import annotations
import math
import numpy as np
from scipy.stats import spearmanr


def dcg_at_k(relevances: list[int], k: int) -> float:
    """DCG@K using 2^rel-1 gain formula (standard for multi-grade labels)."""
    k = min(k, len(relevances))
    return sum((2**r - 1) / math.log2(i + 2) for i, r in enumerate(relevances[:k]))


def ndcg_at_k(predicted: list[int], ideal: list[int] | None = None, k: int | None = None) -> float:
    """
    NDCG@K = DCG@K(predicted) / DCG@K(ideal).

    Supports all call styles:
      ndcg_at_k(pred, ideal, k)      – original 3-arg positional form
      ndcg_at_k(pred, ideal, k=K)    – positional ideal, keyword k
      ndcg_at_k(pred, k=K)           – auto-compute ideal from predicted

    Returns:
        - 1.0 if ideal was explicitly passed and all labels are 0 (trivially perfect).
        - 0.0 if ideal was auto-computed and all labels are 0 (nothing to rank).
    """
    predicted = list(predicted)
    ideal_was_explicit = ideal is not None

    if ideal is None:
        ideal = sorted(predicted, reverse=True)
    else:
        ideal = list(ideal)

    if k is None:
        k = len(predicted)

    idcg = dcg_at_k(ideal, k)
    if idcg > 0:
        return dcg_at_k(predicted, k) / idcg
    return 1.0 if ideal_was_explicit else 0.0


def precision_at_k(relevances: list[int], k: int, threshold: int = 1) -> float:
    """Fraction of top-K items with relevance >= threshold."""
    relevances = list(relevances)
    top = relevances[:k]
    return sum(1 for r in top if r >= threshold) / len(top) if len(top) > 0 else 0.0


def mean_reciprocal_rank(relevances: list[int], threshold: int = 1) -> float:
    """1 / rank of first relevant item. 0.0 if no relevant item found."""
    relevances = list(relevances)
    for i, r in enumerate(relevances):
        if r >= threshold:
            return 1.0 / (i + 1)
    return 0.0


def average_precision(relevances: list[int], threshold: int = 1) -> float:
    """Average Precision for a single query."""
    n_rel, total = 0, 0.0
    for i, r in enumerate(relevances):
        if r >= threshold:
            n_rel += 1
            total += n_rel / (i + 1)
    return total / n_rel if n_rel > 0 else 0.0


def spearman_correlation(pred: list[float], true: list[float]) -> float:
    if len(pred) < 2: return 1.0
    rho, _ = spearmanr(pred, true)
    return float(rho)


def evaluate_ranking(query_groups: list[dict], k_values: list[int] | None = None) -> dict:
    """
    Args:
        query_groups: [{"predicted": [...], "ideal": [...]}]
    Returns: dict of metric -> mean value across queries.
    """
    k_values = k_values or [5, 10]
    acc: dict[str, list] = {}
    for g in query_groups:
        pred, ideal = g["predicted"], g["ideal"]
        for k in k_values:
            acc.setdefault(f"ndcg@{k}", []).append(ndcg_at_k(pred, ideal, k))
            acc.setdefault(f"p@{k}", []).append(precision_at_k(pred, k, threshold=2))
        acc.setdefault("mrr", []).append(mean_reciprocal_rank(pred, threshold=2))
        acc.setdefault("map", []).append(average_precision(pred, threshold=2))
    return {k: round(float(np.mean(v)), 4) for k, v in acc.items()}


def print_report(metrics: dict, name: str = "Model"):
    print(f"\n{'='*50}\n  RANKING EVAL — {name}\n{'='*50}")
    for k, v in sorted(metrics.items()):
        bar = "█" * int(v * 30) + "░" * (30 - int(v * 30))
        print(f"  {k:20s} {bar}  {v:.4f}")
    print("=" * 50)
