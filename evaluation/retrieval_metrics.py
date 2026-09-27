"""Recall, MRR, and NDCG metric implementations for ranked retrieval results."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence


@dataclass(frozen=True)
class RetrievalMetrics:
    """Metrics for one ranked result list against a set of relevant identifiers."""

    recall_at_5: float
    recall_at_10: float
    recall_at_20: float
    mrr: float
    ndcg_at_5: float
    ndcg_at_10: float
    ndcg_at_20: float

    def as_dict(self) -> dict[str, float]:
        return {
            "recall@5": self.recall_at_5,
            "recall@10": self.recall_at_10,
            "recall@20": self.recall_at_20,
            "mrr": self.mrr,
            "ndcg@5": self.ndcg_at_5,
            "ndcg@10": self.ndcg_at_10,
            "ndcg@20": self.ndcg_at_20,
        }


def calculate_retrieval_metrics(
    ranked_ids: Sequence[str],
    relevant_ids: set[str] | frozenset[str],
    *,
    cutoffs: Sequence[int] = (5, 10, 20),
) -> RetrievalMetrics:
    """Calculate recall@k, reciprocal rank, and binary NDCG@k.

    Duplicate result identifiers count once, at their first occurrence. Tied
    retrieval scores use the deterministic order supplied by the retriever. An
    empty relevant set yields zero for all metrics, avoiding an undefined score.
    """
    if tuple(cutoffs) != (5, 10, 20):
        raise ValueError("cutoffs must be exactly (5, 10, 20)")
    if any(not isinstance(identifier, str) for identifier in ranked_ids):
        raise TypeError("ranked identifiers must be strings")
    if any(not isinstance(identifier, str) for identifier in relevant_ids):
        raise TypeError("relevant identifiers must be strings")

    deduplicated: list[str] = []
    seen: set[str] = set()
    for identifier in ranked_ids:
        if identifier not in seen:
            seen.add(identifier)
            deduplicated.append(identifier)

    relevant = set(relevant_ids)
    if not relevant:
        return RetrievalMetrics(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)

    def recall(cutoff: int) -> float:
        found = len(set(deduplicated[:cutoff]) & relevant)
        return found / len(relevant)

    reciprocal_rank = next(
        (1.0 / rank for rank, identifier in enumerate(deduplicated, start=1) if identifier in relevant),
        0.0,
    )

    def ndcg(cutoff: int) -> float:
        observed = math.fsum(
            1.0 / math.log2(rank + 1)
            for rank, identifier in enumerate(deduplicated[:cutoff], start=1)
            if identifier in relevant
        )
        ideal_relevant_count = min(len(relevant), cutoff)
        ideal = math.fsum(
            1.0 / math.log2(rank + 1)
            for rank in range(1, ideal_relevant_count + 1)
        )
        return observed / ideal if ideal else 0.0

    return RetrievalMetrics(
        recall_at_5=recall(5),
        recall_at_10=recall(10),
        recall_at_20=recall(20),
        mrr=reciprocal_rank,
        ndcg_at_5=ndcg(5),
        ndcg_at_10=ndcg(10),
        ndcg_at_20=ndcg(20),
    )
