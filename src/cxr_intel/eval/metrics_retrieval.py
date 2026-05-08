"""Retrieval metrics: Recall@k, MRR, nDCG@k.

Gold construction: for each test image we pick the gold report (its own report)
and treat the query as a salient sentence from FINDINGS. This gives us
1 gold doc per query — Recall@k thus measures "did we retrieve the source?".
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Sequence


@dataclass
class RetrievalMetrics:
    recall_at_1: float = 0.0
    recall_at_5: float = 0.0
    recall_at_10: float = 0.0
    mrr: float = 0.0
    ndcg_at_10: float = 0.0
    n: int = 0
    by_k: dict[int, float] = field(default_factory=dict)


def _reciprocal_rank(retrieved_ids: list[str], gold_id: str) -> float:
    for i, rid in enumerate(retrieved_ids, 1):
        if rid == gold_id:
            return 1.0 / i
    return 0.0


def _ndcg(retrieved_ids: list[str], gold_id: str, k: int = 10) -> float:
    rels = [1.0 if rid == gold_id else 0.0 for rid in retrieved_ids[:k]]
    dcg = sum(r / math.log2(i + 2) for i, r in enumerate(rels))
    idcg = 1.0  # Single gold doc → ideal is 1 / log2(2) = 1
    return dcg / idcg if idcg > 0 else 0.0


def _recall(retrieved_ids: list[str], gold_id: str, k: int) -> float:
    return float(gold_id in retrieved_ids[:k])


def score_retrieval(
    retrieved_ids_per_query: Sequence[list[str]],
    gold_ids: Sequence[str],
    ks: Sequence[int] = (1, 5, 10),
) -> RetrievalMetrics:
    if len(retrieved_ids_per_query) != len(gold_ids):
        raise ValueError("retrieved/gold length mismatch")
    n = len(gold_ids)
    if n == 0:
        return RetrievalMetrics()

    rr = sum(_reciprocal_rank(r, g) for r, g in zip(retrieved_ids_per_query, gold_ids)) / n
    ndcg = sum(_ndcg(r, g, 10) for r, g in zip(retrieved_ids_per_query, gold_ids)) / n
    by_k = {k: sum(_recall(r, g, k) for r, g in zip(retrieved_ids_per_query, gold_ids)) / n for k in ks}

    return RetrievalMetrics(
        recall_at_1=by_k.get(1, 0.0),
        recall_at_5=by_k.get(5, 0.0),
        recall_at_10=by_k.get(10, 0.0),
        mrr=rr,
        ndcg_at_10=ndcg,
        n=n,
        by_k=by_k,
    )
