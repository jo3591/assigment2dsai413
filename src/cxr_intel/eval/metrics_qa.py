"""QA metrics: Exact-Match, token-F1, BERTScore, LLM-judge."""
from __future__ import annotations

import re
import string
from collections import Counter
from dataclasses import dataclass
from typing import Sequence


@dataclass
class QAMetrics:
    exact_match: float = 0.0
    token_f1: float = 0.0
    bertscore_f1: float = 0.0
    llm_judge_mean: float = 0.0
    llm_judge_pass_rate: float = 0.0
    n: int = 0


_PUNC = set(string.punctuation)


def _normalize(text: str) -> str:
    text = text.lower().strip()
    text = "".join(ch for ch in text if ch not in _PUNC)
    text = re.sub(r"\s+", " ", text)
    return text


def _exact_match(pred: str, gold: str) -> float:
    return float(_normalize(pred) == _normalize(gold))


def _token_f1(pred: str, gold: str) -> float:
    pt = _normalize(pred).split()
    gt = _normalize(gold).split()
    if not pt and not gt:
        return 1.0
    if not pt or not gt:
        return 0.0
    common = Counter(pt) & Counter(gt)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0
    precision = num_same / len(pt)
    recall = num_same / len(gt)
    return 2 * precision * recall / (precision + recall)


def score_qa(
    refs: Sequence[str],
    preds: Sequence[str],
    judge_scores: Sequence[float] | None = None,
    pass_threshold: float = 4.0,
    bertscore_model: str = "roberta-large",
) -> QAMetrics:
    if len(refs) != len(preds):
        raise ValueError(f"len(refs)={len(refs)} != len(preds)={len(preds)}")
    if not refs:
        return QAMetrics()

    em = sum(_exact_match(p, r) for p, r in zip(preds, refs)) / len(refs)
    f1 = sum(_token_f1(p, r) for p, r in zip(preds, refs)) / len(refs)

    try:
        from bert_score import score as bs_score

        _, _, bf = bs_score(list(preds), list(refs), model_type=bertscore_model, lang="en", verbose=False)
        bs_f1 = float(bf.mean())
    except Exception:
        bs_f1 = 0.0

    judge_mean = 0.0
    judge_pass = 0.0
    if judge_scores is not None and len(judge_scores) > 0:
        judge_mean = float(sum(judge_scores) / len(judge_scores))
        judge_pass = float(sum(1 for s in judge_scores if s >= pass_threshold) / len(judge_scores))

    return QAMetrics(
        exact_match=em,
        token_f1=f1,
        bertscore_f1=bs_f1,
        llm_judge_mean=judge_mean,
        llm_judge_pass_rate=judge_pass,
        n=len(refs),
    )
