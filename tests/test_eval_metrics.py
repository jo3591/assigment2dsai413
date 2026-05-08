"""Smoke tests for evaluation metrics."""
from __future__ import annotations

from cxr_intel.eval.metrics_qa import _exact_match, _token_f1, score_qa
from cxr_intel.eval.metrics_retrieval import score_retrieval


def test_exact_match() -> None:
    assert _exact_match("Yes.", "yes") == 1.0
    assert _exact_match("Yes.", "no") == 0.0


def test_token_f1() -> None:
    assert _token_f1("hello world", "hello world") == 1.0
    assert _token_f1("foo bar", "baz qux") == 0.0
    assert 0 < _token_f1("the quick brown fox", "the brown fox jumps") < 1


def test_score_qa_perfect() -> None:
    refs = ["yes", "no", "left lower lobe"]
    preds = ["yes", "no", "left lower lobe"]
    m = score_qa(refs, preds, judge_scores=None)
    assert m.exact_match == 1.0
    assert m.token_f1 == 1.0


def test_retrieval_recall_at_k() -> None:
    retrieved = [
        ["s1", "s2", "s3"],
        ["s5", "s2", "s9"],
        ["s9", "s9", "s2"],
    ]
    gold = ["s1", "s2", "s2"]
    m = score_retrieval(retrieved, gold, ks=(1, 3))
    assert m.recall_at_1 == 1 / 3
    assert m.by_k[3] == 1.0


def test_retrieval_mrr() -> None:
    retrieved = [["s1", "s9"], ["s9", "s2"]]
    gold = ["s1", "s2"]
    m = score_retrieval(retrieved, gold, ks=(1,))
    # MRR = (1/1 + 1/2) / 2
    assert abs(m.mrr - 0.75) < 1e-6
