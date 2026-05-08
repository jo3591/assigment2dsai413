"""Report generation metrics: BLEU-1/2/4, ROUGE-L, BERTScore, CheXbert-F1 proxy."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from cxr_intel.data.chexpert_labels import micro_f1, rule_based_label_vector
from cxr_intel.utils.logging import get_logger

log = get_logger(__name__)


@dataclass
class ReportMetrics:
    bleu1: float = 0.0
    bleu2: float = 0.0
    bleu4: float = 0.0
    rouge_l: float = 0.0
    bertscore_f1: float = 0.0
    chexbert_f1: float = 0.0
    radgraph_f1: float | None = None
    n: int = 0


def _compute_bleu(refs: Sequence[str], preds: Sequence[str]) -> tuple[float, float, float]:
    import sacrebleu

    bleu1 = sacrebleu.corpus_bleu(preds, [refs], max_ngram_order=1).score / 100
    bleu2 = sacrebleu.corpus_bleu(preds, [refs], max_ngram_order=2).score / 100
    bleu4 = sacrebleu.corpus_bleu(preds, [refs]).score / 100
    return bleu1, bleu2, bleu4


def _compute_rouge_l(refs: Sequence[str], preds: Sequence[str]) -> float:
    from rouge_score import rouge_scorer

    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    scores = [scorer.score(r, p)["rougeL"].fmeasure for r, p in zip(refs, preds)]
    return sum(scores) / max(1, len(scores))


def _compute_bertscore(refs: Sequence[str], preds: Sequence[str],
                      model_type: str = "microsoft/deberta-xlarge-mnli") -> float:
    try:
        from bert_score import score as bs_score

        _, _, f1 = bs_score(list(preds), list(refs), model_type=model_type, lang="en", verbose=False)
        return float(f1.mean())
    except Exception as e:
        log.warning("BERTScore failed (%s) — falling back to roberta-large", e)
        from bert_score import score as bs_score

        _, _, f1 = bs_score(list(preds), list(refs), model_type="roberta-large", lang="en", verbose=False)
        return float(f1.mean())


def _compute_chexbert_f1(refs: Sequence[str], preds: Sequence[str]) -> float:
    """Rule-based proxy for CheXbert F1 on 14 labels."""
    f1s = []
    for r, p in zip(refs, preds):
        f1s.append(micro_f1(rule_based_label_vector(p), rule_based_label_vector(r)))
    return sum(f1s) / max(1, len(f1s))


def _compute_radgraph_f1(refs: Sequence[str], preds: Sequence[str]) -> float | None:
    try:
        from radgraph import F1RadGraph

        f1 = F1RadGraph(reward_level="partial")
        score, *_ = f1(list(preds), list(refs))
        return float(score)
    except Exception as e:
        log.info("RadGraph unavailable (%s) — skipping", e)
        return None


def score_report(
    refs: Sequence[str],
    preds: Sequence[str],
    bertscore_model: str = "microsoft/deberta-xlarge-mnli",
    enable_radgraph: bool = False,
) -> ReportMetrics:
    if len(refs) != len(preds):
        raise ValueError(f"len(refs)={len(refs)} != len(preds)={len(preds)}")
    if not refs:
        return ReportMetrics()
    bleu1, bleu2, bleu4 = _compute_bleu(refs, preds)
    return ReportMetrics(
        bleu1=bleu1,
        bleu2=bleu2,
        bleu4=bleu4,
        rouge_l=_compute_rouge_l(refs, preds),
        bertscore_f1=_compute_bertscore(refs, preds, bertscore_model),
        chexbert_f1=_compute_chexbert_f1(refs, preds),
        radgraph_f1=_compute_radgraph_f1(refs, preds) if enable_radgraph else None,
        n=len(refs),
    )
