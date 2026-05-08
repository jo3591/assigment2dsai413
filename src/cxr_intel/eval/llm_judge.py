"""LLM-as-judge for QA evaluation.

Scores generated answers against (gold answer, supporting report) on a 0-5
scale across two dimensions: correctness and groundedness. Returns the mean.
"""
from __future__ import annotations

from dataclasses import dataclass
from textwrap import dedent

from cxr_intel.models.llm_router import LLMRouter
from cxr_intel.utils.logging import get_logger

log = get_logger(__name__)


JUDGE_SYSTEM = dedent(
    """\
    You are evaluating a chest X-ray QA model. For each item you will see:
      - the question
      - the gold (reference) answer
      - the model's predicted answer
      - the source radiology report

    Score the prediction on TWO dimensions, each 0-5 integer:
      - correctness: does the prediction agree with the gold answer and report?
      - groundedness: is every claim in the prediction supported by the report?

    Output JSON only with keys: correctness, groundedness, comment (≤ 20 words).
    """
).strip()


JUDGE_USER_TEMPLATE = dedent(
    """\
    Question: {question}
    Gold answer: {gold}
    Predicted answer: {pred}

    Source report:
    \"\"\"
    {report}
    \"\"\"
    """
).strip()


@dataclass
class LLMJudge:
    llm: LLMRouter

    def score_one(self, question: str, gold: str, pred: str, report: str) -> tuple[float, dict]:
        user = JUDGE_USER_TEMPLATE.format(question=question, gold=gold, pred=pred, report=report)
        try:
            obj = self.llm.chat_json(JUDGE_SYSTEM, user)
        except Exception as e:
            log.warning("Judge error: %s", e)
            return 0.0, {}
        c = float(obj.get("correctness", 0))
        g = float(obj.get("groundedness", 0))
        mean = (c + g) / 2
        return mean, obj

    def score_many(
        self,
        questions: list[str],
        golds: list[str],
        preds: list[str],
        reports: list[str],
    ) -> list[float]:
        out: list[float] = []
        for q, g, p, r in zip(questions, golds, preds, reports):
            mean, _ = self.score_one(q, g, p, r)
            out.append(mean)
        return out
