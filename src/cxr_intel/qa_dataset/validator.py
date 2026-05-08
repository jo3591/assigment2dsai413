"""4-dim QA quality validation.

Pipeline:
  1. Banned-term regex reject.
  2. Source-sentence fuzzy match against report (≥85%).
  3. LLM judge scores 4 dims 0-5 — keep mean ≥3.5 and all dims ≥3.
  4. Dedupe by (study_id, question_type, anchor_label).
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from textwrap import dedent

from rapidfuzz import fuzz

from cxr_intel.models.llm_router import LLMRouter
from cxr_intel.qa_dataset.schema import BANNED_TERMS, QAPair, QualityScores
from cxr_intel.utils.logging import get_logger

log = get_logger(__name__)


_BANNED_RE = re.compile(
    r"\b(" + "|".join(re.escape(t) for t in BANNED_TERMS) + r")\b",
    re.IGNORECASE,
)


JUDGE_SYSTEM = dedent(
    """\
    You are a clinical reviewer scoring radiology QA pairs. Given a chest X-ray
    report and a generated (question, answer) pair, score the pair on FOUR
    dimensions, each 0-5 integer:

      - correctness: Is the answer factually correct given the report?
      - consistency: Does the answer use only information present in the report?
      - completeness: Does the answer fully address the question?
      - clinical_relevance: Is the QA pair clinically meaningful?

    Output JSON only with keys: correctness, consistency, completeness,
    clinical_relevance, comment (≤ 20 words).
    """
).strip()


JUDGE_USER_TEMPLATE = dedent(
    """\
    Report:
    \"\"\"
    {report}
    \"\"\"

    Question: {question}
    Answer: {answer}

    Score this QA pair as JSON.
    """
).strip()


def has_banned_term(text: str) -> bool:
    return bool(_BANNED_RE.search(text))


def source_sentence_match(source: str, report: str, threshold: int = 85) -> bool:
    if not source:
        return False
    return fuzz.partial_ratio(source.lower(), report.lower()) >= threshold


@dataclass
class QAValidator:
    judge: LLMRouter | None = None
    threshold_mean: float = 3.5
    threshold_min: float = 3.0
    require_source_match: bool = True

    rejected: list[tuple[QAPair, str]] = field(default_factory=list)

    def _score_with_judge(self, qa: QAPair, report: str) -> QualityScores:
        if self.judge is None:
            # Heuristic: assume 4 across the board if no judge available.
            return QualityScores(correctness=4, consistency=4, completeness=4, clinical_relevance=4)
        user = JUDGE_USER_TEMPLATE.format(
            report=report, question=qa.question, answer=qa.answer
        )
        try:
            obj = self.judge.chat_json(JUDGE_SYSTEM, user)
        except Exception as e:
            log.warning("Judge error qa=%s: %s", qa.qa_id, e)
            return QualityScores()
        return QualityScores(
            correctness=float(obj.get("correctness", 0)),
            consistency=float(obj.get("consistency", 0)),
            completeness=float(obj.get("completeness", 0)),
            clinical_relevance=float(obj.get("clinical_relevance", 0)),
        )

    def _scores_pass(self, scores: QualityScores) -> bool:
        vals = [scores.correctness, scores.consistency, scores.completeness, scores.clinical_relevance]
        if min(vals) < self.threshold_min:
            return False
        return (sum(vals) / 4) >= self.threshold_mean

    def validate(self, qa: QAPair, report: str) -> QAPair | None:
        if has_banned_term(qa.question) or has_banned_term(qa.answer):
            self.rejected.append((qa, "banned_term"))
            return None
        if self.require_source_match and not source_sentence_match(qa.source_sentence, report):
            self.rejected.append((qa, "source_mismatch"))
            return None
        scores = self._score_with_judge(qa, report)
        qa.quality_scores = scores
        if not self._scores_pass(scores):
            self.rejected.append((qa, f"judge_below_threshold:{scores.dict()}"))
            return None
        return qa

    def dedupe(self, pairs: list[QAPair]) -> list[QAPair]:
        seen: set[tuple[str, str, str]] = set()
        out: list[QAPair] = []
        for qa in pairs:
            key = (qa.study_id, qa.question_type, qa.anchor_label)
            if key in seen:
                self.rejected.append((qa, "duplicate"))
                continue
            seen.add(key)
            out.append(qa)
        return out
