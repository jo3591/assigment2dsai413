"""Pydantic schema for QA pairs and shared constants."""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

QuestionType = Literal["existence", "location", "severity", "attribute", "open"]

BANNED_TERMS: list[str] = [
    "unchanged", "new", "interval", "prior", "follow-up", "followup",
    "previously", "compared to", "since", "stable", "improved", "worsened",
]


class QualityScores(BaseModel):
    correctness: float = 0.0
    consistency: float = 0.0
    completeness: float = 0.0
    clinical_relevance: float = 0.0


class QAPair(BaseModel):
    qa_id: str
    study_id: str
    image_path: str
    question: str
    answer: str
    question_type: QuestionType
    anchor_label: str = Field(..., description="One of CHEXPERT_LABELS")
    anchor_value: float = Field(..., description="1.0 positive, 0.0 negative, -1.0 uncertain")
    source_sentence: str = ""
    quality_scores: QualityScores = Field(default_factory=QualityScores)

    @property
    def quality_mean(self) -> float:
        s = self.quality_scores
        return (s.correctness + s.consistency + s.completeness + s.clinical_relevance) / 4
