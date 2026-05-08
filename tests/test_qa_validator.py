"""Smoke tests for the QA validator (banned terms, source match, dedup)."""
from __future__ import annotations

from cxr_intel.qa_dataset.schema import QAPair, QualityScores
from cxr_intel.qa_dataset.validator import (
    QAValidator,
    has_banned_term,
    source_sentence_match,
)


def test_banned_term_detection() -> None:
    assert has_banned_term("There is no interval change.")
    assert has_banned_term("Stable findings since prior.")
    assert not has_banned_term("Cardiac silhouette is enlarged.")


def test_source_sentence_match_fuzzy() -> None:
    report = "FINDINGS: There is mild cardiomegaly with vascular congestion."
    assert source_sentence_match("mild cardiomegaly with vascular congestion", report)
    assert not source_sentence_match("severe pneumothorax with chest tube", report)


def test_validator_rejects_banned_term() -> None:
    qa = QAPair(
        qa_id="q1",
        study_id="s1",
        image_path="/tmp/x.jpg",
        question="Is there a new infiltrate?",   # "new" banned
        answer="Yes.",
        question_type="existence",
        anchor_label="Pneumonia",
        anchor_value=1.0,
        source_sentence="infiltrate observed",
        quality_scores=QualityScores(),
    )
    val = QAValidator(judge=None)
    out = val.validate(qa, "Infiltrate observed in right lower lobe.")
    assert out is None
    assert val.rejected[-1][1] == "banned_term"


def test_validator_dedup() -> None:
    qa1 = QAPair(qa_id="q1", study_id="s1", image_path="/tmp/x.jpg",
                 question="Q?", answer="A.", question_type="existence",
                 anchor_label="Cardiomegaly", anchor_value=1.0)
    qa2 = QAPair(qa_id="q2", study_id="s1", image_path="/tmp/x.jpg",
                 question="Q2?", answer="A2.", question_type="existence",
                 anchor_label="Cardiomegaly", anchor_value=1.0)
    val = QAValidator(judge=None)
    deduped = val.dedupe([qa1, qa2])
    assert len(deduped) == 1
