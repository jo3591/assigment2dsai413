"""Smoke tests for prompt assembly."""
from __future__ import annotations

from cxr_intel.generation.prompts import (
    build_qa_user,
    build_report_user,
    format_retrieved_block,
)
from cxr_intel.retrieval.base import RetrievalHit


def _hits() -> list[RetrievalHit]:
    return [
        RetrievalHit(study_id="s1", image_path="x.jpg",
                     report_text="Mild cardiomegaly.", score=0.85),
        RetrievalHit(study_id="s2", image_path="y.jpg",
                     report_text="Right lower lobe pneumonia.", score=0.79),
    ]


def test_format_retrieved_block_truncates() -> None:
    long = RetrievalHit(study_id="s3", image_path="z.jpg",
                        report_text="x" * 1000, score=0.5)
    block = format_retrieved_block([long], max_chars=200)
    assert "..." in block


def test_build_report_user_includes_retrieved() -> None:
    user = build_report_user(_hits())
    assert "Mild cardiomegaly" in user
    assert "Generate the FINDINGS" in user


def test_build_qa_user_includes_question() -> None:
    user = build_qa_user("Is there pneumothorax?", _hits())
    assert "Is there pneumothorax?" in user
    assert "Right lower lobe" in user


def test_empty_retrieved_returns_clean_prompt() -> None:
    user = build_report_user([])
    assert "Retrieved similar reports" not in user
