"""Smoke tests for report preprocessing."""
from __future__ import annotations

import pandas as pd

from cxr_intel.data.preprocess import (
    clean_report,
    extract_sections,
    preprocess_dataframe,
)


def test_clean_report_strips_deid_and_whitespace() -> None:
    raw = "FINDINGS:\n\n  Lungs are clear. ___ no pneumothorax  \n  IMPRESSION: Normal."
    cleaned = clean_report(raw)
    assert "___" not in cleaned
    assert "  " not in cleaned


def test_extract_sections_basic() -> None:
    raw = "FINDINGS: Cardiac silhouette normal. IMPRESSION: No acute abnormality."
    s = extract_sections(raw)
    assert "cardiac silhouette" in s.findings.lower()
    assert "no acute" in s.impression.lower()


def test_extract_sections_missing_impression() -> None:
    raw = "Lungs are clear. No focal consolidation."
    s = extract_sections(raw)
    assert s.findings  # falls back to whole text
    assert s.impression == ""


def test_preprocess_dataframe_drops_short_reports() -> None:
    df = pd.DataFrame({
        "study_id": ["s1", "s2", "s3"],
        "text": [
            "FINDINGS: " + "lungs clear " * 20 + "IMPRESSION: normal.",
            "short",
            "FINDINGS: " + "edema present " * 15 + "IMPRESSION: pulmonary edema.",
        ],
    })
    out = preprocess_dataframe(df, min_tokens=20)
    assert len(out) == 2
    assert "s2" not in out["study_id"].tolist()
