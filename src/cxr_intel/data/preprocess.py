"""Report cleaning and Findings/Impression extraction."""
from __future__ import annotations

import re
from dataclasses import dataclass

import pandas as pd

from cxr_intel.utils.logging import get_logger

log = get_logger(__name__)

DEID_PATTERN = re.compile(r"_{2,}")
WS_PATTERN = re.compile(r"\s+")
FINDINGS_RE = re.compile(r"(?is)\bfindings\b\s*:?\s*(.*?)(?=\bimpression\b|$)")
IMPRESSION_RE = re.compile(r"(?is)\bimpression\b\s*:?\s*(.*?)$")


@dataclass(slots=True)
class ReportSections:
    findings: str
    impression: str
    full: str


def clean_report(text: str) -> str:
    if not isinstance(text, str):
        return ""
    text = DEID_PATTERN.sub(" ", text)
    text = WS_PATTERN.sub(" ", text).strip()
    return text


def extract_sections(text: str) -> ReportSections:
    cleaned = clean_report(text)
    findings_match = FINDINGS_RE.search(cleaned)
    impression_match = IMPRESSION_RE.search(cleaned)
    findings = (findings_match.group(1).strip() if findings_match else "").strip(" .:;-")
    impression = (impression_match.group(1).strip() if impression_match else "").strip(" .:;-")
    if not findings and not impression:
        # Fallback: treat whole report as findings
        findings = cleaned
    return ReportSections(findings=findings, impression=impression, full=cleaned)


def token_count(text: str) -> int:
    return len(text.split())


def preprocess_dataframe(
    df: pd.DataFrame,
    text_col: str = "text",
    min_tokens: int = 30,
) -> pd.DataFrame:
    """Apply cleaning, section extraction, length filter, and dedup."""
    df = df.copy()
    df["clean_text"] = df[text_col].astype(str).map(clean_report)
    df["n_tokens"] = df["clean_text"].map(token_count)

    sections = df["clean_text"].map(extract_sections)
    df["findings"] = [s.findings for s in sections]
    df["impression"] = [s.impression for s in sections]

    before = len(df)
    df = df[df["n_tokens"] >= min_tokens].copy()
    log.info("Length filter: %d -> %d (drop reports < %d tokens)", before, len(df), min_tokens)

    if "study_id" in df.columns:
        before = len(df)
        df = df.drop_duplicates(subset=["study_id"]).copy()
        log.info("study_id dedup: %d -> %d", before, len(df))

    return df.reset_index(drop=True)
