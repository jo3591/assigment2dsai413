"""Download and load the Kaggle MIMIC-CXR subset.

Dataset: https://www.kaggle.com/datasets/simhadrisadaram/mimic-cxr-dataset
Schema is asserted on load — fail fast if upstream changes.
"""
from __future__ import annotations

import ast
import os
from pathlib import Path

import pandas as pd

from cxr_intel.utils.io import ensure_dir
from cxr_intel.utils.logging import get_logger

log = get_logger(__name__)

EXPECTED_TEXT_COL = "text"


def _parse_image_cell(value: str) -> list[str]:
    """The `image` column in mimic_cxr_aug_train.csv is a stringified Python list.
    Returns the list, or [value] if value is already a plain path."""
    if not isinstance(value, str):
        return []
    value = value.strip()
    if value.startswith("[") and value.endswith("]"):
        try:
            parsed = ast.literal_eval(value)
            if isinstance(parsed, (list, tuple)):
                return [str(p) for p in parsed if isinstance(p, str)]
        except (ValueError, SyntaxError):
            pass
    return [value]


def download_kaggle_dataset(slug: str, dest_dir: str | Path) -> Path:
    """Download and unzip a Kaggle dataset. Requires kaggle.json credentials."""
    dest = ensure_dir(dest_dir)
    log.info("Downloading kaggle dataset %s -> %s", slug, dest)
    # Lazy import so the module doesn't fail without kaggle deps
    from kaggle.api.kaggle_api_extended import KaggleApi

    api = KaggleApi()
    api.authenticate()
    api.dataset_download_files(slug, path=str(dest), unzip=True, quiet=False)
    return dest


def discover_csv(root: str | Path) -> Path:
    """Find the report CSV file in the downloaded archive."""
    root = Path(root)
    candidates = [
        *root.glob("*.csv"),
        *root.rglob("mimic*.csv"),
        *root.rglob("*reports*.csv"),
    ]
    if not candidates:
        raise FileNotFoundError(f"No CSV found under {root}")
    # Pick the largest CSV — that's almost always the reports file.
    return max(candidates, key=lambda p: p.stat().st_size)


def load_reports_csv(path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    log.info("Loaded %d rows from %s | columns=%s", len(df), path, list(df.columns))
    if EXPECTED_TEXT_COL not in df.columns:
        # Try to recover with common alternatives
        alt = next((c for c in ["report", "report_text", "TEXT", "Text"] if c in df.columns), None)
        if alt is None:
            raise ValueError(
                f"Expected column '{EXPECTED_TEXT_COL}' not found. Got: {list(df.columns)}"
            )
        df = df.rename(columns={alt: EXPECTED_TEXT_COL})
        log.warning("Renamed %s -> %s", alt, EXPECTED_TEXT_COL)
    return df


def find_image_for_row(row: pd.Series, image_root: Path) -> Path | None:
    """Best-effort image lookup. The Kaggle subset variants ship images under
    different folder layouts; this tries a few common conventions."""
    candidate_keys = ["image_path", "image", "path", "img_path", "filename", "file"]
    # Common image-root prefixes inside the extracted dataset directory
    candidate_roots = [
        image_root,
        image_root / "files",
        image_root / "images",
        image_root / "mimic-cxr-jpg",
        image_root / "mimic-cxr-jpg" / "files",
        image_root / "official_data_iccv_final",
        image_root / "official_data_iccv_final" / "files",
        image_root / "official_data_iccv_final" / "images",
    ]
    for k in candidate_keys:
        if k not in row or not isinstance(row[k], str) or not row[k].strip():
            continue
        # Cell may be a stringified list of multiple views — try each, pick the first hit.
        for rel in _parse_image_cell(row[k]):
            for root in candidate_roots:
                p = root / rel
                if p.exists():
                    return p
            base = Path(rel).name
            for root in candidate_roots:
                hits = list(root.rglob(base)) if root.exists() else []
                if hits:
                    return hits[0]
    if "study_id" in row:
        for ext in (".jpg", ".png", ".jpeg"):
            for root in candidate_roots:
                p = root / f"{row['study_id']}{ext}"
                if p.exists():
                    return p
    return None
