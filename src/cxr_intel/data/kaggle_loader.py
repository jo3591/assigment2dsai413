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
    """Find the report CSV file in the downloaded archive.

    Avoid rglob on the top-level directory — it can recurse into image
    subdirectories with hundreds of thousands of files. Try top-level first,
    then immediate subdirs only.
    """
    root = Path(root)
    # 1. Top-level CSVs (fast)
    candidates = list(root.glob("*.csv"))
    if not candidates:
        # 2. One-level deep (immediate subdirs only, not recursive)
        for sub in root.iterdir():
            if sub.is_dir():
                candidates.extend(sub.glob("*.csv"))
                if candidates:
                    break
    if not candidates:
        raise FileNotFoundError(f"No CSV found under {root} (top-level or 1 deep)")
    # Prefer "train" file, then largest
    train_hits = [c for c in candidates if "train" in c.name.lower()]
    if train_hits:
        return max(train_hits, key=lambda p: p.stat().st_size)
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


def find_image_for_row(
    row: pd.Series,
    image_root: Path,
    basename_index: dict[str, Path] | None = None,
) -> Path | None:
    """Best-effort image lookup. Direct path-join first (O(1) stat), then
    optional O(1) basename lookup via a pre-built index. No recursive rglob —
    that's catastrophic on directories with 100K+ files."""
    candidate_keys = ["image_path", "image", "path", "img_path", "filename", "file"]
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
        for rel in _parse_image_cell(row[k]):
            # 1. Direct path join (fast)
            for root in candidate_roots:
                p = root / rel
                if p.exists():
                    return p
            # 2. Basename lookup in pre-built index (O(1))
            if basename_index is not None:
                hit = basename_index.get(Path(rel).name)
                if hit is not None:
                    return hit
    if "study_id" in row:
        for ext in (".jpg", ".png", ".jpeg"):
            for root in candidate_roots:
                p = root / f"{row['study_id']}{ext}"
                if p.exists():
                    return p
    return None


def build_basename_index(image_root: Path, suffixes: tuple[str, ...] = (".jpg", ".png", ".jpeg")
                         ) -> dict[str, Path]:
    """One-pass scan that builds {basename: full_path}. Use to accelerate row-level
    image resolution when the path layout in the CSV differs from on-disk layout."""
    log.info("Building basename index under %s ...", image_root)
    index: dict[str, Path] = {}
    # Scan known sub-roots first (cheaper than a top-level rglob)
    for sub in ["official_data_iccv_final", "files", "images", "mimic-cxr-jpg", "."]:
        root = image_root / sub if sub != "." else image_root
        if not root.exists():
            continue
        for ext in suffixes:
            for p in root.rglob(f"*{ext}"):
                # Don't overwrite — keep first hit per basename
                if p.name not in index:
                    index[p.name] = p
        # Stop once we've found a healthy chunk so we don't double-scan
        if len(index) > 1000:
            break
    log.info("Basename index: %d entries", len(index))
    return index
