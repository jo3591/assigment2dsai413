"""Download + preprocess the Kaggle MIMIC-CXR subset.

Usage:
    python scripts/download_data.py --config configs/data.yaml [--limit 50]
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

from cxr_intel.data.chexpert_labels import (
    label_vec_to_array,
    primary_label,
    rule_based_label_vector,
)
from cxr_intel.data.kaggle_loader import (
    build_basename_index,
    discover_csv,
    download_kaggle_dataset,
    find_image_for_row,
    load_reports_csv,
)
from cxr_intel.data.preprocess import preprocess_dataframe
from cxr_intel.data.splits import patient_level_split, stratified_subsample
from cxr_intel.utils.io import ensure_dir, load_yaml, save_json
from cxr_intel.utils.logging import get_logger

log = get_logger("download_data")


def _derive_study_id_from_path(path: str | None) -> str:
    """MIMIC-CXR paths look like .../s56476430/uuid.jpg — extract the 's...' segment."""
    if not path:
        return ""
    for p in Path(str(path)).parts:
        if p.startswith("s") and len(p) > 1 and p[1:].isdigit():
            return p
    return ""


def main() -> None:
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--config", default="configs/data.yaml")
    parser.add_argument("--limit", type=int, default=None,
                        help="Cap rows after preprocessing (smoke test)")
    parser.add_argument("--skip-download", action="store_true")
    parser.add_argument("--no-basename-index", action="store_true",
                        help="Skip slow basename-index fallback (recommended on Kaggle's network mount)")
    args = parser.parse_args()

    load_dotenv()
    cfg = load_yaml(args.config)
    raw_dir = Path(cfg["kaggle"]["download_root"])
    ensure_dir(raw_dir)

    if not args.skip_download:
        download_kaggle_dataset(cfg["kaggle"]["dataset"], raw_dir)

    csv_path = discover_csv(raw_dir)
    log.info("Reports CSV: %s", csv_path)
    df = load_reports_csv(csv_path)

    df = preprocess_dataframe(
        df,
        text_col=cfg["preprocess"]["text_column"],
        min_tokens=cfg["preprocess"]["min_report_tokens"],
    )

    log.info("Computing rule-based CheXpert labels…")
    label_records: list[dict] = []
    for _, row in df.iterrows():
        vec = rule_based_label_vector(row["clean_text"])
        label_records.append({
            "primary_chexpert_label": primary_label(vec),
            "chexpert_vec": label_vec_to_array(vec).tolist(),
        })
    labels_df = pd.DataFrame(label_records)
    df = pd.concat([df.reset_index(drop=True), labels_df], axis=1)

    if args.limit:
        df = df.head(args.limit).copy()
        log.info("Limited to %d rows", args.limit)
    elif cfg["sample"]["n_total"] and cfg["sample"]["n_total"] < len(df):
        df = stratified_subsample(
            df,
            n=cfg["sample"]["n_total"],
            by=cfg["sample"]["stratify_by"],
            seed=cfg["sample"]["random_seed"],
        )
        log.info("Stratified subsample to %d rows", len(df))

    log.info("Resolving image paths…")
    # First pass: direct path-join only (fast, no scan). If hit-rate is too low,
    # we fall back to building a basename index. Skip via --no-basename-index.
    df["image_path"] = df.apply(
        lambda r: find_image_for_row(r, raw_dir, basename_index=None), axis=1
    )
    n_with_image = int(df["image_path"].notna().sum())
    log.info("Images located (direct path): %d / %d", n_with_image, len(df))
    if not args.no_basename_index and n_with_image / max(1, len(df)) < 0.5:
        log.info("Hit rate < 50%% — building basename index as fallback (slow on network mounts)")
        basename_index = build_basename_index(raw_dir)
        missing_mask = df["image_path"].isna()
        df.loc[missing_mask, "image_path"] = df[missing_mask].apply(
            lambda r: find_image_for_row(r, raw_dir, basename_index=basename_index), axis=1
        )
        n_with_image = int(df["image_path"].notna().sum())
        log.info("Images located (after basename index): %d / %d", n_with_image, len(df))
    df = df[df["image_path"].notna()].copy()
    df["image_path"] = df["image_path"].astype(str)

    # Derive study_id from the image path if the CSV didn't carry one
    if "study_id" not in df.columns or df["study_id"].isna().all():
        df["study_id"] = df["image_path"].apply(_derive_study_id_from_path)
        # Drop rows where extraction failed
        before = len(df)
        df = df[df["study_id"].astype(bool)].copy()
        if before != len(df):
            log.warning("Dropped %d rows with unparseable study_id", before - len(df))
        # Dedup at study level (multiple rows can map to the same study)
        before = len(df)
        df = df.drop_duplicates(subset=["study_id"]).copy()
        log.info("study_id dedup (post-resolve): %d -> %d", before, len(df))

    if "subject_id" not in df.columns:
        # Synthesize from study_id when subject_id missing
        df["subject_id"] = df.get("study_id", df.index.astype(str))

    splits = patient_level_split(
        df,
        train_frac=cfg["splits"]["train_frac"],
        val_frac=cfg["splits"]["val_frac"],
        test_frac=cfg["splits"]["test_frac"],
        by=cfg["splits"]["by"],
        seed=cfg["sample"]["random_seed"],
    )
    save_json(splits, cfg["paths"]["splits"])
    log.info("Splits: train=%d, val=%d, test=%d",
             len(splits["train"]), len(splits["val"]), len(splits["test"]))

    out_path = Path(cfg["paths"]["processed"])
    ensure_dir(out_path.parent)
    df.to_parquet(out_path, index=False)
    log.info("Wrote %d rows -> %s", len(df), out_path)

    samples_dir = ensure_dir(cfg["paths"]["samples_dir"])
    for i, row in df.head(5).iterrows():
        try:
            target = samples_dir / f"sample{i+1}{Path(row['image_path']).suffix}"
            if not target.exists():
                target.write_bytes(Path(row["image_path"]).read_bytes())
        except Exception as e:
            log.warning("Could not copy sample %d: %s", i, e)


if __name__ == "__main__":
    main()
