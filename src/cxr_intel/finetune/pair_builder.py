"""Contrastive (image, report) pair construction with hard negatives."""
from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Iterable, Sequence

import pandas as pd

from cxr_intel.utils.logging import get_logger

log = get_logger(__name__)


@dataclass(slots=True)
class ContrastiveExample:
    image_path: str
    positive_text: str
    negative_texts: list[str]
    study_id: str
    primary_label: str


def build_contrastive_pairs(
    df: pd.DataFrame,
    n_hard_negatives: int = 3,
    seed: int = 42,
) -> list[ContrastiveExample]:
    """For each row in df (must contain image_path, study_id, findings, impression,
    primary_chexpert_label), build a positive (image, FINDINGS+IMPRESSION) and
    n_hard_negatives drawn from the same primary label but different study_id."""
    required = {"image_path", "study_id", "findings", "impression", "primary_chexpert_label"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"build_contrastive_pairs missing columns: {missing}")

    rng = random.Random(seed)
    by_label: dict[str, list[int]] = {}
    for i, row in df.iterrows():
        by_label.setdefault(row["primary_chexpert_label"], []).append(i)

    examples: list[ContrastiveExample] = []
    for i, row in df.iterrows():
        label = row["primary_chexpert_label"]
        pool = [j for j in by_label.get(label, []) if df.at[j, "study_id"] != row["study_id"]]
        if len(pool) < n_hard_negatives:
            # Top up from random pool of any label
            extra = [
                j for j in df.index.tolist()
                if df.at[j, "study_id"] != row["study_id"] and j not in pool
            ]
            rng.shuffle(extra)
            pool = pool + extra[: max(0, n_hard_negatives - len(pool))]
        rng.shuffle(pool)
        neg_idx = pool[:n_hard_negatives]
        pos_text = (str(row["findings"]) + " " + str(row["impression"])).strip()
        neg_texts = [
            (str(df.at[j, "findings"]) + " " + str(df.at[j, "impression"])).strip()
            for j in neg_idx
        ]
        examples.append(
            ContrastiveExample(
                image_path=str(row["image_path"]),
                positive_text=pos_text,
                negative_texts=neg_texts,
                study_id=str(row["study_id"]),
                primary_label=str(label),
            )
        )

    log.info("Built %d contrastive examples (n_hard_neg=%d)", len(examples), n_hard_negatives)
    return examples
