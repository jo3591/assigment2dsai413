"""Patient-level train/val/test splits."""
from __future__ import annotations

import random
from typing import Iterable

import pandas as pd


def patient_level_split(
    df: pd.DataFrame,
    train_frac: float = 0.8,
    val_frac: float = 0.1,
    test_frac: float = 0.1,
    by: str = "subject_id",
    seed: int = 42,
) -> dict[str, list]:
    """Return dict with study_id lists for train/val/test, disjoint by subject_id."""
    assert abs(train_frac + val_frac + test_frac - 1.0) < 1e-6
    if by not in df.columns:
        raise ValueError(f"Column {by!r} required for patient-level split. Got {list(df.columns)}")
    rng = random.Random(seed)
    patients = sorted(df[by].astype(str).unique().tolist())
    rng.shuffle(patients)
    n = len(patients)
    n_train = int(n * train_frac)
    n_val = int(n * val_frac)
    train_p = set(patients[:n_train])
    val_p = set(patients[n_train : n_train + n_val])
    test_p = set(patients[n_train + n_val :])
    return {
        "train": df[df[by].astype(str).isin(train_p)]["study_id"].astype(str).tolist()
        if "study_id" in df.columns
        else df.index[df[by].astype(str).isin(train_p)].tolist(),
        "val": df[df[by].astype(str).isin(val_p)]["study_id"].astype(str).tolist()
        if "study_id" in df.columns
        else df.index[df[by].astype(str).isin(val_p)].tolist(),
        "test": df[df[by].astype(str).isin(test_p)]["study_id"].astype(str).tolist()
        if "study_id" in df.columns
        else df.index[df[by].astype(str).isin(test_p)].tolist(),
    }


def stratified_subsample(
    df: pd.DataFrame,
    n: int,
    by: str = "primary_chexpert_label",
    seed: int = 42,
) -> pd.DataFrame:
    """Stratified subsample by a categorical column. Falls back to random if column missing."""
    if by not in df.columns:
        return df.sample(n=min(n, len(df)), random_state=seed).reset_index(drop=True)
    groups = df.groupby(by, group_keys=False)
    counts = groups.size()
    fracs = (counts / counts.sum()).clip(lower=1 / len(df))
    take = (fracs * n).round().astype(int).to_dict()
    out: list[pd.DataFrame] = []
    rng = seed
    for label, sub in groups:
        k = min(take.get(label, 0), len(sub))
        if k > 0:
            out.append(sub.sample(n=k, random_state=rng))
            rng += 1
    if not out:
        return df.sample(n=min(n, len(df)), random_state=seed).reset_index(drop=True)
    return pd.concat(out, ignore_index=True).sample(frac=1.0, random_state=seed).reset_index(drop=True)
