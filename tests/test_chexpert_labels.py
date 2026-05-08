"""Smoke tests for the rule-based CheXpert labeler."""
from __future__ import annotations

from cxr_intel.data.chexpert_labels import (
    CHEXPERT_LABELS,
    micro_f1,
    primary_label,
    rule_based_label_vector,
)


def test_label_vector_keys() -> None:
    vec = rule_based_label_vector("FINDINGS: Mild cardiomegaly. No effusion.")
    assert set(vec.keys()) == set(CHEXPERT_LABELS)


def test_positive_cardiomegaly_negative_effusion() -> None:
    vec = rule_based_label_vector("FINDINGS: Mild cardiomegaly. No pleural effusion.")
    assert vec["Cardiomegaly"] >= 1.0
    assert vec["Pleural Effusion"] == 0.0


def test_no_finding() -> None:
    vec = rule_based_label_vector("IMPRESSION: No acute abnormality.")
    assert vec["No Finding"] >= 1.0


def test_primary_label_prefers_positives() -> None:
    vec = rule_based_label_vector("FINDINGS: Pneumothorax on the left.")
    assert primary_label(vec) == "Pneumothorax"


def test_micro_f1_self_consistent() -> None:
    text = "FINDINGS: Cardiomegaly with pulmonary edema."
    vec = rule_based_label_vector(text)
    assert micro_f1(vec, vec) >= 0.99
