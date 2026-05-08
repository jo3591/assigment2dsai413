"""CheXpert 14-label rule-based parser.

The official CheXpert labeler requires Java/license; this is a regex fallback
that recognizes the 14 conditions and basic negation patterns. Sufficient for
stratified sampling and CheXbert-F1 estimation in a graduate assignment.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

import numpy as np

CHEXPERT_LABELS: list[str] = [
    "Atelectasis",
    "Cardiomegaly",
    "Consolidation",
    "Edema",
    "Enlarged Cardiomediastinum",
    "Fracture",
    "Lung Lesion",
    "Lung Opacity",
    "No Finding",
    "Pleural Effusion",
    "Pleural Other",
    "Pneumonia",
    "Pneumothorax",
    "Support Devices",
]

# Synonym/keyword patterns per label (lowercase, regex-friendly).
LABEL_KEYWORDS: dict[str, list[str]] = {
    "Atelectasis": [r"atelecta", r"collapse"],
    "Cardiomegaly": [r"cardiomegaly", r"enlarged heart", r"cardiac (silhouette )?enlarg"],
    "Consolidation": [r"consolidat"],
    "Edema": [r"edema", r"vascular congestion", r"interstitial fluid"],
    "Enlarged Cardiomediastinum": [r"mediastin\w* (widen|enlarg)", r"enlarged mediastinum"],
    "Fracture": [r"fracture"],
    "Lung Lesion": [r"lesion", r"mass", r"nodule"],
    "Lung Opacity": [r"opacit"],
    "No Finding": [r"no acute\w* (finding|abnormalit)", r"normal\w* (chest|cxr|radiograph)"],
    "Pleural Effusion": [r"effusion"],
    "Pleural Other": [r"pleural thickening", r"pleural plaque", r"calcified pleura"],
    "Pneumonia": [r"pneumonia", r"infection of the lung", r"infiltrat"],
    "Pneumothorax": [r"pneumothorax", r"ptx"],
    "Support Devices": [
        r"endotracheal tube", r"\bett\b", r"central line", r"picc", r"pacemaker",
        r"feeding tube", r"chest tube", r"nasogastric", r"\bng tube\b",
    ],
}

NEGATION_WINDOW = 4  # tokens before a keyword to scan for negation
NEGATION_TOKENS = {
    "no", "not", "without", "denies", "negative", "absent", "ruled out",
    "free of", "rule out", "unremarkable", "clear of",
}
UNCERTAIN_TOKENS = {
    "possible", "possibly", "may", "might", "suggests", "suspicious",
    "questionable", "concern for", "cannot exclude", "could represent",
    "likely", "probable",
}


@dataclass(slots=True)
class LabelHit:
    label: str
    value: float            # 1.0 positive, 0.0 negative, -1.0 uncertain
    matched_term: str
    sentence: str


def _split_sentences(text: str) -> list[str]:
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]


def _scan_window(words: list[str], idx: int, window: int) -> list[str]:
    start = max(0, idx - window)
    return [w.lower() for w in words[start:idx]]


def _classify_match(sentence: str, term: str) -> float:
    lower = sentence.lower()
    pos = lower.find(term.lower().split()[0]) if term else -1
    if pos < 0:
        return 1.0
    words = re.findall(r"\w+|[^\w\s]", lower)
    # Approximate token-level scan for negation/uncertainty in a small window
    flat = " ".join(words[:max(1, pos)])
    for neg in NEGATION_TOKENS:
        if re.search(rf"\b{re.escape(neg)}\b", flat[-80:]):
            return 0.0
    for unc in UNCERTAIN_TOKENS:
        if re.search(rf"\b{re.escape(unc)}\b", flat[-80:]):
            return -1.0
    return 1.0


def rule_based_label_vector(report_text: str) -> dict[str, float]:
    """Return a CheXpert-style vector. Missing labels default to NaN-equivalent 0.0
    only if "No Finding" is matched; otherwise we leave them at 0.0 (negative-by-default)
    which matches the MIMIC-CXR-VQA convention."""
    out: dict[str, float] = {label: 0.0 for label in CHEXPERT_LABELS}
    sentences = _split_sentences(report_text)
    for sent in sentences:
        for label, patterns in LABEL_KEYWORDS.items():
            for pat in patterns:
                m = re.search(pat, sent, flags=re.IGNORECASE)
                if m:
                    # "No Finding" patterns already encode negation as the predicate;
                    # don't re-apply negation analysis to them.
                    if label == "No Finding":
                        val = 1.0
                    else:
                        val = _classify_match(sent, m.group(0))
                    # Take strongest positive evidence (prefer 1.0 over -1.0 over 0.0)
                    prev = out[label]
                    out[label] = max(prev, val) if val > 0 else (val if prev == 0 else prev)
    if out["No Finding"] >= 1.0:
        # Negate other findings if explicitly normal
        for k in CHEXPERT_LABELS:
            if k != "No Finding" and out[k] == 0.0:
                pass  # already zero
    return out


def primary_label(label_vec: dict[str, float]) -> str:
    """Pick the primary CheXpert label for stratified sampling."""
    positives = [k for k, v in label_vec.items() if v >= 1.0 and k != "No Finding"]
    if positives:
        return positives[0]
    if label_vec.get("No Finding", 0.0) >= 1.0:
        return "No Finding"
    return "Other"


def label_vec_to_array(vec: dict[str, float]) -> np.ndarray:
    return np.array([vec[l] for l in CHEXPERT_LABELS], dtype=np.float32)


def micro_f1(pred: dict[str, float], gold: dict[str, float]) -> float:
    """Micro F1 over positive labels only (1.0 == positive). Used as a CheXbert-F1 proxy."""
    p = label_vec_to_array(pred) >= 1.0
    g = label_vec_to_array(gold) >= 1.0
    tp = float((p & g).sum())
    fp = float((p & ~g).sum())
    fn = float((~p & g).sum())
    if tp == 0 and fp == 0 and fn == 0:
        return 1.0
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    return 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
