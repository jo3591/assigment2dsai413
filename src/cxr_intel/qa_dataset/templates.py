"""Question templates per CheXpert label, structured by question type.

Used both as scaffold for the LLM (suggested phrasings to vary) and as a
template-only fallback when API budget is exhausted.
"""
from __future__ import annotations

from cxr_intel.data.chexpert_labels import CHEXPERT_LABELS

EXISTENCE_TEMPLATES: list[str] = [
    "Is there evidence of {label}?",
    "Are signs of {label} visible on this radiograph?",
    "Can {label} be identified?",
]

LOCATION_TEMPLATES: list[str] = [
    "In which lung zone is the {label} located?",
    "Where on the radiograph is the {label} most apparent?",
    "Which side shows the {label}?",
]

SEVERITY_TEMPLATES: list[str] = [
    "How severe is the {label}?",
    "What is the extent of the {label}?",
    "Describe the magnitude of the {label}.",
]

ATTRIBUTE_TEMPLATES: list[str] = [
    "Describe the appearance of the {label}.",
    "What are the radiographic features of the {label}?",
]

OPEN_TEMPLATES: list[str] = [
    "What are the principal findings on this chest radiograph?",
    "Summarize the main observations from this chest X-ray.",
]


def templates_for(question_type: str, label: str) -> list[str]:
    table = {
        "existence": EXISTENCE_TEMPLATES,
        "location": LOCATION_TEMPLATES,
        "severity": SEVERITY_TEMPLATES,
        "attribute": ATTRIBUTE_TEMPLATES,
        "open": OPEN_TEMPLATES,
    }
    pool = table[question_type]
    if question_type == "open":
        return pool
    return [t.format(label=label.lower()) for t in pool]


__all__ = [
    "CHEXPERT_LABELS",
    "templates_for",
    "EXISTENCE_TEMPLATES",
    "LOCATION_TEMPLATES",
    "SEVERITY_TEMPLATES",
    "ATTRIBUTE_TEMPLATES",
    "OPEN_TEMPLATES",
]
