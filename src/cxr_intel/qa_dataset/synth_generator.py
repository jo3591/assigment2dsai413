"""Synthetic QA generator backed by an OpenRouter / NVIDIA NIM LLM.

For each report, asks the LLM to write {question, answer, source_sentence}
grounded in the supplied report text and CheXpert vector. Banned-term constraint
is enforced both in the system prompt and post-hoc by the validator.
"""
from __future__ import annotations

import hashlib
import json
import random
from dataclasses import dataclass, field
from pathlib import Path
from textwrap import dedent
from typing import Iterable

from cxr_intel.data.chexpert_labels import primary_label, rule_based_label_vector
from cxr_intel.models.llm_router import LLMRouter
from cxr_intel.qa_dataset.schema import BANNED_TERMS, QAPair, QualityScores
from cxr_intel.qa_dataset.templates import templates_for
from cxr_intel.utils.io import ensure_dir, save_json
from cxr_intel.utils.logging import get_logger

log = get_logger(__name__)


SYSTEM_PROMPT = dedent(
    f"""\
    You are a senior radiologist writing question-answer pairs from chest X-ray
    reports for an educational dataset. Follow these rules strictly:

    1. Ground every answer in the supplied report text. If the report does not
       support an answer, return {{ "skip": true }} with a brief reason.
    2. Pretend you are observing the radiograph for the first time. Do NOT use
       any of these comparative terms: {", ".join(BANNED_TERMS)}.
    3. Use observational verbs (seen, demonstrated, identified, shown).
       Refer to the image as "the radiograph", not "the report".
    4. Your output MUST be valid JSON with exactly these keys:
       {{ "question": str, "answer": str, "source_sentence": str }}
       — or {{ "skip": true, "reason": str }} if you cannot ground the answer.
    5. Keep answers concise (≤ 60 words).
    """
).strip()


USER_TEMPLATE = dedent(
    """\
    Report:
    \"\"\"
    {report}
    \"\"\"

    CheXpert vector (label: value where 1.0=positive, 0.0=negative, -1.0=uncertain):
    {chexpert}

    Target question type: {qtype}
    Anchor label: {anchor_label} (value={anchor_value})
    Suggested phrasings (vary, do not copy verbatim):
    {suggestions}

    Write a single QA pair grounded in the report. Output JSON only.
    """
).strip()


@dataclass
class SynthGenerator:
    llm: LLMRouter = field(default_factory=LLMRouter)
    cache_dir: Path | None = None
    questions_per_report: int = 4

    def _cache_key(self, report: str, qtype: str, anchor: str) -> str:
        h = hashlib.sha256(f"{report}::{qtype}::{anchor}::{self.llm.model}".encode()).hexdigest()
        return h[:16]

    def _read_cache(self, key: str) -> dict | None:
        if not self.cache_dir:
            return None
        p = self.cache_dir / f"{key}.json"
        if p.exists():
            try:
                return json.loads(p.read_text(encoding="utf-8"))
            except Exception:
                return None
        return None

    def _write_cache(self, key: str, obj: dict) -> None:
        if not self.cache_dir:
            return
        ensure_dir(self.cache_dir)
        (self.cache_dir / f"{key}.json").write_text(json.dumps(obj), encoding="utf-8")

    def _pick_question_plan(self, anchor_label: str, anchor_value: float, rng: random.Random
                            ) -> list[tuple[str, str, float]]:
        """Decide which question_types to ask for this report. Returns list of
        (qtype, anchor_label, anchor_value)."""
        plan: list[tuple[str, str, float]] = [("existence", anchor_label, anchor_value)]
        if anchor_value >= 1.0:
            plan.append(("location", anchor_label, anchor_value))
            plan.append(("severity", anchor_label, anchor_value))
        else:
            plan.append(("attribute", anchor_label, anchor_value))
        plan.append(("open", "No Finding" if anchor_value < 1.0 else anchor_label, anchor_value))
        rng.shuffle(plan)
        return plan[: self.questions_per_report]

    def generate_for_report(
        self,
        study_id: str,
        image_path: str,
        report_text: str,
        seed: int = 42,
    ) -> list[QAPair]:
        rng = random.Random(f"{study_id}:{seed}")
        chex = rule_based_label_vector(report_text)
        anchor = primary_label(chex)
        anchor_val = chex.get(anchor, 0.0)

        plan = self._pick_question_plan(anchor, anchor_val, rng)
        out: list[QAPair] = []
        for q_idx, (qtype, alabel, aval) in enumerate(plan):
            suggestions = templates_for(qtype, alabel)
            user = USER_TEMPLATE.format(
                report=report_text,
                chexpert=json.dumps(chex, indent=0),
                qtype=qtype,
                anchor_label=alabel,
                anchor_value=aval,
                suggestions="\n".join(f"- {s}" for s in suggestions),
            )
            key = self._cache_key(report_text, qtype, alabel)
            cached = self._read_cache(key)
            try:
                obj = cached or self.llm.chat_json(SYSTEM_PROMPT, user)
            except Exception as e:
                log.warning("LLM error on study=%s qtype=%s: %s", study_id, qtype, e)
                continue
            if not cached:
                self._write_cache(key, obj)

            if obj.get("skip"):
                continue
            try:
                qa = QAPair(
                    qa_id=f"{study_id}_{qtype}_{q_idx}",
                    study_id=str(study_id),
                    image_path=str(image_path),
                    question=obj["question"].strip(),
                    answer=obj["answer"].strip(),
                    question_type=qtype,
                    anchor_label=alabel,
                    anchor_value=float(aval),
                    source_sentence=obj.get("source_sentence", "").strip(),
                    quality_scores=QualityScores(),
                )
                out.append(qa)
            except (KeyError, ValueError) as e:
                log.warning("Bad LLM JSON for study=%s: %s", study_id, e)
        return out

    def generate_many(self, rows: Iterable[dict], seed: int = 42) -> Iterable[QAPair]:
        for row in rows:
            yield from self.generate_for_report(
                study_id=str(row["study_id"]),
                image_path=str(row["image_path"]),
                report_text=str(row.get("clean_text", row.get("text", ""))),
                seed=seed,
            )
