"""Prompt templates for report generation, QA, and the text-only LLM ablation."""
from __future__ import annotations

from textwrap import dedent

from cxr_intel.retrieval.base import RetrievalHit


def format_retrieved_block(hits: list[RetrievalHit], max_chars: int = 1200) -> str:
    if not hits:
        return ""
    lines = ["Retrieved similar reports (use only as terminology reference):"]
    budget = max_chars
    for i, h in enumerate(hits, 1):
        snippet = h.report_text.strip().replace("\n", " ")
        # Inner truncate to keep any single retrieved snippet bounded
        cap = min(400, max(80, budget - 32))
        if len(snippet) > cap:
            snippet = snippet[: max(1, cap - 3)] + "..."
        block = f"[{i}] (similarity={h.score:.3f}) {snippet}"
        if budget - len(block) < 0 and len(lines) > 1:
            break
        budget -= len(block)
        lines.append(block)
    return "\n".join(lines)


REPORT_SYSTEM = dedent(
    """\
    You are an experienced radiologist drafting a chest X-ray report.
    Produce two sections only: FINDINGS and IMPRESSION.
    Be concise, clinically accurate, and use neutral observational language.
    Do not use comparative terms (unchanged, new, interval, prior, follow-up, since).
    Only describe what is visible in the supplied radiograph. If retrieval context is
    provided, you may use it to inform terminology — do not copy verbatim and do not
    invent findings absent from the image.
    """
).strip()


REPORT_USER_TEMPLATE = dedent(
    """\
    Generate the FINDINGS and IMPRESSION sections for this chest X-ray.
    {retrieved_block}
    """
).strip()


QA_SYSTEM = dedent(
    """\
    You are an experienced radiologist answering a clinical question about a chest X-ray.
    Ground every answer strictly in what is visible in the radiograph and any provided
    retrieved evidence. Do not use comparative terms (unchanged, new, interval, prior,
    follow-up, since). If you cannot answer from the image and evidence, say so explicitly.
    Keep answers under 60 words.
    """
).strip()


QA_USER_TEMPLATE = dedent(
    """\
    Question: {question}
    {retrieved_block}
    Answer:
    """
).strip()


# Text-only LLM ablation: no image, only retrieved reports + question.
TEXT_ONLY_SYSTEM = dedent(
    """\
    You are a radiology assistant. You will be given retrieved radiology report
    excerpts and a question. Answer ONLY using information present in the excerpts.
    If the answer is not in the excerpts, reply: "Not stated in retrieved reports."
    Do not use comparative terms (unchanged, new, interval, prior, follow-up, since).
    """
).strip()


TEXT_ONLY_USER_TEMPLATE = dedent(
    """\
    {retrieved_block}

    Question: {question}
    Answer:
    """
).strip()


def build_report_user(retrieved: list[RetrievalHit] | None) -> str:
    block = format_retrieved_block(retrieved or [])
    return REPORT_USER_TEMPLATE.format(retrieved_block=block)


def build_qa_user(question: str, retrieved: list[RetrievalHit] | None) -> str:
    block = format_retrieved_block(retrieved or [])
    return QA_USER_TEMPLATE.format(question=question, retrieved_block=block)


def build_text_only_user(question: str, retrieved: list[RetrievalHit] | None) -> str:
    block = format_retrieved_block(retrieved or [])
    return TEXT_ONLY_USER_TEMPLATE.format(retrieved_block=block, question=question)
