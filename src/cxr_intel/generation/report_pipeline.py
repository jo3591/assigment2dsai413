"""Report Generation Mode pipeline.

Five configurations:
  1. medgemma_only           — pure VLM, no retrieval.
  2. biomedclip_rag          — BiomedCLIP retrieves, MedGemma generates.
  3. colpali_zs_rag          — Zero-shot ColPali retrieves, MedGemma generates.
  4. colpali_lora_rag        — LoRA-tuned ColPali retrieves, MedGemma generates.
  5. colpali_lora_text_llm   — LoRA-tuned ColPali retrieves, OpenRouter LLM generates from text only.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Literal

from PIL import Image

from cxr_intel.generation.prompts import (
    REPORT_SYSTEM,
    TEXT_ONLY_SYSTEM,
    build_report_user,
    build_text_only_user,
)
from cxr_intel.models.llm_router import LLMRouter
from cxr_intel.models.medgemma_runner import MedGemmaRunner
from cxr_intel.retrieval.base import Retriever, RetrievalHit
from cxr_intel.utils.logging import get_logger

log = get_logger(__name__)

ConfigName = Literal[
    "medgemma_only",
    "biomedclip_rag",
    "colpali_zs_rag",
    "colpali_lora_rag",
    "colpali_lora_text_llm",
]


@dataclass
class ReportOutput:
    config: str
    report_text: str
    retrieved: list[RetrievalHit] = field(default_factory=list)
    latency_s: float = 0.0


@dataclass
class ReportPipeline:
    config: ConfigName
    retriever: Retriever | None = None
    medgemma: MedGemmaRunner | None = None
    llm: LLMRouter | None = None
    top_k: int = 3

    def run(self, image: Image.Image) -> ReportOutput:
        t0 = time.time()
        hits: list[RetrievalHit] = []
        if self.config != "medgemma_only" and self.retriever is not None:
            hits = self.retriever.search_image(image, k=self.top_k)

        if self.config == "colpali_lora_text_llm":
            if self.llm is None:
                raise ValueError("llm is required for colpali_lora_text_llm")
            user = build_text_only_user("Generate a chest X-ray report (FINDINGS + IMPRESSION).", hits)
            text = self.llm.chat(TEXT_ONLY_SYSTEM, user)
        else:
            if self.medgemma is None:
                raise ValueError("medgemma is required for VLM configs")
            user = build_report_user(hits)
            text = self.medgemma.generate([image], REPORT_SYSTEM, user)

        return ReportOutput(
            config=self.config,
            report_text=text,
            retrieved=hits,
            latency_s=time.time() - t0,
        )
