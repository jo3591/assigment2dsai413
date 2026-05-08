"""QA Mode pipeline. Same configs as ReportPipeline, parameterized by question."""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Literal

from PIL import Image

from cxr_intel.generation.prompts import (
    QA_SYSTEM,
    TEXT_ONLY_SYSTEM,
    build_qa_user,
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
class QAOutput:
    config: str
    question: str
    answer: str
    retrieved: list[RetrievalHit] = field(default_factory=list)
    latency_s: float = 0.0


@dataclass
class QAPipeline:
    config: ConfigName
    retriever: Retriever | None = None
    medgemma: MedGemmaRunner | None = None
    llm: LLMRouter | None = None
    top_k: int = 3

    def run(self, image: Image.Image, question: str) -> QAOutput:
        t0 = time.time()
        hits: list[RetrievalHit] = []
        if self.config != "medgemma_only" and self.retriever is not None:
            # Retrieve by image (visual similarity); could also retrieve by question text.
            hits = self.retriever.search_image(image, k=self.top_k)

        if self.config == "colpali_lora_text_llm":
            if self.llm is None:
                raise ValueError("llm is required for colpali_lora_text_llm")
            user = build_text_only_user(question, hits)
            text = self.llm.chat(TEXT_ONLY_SYSTEM, user)
        else:
            if self.medgemma is None:
                raise ValueError("medgemma is required for VLM configs")
            user = build_qa_user(question, hits)
            text = self.medgemma.generate([image], QA_SYSTEM, user, max_new_tokens=200)

        return QAOutput(
            config=self.config,
            question=question,
            answer=text,
            retrieved=hits,
            latency_s=time.time() - t0,
        )
