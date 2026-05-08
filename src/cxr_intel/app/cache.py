"""Streamlit-aware lazy loaders for retrievers, MedGemma, and the LLM router.

@st.cache_resource ensures models are loaded once per Streamlit process.
Loaders gracefully degrade: if a checkpoint isn't accessible (no GPU, no HF
token, no LoRA adapter on disk), the loader returns None and the page shows
a clear error.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import streamlit as st


@st.cache_resource(show_spinner="Loading MedGemma…")
def load_medgemma(checkpoint: str = "google/medgemma-4b-it",
                  quantization: Optional[str] = None):
    try:
        from cxr_intel.models.medgemma_runner import MedGemmaRunner

        runner = MedGemmaRunner(checkpoint=checkpoint, quantization=quantization)
        runner.load()
        return runner
    except Exception as e:
        st.warning(f"MedGemma not available: {e}")
        return None


@st.cache_resource(show_spinner="Loading ColPali (zero-shot)…")
def load_colpali_zs(index_dir: str = "data/indices/colpali_zs"):
    try:
        from cxr_intel.retrieval.colpali_index import ColPaliRetriever

        r = ColPaliRetriever(name="colpali_zs")
        if Path(index_dir).exists():
            r.load(index_dir)
        return r
    except Exception as e:
        st.warning(f"ColPali zero-shot not available: {e}")
        return None


@st.cache_resource(show_spinner="Loading ColPali-LoRA…")
def load_colpali_lora(
    lora_path: str = "models/colpali-cxr-lora/final",
    index_dir: str = "data/indices/colpali_lora",
):
    try:
        from cxr_intel.retrieval.colpali_index import ColPaliRetriever

        r = ColPaliRetriever(name="colpali_lora", lora_path=lora_path)
        if Path(index_dir).exists():
            r.load(index_dir)
        return r
    except Exception as e:
        st.warning(f"ColPali-LoRA not available: {e}")
        return None


@st.cache_resource(show_spinner="Loading BiomedCLIP…")
def load_biomedclip(index_dir: str = "data/indices/biomedclip"):
    try:
        from cxr_intel.retrieval.biomedclip_index import BiomedCLIPRetriever

        r = BiomedCLIPRetriever()
        if Path(index_dir).exists():
            r.load(index_dir)
        return r
    except Exception as e:
        st.warning(f"BiomedCLIP not available: {e}")
        return None


@st.cache_resource
def load_llm():
    try:
        from cxr_intel.models.llm_router import LLMRouter

        return LLMRouter()
    except Exception as e:
        st.warning(f"LLM router not available: {e}")
        return None


def get_retriever(name: str):
    return {
        "ColPali-LoRA": load_colpali_lora(),
        "ColPali-zero-shot": load_colpali_zs(),
        "BiomedCLIP": load_biomedclip(),
        "None (Pure VLM)": None,
    }.get(name)
