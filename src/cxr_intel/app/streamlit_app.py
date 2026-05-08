"""Streamlit launcher for the dual-mode CXR Intelligence System.

Run:
    streamlit run src/cxr_intel/app/streamlit_app.py

Pages live under src/cxr_intel/app/pages/ — Streamlit auto-discovers them.
"""
from __future__ import annotations

import streamlit as st

st.set_page_config(
    page_title="CXR Intelligence — DSAI 413",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("Multi-Modal Chest X-Ray Intelligence System")
st.caption("DSAI 413 Assignment 2 — Report Generation + RAG QA")

st.markdown(
    """
This demo runs two modes side-by-side:

- **Report Generation**: upload a chest X-ray and the system drafts FINDINGS + IMPRESSION.
- **QA Mode**: ask a clinical question about the X-ray and get a grounded answer.

Both modes can be backed by:

| Retriever | Generator |
|---|---|
| None (Pure VLM) | MedGemma-4B-IT |
| BiomedCLIP | MedGemma-4B-IT |
| ColPali zero-shot | MedGemma-4B-IT |
| ColPali LoRA-tuned | MedGemma-4B-IT |
| ColPali LoRA-tuned | OpenRouter LLM (text-only ablation) |

Use the sidebar to navigate. The **Compare** page runs all configurations on the
same input for direct visual comparison.
"""
)

st.info(
    "First run will download MedGemma (~10 GB) and ColPali (~4 GB) and may take "
    "several minutes. Pre-built retrieval indices are expected under `data/indices/`. "
    "See README.md for setup."
)

with st.sidebar:
    st.header("Status")
    from cxr_intel.app.cache import (
        load_biomedclip,
        load_colpali_lora,
        load_colpali_zs,
        load_llm,
        load_medgemma,
    )

    st.write("MedGemma:", "✅" if load_medgemma() else "⚠️")
    st.write("ColPali (zs):", "✅" if load_colpali_zs() else "⚠️")
    st.write("ColPali (LoRA):", "✅" if load_colpali_lora() else "⚠️")
    st.write("BiomedCLIP:", "✅" if load_biomedclip() else "⚠️")
    st.write("LLM router:", "✅" if load_llm() else "⚠️")
