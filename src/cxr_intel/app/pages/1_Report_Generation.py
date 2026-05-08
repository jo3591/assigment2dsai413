"""Page 1 — Report Generation."""
from __future__ import annotations

import streamlit as st
from PIL import Image

from cxr_intel.app.cache import (
    get_retriever,
    load_llm,
    load_medgemma,
)
from cxr_intel.generation.report_pipeline import ReportPipeline

st.title("Report Generation")
st.caption("Upload a chest X-ray. The system drafts FINDINGS + IMPRESSION.")

with st.sidebar:
    retriever_name = st.radio(
        "Retriever",
        ["ColPali-LoRA", "ColPali-zero-shot", "BiomedCLIP", "None (Pure VLM)"],
        index=0,
    )
    generator_name = st.radio("Generator", ["MedGemma", "OpenRouter LLM (text-only)"], index=0)
    top_k = st.slider("Top-K retrievals", 0, 5, 3)

uploaded = st.file_uploader("Chest X-ray (PNG/JPG)", type=["png", "jpg", "jpeg"])
if not uploaded:
    st.info("Upload a CXR to begin.")
    st.stop()

image = Image.open(uploaded).convert("RGB")
col_a, col_b = st.columns([1, 2])
col_a.image(image, caption="Input radiograph", use_column_width=True)

if not col_b.button("Generate report", type="primary"):
    st.stop()

retriever = get_retriever(retriever_name)
medgemma = load_medgemma()
llm = load_llm()

if generator_name == "OpenRouter LLM (text-only)":
    config = "colpali_lora_text_llm"
    if retriever is None:
        st.error("Text-only generator needs a retriever; pick ColPali-LoRA in the sidebar.")
        st.stop()
elif retriever_name == "None (Pure VLM)":
    config = "medgemma_only"
elif retriever_name == "BiomedCLIP":
    config = "biomedclip_rag"
elif retriever_name == "ColPali-zero-shot":
    config = "colpali_zs_rag"
else:
    config = "colpali_lora_rag"

if config != "colpali_lora_text_llm" and medgemma is None:
    st.error("MedGemma is not available. Check HF token and GPU availability.")
    st.stop()

pipe = ReportPipeline(
    config=config,
    retriever=retriever,
    medgemma=medgemma,
    llm=llm,
    top_k=top_k,
)

with col_b, st.spinner("Generating report…"):
    out = pipe.run(image)

col_b.subheader("Generated report")
col_b.write(out.report_text)
col_b.caption(f"Latency: {out.latency_s:.1f}s · config: `{out.config}` · top_k={top_k}")

if out.retrieved:
    with col_b.expander(f"Retrieved evidence ({len(out.retrieved)})"):
        for i, hit in enumerate(out.retrieved, 1):
            st.markdown(f"**[{i}] study={hit.study_id} · score={hit.score:.3f}**")
            st.caption(hit.report_text[:500] + ("..." if len(hit.report_text) > 500 else ""))
