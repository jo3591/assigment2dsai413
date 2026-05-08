"""Page 3 — three-way comparison."""
from __future__ import annotations

import streamlit as st
from PIL import Image

from cxr_intel.app.cache import (
    load_biomedclip,
    load_colpali_lora,
    load_colpali_zs,
    load_llm,
    load_medgemma,
)
from cxr_intel.generation.qa_pipeline import QAPipeline
from cxr_intel.generation.report_pipeline import ReportPipeline

st.title("Compare Models")
st.caption("Run the same input through three configurations side-by-side.")

mode = st.radio("Mode", ["Report Generation", "QA"], horizontal=True)
uploaded = st.file_uploader("Chest X-ray", type=["png", "jpg", "jpeg"])
question = st.text_input("Question (QA mode only)")
top_k = st.slider("Top-K retrievals", 0, 5, 3)

if not uploaded:
    st.info("Upload a CXR to begin.")
    st.stop()
if mode == "QA" and not question:
    st.info("Enter a question to begin.")
    st.stop()

image = Image.open(uploaded).convert("RGB")
st.image(image, width=320)

medgemma = load_medgemma()
llm = load_llm()
configs = [
    ("MedGemma-only", None, "medgemma_only"),
    ("ColPali-zs RAG", load_colpali_zs(), "colpali_zs_rag"),
    ("ColPali-LoRA RAG", load_colpali_lora(), "colpali_lora_rag"),
]

cols = st.columns(len(configs))
for col, (name, retriever, cfg) in zip(cols, configs):
    with col, st.spinner(f"{name}…"):
        if mode == "Report Generation":
            pipe = ReportPipeline(config=cfg, retriever=retriever, medgemma=medgemma,
                                  llm=llm, top_k=top_k)
            out = pipe.run(image)
            col.subheader(name)
            col.write(out.report_text)
            col.caption(f"{out.latency_s:.1f}s · {len(out.retrieved)} hits")
        else:
            pipe = QAPipeline(config=cfg, retriever=retriever, medgemma=medgemma,
                              llm=llm, top_k=top_k)
            out = pipe.run(image, question)
            col.subheader(name)
            col.write(out.answer)
            col.caption(f"{out.latency_s:.1f}s · {len(out.retrieved)} hits")
