"""Page 2 — QA Mode."""
from __future__ import annotations

import streamlit as st
from PIL import Image

from cxr_intel.app.cache import get_retriever, load_llm, load_medgemma
from cxr_intel.app.components.heatmap import render_heatmap
from cxr_intel.generation.qa_pipeline import QAPipeline

st.title("QA Mode")
st.caption("Ask a clinical question about the chest X-ray.")

with st.sidebar:
    retriever_name = st.radio(
        "Retriever",
        ["ColPali-LoRA", "ColPali-zero-shot", "BiomedCLIP", "None (Pure VLM)"],
        index=0,
    )
    generator_name = st.radio("Generator", ["MedGemma", "OpenRouter LLM (text-only)"], index=0)
    top_k = st.slider("Top-K retrievals", 0, 5, 3)
    show_heatmap = st.checkbox("Show ColPali heatmap", value=True)

uploaded = st.file_uploader("Chest X-ray", type=["png", "jpg", "jpeg"])
question = st.text_input("Question",
                         placeholder="e.g., Is there pleural effusion in this radiograph?")
qtype = st.selectbox(
    "Question type (optional)",
    ["", "existence", "location", "severity", "attribute", "open"],
    index=0,
)

if not (uploaded and question):
    st.info("Upload a CXR and enter a question to begin.")
    st.stop()

image = Image.open(uploaded).convert("RGB")
col_a, col_b = st.columns([1, 2])
col_a.image(image, caption="Input radiograph", use_column_width=True)

if not col_b.button("Answer", type="primary"):
    st.stop()

retriever = get_retriever(retriever_name)
medgemma = load_medgemma()
llm = load_llm()

if generator_name == "OpenRouter LLM (text-only)":
    config = "colpali_lora_text_llm"
elif retriever_name == "None (Pure VLM)":
    config = "medgemma_only"
elif retriever_name == "BiomedCLIP":
    config = "biomedclip_rag"
elif retriever_name == "ColPali-zero-shot":
    config = "colpali_zs_rag"
else:
    config = "colpali_lora_rag"

pipe = QAPipeline(config=config, retriever=retriever, medgemma=medgemma, llm=llm, top_k=top_k)
with col_b, st.spinner("Reasoning…"):
    out = pipe.run(image, question)

col_b.subheader("Answer")
col_b.write(out.answer)
col_b.caption(f"Latency: {out.latency_s:.1f}s · config: `{out.config}` · qtype={qtype or 'auto'}")

if out.retrieved:
    with col_b.expander(f"Evidence ({len(out.retrieved)})"):
        for i, hit in enumerate(out.retrieved, 1):
            st.markdown(f"**[{i}] study={hit.study_id} · score={hit.score:.3f}**")
            st.caption(hit.report_text[:400] + ("..." if len(hit.report_text) > 400 else ""))
            if show_heatmap and "colpali" in (retriever_name or "").lower() and retriever is not None:
                try:
                    render_heatmap(retriever, image, Image.open(hit.image_path).convert("RGB"))
                except Exception as e:
                    st.caption(f"(heatmap unavailable: {e})")
