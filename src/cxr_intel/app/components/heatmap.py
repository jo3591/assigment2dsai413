"""ColPali MaxSim heatmap component for the Streamlit app."""
from __future__ import annotations

import streamlit as st
from PIL import Image

from cxr_intel.retrieval.colpali_search import extract_query_doc_heatmap
from cxr_intel.utils.viz import overlay_heatmap


def render_heatmap(retriever, query_image: Image.Image, doc_image: Image.Image) -> None:
    """Render an alpha-blended heatmap on the doc image showing which patches
    drove the retrieval score for the query."""
    retriever._ensure_model()
    grid = extract_query_doc_heatmap(
        retriever._model, retriever._processor, query_image, doc_image
    )
    blended = overlay_heatmap(doc_image, grid, alpha=0.45)
    st.image(blended, caption="ColPali MaxSim heatmap (red = high relevance)",
             use_column_width=True)
