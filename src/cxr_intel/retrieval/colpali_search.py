"""Convenience helpers around ColPali retrieval (heatmap extraction, batched eval)."""
from __future__ import annotations

from typing import Sequence

import numpy as np
from PIL import Image


def extract_query_doc_heatmap(model, processor, query_image: Image.Image,
                              doc_image: Image.Image) -> np.ndarray:
    """Compute a (H_p, W_p) score grid over doc patches for a single (query, doc) pair.

    Used by the Streamlit heatmap component to highlight which doc regions
    drove the MaxSim retrieval score.
    """
    import torch

    q_inputs = processor.process_images([query_image]).to(model.device)
    d_inputs = processor.process_images([doc_image]).to(model.device)
    with torch.no_grad():
        q = model(**q_inputs).cpu().to(torch.float32)[0]   # (Lq, D)
        d = model(**d_inputs).cpu().to(torch.float32)[0]   # (Lp, D)
    sim = torch.einsum("qd,ld->ql", q, d)  # (Lq, Lp)
    per_patch = sim.max(dim=0).values.numpy()  # (Lp,)
    side = int(np.sqrt(per_patch.shape[0]))
    if side * side != per_patch.shape[0]:
        side = int(np.ceil(np.sqrt(per_patch.shape[0])))
        padded = np.zeros(side * side, dtype=per_patch.dtype)
        padded[: per_patch.shape[0]] = per_patch
        per_patch = padded
    return per_patch.reshape(side, side)


def batched_search(retriever, queries: Sequence[Image.Image], k: int = 5) -> list[list]:
    return [retriever.search_image(q, k=k) for q in queries]
