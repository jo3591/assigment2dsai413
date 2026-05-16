"""ColPali multi-vector indexing + MaxSim search."""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

import numpy as np
from PIL import Image

from cxr_intel.retrieval.base import Retriever, RetrievalHit
from cxr_intel.utils.io import ensure_dir, load_json, save_json
from cxr_intel.utils.logging import get_logger

log = get_logger(__name__)


@dataclass
class ColPaliRetriever:
    """Direct colpali-engine implementation. Stores per-image multi-vector tensors."""

    name: str = "colpali"
    checkpoint: str = "vidore/colpali-v1.3"
    lora_path: str | None = None
    torch_dtype: str = "bfloat16"
    device_map: str = "auto"
    image_max_side: int = 448
    batch_size: int = 2
    metadata: list[dict] = field(default_factory=list)
    _model = None
    _processor = None
    _doc_embeddings = None

    def _ensure_model(self) -> None:
        if self._model is not None:
            return
        import torch
        from colpali_engine.models import ColPali, ColPaliProcessor

        dtype = getattr(torch, self.torch_dtype)
        log.info("Loading ColPali %s (dtype=%s)", self.checkpoint, self.torch_dtype)
        # device_map="auto" can hit a "meta device" bug with multi-GPU + accelerate
        # when the model fits on one card. Force single GPU when CUDA is available.
        effective_device_map = self.device_map
        if effective_device_map == "auto" and torch.cuda.is_available():
            effective_device_map = {"": "cuda:0"}
        self._model = ColPali.from_pretrained(
            self.checkpoint, torch_dtype=dtype, device_map=effective_device_map
        ).eval()
        self._processor = ColPaliProcessor.from_pretrained(self.checkpoint)

        if self.lora_path:
            from peft import PeftModel

            log.info("Applying LoRA adapter from %s", self.lora_path)
            self._model = PeftModel.from_pretrained(self._model, self.lora_path)
            self._model = self._model.merge_and_unload()

    def _encode_images(self, images: Sequence[Image.Image]):
        import torch
        from tqdm.auto import tqdm

        self._ensure_model()
        out = []
        n_batches = (len(images) + self.batch_size - 1) // self.batch_size
        for i in tqdm(range(0, len(images), self.batch_size),
                      total=n_batches, desc="ColPali encode"):
            batch = images[i : i + self.batch_size]
            inputs = self._processor.process_images(batch).to(self._model.device)
            with torch.no_grad():
                emb = self._model(**inputs)
            out.append(emb.cpu().to(torch.float32))
        return torch.cat(out, dim=0)  # (N, num_patches, dim)

    def _encode_queries(self, queries: Sequence[str]):
        import torch

        self._ensure_model()
        inputs = self._processor.process_queries(list(queries)).to(self._model.device)
        with torch.no_grad():
            emb = self._model(**inputs)
        return emb.cpu().to(torch.float32)

    def index(self, items: Sequence[dict], out_dir: str | Path) -> None:
        """Items: [{study_id, image_path, report_text}]"""
        out = ensure_dir(out_dir)
        log.info("Indexing %d images with ColPali -> %s", len(items), out)
        images = [Image.open(it["image_path"]).convert("RGB") for it in items]
        embeddings = self._encode_images(images)
        self._doc_embeddings = embeddings
        self.metadata = list(items)
        np.save(out / "doc_embeddings.npy", embeddings.numpy())
        save_json([dict(it) for it in items], out / "metadata.json")
        log.info("Saved %d docs to %s", len(items), out)

    def load(self, index_dir: str | Path) -> None:
        import torch

        index_dir = Path(index_dir)
        emb = np.load(index_dir / "doc_embeddings.npy")
        self._doc_embeddings = torch.from_numpy(emb)
        self.metadata = load_json(index_dir / "metadata.json")
        log.info("Loaded ColPali index: %d docs from %s", len(self.metadata), index_dir)

    def _maxsim(self, query_emb, doc_emb) -> np.ndarray:
        """ColBERT-style late interaction scoring. q: (Lq, D), docs: (N, Lp, D)."""
        import torch

        # einsum over query tokens × doc patches
        sim = torch.einsum("qd,nld->nql", query_emb, doc_emb)  # (N, Lq, Lp)
        max_per_qtok = sim.max(dim=-1).values  # (N, Lq)
        scores = max_per_qtok.sum(dim=-1)  # (N,)
        return scores.cpu().numpy()

    def search_image(self, query: Image.Image, k: int = 5) -> list[RetrievalHit]:
        if self._doc_embeddings is None:
            raise RuntimeError("Index not loaded. Call .index() or .load() first.")
        q_emb = self._encode_images([query])  # (1, Lp, D)
        # For image-as-query MaxSim, treat the query patches as tokens
        scores = self._maxsim(q_emb[0], self._doc_embeddings)
        return self._top_k(scores, k)

    def search_text(self, query: str, k: int = 5) -> list[RetrievalHit]:
        if self._doc_embeddings is None:
            raise RuntimeError("Index not loaded.")
        q_emb = self._encode_queries([query])  # (1, Lq, D)
        scores = self._maxsim(q_emb[0], self._doc_embeddings)
        return self._top_k(scores, k)

    def _top_k(self, scores: np.ndarray, k: int) -> list[RetrievalHit]:
        idx = np.argsort(-scores)[:k]
        return [
            RetrievalHit(
                study_id=str(self.metadata[i]["study_id"]),
                image_path=str(self.metadata[i]["image_path"]),
                report_text=str(self.metadata[i].get("report_text", "")),
                score=float(scores[i]),
            )
            for i in idx
        ]

