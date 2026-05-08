"""BiomedCLIP FAISS-IP retrieval baseline. Single-vector image embeddings."""
from __future__ import annotations

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
class BiomedCLIPRetriever:
    name: str = "biomedclip"
    checkpoint: str = "hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224"
    image_size: int = 224
    batch_size: int = 32
    metadata: list[dict] = field(default_factory=list)
    _model = None
    _preprocess = None
    _tokenizer = None
    _index = None

    def _ensure_model(self) -> None:
        if self._model is not None:
            return
        import open_clip
        import torch

        log.info("Loading BiomedCLIP %s", self.checkpoint)
        self._model, _, self._preprocess = open_clip.create_model_and_transforms(self.checkpoint)
        self._tokenizer = open_clip.get_tokenizer(self.checkpoint)
        self._model.eval()
        if torch.cuda.is_available():
            self._model = self._model.cuda()

    def _encode_images(self, images: Sequence[Image.Image]) -> np.ndarray:
        import torch

        self._ensure_model()
        out = []
        for i in range(0, len(images), self.batch_size):
            batch = images[i : i + self.batch_size]
            tensors = torch.stack([self._preprocess(img) for img in batch])
            if torch.cuda.is_available():
                tensors = tensors.cuda()
            with torch.no_grad():
                feats = self._model.encode_image(tensors)
                feats = feats / feats.norm(dim=-1, keepdim=True)
            out.append(feats.cpu().numpy().astype(np.float32))
        return np.concatenate(out, axis=0)

    def _encode_text(self, texts: Sequence[str]) -> np.ndarray:
        import torch

        self._ensure_model()
        tokens = self._tokenizer(list(texts))
        if torch.cuda.is_available():
            tokens = tokens.cuda()
        with torch.no_grad():
            feats = self._model.encode_text(tokens)
            feats = feats / feats.norm(dim=-1, keepdim=True)
        return feats.cpu().numpy().astype(np.float32)

    def index(self, items: Sequence[dict], out_dir: str | Path) -> None:
        import faiss

        out = ensure_dir(out_dir)
        log.info("Indexing %d images with BiomedCLIP", len(items))
        images = [Image.open(it["image_path"]).convert("RGB") for it in items]
        feats = self._encode_images(images)
        d = feats.shape[1]
        idx = faiss.IndexFlatIP(d)
        idx.add(feats)
        faiss.write_index(idx, str(out / "index.faiss"))
        np.save(out / "embeddings.npy", feats)
        self.metadata = list(items)
        save_json([dict(it) for it in items], out / "metadata.json")
        self._index = idx
        log.info("Saved BiomedCLIP index: dim=%d, n=%d -> %s", d, len(items), out)

    def load(self, index_dir: str | Path) -> None:
        import faiss

        index_dir = Path(index_dir)
        self._index = faiss.read_index(str(index_dir / "index.faiss"))
        self.metadata = load_json(index_dir / "metadata.json")
        log.info("Loaded BiomedCLIP index: %d docs", len(self.metadata))

    def search_image(self, query: Image.Image, k: int = 5) -> list[RetrievalHit]:
        if self._index is None:
            raise RuntimeError("Index not loaded.")
        q = self._encode_images([query])
        scores, ids = self._index.search(q, k)
        return self._format(scores[0], ids[0])

    def search_text(self, query: str, k: int = 5) -> list[RetrievalHit]:
        if self._index is None:
            raise RuntimeError("Index not loaded.")
        q = self._encode_text([query])
        scores, ids = self._index.search(q, k)
        return self._format(scores[0], ids[0])

    def _format(self, scores: np.ndarray, ids: np.ndarray) -> list[RetrievalHit]:
        out: list[RetrievalHit] = []
        for s, i in zip(scores, ids):
            if i < 0 or i >= len(self.metadata):
                continue
            md = self.metadata[i]
            out.append(
                RetrievalHit(
                    study_id=str(md["study_id"]),
                    image_path=str(md["image_path"]),
                    report_text=str(md.get("report_text", "")),
                    score=float(s),
                )
            )
        return out
