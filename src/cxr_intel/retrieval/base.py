"""Common Retriever protocol so the rest of the system is backend-agnostic."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, Sequence, runtime_checkable

from PIL import Image


@dataclass(slots=True)
class RetrievalHit:
    study_id: str
    image_path: str
    report_text: str
    score: float


@runtime_checkable
class Retriever(Protocol):
    """A retrieval backend indexed over (image, report) pairs."""

    name: str

    def index(self, items: Sequence[dict], out_dir: str | Path) -> None: ...
    def load(self, index_dir: str | Path) -> None: ...
    def search_image(self, query: Image.Image, k: int = 5) -> list[RetrievalHit]: ...
    def search_text(self, query: str, k: int = 5) -> list[RetrievalHit]: ...
