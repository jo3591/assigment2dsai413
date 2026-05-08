"""Collator that produces (image_inputs, query_inputs) batches for ColPali training."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from PIL import Image


@dataclass
class ColPaliCollator:
    processor: Any                  # colpali_engine.models.ColPaliProcessor
    image_max_side: int = 448

    def __call__(self, batch: list[dict[str, Any]]) -> dict[str, Any]:
        images = [Image.open(b["image_path"]).convert("RGB") for b in batch]
        positives = [b["positive_text"] for b in batch]
        negatives_flat: list[str] = []
        for b in batch:
            negatives_flat.extend(b["negative_texts"])

        image_inputs = self.processor.process_images(images)
        pos_inputs = self.processor.process_queries(positives)
        neg_inputs = self.processor.process_queries(negatives_flat)
        return {
            "image_inputs": image_inputs,
            "pos_query_inputs": pos_inputs,
            "neg_query_inputs": neg_inputs,
            "batch_size": len(batch),
            "n_negatives_per_pos": len(negatives_flat) // len(batch) if batch else 0,
        }
