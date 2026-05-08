"""MedGemma 4B-IT wrapper for image+text generation."""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any

from PIL import Image

from cxr_intel.utils.logging import get_logger

log = get_logger(__name__)


@dataclass
class MedGemmaRunner:
    checkpoint: str = "google/medgemma-4b-it"
    torch_dtype: str = "bfloat16"
    device_map: str = "auto"
    quantization: str | None = None     # "int8" | "int4" | None

    _model: Any = None
    _processor: Any = None

    def load(self) -> None:
        if self._model is not None:
            return
        import torch
        from transformers import AutoModelForImageTextToText, AutoProcessor

        token = os.getenv("HF_TOKEN")
        dtype = getattr(torch, self.torch_dtype)
        kwargs: dict[str, Any] = dict(torch_dtype=dtype, device_map=self.device_map)
        if self.quantization == "int8":
            from transformers import BitsAndBytesConfig

            kwargs["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)
        elif self.quantization == "int4":
            from transformers import BitsAndBytesConfig

            kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=dtype,
                bnb_4bit_quant_type="nf4",
            )

        log.info("Loading MedGemma %s (dtype=%s, quant=%s)",
                 self.checkpoint, self.torch_dtype, self.quantization)
        self._model = AutoModelForImageTextToText.from_pretrained(
            self.checkpoint, token=token, **kwargs
        )
        self._processor = AutoProcessor.from_pretrained(self.checkpoint, token=token)

    def generate(
        self,
        images: list[Image.Image],
        system_prompt: str,
        user_text: str,
        max_new_tokens: int = 512,
        do_sample: bool = False,
        temperature: float = 0.0,
    ) -> str:
        import torch

        self.load()
        content_user = [{"type": "image", "image": img} for img in images]
        content_user.append({"type": "text", "text": user_text})
        messages = [
            {"role": "system", "content": [{"type": "text", "text": system_prompt}]},
            {"role": "user", "content": content_user},
        ]
        inputs = self._processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        ).to(self._model.device)

        gen_kwargs: dict[str, Any] = dict(
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
        )
        if do_sample:
            gen_kwargs["temperature"] = temperature

        with torch.inference_mode():
            output_ids = self._model.generate(**inputs, **gen_kwargs)

        prompt_len = inputs["input_ids"].shape[-1]
        new_tokens = output_ids[0][prompt_len:]
        return self._processor.decode(new_tokens, skip_special_tokens=True).strip()
