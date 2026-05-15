"""LoRA fine-tune ColPali on (CXR image, report) contrastive pairs.

Designed for Colab A100. Run via `notebooks/03_colpali_lora_train.ipynb` or
`scripts/train_colpali_lora.py`.

Loss: late-interaction InfoNCE (ColbertLoss) — for each (image, pos_query) pair
we compute MaxSim score; logits are formed with N hard negatives in the batch
plus in-batch negatives.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm

from cxr_intel.finetune.collator import ColPaliCollator
from cxr_intel.finetune.pair_builder import ContrastiveExample
from cxr_intel.utils.io import ensure_dir, save_json
from cxr_intel.utils.logging import get_logger

log = get_logger(__name__)


class ContrastiveDataset(Dataset):
    def __init__(self, examples: list[ContrastiveExample]):
        self.items = [
            {
                "image_path": e.image_path,
                "positive_text": e.positive_text,
                "negative_texts": e.negative_texts,
                "study_id": e.study_id,
            }
            for e in examples
        ]

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, i: int) -> dict[str, Any]:
        return self.items[i]


@dataclass
class TrainConfig:
    base_checkpoint: str = "vidore/colpali-v1.3"
    output_dir: str = "models/colpali-cxr-lora"
    epochs: int = 3
    batch_size: int = 4
    grad_accum: int = 4
    lr: float = 5e-5
    warmup_ratio: float = 0.1
    weight_decay: float = 0.01
    log_every: int = 25
    eval_every: int = 250
    save_every: int = 250
    bf16: bool = True
    gradient_checkpointing: bool = True
    lora_r: int = 32
    lora_alpha: int = 32
    lora_dropout: float = 0.1
    target_modules: tuple[str, ...] = ("q_proj", "k_proj", "v_proj", "o_proj")


def maxsim(query_emb: torch.Tensor, doc_emb: torch.Tensor) -> torch.Tensor:
    """ColBERT MaxSim. q: (B, Lq, D), d: (B, Lp, D) → scores: (B,)."""
    sim = torch.einsum("bqd,bld->bql", query_emb, doc_emb)
    return sim.max(dim=-1).values.sum(dim=-1)


def cross_maxsim(query_emb: torch.Tensor, doc_emb: torch.Tensor) -> torch.Tensor:
    """All-pairs MaxSim. q: (Bq, Lq, D), d: (Bd, Lp, D) → (Bq, Bd)."""
    sim = torch.einsum("aqd,bld->abql", query_emb, doc_emb)
    return sim.max(dim=-1).values.sum(dim=-1)


def _select_dtype(prefer_bf16: bool) -> torch.dtype:
    """Pick the best dtype the current GPU actually supports.

    - Ampere+ (A100, A6000, RTX 30/40-series, L4): bf16 if requested.
    - Pascal/Turing/Volta (P100, T4, V100): fall back to fp16; bf16 silently
      converts to fp32 on these GPUs which kills training speed and OOMs.
    """
    if not torch.cuda.is_available():
        return torch.float32
    if prefer_bf16 and torch.cuda.is_bf16_supported():
        log.info("bf16 supported on this GPU — using bfloat16")
        return torch.bfloat16
    log.info("Using float16 (bf16 unavailable or disabled)")
    return torch.float16


def train(
    examples: list[ContrastiveExample],
    val_examples: list[ContrastiveExample] | None = None,
    cfg: TrainConfig = TrainConfig(),
) -> str:
    from colpali_engine.models import ColPali, ColPaliProcessor
    from peft import LoraConfig, get_peft_model
    from transformers import get_cosine_schedule_with_warmup

    ensure_dir(cfg.output_dir)
    log.info("Loading base ColPali %s", cfg.base_checkpoint)
    dtype = _select_dtype(cfg.bf16)
    model = ColPali.from_pretrained(cfg.base_checkpoint, torch_dtype=dtype, device_map="auto")
    if cfg.gradient_checkpointing:
        model.gradient_checkpointing_enable()
    processor = ColPaliProcessor.from_pretrained(cfg.base_checkpoint)

    lora_cfg = LoraConfig(
        r=cfg.lora_r,
        lora_alpha=cfg.lora_alpha,
        lora_dropout=cfg.lora_dropout,
        target_modules=list(cfg.target_modules),
        bias="none",
        task_type="FEATURE_EXTRACTION",
    )
    model = get_peft_model(model, lora_cfg)
    model.print_trainable_parameters()

    train_ds = ContrastiveDataset(examples)
    collator = ColPaliCollator(processor=processor)
    loader = DataLoader(
        train_ds, batch_size=cfg.batch_size, shuffle=True, collate_fn=collator, num_workers=0
    )

    optim = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=cfg.lr,
        weight_decay=cfg.weight_decay,
    )
    total_steps = (len(loader) * cfg.epochs) // cfg.grad_accum
    scheduler = get_cosine_schedule_with_warmup(
        optim, num_warmup_steps=int(cfg.warmup_ratio * total_steps), num_training_steps=total_steps
    )

    step = 0
    history: list[dict[str, Any]] = []
    model.train()

    for epoch in range(cfg.epochs):
        pbar = tqdm(loader, desc=f"epoch {epoch+1}/{cfg.epochs}")
        for batch in pbar:
            img_in = {k: v.to(model.device) for k, v in batch["image_inputs"].items()}
            pos_in = {k: v.to(model.device) for k, v in batch["pos_query_inputs"].items()}
            neg_in = {k: v.to(model.device) for k, v in batch["neg_query_inputs"].items()}

            doc_emb = model(**img_in)             # (B, Lp, D)
            pos_emb = model(**pos_in)             # (B, Lq, D)
            neg_emb = model(**neg_in)             # (B*N, Lq, D)

            # logits: positive on diagonal, in-batch negatives, plus hard negatives
            B = doc_emb.shape[0]
            n_neg = batch["n_negatives_per_pos"]
            pos_logits = cross_maxsim(pos_emb, doc_emb)        # (B, B) - diag is positive
            # Score hard negatives against their corresponding doc only
            neg_emb_reshaped = neg_emb.view(B, n_neg, *neg_emb.shape[1:])
            neg_logits_per_doc = torch.stack(
                [
                    maxsim(neg_emb_reshaped[i], doc_emb[i].unsqueeze(0).expand(n_neg, -1, -1))
                    for i in range(B)
                ],
                dim=0,
            )  # (B, n_neg)

            logits = torch.cat([pos_logits, neg_logits_per_doc], dim=1)  # (B, B+n_neg)
            labels = torch.arange(B, device=logits.device)
            loss = torch.nn.functional.cross_entropy(logits, labels) / cfg.grad_accum
            loss.backward()

            if (step + 1) % cfg.grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optim.step()
                scheduler.step()
                optim.zero_grad(set_to_none=True)

            if step % cfg.log_every == 0:
                pbar.set_postfix(loss=float(loss.detach() * cfg.grad_accum))
                history.append({"step": step, "loss": float(loss.detach() * cfg.grad_accum)})

            if val_examples and step > 0 and step % cfg.eval_every == 0:
                m = quick_recall_at_k(model, processor, val_examples, k=5, max_eval=64)
                log.info("step=%d val_recall@5=%.3f", step, m)
                history.append({"step": step, "val_recall@5": m})

            if step > 0 and step % cfg.save_every == 0:
                save_path = Path(cfg.output_dir) / f"step-{step}"
                model.save_pretrained(save_path)
                log.info("Saved adapter to %s", save_path)

            step += 1

    final_path = Path(cfg.output_dir) / "final"
    model.save_pretrained(final_path)
    save_json(history, Path(cfg.output_dir) / "history.json")
    log.info("Training complete. Adapter at %s", final_path)
    return str(final_path)


def quick_recall_at_k(model, processor, examples: list[ContrastiveExample],
                      k: int = 5, max_eval: int = 64) -> float:
    """Tiny sanity-check Recall@k on a subset of val examples."""
    model.eval()
    sample = examples[:max_eval]
    from PIL import Image

    images = [Image.open(e.image_path).convert("RGB") for e in sample]
    queries = [e.positive_text for e in sample]
    with torch.no_grad():
        img_in = {k_: v.to(model.device) for k_, v in processor.process_images(images).items()}
        q_in = {k_: v.to(model.device) for k_, v in processor.process_queries(queries).items()}
        d_emb = model(**img_in)
        q_emb = model(**q_in)
        scores = cross_maxsim(q_emb, d_emb)  # (N, N), diag = correct
    topk = scores.topk(k, dim=1).indices.cpu().numpy()
    correct = sum(1 for i, idx in enumerate(topk) if i in idx)
    model.train()
    return correct / len(sample)
