"""Train the ColPali LoRA adapter on (image, report) contrastive pairs.

Run on Colab A100 (or local A6000+). Wall clock ≈ 3 hours for 3 epochs on 2400
training examples.

    python scripts/train_colpali_lora.py --epochs 3
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

from cxr_intel.finetune.pair_builder import build_contrastive_pairs
from cxr_intel.finetune.train_colpali_lora import TrainConfig, train
from cxr_intel.utils.io import load_json, load_yaml
from cxr_intel.utils.logging import get_logger

log = get_logger("train_colpali_lora")


def main() -> None:
    p = argparse.ArgumentParser(__doc__)
    p.add_argument("--config", default="configs/colpali.yaml")
    p.add_argument("--data-config", default="configs/data.yaml")
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--limit-train", type=int, default=None)
    args = p.parse_args()

    load_dotenv()
    cfg = load_yaml(args.config)
    data_cfg = load_yaml(args.data_config)

    df = pd.read_parquet(data_cfg["paths"]["processed"])
    splits = load_json(data_cfg["paths"]["splits"])
    train_df = df[df["study_id"].astype(str).isin(splits["train"])].copy()
    val_df = df[df["study_id"].astype(str).isin(splits["val"])].copy()
    if args.limit_train:
        train_df = train_df.head(args.limit_train)

    train_examples = build_contrastive_pairs(
        train_df, n_hard_negatives=cfg["train"]["hard_negatives_per_positive"]
    )
    val_examples = build_contrastive_pairs(val_df, n_hard_negatives=1)
    log.info("Pairs: train=%d val=%d", len(train_examples), len(val_examples))

    tcfg = TrainConfig(
        base_checkpoint=cfg["checkpoint"],
        output_dir=cfg["lora"]["output_dir"],
        epochs=args.epochs or cfg["train"]["epochs"],
        batch_size=args.batch_size or cfg["train"]["per_device_batch_size"],
        grad_accum=cfg["train"]["grad_accumulation"],
        lr=cfg["train"]["learning_rate"],
        warmup_ratio=cfg["train"]["warmup_ratio"],
        weight_decay=cfg["train"]["weight_decay"],
        log_every=cfg["train"]["logging_steps"],
        eval_every=cfg["train"]["eval_steps"],
        save_every=cfg["train"]["save_steps"],
        bf16=cfg["train"]["bf16"],
        gradient_checkpointing=cfg["train"]["gradient_checkpointing"],
        lora_r=cfg["lora"]["r"],
        lora_alpha=cfg["lora"]["alpha"],
        lora_dropout=cfg["lora"]["dropout"],
        target_modules=tuple(cfg["lora"]["target_modules"]),
    )
    final_path = train(train_examples, val_examples=val_examples, cfg=tcfg)
    log.info("LoRA adapter saved to %s", final_path)


if __name__ == "__main__":
    main()
