"""Build retrieval indices (BiomedCLIP + ColPali zero-shot + ColPali LoRA)."""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

from cxr_intel.retrieval.biomedclip_index import BiomedCLIPRetriever
from cxr_intel.retrieval.colpali_index import ColPaliRetriever
from cxr_intel.utils.io import ensure_dir, load_json, load_yaml
from cxr_intel.utils.logging import get_logger

log = get_logger("build_indices")


def main() -> None:
    p = argparse.ArgumentParser(__doc__)
    p.add_argument("--config-data", default="configs/data.yaml")
    p.add_argument("--config-colpali", default="configs/colpali.yaml")
    p.add_argument("--config-biomedclip", default="configs/biomedclip.yaml")
    p.add_argument("--backend", action="append",
                   choices=["biomedclip", "colpali_zs", "colpali_lora"],
                   help="Which backends to build (repeatable). Default: all.")
    p.add_argument("--lora-path", default="models/colpali-cxr-lora/final")
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--split", default="all", choices=["train", "all"])
    args = p.parse_args()

    load_dotenv()
    backends = args.backend or ["biomedclip", "colpali_zs", "colpali_lora"]

    data_cfg = load_yaml(args.config_data)
    colpali_cfg = load_yaml(args.config_colpali)
    biomed_cfg = load_yaml(args.config_biomedclip)

    df = pd.read_parquet(data_cfg["paths"]["processed"])
    if args.split == "train":
        splits = load_json(data_cfg["paths"]["splits"])
        df = df[df["study_id"].astype(str).isin(splits["train"])].copy()
    if args.limit:
        df = df.head(args.limit)
    items = [
        {
            "study_id": str(r["study_id"]),
            "image_path": str(r["image_path"]),
            "report_text": (str(r.get("findings", "")) + " " + str(r.get("impression", ""))).strip()
            or str(r.get("clean_text", "")),
        }
        for _, r in df.iterrows()
    ]
    log.info("Indexing %d items across backends=%s", len(items), backends)

    if "biomedclip" in backends:
        out = ensure_dir(biomed_cfg["index"]["out_path"]).with_suffix("")  # parent dir
        out_dir = Path(biomed_cfg["index"]["out_path"]).parent if biomed_cfg["index"]["out_path"].endswith(".faiss") else out
        out_dir = Path("data/indices/biomedclip")
        ensure_dir(out_dir)
        r = BiomedCLIPRetriever(
            checkpoint=biomed_cfg["checkpoint"],
            image_size=biomed_cfg["image_size"],
            batch_size=biomed_cfg["index"]["batch_size"],
        )
        r.index(items, out_dir)

    if "colpali_zs" in backends:
        out_dir = Path("data/indices/colpali_zs")
        ensure_dir(out_dir)
        r = ColPaliRetriever(
            checkpoint=colpali_cfg["checkpoint"],
            torch_dtype=colpali_cfg["torch_dtype"],
            device_map=colpali_cfg["device_map"],
            batch_size=colpali_cfg["index"]["batch_size"],
            image_max_side=colpali_cfg["index"]["image_max_side"],
        )
        r.index(items, out_dir)

    if "colpali_lora" in backends:
        if not Path(args.lora_path).exists():
            log.warning("LoRA path %s not found — skipping ColPali LoRA index", args.lora_path)
        else:
            out_dir = Path("data/indices/colpali_lora")
            ensure_dir(out_dir)
            r = ColPaliRetriever(
                checkpoint=colpali_cfg["checkpoint"],
                lora_path=args.lora_path,
                torch_dtype=colpali_cfg["torch_dtype"],
                device_map=colpali_cfg["device_map"],
                batch_size=colpali_cfg["index"]["batch_size"],
                image_max_side=colpali_cfg["index"]["image_max_side"],
            )
            r.index(items, out_dir)


if __name__ == "__main__":
    main()
