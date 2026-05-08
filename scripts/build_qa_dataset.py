"""Generate + validate the synthetic QA dataset from preprocessed reports.

Usage:
    python scripts/build_qa_dataset.py --config configs/data.yaml --limit 100
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from tqdm.auto import tqdm

from cxr_intel.models.llm_router import LLMRouter
from cxr_intel.qa_dataset.schema import QAPair
from cxr_intel.qa_dataset.synth_generator import SynthGenerator
from cxr_intel.qa_dataset.validator import QAValidator
from cxr_intel.utils.io import ensure_dir, load_json, load_yaml, save_jsonl
from cxr_intel.utils.logging import get_logger

log = get_logger("build_qa_dataset")


def main() -> None:
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--config", default="configs/data.yaml")
    parser.add_argument("--processed", default=None,
                        help="Override processed parquet path")
    parser.add_argument("--splits", default=None, help="Override splits json path")
    parser.add_argument("--out", default="data/qa/qa_v1.jsonl")
    parser.add_argument("--cache-dir", default="data/qa/cache")
    parser.add_argument("--limit", type=int, default=None,
                        help="Limit reports for smoke test")
    parser.add_argument("--no-validate", action="store_true")
    parser.add_argument("--judge-model", default=None,
                        help="LLM model id for judging (default = QA_JUDGE_MODEL env)")
    parser.add_argument("--synth-model", default=None,
                        help="LLM model id for synthesis (default = QA_SYNTH_MODEL env)")
    args = parser.parse_args()

    load_dotenv()
    cfg = load_yaml(args.config)
    parquet = args.processed or cfg["paths"]["processed"]
    splits = load_json(args.splits or cfg["paths"]["splits"])

    df = pd.read_parquet(parquet)
    train_ids = set(splits["train"])
    test_ids = set(splits["test"])

    train_df = df[df["study_id"].astype(str).isin(train_ids)].copy()
    test_df = df[df["study_id"].astype(str).isin(test_ids)].copy()
    if args.limit:
        train_df = train_df.head(args.limit)
        test_df = test_df.head(max(10, args.limit // 4))

    synth_llm = LLMRouter(model=args.synth_model) if args.synth_model else LLMRouter()
    judge_llm = (
        LLMRouter(model=args.judge_model) if args.judge_model else None
    ) if not args.no_validate else None

    gen = SynthGenerator(
        llm=synth_llm,
        cache_dir=Path(args.cache_dir),
        questions_per_report=4,
    )
    validator = QAValidator(judge=judge_llm)

    out_train = Path(args.out)
    out_test = out_train.with_name("qa_test.jsonl")
    ensure_dir(out_train.parent)

    def run_one(df_part: pd.DataFrame, out_path: Path) -> None:
        rows = df_part.to_dict(orient="records")
        all_pairs: list[QAPair] = []
        for row in tqdm(rows, desc=f"synth->{out_path.name}"):
            pairs = gen.generate_for_report(
                study_id=str(row["study_id"]),
                image_path=str(row["image_path"]),
                report_text=str(row.get("clean_text", row.get("text", ""))),
            )
            if not args.no_validate:
                pairs = [
                    qa for qa in pairs
                    if validator.validate(qa, row.get("clean_text", row.get("text", ""))) is not None
                ]
            all_pairs.extend(pairs)
        all_pairs = validator.dedupe(all_pairs) if not args.no_validate else all_pairs
        n = save_jsonl((p.dict() for p in all_pairs), out_path)
        log.info("Wrote %d QA pairs -> %s", n, out_path)

    run_one(train_df, out_train)
    run_one(test_df, out_test)
    log.info("Done. Rejected reasons summary: %d total", len(validator.rejected))


if __name__ == "__main__":
    main()
