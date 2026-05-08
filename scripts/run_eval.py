"""Run the full evaluation matrix and write results to results/tables/.

Usage:
    python scripts/run_eval.py --mode report --mode qa --mode retrieval --configs all
"""
from __future__ import annotations

import argparse
import csv
import dataclasses as dc
import json
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from PIL import Image
from tqdm.auto import tqdm

from cxr_intel.eval.metrics_qa import score_qa
from cxr_intel.eval.metrics_report import score_report
from cxr_intel.eval.metrics_retrieval import score_retrieval
from cxr_intel.eval.llm_judge import LLMJudge
from cxr_intel.generation.qa_pipeline import QAPipeline
from cxr_intel.generation.report_pipeline import ReportPipeline
from cxr_intel.models.llm_router import LLMRouter
from cxr_intel.models.medgemma_runner import MedGemmaRunner
from cxr_intel.retrieval.biomedclip_index import BiomedCLIPRetriever
from cxr_intel.retrieval.colpali_index import ColPaliRetriever
from cxr_intel.utils.io import ensure_dir, load_json, load_yaml, load_jsonl
from cxr_intel.utils.logging import get_logger

log = get_logger("run_eval")


def build_retriever(name: str, lora_path: str = "models/colpali-cxr-lora/final"):
    if name == "biomedclip":
        r = BiomedCLIPRetriever()
        r.load("data/indices/biomedclip")
        return r
    if name == "colpali_zs":
        r = ColPaliRetriever(name="colpali_zs")
        r.load("data/indices/colpali_zs")
        return r
    if name == "colpali_lora":
        r = ColPaliRetriever(name="colpali_lora", lora_path=lora_path)
        r.load("data/indices/colpali_lora")
        return r
    return None


CONFIG_RETRIEVERS = {
    "medgemma_only": None,
    "biomedclip_rag": "biomedclip",
    "colpali_zs_rag": "colpali_zs",
    "colpali_lora_rag": "colpali_lora",
    "colpali_lora_text_llm": "colpali_lora",
}


def write_table(rows: list[dict], path: str | Path) -> None:
    if not rows:
        return
    ensure_dir(Path(path).parent)
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    log.info("Wrote %s (%d rows)", path, len(rows))


def eval_report(args, df: pd.DataFrame, configs: list[str]) -> None:
    medgemma = MedGemmaRunner()
    medgemma.load()
    llm = LLMRouter() if "colpali_lora_text_llm" in configs else None
    rows = []
    test_df = df[df["study_id"].astype(str).isin(load_json("data/processed/splits.json")["test"])]
    test_df = test_df.head(args.test_size)
    refs = (test_df["findings"].fillna("") + " " + test_df["impression"].fillna("")).str.strip().tolist()

    predictions: dict[str, list[str]] = {}
    for cfg in configs:
        retriever_name = CONFIG_RETRIEVERS[cfg]
        retriever = build_retriever(retriever_name) if retriever_name else None
        pipe = ReportPipeline(config=cfg, retriever=retriever, medgemma=medgemma, llm=llm,
                              top_k=args.top_k)
        preds: list[str] = []
        for _, row in tqdm(test_df.iterrows(), total=len(test_df), desc=f"report::{cfg}"):
            img = Image.open(row["image_path"]).convert("RGB")
            out = pipe.run(img)
            preds.append(out.report_text)
        predictions[cfg] = preds

        m = score_report(refs, preds, enable_radgraph=args.radgraph)
        rows.append({"config": cfg, **dc.asdict(m)})

    write_table(rows, "results/tables/report_metrics.csv")
    Path("results/predictions").mkdir(parents=True, exist_ok=True)
    with open("results/predictions/report.json", "w", encoding="utf-8") as f:
        json.dump({"refs": refs, "predictions": predictions}, f, indent=2)


def eval_qa(args, configs: list[str]) -> None:
    qa_test = list(load_jsonl("data/qa/qa_test.jsonl"))[: args.test_size]
    df = pd.read_parquet("data/processed/reports.parquet").set_index("study_id")
    medgemma = MedGemmaRunner()
    medgemma.load()
    llm = LLMRouter() if "colpali_lora_text_llm" in configs else None
    judge = LLMJudge(LLMRouter())

    rows = []
    predictions: dict[str, list[dict]] = {}
    for cfg in configs:
        retriever_name = CONFIG_RETRIEVERS[cfg]
        retriever = build_retriever(retriever_name) if retriever_name else None
        pipe = QAPipeline(config=cfg, retriever=retriever, medgemma=medgemma, llm=llm,
                          top_k=args.top_k)

        preds: list[str] = []
        golds: list[str] = []
        questions: list[str] = []
        reports: list[str] = []
        records: list[dict] = []
        for qa in tqdm(qa_test, desc=f"qa::{cfg}"):
            img = Image.open(qa["image_path"]).convert("RGB")
            out = pipe.run(img, qa["question"])
            preds.append(out.answer)
            golds.append(qa["answer"])
            questions.append(qa["question"])
            reports.append(str(df.loc[qa["study_id"]]["clean_text"])
                           if qa["study_id"] in df.index else "")
            records.append({"qa_id": qa["qa_id"], "pred": out.answer, "gold": qa["answer"]})
        predictions[cfg] = records

        judge_scores = judge.score_many(questions, golds, preds, reports) if args.llm_judge else None
        m = score_qa(golds, preds, judge_scores=judge_scores)
        rows.append({"config": cfg, **dc.asdict(m)})

    write_table(rows, "results/tables/qa_metrics.csv")
    Path("results/predictions").mkdir(parents=True, exist_ok=True)
    with open("results/predictions/qa.json", "w", encoding="utf-8") as f:
        json.dump(predictions, f, indent=2)


def eval_retrieval(args) -> None:
    df = pd.read_parquet("data/processed/reports.parquet")
    splits = load_json("data/processed/splits.json")
    test_df = df[df["study_id"].astype(str).isin(splits["test"])].head(args.test_size)
    queries = []
    gold_ids = []
    for _, row in test_df.iterrows():
        sent = (row.get("findings") or row.get("clean_text") or "").split(".")[0].strip()
        if not sent:
            continue
        queries.append(sent)
        gold_ids.append(str(row["study_id"]))

    rows = []
    for backend in ["biomedclip", "colpali_zs", "colpali_lora"]:
        try:
            retriever = build_retriever(backend)
        except Exception as e:
            log.warning("Skipping %s: %s", backend, e)
            continue
        retrieved_ids = []
        for q in tqdm(queries, desc=f"retrieval::{backend}"):
            hits = retriever.search_text(q, k=10)
            retrieved_ids.append([h.study_id for h in hits])
        m = score_retrieval(retrieved_ids, gold_ids)
        rows.append({"backend": backend, **dc.asdict(m)})
    write_table(rows, "results/tables/retrieval_metrics.csv")


def main() -> None:
    p = argparse.ArgumentParser(__doc__)
    p.add_argument("--mode", action="append", choices=["report", "qa", "retrieval"], required=True)
    p.add_argument(
        "--configs", default="all",
        help="Comma-separated config names or 'all'",
    )
    p.add_argument("--test-size", type=int, default=200)
    p.add_argument("--top-k", type=int, default=3)
    p.add_argument("--radgraph", action="store_true")
    p.add_argument("--llm-judge", action="store_true", default=True)
    args = p.parse_args()

    load_dotenv()
    all_configs = [
        "medgemma_only",
        "biomedclip_rag",
        "colpali_zs_rag",
        "colpali_lora_rag",
        "colpali_lora_text_llm",
    ]
    configs = all_configs if args.configs == "all" else args.configs.split(",")

    if "report" in args.mode:
        df = pd.read_parquet("data/processed/reports.parquet")
        eval_report(args, df, configs)
    if "qa" in args.mode:
        eval_qa(args, configs)
    if "retrieval" in args.mode:
        eval_retrieval(args)


if __name__ == "__main__":
    main()
