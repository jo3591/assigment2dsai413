# Architecture

## Module map

```
src/cxr_intel/
├── data/
│   ├── kaggle_loader.py     ← downloads + locates the Kaggle subset
│   ├── preprocess.py        ← report cleaning + Findings/Impression split
│   ├── chexpert_labels.py   ← rule-based 14-label parser + micro-F1
│   └── splits.py            ← patient-level + stratified subsample
├── retrieval/
│   ├── base.py              ← Retriever Protocol + RetrievalHit
│   ├── colpali_index.py     ← ColPali multi-vector index + MaxSim search
│   ├── colpali_search.py    ← heatmap helper for the Streamlit UI
│   └── biomedclip_index.py  ← BiomedCLIP FAISS-IP single-vector baseline
├── models/
│   ├── medgemma_runner.py   ← MedGemma-4B-IT generation wrapper
│   └── llm_router.py        ← OpenRouter / NVIDIA NIM client (OpenAI-compatible)
├── generation/
│   ├── prompts.py           ← system + user templates for both modes
│   ├── report_pipeline.py   ← Mode A: image → report
│   └── qa_pipeline.py       ← Mode B: (image, question) → grounded answer
├── qa_dataset/
│   ├── schema.py            ← QAPair pydantic model + banned-terms list
│   ├── templates.py         ← per-CheXpert-label template scaffolds
│   ├── synth_generator.py   ← LLM-driven QA synthesis with cache
│   └── validator.py         ← banned-term + source-match + judge + dedup
├── finetune/
│   ├── pair_builder.py      ← contrastive (image, report, hard negatives) builder
│   ├── collator.py          ← ColPaliProcessor batch collator
│   └── train_colpali_lora.py ← PEFT LoRA training loop with InfoNCE
├── eval/
│   ├── metrics_report.py    ← BLEU, ROUGE, BERTScore, CheXbert-F1, RadGraph-F1
│   ├── metrics_qa.py        ← EM, token-F1, BERTScore, LLM-judge mean/pass-rate
│   ├── metrics_retrieval.py ← Recall@k, MRR, nDCG@k
│   └── llm_judge.py         ← Claude-as-judge wrapper
├── app/
│   ├── streamlit_app.py     ← multipage launcher + status sidebar
│   ├── pages/
│   │   ├── 1_Report_Generation.py
│   │   ├── 2_QA_Mode.py
│   │   └── 3_Compare_Models.py
│   ├── components/heatmap.py
│   └── cache.py             ← @st.cache_resource model loaders
└── utils/
    ├── io.py, logging.py, viz.py
```

## Data flow

```
            scripts/                     src/cxr_intel/                results/
download_data.py  ─►  data/processed/reports.parquet
                     ─►  data/processed/splits.json
                     ─►  data/samples/*.jpg

build_qa_dataset.py ─►  data/qa/qa_v1.jsonl  (train)
                     ─►  data/qa/qa_test.jsonl
                     ─►  data/qa/cache/*.json (LLM cache)

build_indices.py    ─►  data/indices/biomedclip/{index.faiss, metadata.json}
                     ─►  data/indices/colpali_zs/{doc_embeddings.npy, metadata.json}
                     ─►  data/indices/colpali_lora/{...}

train_colpali_lora.py ─► models/colpali-cxr-lora/{step-N, final}/

run_eval.py         ─►  results/tables/{report,qa,retrieval}_metrics.csv
                     ─►  results/predictions/{report,qa}.json

notebooks/06        ─►  results/figures/*.png
```

## Configuration surface

`configs/data.yaml` — corpus size, splits, preprocessing thresholds.
`configs/colpali.yaml` — checkpoint, LoRA hyperparameters, training schedule.
`configs/medgemma.yaml` — generation parameters, prompt templates.
`configs/biomedclip.yaml` — baseline retriever knobs.
`configs/eval.yaml` — metric toggles, judge model, output paths.

All scripts accept `--config PATH` to override.
