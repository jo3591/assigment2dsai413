# Multi-Modal Chest X-Ray Intelligence System

> DSAI 413 Assignment 2 — dual-mode medical AI: **Report Generation** + **RAG-based QA** for chest X-rays. Compares **ColPali** (patched v1.3 adapter), **BiomedCLIP**, and pure **MedGemma-4B-IT** across both modes.

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## Overview

Two independent modes share one retrieval + generation backbone:

```
                CXR Image (+ optional Question)
                         │
          ┌──────────────┴──────────────┐
          │                             │
     Mode A: Report                Mode B: QA
          │                             │
   [Retriever]                    [Retriever]
   ColPali  /  BiomedCLIP  /  none
          │                             │
   Top-K reports                  Top-K reports
          │                             │
   [Generator]                    [Generator]
   MedGemma-4B-IT (INT4)          MedGemma-4B-IT (INT4)
          │                             │
   FINDINGS + IMPRESSION           Grounded Answer
                                    + retrieved evidence
```

Three configurations are evaluated head-to-head:
1. `medgemma_only` — pure VLM, no retrieval (baseline)
2. `biomedclip_rag` — BiomedCLIP retrieves → MedGemma generates
3. `colpali_zs_rag` — ColPali (v1.3 with patched LoRA adapter) retrieves → MedGemma generates

## Headline Results

> Test set: **50 patient-disjoint studies** (Report), **50 patient-disjoint QA pairs** (QA). Full eval matrix in [`results/tables/`](results/tables/).

**Report Generation**

| Config | BLEU-4 | ROUGE-L | BERTScore F1 | **CheXbert F1** |
|---|---|---|---|---|
| `medgemma_only` (no retrieval) | 0.0002 | 0.186 | 0.843 | 0.301 |
| `biomedclip_rag` | 0.0012 | 0.261 | 0.864 | 0.352 |
| **`colpali_zs_rag`** | **0.0016** | 0.260 | **0.865** | **0.429** |

**QA Mode**

| Config | Token-F1 | BERTScore F1 | LLM-judge mean | **Pass-rate (≥4)** |
|---|---|---|---|---|
| `medgemma_only` (no retrieval) | 0.259 | 0.887 | 3.42 | 0.66 |
| `biomedclip_rag` | 0.448 | 0.909 | **4.39** | 0.88 |
| **`colpali_zs_rag`** | **0.454** | **0.911** | 4.34 | **0.90** |

**ColPali-RAG wins on the clinical accuracy (CheXbert F1 +43% over no-RAG) and QA pass-rate.** See [`report/report.md`](report/report.md) §8 for discussion.

## Repo Layout

```
413assigment2/
├── README.md                         # this file
├── report/report.md                  # short written report (architecture, results, discussion)
├── report/demo_video_link.md         # YouTube link
├── configs/                          # 5 YAML configs (data, colpali, medgemma, biomedclip, eval)
├── notebooks/
│   ├── kaggle_main.ipynb             # ★ THE notebook — full pipeline executed on Kaggle T4×2
│   └── colabpart1.ipynb              # early Colab session (data download exploration)
├── scripts/                          # CLI entry points called by the notebook
│   ├── download_data.py
│   ├── build_qa_dataset.py
│   ├── build_indices.py
│   ├── run_eval.py
│   └── colab_bootstrap.sh
├── src/cxr_intel/                    # the importable package
│   ├── data/                         # loader, preprocess, CheXpert labeler, splits
│   ├── retrieval/                    # ColPali + BiomedCLIP retrievers
│   ├── models/                       # MedGemma runner + LLM router (NVIDIA NIM / OpenRouter)
│   ├── generation/                   # Report + QA pipelines + prompt templates
│   ├── qa_dataset/                   # synthetic QA generation + 4-step validator
│   ├── eval/                         # metrics + LLM-as-judge
│   └── utils/                        # io, logging
├── results/
│   ├── tables/{report,qa,retrieval}_metrics.csv
│   ├── predictions/{report,qa}.json
│   └── *.png                         # demo screenshots
├── docs/{architecture,ethics}.md
└── data/                             # gitignored; populated by the pipeline
```

## How to Reproduce

**Single source of truth**: open [`notebooks/kaggle_main.ipynb`](notebooks/kaggle_main.ipynb) on Kaggle (the actual notebook that produced every artifact in this repo). Prerequisites:

1. Kaggle Notebook → Settings → Accelerator = **GPU T4 ×2**
2. **+ Add Input** → `simhadrisadaram/mimic-cxr-dataset` → Add
3. **Add-ons → Secrets** — add:
   - `HF_TOKEN` (accept the [MedGemma TOS](https://huggingface.co/google/medgemma-4b-it) first)
   - `NVIDIA_API_KEY` (free at https://build.nvidia.com)
   - `KAGGLE_USERNAME`, `KAGGLE_KEY` (from https://www.kaggle.com/settings)
4. Run all cells in order. Total wall-clock ≈ 3 hours on T4×2 free tier.

## QA Dataset (built by `scripts/build_qa_dataset.py`)

- **Source**: 2,268 train + 284 test studies from the Kaggle MIMIC-CXR subset, patient-disjoint
- **Anchors**: 14 CheXpert labels (rule-based parser in `src/cxr_intel/data/chexpert_labels.py`)
- **Question types**: existence / location / severity / attribute / open
- **Generator**: `meta/llama-3.1-8b-instruct` via NVIDIA NIM, prompted with report + CheXpert vector + target type. System prompt forbids comparative language (`unchanged, new, interval, prior, follow-up, since, …`).
- **Validator** (`src/cxr_intel/qa_dataset/validator.py`): banned-term regex reject → source-sentence fuzzy match (≥85%) → LLM judge 4-dim quality score → dedupe by `(study_id, question_type, anchor_label)`
- **Output**: **6,216 train + 811 test validated pairs**

## Evaluation Methodology

| Mode | Metrics | Test set |
|---|---|---|
| Report Generation | BLEU-1/2/4, ROUGE-L, BERTScore-F1, CheXbert-F1 | 50 patient-disjoint studies |
| QA | Exact-Match, token-F1, BERTScore, LLM-judge mean + pass-rate (≥ 4) | 50 patient-disjoint QA pairs |
| Retrieval | Recall@1/5/10, MRR, nDCG@10 | 50 sentence queries; gold = source study_id (see report §7.3 for caveat) |

## Demo

See [`report/demo_video_link.md`](report/demo_video_link.md) for the unlisted YouTube walkthrough and the live Gradio public URL (Kaggle + ngrok). Screenshots are in [`results/`](results/).

## Limitations & Ethics

- **Not for clinical decision-making.** Generated reports/answers are research artifacts.
- **Rule-based CheXpert labeler** (regex + negation heuristics) is a proxy for the official CheXpert/CheXbert labeler.
- **Synthetic QA** is grounded in report text, not directly in image content — it inherits any errors in the source reports.
- **MIMIC-CXR data** is PhysioNet credentialed-use; the Kaggle subset is a derivative.
- **ColPali v1.3 adapter** had to be manually key-remapped to load with transformers ≥ 4.51 — see Stage 4 in `notebooks/kaggle_main.ipynb`.

## Citations

- Faysse et al. **ColPali: Efficient Document Retrieval with Vision Language Models.** arXiv:2407.01449
- Google DeepMind. **MedGemma 4B IT.** https://huggingface.co/google/medgemma-4b-it
- Zhang et al. **BiomedCLIP: a multimodal biomedical foundation model.** arXiv:2303.00915
- Ranjit et al. **Retrieval-Augmented Chest X-Ray Report Generation using OpenAI GPT models.** arXiv:2305.03660
- MIMIC-CXR-VQA dataset construction: https://github.com/LightVED-prhlt/MIMIC-CXR-VQA-Dataset_Creation

## License

MIT for code. Data per the PhysioNet / MIMIC-CXR license.
