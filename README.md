# Multi-Modal Chest X-Ray Intelligence System

> DSAI 413 Assignment 2 — dual-mode medical AI: **Report Generation** + **RAG-based QA** for chest X-rays. Compares ColPali (zero-shot + LoRA-tuned), BiomedCLIP, and pure MedGemma-4B-IT across both modes.

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/UI-Streamlit-ff4b4b)](https://streamlit.io)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## Overview

Two independent modes share one retrieval+generation backbone:

```
                CXR Image (+ optional Question)
                         │
          ┌──────────────┴──────────────┐
          │                             │
     Mode A: Report                Mode B: QA
          │                             │
   [Retriever]                    [Retriever]
   ColPali-LoRA / ColPali-zs / BiomedCLIP
          │                             │
   Top-K reports                  Top-K reports
          │                             │
   [Generator]                    [Generator]
   MedGemma-4B-IT                 MedGemma-4B-IT
          │                             │
   FINDINGS + IMPRESSION           Grounded Answer
                                    + retrieved evidence
```

**Five configurations** are evaluated head-to-head:
1. `medgemma_only` — pure VLM, no retrieval (baseline).
2. `biomedclip_rag` — BiomedCLIP retrieves → MedGemma generates.
3. `colpali_zs_rag` — Zero-shot ColPali retrieves → MedGemma generates.
4. `colpali_lora_rag` — LoRA-tuned ColPali retrieves → MedGemma generates (recommended).
5. `colpali_lora_text_llm` — LoRA-tuned ColPali → text-only OpenRouter LLM (ablation).

## Headline Results

> Test set: 50 patient-disjoint studies (Report), 50 patient-disjoint QA pairs (QA).
> Populate from `results/tables/*.csv`.

**Report Generation**

| Config | BLEU-4 | ROUGE-L | BERTScore F1 | CheXbert F1 |
|---|---|---|---|---|
| `medgemma_only` | _R3_ | _R4_ | _R5_ | _R6_ |
| `biomedclip_rag` | _R9_ | _R10_ | _R11_ | _R12_ |
| **`colpali_zs_rag`** | **_R15_** | **_R16_** | **_R17_** | **_R18_** |

**QA Mode**

| Config | Exact Match | Token-F1 | BERTScore F1 | LLM-judge mean |
|---|---|---|---|---|
| `medgemma_only` | _Q1_ | _Q2_ | _Q3_ | _Q4_ |
| `biomedclip_rag` | _Q6_ | _Q7_ | _Q8_ | _Q9_ |
| **`colpali_zs_rag`** | **_Q11_** | **_Q12_** | **_Q13_** | **_Q14_** |

**Retrieval**

| Retriever | Recall@1 | Recall@5 | Recall@10 | MRR | nDCG@10 |
|---|---|---|---|---|---|
| BiomedCLIP | _T1_ | _T2_ | _T3_ | _T4_ | _T5_ |
| **ColPali (v1.3 patched)** | **_T6_** | **_T7_** | **_T8_** | **_T9_** | **_T10_** |

## Repo Layout

```
413assigment2/
├── README.md                        # this file
├── pyproject.toml + requirements*   # deps (local + Colab)
├── .env.example                     # API keys template
├── configs/                         # YAML configs (data, models, eval)
├── src/cxr_intel/                   # main package
│   ├── data/                        # loaders, preprocess, CheXpert labeler, splits
│   ├── retrieval/                   # ColPali + BiomedCLIP retrievers
│   ├── models/                      # MedGemma runner + LLM router
│   ├── generation/                  # Report + QA pipelines + prompt templates
│   ├── qa_dataset/                  # synthetic QA generation + validation
│   ├── finetune/                    # ColPali LoRA training
│   ├── eval/                        # metrics + LLM-as-judge
│   ├── app/                         # Streamlit demo
│   └── utils/                       # io, logging, viz helpers
├── notebooks/01..06                 # end-to-end Colab pipeline
├── scripts/                         # CLI entry points
├── tests/                           # pytest smoke tests
├── data/                            # gitignored except samples/
├── models/colpali-cxr-lora/         # the trained LoRA adapter
├── results/                         # CSV tables + PNG figures
└── report/                          # written report + demo video link
```

## Installation

### Local (CPU dev / running tests)

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -e .[dev]
pytest tests/
```

### Free-tier GPU pipeline (Colab T4 + Kaggle P100)

This repo targets the **free** versions of both services. Memory budget on Colab T4
(15 GB) is tight, so MedGemma is loaded with INT4 quantization by default.

**Colab T4** — used for data prep, QA synthesis, indexing, evaluation, and the
Streamlit demo:

```python
from google.colab import drive; drive.mount('/content/drive')
!git clone https://github.com/jo3591/assigment2dsai413 /content/cxr-intel
%cd /content/cxr-intel
!bash scripts/colab_bootstrap.sh
```

**Kaggle P100** — used only for the ColPali LoRA fine-tune (16 GB VRAM, 30 h/week
free). See [docs/kaggle_setup.md](docs/kaggle_setup.md) for the full walkthrough and
run [`notebooks/03_colpali_lora_train_kaggle.ipynb`](notebooks/03_colpali_lora_train_kaggle.ipynb).

> If you have access to an A100 or H100, set `torch_dtype: bfloat16` in
> `configs/colpali.yaml` and `configs/medgemma.yaml`, and `quantization: null` in
> the latter for slightly better quality.

## Setup — Secrets

Copy `.env.example` to `.env` and fill in:

| Variable | Where to get it |
|---|---|
| `HF_TOKEN` | https://huggingface.co/settings/tokens — also accept the [MedGemma TOS](https://huggingface.co/google/medgemma-4b-it) |
| `OPENROUTER_API_KEY` | https://openrouter.ai/keys |
| `NVIDIA_API_KEY` | (optional) https://build.nvidia.com/ |
| `KAGGLE_USERNAME`, `KAGGLE_KEY` | https://www.kaggle.com/settings → Create API Token |

## Reproducing Results — Step by Step

The pipeline straddles two services to stay on free tiers.

### On Colab T4

```bash
# 1. Pull the Kaggle MIMIC-CXR subset, preprocess, compute CheXpert labels, split
python scripts/download_data.py --config configs/data.yaml

# 2. Generate the synthetic QA dataset (LLM call ≈ 1-2h on full corpus)
python scripts/build_qa_dataset.py --config configs/data.yaml

# 3. Build retrieval indices for the two non-LoRA backends
python scripts/build_indices.py --backend biomedclip --backend colpali_zs
```

### On Kaggle P100  (see [docs/kaggle_setup.md](docs/kaggle_setup.md))

```bash
# 4. Fine-tune ColPali LoRA → models/colpali-cxr-lora/final/
#    Use notebooks/03_colpali_lora_train_kaggle.ipynb (P100, ~3-4 h, 2 epochs)
python scripts/train_colpali_lora.py --epochs 2 --batch-size 2
```

Download the LoRA adapter, commit it back to the repo, then return to Colab.

### Back on Colab T4

```bash
# 5. Build the LoRA-aware index
python scripts/build_indices.py --backend colpali_lora

# 6. Run the full evaluation matrix (Report + QA + Retrieval, all 5 configs)
python scripts/run_eval.py --mode report --mode qa --mode retrieval --configs all

# 7. Generate figures
jupyter nbconvert --to notebook --execute notebooks/06_results_figures.ipynb
```

## Running the Streamlit Demo

```powershell
streamlit run src/cxr_intel/app/streamlit_app.py
```

- **Page 1 — Report Generation**: upload an X-ray, choose retriever + generator, get a structured report.
- **Page 2 — QA Mode**: same inputs + a question. Optional ColPali MaxSim heatmap overlay shows what drove retrieval.
- **Page 3 — Compare Models**: runs three configs side-by-side on the same input.

## QA Dataset Card

- **Source**: synthesized from MIMIC-CXR reports in the Kaggle subset (`text` column).
- **Anchors**: 14 CheXpert labels parsed by `src/cxr_intel/data/chexpert_labels.py` (rule-based fallback).
- **Question types**: `existence`, `location`, `severity`, `attribute`, `open` (templates in `src/cxr_intel/qa_dataset/templates.py`).
- **Generator**: `meta-llama/llama-3.3-70b-instruct` via OpenRouter, prompted with the report + CheXpert vector + target type. Banned-terms list (`unchanged, new, interval, prior, follow-up, since, …`) baked into the system prompt.
- **Validation**: 4-step pipeline in `src/cxr_intel/qa_dataset/validator.py`:
  1. Banned-term regex reject.
  2. Source-sentence fuzzy match (≥85%) against the report.
  3. LLM judge scores 4 dims 0-5; keep mean ≥3.5 and all dims ≥3.
  4. Dedupe by `(study_id, question_type, anchor_label)`.
- **Output**: `data/qa/qa_v1.jsonl` (train) and `data/qa/qa_test.jsonl` (test, patient-disjoint). Schema in `src/cxr_intel/qa_dataset/schema.py`.

## Evaluation Methodology

| Mode | Metrics | Test set |
|---|---|---|
| Report Generation | BLEU-1/2/4, ROUGE-L, BERTScore-F1, CheXbert-F1 (rule-based proxy), RadGraph-F1 (optional) | 200 patient-disjoint studies |
| QA | Exact-Match, token-F1, BERTScore, LLM-as-judge mean + pass-rate (≥4) | 200 patient-disjoint QA pairs |
| Retrieval | Recall@1/5/10, MRR, nDCG@10 | 200 sentence queries; gold = source study_id |

## Demo Video

[`report/demo_video_link.md`](report/demo_video_link.md) — 10 min unlisted YouTube walkthrough.

## Limitations & Ethics

- **Not for clinical decision-making.** Generated reports/answers are research artifacts and must not inform patient care.
- **Rule-based CheXpert labeler** (regex + negation heuristics) is a proxy; the official CheXpert labeler may give slightly different stats.
- **Synthetic QA** is grounded in radiology reports, not in the images themselves — it inherits report errors.
- **MIMIC-CXR data** is PhysioNet credentialed-use; the Kaggle subset is a derivative; downstream users must comply with the MIMIC license.

## Citations

- Faysse et al. **ColPali: Efficient Document Retrieval with Vision Language Models.** arXiv:2407.01449.
- Sellergren et al. **MedGemma.** Google DeepMind, 2024–2025.
- Zhang et al. **BiomedCLIP: a multimodal biomedical foundation model.** arXiv:2303.00915.
- Ranjit et al. **Retrieval Augmented Chest X-Ray Report Generation using OpenAI GPT models.** arXiv:2305.03660.
- Pellegrini et al. (MIMIC-CXR-VQA dataset construction): https://github.com/LightVED-prhlt/MIMIC-CXR-VQA-Dataset_Creation.
- Hugging Face **Multimodal RAG cookbook.** https://huggingface.co/learn/cookbook/multimodal_rag_using_document_retrieval_and_vlms.

## License

MIT for code (see [`LICENSE`](LICENSE)). Data redistribution governed by the underlying MIMIC-CXR / PhysioNet license.
