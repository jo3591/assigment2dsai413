# Multi-Modal Chest X-Ray Intelligence System
*DSAI 413 — Assignment 2 — Short Report*

## 1. Problem Setting

Chest X-rays are the most common diagnostic imaging study; clinicians read them under
time pressure and rely on structured reports (FINDINGS + IMPRESSION) and follow-up Q&A
with peers. Two complementary AI tasks fall out of this workflow:

- **Report Generation** — given a CXR, draft a radiology report.
- **Clinical QA** — given a CXR and a clinical question, answer it grounded in the
  visible evidence and (optionally) the corpus of previously written reports.

Both tasks require fluent vision–language reasoning, but they have different cost
profiles: report generation is open-ended; QA is bounded but must be precise. We build
**one retrieval+generation backbone** that serves both modes, with five swappable
configurations so we can isolate the contribution of each component.

## 2. Architecture

The full pipeline lives in `src/cxr_intel/`. A high-level diagram:

```
                CXR Image (+ optional Question)
                         │
          ┌──────────────┴──────────────┐
     Mode A: Report                Mode B: QA
          │                             │
          ▼                             ▼
   ┌──────────────────────────────────────────────────┐
   │  Retriever (one of):                             │
   │    BiomedCLIP · ColPali (zs) · ColPali (LoRA)    │
   │    or none (pure VLM)                            │
   └──────────────────────────────────────────────────┘
                         │
                         ▼
   ┌──────────────────────────────────────────────────┐
   │  Generator: MedGemma-4B-IT  (or text-only LLM)   │
   └──────────────────────────────────────────────────┘
                         │
                         ▼
            FINDINGS + IMPRESSION  /  Grounded Answer
```

- **ColPali** ([Faysse et al. 2024](https://arxiv.org/abs/2407.01449)) is a multi-vector
  PaliGemma-based retriever using ColBERT-style late interaction (MaxSim). It treats
  page-as-image: the CXR is encoded into a grid of patch embeddings and queries are
  scored against all patches.
- **MedGemma-4B-IT** ([Google DeepMind](https://deepmind.google/models/gemma/medgemma/))
  is the multimodal generator. We chose 4B over 27B for Colab feasibility.
- **BiomedCLIP** is the lightweight CLIP baseline for retrieval, providing a single
  vector per image — useful for highlighting where multi-vector ColPali wins.
- **Text-only LLM ablation** (Llama-3.3-70B via OpenRouter) tests whether the VLM
  contributes more than the retrieved evidence.

## 3. Dataset

We use the Kaggle subset `simhadrisadaram/mimic-cxr-dataset` of MIMIC-CXR. The `text`
column holds the radiology report. Preprocessing in `src/cxr_intel/data/preprocess.py`:

1. Strip MIMIC de-identification markers (`___`).
2. Regex-extract FINDINGS / IMPRESSION sections.
3. Drop reports < 30 tokens; dedupe by `study_id`.
4. Compute a 14-label CheXpert vector via the rule-based parser in `chexpert_labels.py`
   (Java-licensed CheXbert labeler is unavailable in our pipeline; the regex parser uses
   negation/uncertainty windows and matches MIMIC-CXR-VQA conventions).
5. Stratified subsample to ~4 000 reports by primary CheXpert label.
6. Patient-level 80/10/10 split.

## 4. QA Dataset Construction

The assignment ships no QA dataset, so we synthesize one. The protocol mirrors the
MIMIC-CXR-VQA paper:

- **Anchors**: 14 CheXpert labels + a "No Finding" anchor.
- **Question types**: existence, location, severity, attribute, open (templates in
  `qa_dataset/templates.py`).
- **Generation prompt** (full text in `qa_dataset/synth_generator.py`): the LLM gets
  the cleaned report, the CheXpert vector, the target question type, and a list of
  suggested phrasings. Banned terms (`unchanged, new, interval, prior, follow-up, since,
  …`) are listed in the system prompt — these violate the single-study assumption.
- **Validation** (`qa_dataset/validator.py`):
  1. Banned-term regex reject.
  2. Source-sentence fuzzy match (≥85% via `rapidfuzz`) against the report.
  3. LLM judge (Claude Sonnet via OpenRouter) scores correctness, consistency,
     completeness, and clinical-relevance on 0–5; keep mean ≥3.5 and all dims ≥3.
  4. Dedupe by `(study_id, question_type, anchor_label)`.
- **Output**: ~1 200 validated pairs across train/val, ~200 held-out test pairs.
  Distribution: 35% existence, 25% location, 20% severity, 10% attribute, 10% open.

## 5. ColPali LoRA Fine-tune

Even strong zero-shot ColPali is trained on academic-document corpora; the radiology
domain has different visual patterns (low-contrast lung markings, ribs, devices). We
fine-tune a rank-32 LoRA adapter on (CXR image, FINDINGS+IMPRESSION) pairs:

- **Pairs**: 2 400 train / 300 val. Hard negatives = 3 reports sharing the same
  primary CheXpert label but a different `study_id`.
- **Loss**: late-interaction InfoNCE (ColbertLoss).
- **Training**: 3 epochs, AdamW lr=5e-5, cosine schedule, bf16, gradient checkpointing,
  effective batch 16. ≈ 3 hours on Colab A100.

## 6. Evaluation

| Mode | Metrics |
|---|---|
| Report | BLEU-1/2/4, ROUGE-L, BERTScore-F1, CheXbert-F1 (rule-based), RadGraph-F1 (opt-in) |
| QA | Exact Match, token-F1, BERTScore-F1, LLM-judge mean + pass-rate (≥ 4) |
| Retrieval | Recall@1/5/10, MRR, nDCG@10 |

## 7. Results

> Numbers populated after running notebooks 04 and 05. See `results/tables/*.csv` and
> figures in `results/figures/`.

### 7.1 Report Generation

| Config | BLEU-4 | ROUGE-L | BERTScore | CheXbert F1 |
|---|---|---|---|---|
| `medgemma_only` | _tbd_ | _tbd_ | _tbd_ | _tbd_ |
| `biomedclip_rag` | _tbd_ | _tbd_ | _tbd_ | _tbd_ |
| `colpali_zs_rag` | _tbd_ | _tbd_ | _tbd_ | _tbd_ |
| **`colpali_lora_rag`** | **_tbd_** | **_tbd_** | **_tbd_** | **_tbd_** |
| `colpali_lora_text_llm` | _tbd_ | _tbd_ | _tbd_ | _tbd_ |

### 7.2 QA

| Config | EM | Token-F1 | BERTScore | LLM-judge mean |
|---|---|---|---|---|
| `medgemma_only` | _tbd_ | _tbd_ | _tbd_ | _tbd_ |
| `biomedclip_rag` | _tbd_ | _tbd_ | _tbd_ | _tbd_ |
| `colpali_zs_rag` | _tbd_ | _tbd_ | _tbd_ | _tbd_ |
| **`colpali_lora_rag`** | **_tbd_** | **_tbd_** | **_tbd_** | **_tbd_** |
| `colpali_lora_text_llm` | _tbd_ | _tbd_ | _tbd_ | _tbd_ |

### 7.3 Retrieval

| Backend | Recall@1 | Recall@5 | Recall@10 | MRR | nDCG@10 |
|---|---|---|---|---|---|
| BiomedCLIP | _tbd_ | _tbd_ | _tbd_ | _tbd_ | _tbd_ |
| ColPali (zs) | _tbd_ | _tbd_ | _tbd_ | _tbd_ | _tbd_ |
| **ColPali (LoRA)** | **_tbd_** | **_tbd_** | **_tbd_** | **_tbd_** | **_tbd_** |

## 8. Discussion

> Filled in after evaluation. Expected findings:
> - LoRA-tuned ColPali should outperform zero-shot ColPali on retrieval (Recall@5 +5–15
>   points), and feed better evidence into MedGemma.
> - RAG should help QA more than report generation: QA has localized targets, reports
>   are generic enough that pure MedGemma already covers ground.
> - The text-only LLM ablation is informative: if it matches MedGemma+RAG, the VLM
>   isn't pulling much weight beyond the retrieved evidence.

## 9. Limitations

- **Not clinical-grade.** All output is for research only; MedGemma's TOS forbids
  clinical decision-making and the training data has known biases.
- **Rule-based CheXpert labeler** is a proxy; expect ±5% delta from CheXbert numbers.
- **Synthetic QA** inherits any errors in the source reports; it is not an independent
  ground truth.
- **Patient-level split** prevents trivial leakage but doesn't simulate distribution
  shift — performance on truly unseen institutions is likely lower.

## 10. Future Work

- Replace the rule-based labeler with the official CheXbert if license permits.
- Add a second VLM (e.g., LLaVA-Med, Qwen2.5-VL) for direct comparison with MedGemma.
- Index sentence-level chunks rather than full reports — better fine-grained retrieval.
- RadGraph-F1 evaluation once the upstream package stabilizes.
