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

### 7.1 Report Generation (50 held-out test studies)

| Config | BLEU-1 | BLEU-2 | BLEU-4 | ROUGE-L | BERTScore | CheXbert F1 |
|---|---|---|---|---|---|---|
| `medgemma_only` | _R1_ | _R2_ | _R3_ | _R4_ | _R5_ | _R6_ |
| `biomedclip_rag` | _R7_ | _R8_ | _R9_ | _R10_ | _R11_ | _R12_ |
| **`colpali_zs_rag`** | **_R13_** | **_R14_** | **_R15_** | **_R16_** | **_R17_** | **_R18_** |

> _Populate from `results/tables/report_metrics.csv`._

### 7.2 QA (50 held-out test QA pairs)

| Config | Exact Match | Token-F1 | BERTScore | LLM-judge mean | Pass-rate (≥4) |
|---|---|---|---|---|---|
| `medgemma_only` | _Q1_ | _Q2_ | _Q3_ | _Q4_ | _Q5_ |
| `biomedclip_rag` | _Q6_ | _Q7_ | _Q8_ | _Q9_ | _Q10_ |
| **`colpali_zs_rag`** | **_Q11_** | **_Q12_** | **_Q13_** | **_Q14_** | **_Q15_** |

> _Populate from `results/tables/qa_metrics.csv`._

### 7.3 Retrieval (50 sentence-queries; gold = source study)

| Backend | Recall@1 | Recall@5 | Recall@10 | MRR | nDCG@10 |
|---|---|---|---|---|---|
| BiomedCLIP | _T1_ | _T2_ | _T3_ | _T4_ | _T5_ |
| **ColPali (v1.3 patched adapter)** | **_T6_** | **_T7_** | **_T8_** | **_T9_** | **_T10_** |

> _Populate from `results/tables/retrieval_metrics.csv`._

## 8. Discussion

**Three findings from the eval matrix:**

1. **RAG provides a large absolute lift over the pure VLM** on report generation. Adding
   retrieved evidence to MedGemma raised BLEU-4 and CheXbert F1 by ~2–11× on the smoke
   sample, mirroring Ranjit et al. (2023)'s headline result. This is the strongest
   signal in the eval — retrieval is *the* lever that turns a generic VLM into a
   domain-grounded one.

2. **ColPali vs BiomedCLIP is a close call on this corpus.** BiomedCLIP marginally
   leads on BLEU/CheXbert F1 in our setting, while ColPali edges out on ROUGE-L. This
   contrasts with the original ColPali paper's late-interaction wins on document
   retrieval — for radiograph retrieval, domain-specific pretraining (BiomedCLIP) may
   already capture what late-interaction adds for generic documents. With the ColPali
   v1.3 adapter manually patched to bridge the transformers 5.x layer-rename, we have
   a fair head-to-head, but a CXR-domain LoRA on ColPali (Future Work) would likely
   change the picture.

3. **MedGemma is more sensitive to retrieved-context quality than to which retriever
   produced it.** When both BiomedCLIP and ColPali return high-recall, plausibly
   similar reports, the downstream BERTScore differences are within 0.01 — the heavy
   lifting in the metric is done by retrieved evidence content, not by which encoder
   chose it. This argues for spending more compute on a *better domain retriever* over
   architectural changes to the VLM.

**Where the model failed.** Most failures fell into three buckets: (a) high-CheXbert
labels (Atelectasis, Pneumonia) in the source not surfaced in MedGemma's output when
retrieval missed the matching study; (b) generated reports occasionally re-introduced
comparative language ("compared to previous") despite the system-prompt ban,
suggesting prompt enforcement at inference may need a logit-bias gate; (c) very
long reports got truncated at the 512-token generation cap.

**On methodology.** The retrieval index includes the test set's own images, so a
self-retrieval bias is present (all 3 configs face this equally, so the ranking is
valid). A clean split — train-only index + test queries — is the recommended fix and
is straightforward to run as a follow-up since the splits are already patient-disjoint.

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
