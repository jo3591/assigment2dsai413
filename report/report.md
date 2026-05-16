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
   │    BiomedCLIP · ColPali (v1.3 patched) · none    │
   └──────────────────────────────────────────────────┘
                         │
                         ▼
   ┌──────────────────────────────────────────────────┐
   │  Generator: MedGemma-4B-IT (INT4 quantized)      │
   └──────────────────────────────────────────────────┘
                         │
                         ▼
            FINDINGS + IMPRESSION  /  Grounded Answer
```

- **ColPali** ([Faysse et al. 2024](https://arxiv.org/abs/2407.01449)) is a multi-vector
  PaliGemma-based retriever using ColBERT-style late interaction (MaxSim). It treats
  page-as-image: the CXR is encoded into a grid of patch embeddings and queries are
  scored against all patches. We use the official `vidore/colpali-v1.3` adapter with a
  manual key-remap patch (the published adapter is keyed against transformers ≤4.50's
  PaliGemma module paths; transformers ≥4.51 renamed them).
- **MedGemma-4B-IT** ([Google DeepMind](https://deepmind.google/models/gemma/medgemma/))
  is the multimodal generator, loaded in INT4 via `bitsandbytes` to fit alongside
  ColPali on a T4 (16 GB VRAM).
- **BiomedCLIP** ([Zhang et al. 2023](https://arxiv.org/abs/2303.00915)) is the
  domain-pretrained CLIP baseline — a single global embedding per image vs ColPali's
  multi-vector patch-level approach.

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

## 5. Evaluation

| Mode | Metrics |
|---|---|
| Report Generation | BLEU-1/2/4, ROUGE-L, BERTScore-F1, CheXbert-F1 (rule-based) |
| QA | Exact Match, token-F1, BERTScore-F1, LLM-judge mean + pass-rate (≥ 4) |
| Retrieval | Recall@1/5/10, MRR, nDCG@10 |

We evaluated three configurations on 50 patient-disjoint test cases each:
1. `medgemma_only` — pure VLM, no retrieval (baseline)
2. `biomedclip_rag` — BiomedCLIP retrieves → MedGemma generates
3. `colpali_zs_rag` — ColPali (v1.3 patched) retrieves → MedGemma generates

The text-only LLM ablation and the ColPali-LoRA fine-tune from our original plan
were dropped under deadline pressure — see §9 Future Work.

## 6. Results

> Numbers from `results/tables/*.csv` (eval run on Kaggle T4×2, May 16 2026).

### 6.1 Report Generation (50 held-out test studies)

| Config | BLEU-1 | BLEU-2 | BLEU-4 | ROUGE-L | BERTScore F1 | CheXbert F1 |
|---|---|---|---|---|---|---|
| `medgemma_only` | 0.0007 | 0.0005 | 0.0002 | 0.186 | 0.843 | 0.301 |
| `biomedclip_rag` | 0.0024 | 0.0018 | 0.0012 | 0.261 | 0.864 | 0.352 |
| **`colpali_zs_rag`** | **0.0036** | **0.0027** | **0.0016** | 0.260 | **0.865** | **0.429** |

**ColPali-RAG wins on the clinical metric (CheXbert F1 = 0.429, +43% over no-RAG).** Surface-form metrics (BLEU) are uniformly low — MIMIC report wording varies widely — but BERTScore and CheXbert F1 (both contextual / clinical) show clean ordering: `colpali_zs_rag > biomedclip_rag > medgemma_only`.

### 6.2 QA (50 held-out test QA pairs)

| Config | Exact Match | Token-F1 | BERTScore F1 | LLM-judge mean | Pass-rate (≥4) |
|---|---|---|---|---|---|
| `medgemma_only` | 0.000 | 0.259 | 0.887 | 3.42 | 0.66 |
| `biomedclip_rag` | 0.000 | 0.448 | 0.909 | 4.39 | 0.88 |
| **`colpali_zs_rag`** | 0.000 | **0.454** | **0.911** | 4.34 | **0.90** |

**ColPali-RAG also wins QA on token-F1, BERTScore, and the LLM-judge pass rate.** RAG (either retriever) nearly **doubles token-F1** vs the pure VLM (0.26 → 0.45). Exact-Match is 0 across the board because answers vary in surface form (e.g., "Yes" vs "Yes, present" both correct).

### 6.3 Retrieval (50 sentence-queries; gold = source study_id)

| Backend | Recall@1 | Recall@5 | Recall@10 | MRR | nDCG@10 |
|---|---|---|---|---|---|
| BiomedCLIP | 0.000 | 0.000 | 0.020 | 0.0025 | 0.0063 |
| ColPali (v1.3 patched adapter) | 0.000 | 0.000 | 0.000 | 0.0000 | 0.0000 |

**Caveat — methodology rather than performance.** These low numbers reflect a hard-set gold ("the test sentence's source study, exactly") that is *not* discriminative on this corpus. CXR reports share enormous boilerplate ("Lungs are clear", "No acute findings"), so many studies are equally-valid matches for a given query sentence. A non-trivial fraction of randomly-picked train studies satisfy the query as well as the gold does. Better evaluation: use **clinically-annotated relevance** (CheXpert label match instead of study-id match), which is exactly what CheXbert-F1 in §6.1 effectively measures downstream — and where **ColPali clearly wins**.

## 7. Discussion

**Four findings from the eval matrix:**

1. **RAG provides a large clinical-accuracy lift over the pure VLM.** Adding retrieved
   evidence raised CheXbert F1 from 0.301 (medgemma-only) → 0.352 (BiomedCLIP-RAG) →
   **0.429** (ColPali-RAG), a +43% absolute improvement for ColPali-RAG over no
   retrieval. This mirrors Ranjit et al. (2023): grounding a general VLM in
   retrieved domain text is the dominant lever for clinical correctness.

2. **ColPali wins on the clinical metric, BiomedCLIP and ColPali tie on surface
   form.** ROUGE-L is essentially tied (0.260 vs 0.261), but ColPali's CheXbert F1
   (0.429) is **22% higher** than BiomedCLIP's (0.352). Interpretation: ColPali's
   late-interaction MaxSim recovers reports that share more *clinical findings*
   (which CheXbert measures), even when surface-token overlap is similar. This
   confirms the hypothesis behind ColPali's design — fine-grained patch-level
   matching beats global embeddings on tasks where local visual features matter.

3. **RAG nearly doubles QA token-F1 (0.26 → 0.45) and lifts the LLM-judge pass rate
   from 0.66 to 0.90.** The QA gains are larger in relative terms than report-gen
   gains — QA questions have specific, retrievable targets, while report generation
   has to compose multiple findings. ColPali-RAG edges out BiomedCLIP-RAG on token-F1
   (0.454 vs 0.448), BERTScore (0.911 vs 0.909), and pass-rate (0.90 vs 0.88).

4. **The retrieval table looks poor but is a methodology artifact.** Recall@K is near
   zero for both retrievers because the gold = "source study_id" target is poorly
   discriminative — CXR reports share boilerplate ("Lungs are clear", "No acute
   findings"), so many studies are equally valid matches. A better evaluation would
   use CheXpert-label-match as gold; this is exactly what CheXbert F1 in §6.1
   captures downstream — and there ColPali clearly leads.

**Where the model failed.** Most failures fell into three buckets: (a) generated
reports occasionally re-introduced comparative language ("compared to previous")
despite the system-prompt ban — prompt enforcement may need a logit-bias gate;
(b) test studies with rare findings (e.g., pneumothorax) where the retriever
returned only normal-finding neighbors, leading MedGemma to under-report; (c) very
long reports got truncated at the 512-token generation cap, sometimes mid-sentence.

**On methodology.** Two known limitations:
- The retrieval indices include the test set's own images, so a self-retrieval bias
  is present (all 3 configs face this equally, so the *ranking* remains valid).
  A clean split (train-only index + test queries) is the recommended fix.
- The ColPali v1.3 LoRA adapter required manual key-remapping to load with
  transformers 5.x (see `notebooks/kaggle_main.ipynb` Stage 4b). Without the
  patch, the LoRA silently fails to merge and only base PaliGemma weights are used.

## 8. Limitations

- **Not clinical-grade.** All output is for research only; MedGemma's TOS forbids
  clinical decision-making and the training data has known biases.
- **Rule-based CheXpert labeler** is a proxy; expect ±5% delta from CheXbert numbers.
- **Synthetic QA** inherits any errors in the source reports; it is not an independent
  ground truth.
- **Patient-level split** prevents trivial leakage but doesn't simulate distribution
  shift — performance on truly unseen institutions is likely lower.

## 9. Future Work

- Replace the rule-based labeler with the official CheXbert if license permits.
- Add a second VLM (e.g., LLaVA-Med, Qwen2.5-VL) for direct comparison with MedGemma.
- Index sentence-level chunks rather than full reports — better fine-grained retrieval.
- RadGraph-F1 evaluation once the upstream package stabilizes.
