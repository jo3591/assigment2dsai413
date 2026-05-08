# Demo Video

> Recorded with OBS Studio at 1080p; unlisted YouTube link (replace with the real one
> after recording).

**Link:** https://youtu.be/TBD-replace-after-recording

## Script (10 minutes)

| Time | Segment |
|---|---|
| 0:00 – 0:45 | Title slide + framing — chest X-ray AI as a dual-mode problem (report generation + QA). |
| 0:45 – 2:00 | Architecture diagram walk-through. ColPali for retrieval, MedGemma for generation, BiomedCLIP as baseline. |
| 2:00 – 3:30 | Live: Streamlit Page 1 — upload a CXR, run `medgemma_only` then `colpali_lora_rag`. Narrate retrieved evidence. |
| 3:30 – 5:30 | Live: Streamlit Page 2 — three QA examples (existence, location, open-ended). Show ColPali heatmap overlay. |
| 5:30 – 6:30 | Live: Streamlit Page 3 — three-way compare. Point to timing + retrieved-hit count. |
| 6:30 – 8:00 | Slides: results tables (Report, QA, Retrieval). Highlight LoRA-vs-zero-shot delta and RAG-vs-no-RAG delta. |
| 8:00 – 9:15 | QA dataset construction summary; show two validated QA pairs and a banned-term reject. |
| 9:15 – 10:00 | Limitations, ethics statement, repo + report pointer. |

## Recording Checklist

- [ ] OBS Studio canvas at 1920×1080 / 30 fps.
- [ ] Two scenes: "slides" + "Streamlit live".
- [ ] Mic test (ambient noise gate ≤ -45 dB).
- [ ] Pre-warm the Streamlit caches before recording (one full pass) so model loads don't show during the demo.
- [ ] Export MP4 H.264; upload as Unlisted to YouTube; paste link above.
