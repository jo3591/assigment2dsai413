# Ethics & Responsible Use

This system is a **research artifact** for an academic assignment. It must NOT be used
for clinical decision-making, patient triage, or any application involving real patient
care. Specific limits:

1. **MedGemma TOS.** The MedGemma model card explicitly forbids clinical decision use.
   We comply by treating outputs as research data only.
2. **Synthetic QA.** Our QA pairs are derived from MIMIC-CXR reports via an LLM, then
   filtered. Errors in the source reports propagate; the dataset is not a clinically
   validated benchmark.
3. **Rule-based CheXpert labeler.** The official CheXbert labeler is licensed and not
   distributed here. Our regex parser is a proxy with documented limits — expect ±5%
   delta from the reference labeler.
4. **De-identification.** MIMIC-CXR is already de-identified per HIPAA Safe Harbor; we
   strip residual `___` tokens during preprocessing. We do not redistribute raw
   patient images — `data/raw/` is gitignored.
5. **Fairness.** MIMIC-CXR is sourced from a single Boston-area hospital system; our
   models will show distribution shift outside this population. Reported metrics are
   in-distribution only.
6. **Accountability.** Every generated report and answer should be flagged in any UI
   with: "Research output; not for clinical decision support."
