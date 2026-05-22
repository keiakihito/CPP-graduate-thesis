# Thesis Update Plan — Dr. Marin Feedback

Priority order: correctness/clarity issues first, then expansions, then new content.

---

## Priority 1 — Structural / Correctness (must fix)

### [DONE] #01 — Architecture figures
Add PANNs and MERT architecture diagrams to Section 3.2 (Model Setup).
- Added `\subsubsection{Model Architectures}` with `fig:pann_arch` and `fig:mert_arch`.

### [DONE] #10 — Add MAP@K as new metric + update tables
- Added `map_at_k()` to `metrics.py` in both production pipeline and pseudo-tag source.
- Added `map@5` to `run_retrieval_eval.py` and `run_batch_eval.py` in both pipelines.
- Re-ran evaluation on full 203-track corpus; updated all result tables in thesis and RecSys paper with MAP@5 column placed after NDCG@5.
- Updated result discussion paragraphs to reference MAP@5 alongside NDCG@5.
- Formal mathematical definitions of all metrics (NDCG, MAP, Hit, Recall, F1) still to be added to §3.4.

### [DONE] #02 — Figures and tables must be referenced before appearance
Added explicit `Figure~\ref{...}` / `Table~\ref{...}` calls before all six unreferenced items: `fig:pipeline`, `fig:va_scatter`, `tab:sanity_results`, `tab:musical_results`, `tab:latency_results`, `fig:cost_quality`.
- Applied to thesis only (RecSys paper already had inline references).

---

## Priority 2 — Justification / Clarification (strengthen rigor)

### [DONE] #06 — Clarify top-K focus and justify K=5
Added a paragraph in Section 3.4 (Evaluation Protocol) explaining that K=5 targets the candidate generation stage (top-5 ~ 2.5% of 203-track catalog), why smaller/larger cutoffs were not chosen, and that metrics assess the top portion of the ranking rather than the full list.
- Applied to thesis only; RecSys paper to follow.

### [DONE] #07 — Justify NDCG with binary relevance; was MAP@K considered?
Added paragraph in §3.4: NDCG rewards rank position even with binary labels (unlike Recall/F1); MAP@K behaves similarly but NDCG is standard in music retrieval literature. Cites Urbano et al. and Tamm et al.
- Applied to thesis; RecSys paper to follow.

### [DONE] #05 — Justify Hit@K as a primary metric
Added paragraph in §3.4: Hit@K is a binary any-relevant-in-top-K signal, robust to dense-relevance suppression (independent of |R|). Distinguishes it from NDCG's role (fine-grained rank discrimination) and Recall/F1's limitation.
- Applied to thesis; RecSys paper to follow.

### [DONE] #09 — Justify cosine similarity across differently pretrained models
Added paragraph in §3.1 (Problem Formulation): cosine similarity is magnitude-invariant and measures angular alignment (standard in content-based retrieval); it is held constant across all models so it does not confound capacity comparisons; alternative similarity functions flagged as future work.
- Applied to thesis; RecSys paper to follow.

---

## Priority 3 — Discussion Expansions (deepen analysis)

### [DONE] #08 — Justify "at least one shared label" relevance; stricter criteria
Added paragraph in §3.3 (Label Construction): at-least-one chosen to avoid relevance-empty queries with only 4 tags; stricter criteria (2+ shared labels, AV similarity thresholds) flagged as future work. Explains why broad relevance causes Recall/F1 suppression.
- Applied to both thesis and RecSys paper.

### [DONE] #04 — Discuss metadata complementing audio embeddings
Added paragraph in §5.4 (Directions for Future Work): editorial metadata (instrument, period, ensemble) as hybrid retrieval signal; study isolates audio contribution as baseline; no user data required for this extension.
- Applied to both thesis and RecSys paper.

---

## Priority 4 — New Content (requires new material)

### [DONE] #03 — Qualitative retrieval examples / case studies
Full case study deferred (insufficient user interaction data). Added a Future Work sentence in §5.4 of both documents: once user data is available, top-5 side-by-side comparisons across capacity tiers would show whether small aggregate metric differences translate to meaningfully different candidate lists.

---

## Status Tracker

| # | Description | Priority | Target section | Status |
|---|-------------|----------|----------------|--------|
| 01 | Architecture figures | P1 | §3.2 Model Setup | Done |
| 10 | MAP@K added + formal metric definitions | P1 | §3.4 Eval Protocol | Done |
| 02 | Figure/table references in text | P1 | §3–5 full pass | Done |
| 06 | Justify K=5 and top-K focus | P2 | §3.4 Eval Protocol | Done |
| 07 | Justify NDCG with binary relevance | P2 | §3.4 / Discussion | Done |
| 05 | Justify Hit@K as primary metric | P2 | §3.4 Eval Protocol | Done |
| 09 | Justify cosine similarity | P2 | §3.1 Problem Form. | Done |
| 08 | Justify at-least-one relevance | P3 | §3.3 / §5.4 | Done |
| 04 | Metadata discussion | P3 | §5.4 Future Work | Done |
| 03 | Qualitative case study → Future Work | P4 | §5.4 Future Work | Done |
