Timeline (two semesters; adaptable week‑by‑week)

Semester 1

- Weeks 1–3: Formal problem statement; thesis proposal draft; reading list & summaries.
- Weeks 4–6: Build preprocessing & featuxjt0bqd_GWA3qke2edjre pipelines (mel/CQT); stand up RDS schema + S3 storage; write data loaders.
- Weeks 7–9: Implement baselines: handcrafted features + kNN; pretrained embedding baseline; FAISS index; minimal UI.
- Weeks 10–13: Train CNN & CRNN; run first retrieval eval; set up MLflow/W&B for tracking.
- Weeks 14–16: Add Transformer backbone; ablations (crop length, augmentations); draft Methods/Data chapters.

Semester 2

- Weeks 1–3: Hybrid re‑ranker and diversity experiments; efficiency benchmarks.
- Weeks 4–6: Generalization and robustness studies; error analysis with listening sessions.
- Weeks 7–8: User study (IRB/consent, recruit, run, analyze).
- Weeks 9–11: Write Results/Discussion; finalize figures/tables; tighten Abstract/Intro/Related Work.
- Weeks 12–14: Full thesis polish; practice defense; archive code + data cards; submit.


Agile way
Agile Thesis Timeline (Two Semesters)
Semester 1 – MVP Foundations
Sprint 0 (Week 1–2) — Kickoff

Write problem statement + proposal.

Collect small subset of audio files + metadata.

MVP0: A track plays in UI, metadata shows up.

## Sprint 1 (Weeks 3–4) — Baseline pipeline
```
Build preprocessing pipeline (mel + CQT spectrograms).
Store metadata in RDS, audio paths in S3/local.
MVP1: User picks a track → returns top-K using simple MFCC/chroma features + cosine similarity.
```

## Sprint 2 (Weeks 5–6) — Pretrained baseline
```
Integrate pretrained embeddings (OpenL3/BEATs).
Build FAISS index for fast nearest neighbor search.
MVP2: Pretrained similarity search running end-to-end, query <1s.
```
## Sprint 3 (Weeks 7–9) — Classical baselines (CNN, CRNN)
```
Train a small CNN from scratch, then CRNN.
Evaluate vs pretrained baseline.
MVP3: Side-by-side comparison UI (query → lists from CNN, CRNN, Pretrained).
```

## Sprint 4 (Weeks 10–12) — Reporting loop
```
Run first proper evaluation (precision@K, recall@K).
Start ablations (mel vs CQT).
MVP4: First results table + draft Methods/Data chapters.
```

Sprint 5 (Weeks 13–15) — Stretch goal
```
Add Transformer backbone (if compute allows).
Polish MLflow/W&B experiment tracking.
MVP5: Transformer results compared to others, draft discussion.
```

Semester 2 – Iterative Refinement & User Feedback
## Sprint 6 (Weeks 1–2) — Hybrid re-ranking
```
Implement re-ranking (MMR, metadata filters).
MVP6: UI toggle: “raw similarity” vs “hybrid re-ranked” lists.
```
## Sprint 7 (Weeks 3–4) — Generalization checks

```
Test across instrumentation/era splits.
Add robustness tests (noise, old recordings).
MVP7: Generalization results figure.
```
## Sprint 8 (Weeks 5–6) — Human loop
```
Prepare and run small user study (pairwise list preference).
Analyze results with basic stats.
MVP8: User study graph: which list users prefer.
```

## Sprint 9 (Weeks 7–8) — Consolidation
```
Clean repo, archive data splits, finalize evaluation pipeline.
MVP9: All experiments reproducible with one script/command.
```

## Sprint 10 (Weeks 9–10) — Writing integration
```
Results + Discussion chapters drafted.
Figures and tables integrated into LaTeX/Markdown.
MVP10: “Mini-thesis draft” with all methods, results, and discussion.
```

## Sprint 11 (Weeks 11–12) — Defense readiness
```
Polish intro, related work, abstract.
Practice presentations.
MVP11: Full thesis draft + 1-slide pitch deck.
```

## Sprint 12 (Weeks 13–14) — Final release
```
Submit thesis, repo archived with docs + Docker.
MVP12: Final “thesis release” = doc + reproducible code + demo.
```

## ⚡ Agile Principles applied:
- Each sprint delivers something runnable/demoable.
- You always have a working system, just improved each time.
- Regular checkpoints (end of sprint = demo to advisor).
- Early feedback → less risk of surprises in final semester.