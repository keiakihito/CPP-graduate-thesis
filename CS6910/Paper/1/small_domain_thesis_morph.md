# Small-Domain Thesis Plan — Morph Summary

## TL;DR
Yes — we’re **scaling the paper’s pipeline down** to a **small, domain‑specific (classical)** dataset and **changing the research emphasis** from “broad benchmarking” to **robustness under data scarcity**.

---

## 3-Point Morph (Paper → Our Project)

1) **Scope & Data**
- **Paper:** Large, multi‑genre (Music4All‑Onion) + rich user logs.
- **Ours:** Small, classical‑only CD set + lightweight user history (mocked or curated).  
  → Goal: test whether pretrained audio embeddings still work when data are scarce and homogeneous.

2) **Modeling & Compute**
- **Paper:** 6 embedding models × 3 recommenders (KNN / Shallow / BERT4Rec) → broad ranking.
- **Ours:** Start minimal: **1–2 embeddings (MusicNN, MERT)** × **KNN → Shallow**; add BERT4Rec only if time/compute allows.  
  → Goal: analyze **overfitting/generalization** and **scaling sensitivity** with small data.

3) **Evaluation Focus**
- **Paper:** Which combo wins overall?
- **Ours:** **How performance changes as data shrink** (100% → 50% → 10%), and whether rankings **flip** under small-data constraints.  
  → Deliverables: HitRate@50 / Recall@50 / NDCG@50 vs. dataset size, with clear “do’s & don’ts” for low‑resource MRS.

---

## Minimal Viable Experiment (MVE)

- **Data:** 200–500 tracks from classical CDs; create/play-count logs (mock or curated) with ≥ 20 users × ≥ 20 items/user.
- **Embeddings:** Extract **MusicNN** (low-dim) + **MERT** (high-dim). Freeze them.
- **Models:** (A) **KNN baseline** → (B) **Shallow Net** (users learnable, items frozen).
- **Scaling Test:** Train/eval on 100%, 50%, 10% of interactions.
- **Metrics:** HitRate@50, Recall@50, NDCG@50; report mean ± CI.
- **Outcome:** Plot metric vs. data size; state whether embeddings remain useful and which degrades least.

---

## What We’ll Defer (unless needed)
- Full 6‑model sweep, heavy **BERT4Rec**, and cross‑genre generalization.
- Fine‑tuning audio backends (keep them frozen to isolate representation quality).

---

## Success Criteria (Thesis‑ready)
- A clear **RQ answer**: “Pretrained embeddings (X) remain effective/robust down to ~Y% data; (Z) degrades quickly.”
- **Guidelines** for teams with tiny catalogs: which embedding to choose, which model to start with, expected metric bands.

---

## Model Architectures (2–3 sentences each)

**KNN (Content-KNN over Item Embeddings).**  
Treat each track as a point in the pretrained embedding space. For a user, aggregate vectors of positive items (e.g., average or weighted sum) and retrieve the **k** nearest items by cosine similarity; no gradient training, strong baseline for small data.

**Shallow Net (Frozen Items, Learnable Users).**  
Keep **item embeddings frozen** (from the audio backend) and learn a **user embedding** per user. Score by **cosine similarity** and train with a **hinge-margin loss** using **in-batch or sampled negatives**; this gives a light‑weight collaborative layer that adapts users to the fixed content space and resists overfitting on tiny catalogs.

**BERT4Rec (Masked Sequential Transformer with Frozen Projection).**  
Model each user’s listening history as a sequence and apply **bidirectional masking** like BERT to predict hidden items. We keep the **audio→item projection frozen** so the transformer learns sequence patterns without overfitting the content geometry; stronger, but compute‑heavier than Shallow/KNN.

---

## Training & Evaluation — Quick Reference

**Entry points**
- `train.py` → KNN (index/build & eval) and **Shallow Net** training loop.  
- `train_bert.py` → **BERT4Rec** training loop (masked‑item prediction).

**Hyperparameters at a glance**
- Optimizer: **Adam** (β1=0.9, β2=0.999), lr ∈ {1e‑3, 5e‑4}.  
- Losses: **hinge (margin m=0.2)** for Shallow; **cross‑entropy** over masked positions for BERT4Rec.  
- Negatives: **in‑batch negatives** + optional **uniform sampled** negatives per positive (n_neg ∈ {5, 10}).  
- Regularization: L2 on user embeddings (λ ∈ {1e‑4, 5e‑4}); dropout on BERT layers (p=0.1–0.3).  
- Batch sizes: Shallow (256–1024 users*items), BERT (64–128 sequences).  
- Early stopping: patience=5 on **NDCG@50** (validation).

**Metrics (Top‑N)**  
- **HitRate@50**, **Recall@50**, **NDCG@50** computed per user; report mean ± 95% CI.

**Aggregation utilities**  
- `metrics.py` → per‑batch metric updates, distributed‑safe running means.  
- `table.py` → collate runs (seeds × data‑splits), compute mean/CI, and render CSV/Markdown tables for slides.

**Scaling Protocol**  
- Train/evaluate on {**100%**, **50%**, **10%**} interaction subsets; hold item set constant.  
- Keep backend (e.g., **MusicNN**, **MERT**) **frozen** across splits to isolate small‑data effects.

