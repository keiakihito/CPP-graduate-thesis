# Architecture & Preparation Note for Thesis Meeting

## 1. Comparing Slide Outline and Implementation (Architecture Mapping)

This section clarifies what “architecture” means and how to map your presentation slides to the actual implementation in the reference repository [Darel13712/pretrained-audio-representations](https://github.com/Darel13712/pretrained-audio-representations).

### 1.1 Three Layers of Architecture

| Layer | Description | Examples / Files |
|--------|--------------|------------------|
| **Pipeline Architecture** | Overall data flow from raw input to evaluation | Dataset split → embeddings → recommenders → metrics |
| **Model Architecture** | Inner structure of each recommender model | KNN, Shallow Net, BERT4Rec |
| **Code/Module Architecture** | Folder structure and implementation mapping | `extract_item_embeddings`, `preprocess`, `train.py`, `train_bert.py`, `metrics.py`, etc. |

### 1.2 Slide-to-Code Map

| Slide Concept | Repository Module / Script | Notes |
|----------------|-----------------------------|--------|
| Stage 1 – Pretrained Embeddings | `extract_item_embeddings/model.py`, `model.sh` | Generates `.npy` vectors for each model (MusiCNN, MERT, EncodecMAE, etc.) |
| Dataset Split | `preprocess/0_get_plays_pqt.py`, `1_train_test_split.py` | Implements Music4All-Onion split: last month = val/test; previous 12 months = train |
| Stage 2 – Recommender Models | `knn.py`, `model.py` (Shallow Net), `bert4rec.py` | Core architectures for comparison |
| Training Scripts | `train.py`, `train_bert.py`, `train.sh`, `train_bert.sh` | Controls optimizer, epochs, loss, sampling |
| Metrics / Evaluation | `metrics.py`, `table.py` | Computes HitRate@50, Recall@50, NDCG@50 |

### 1.3 Key Architectural Highlights
- **KNN:** computes user embedding as the average of listened item vectors.
- **Shallow Net:** frozen item embeddings + learnable user embeddings, cosine similarity + hinge loss, 20 negative samples.
- **BERT4Rec:** sequential transformer with masked prediction, frozen projection for item embeddings, sequence length capped at 300.

---

## 2. Exploring the Dataset and User History (Music4All-Onion)

### 2.1 Dataset Policy from Paper
- **Train/Val/Test Split:**
  - Last month (after **2020-02-20**) → validation & test (split users 50/50)
  - Previous 12 months → training
  - Remove cold users/items (ensures all appear in training)
- **Sequence Length:** Capped at 300 events per user for BERT4Rec.
- **Metrics:** HitRate@50, Recall@50, NDCG@50 (omit MRR, Precision)

### 2.2 Insights to Bring Up
- Replicate or check these split rules in code.
- Note how user–item interactions are encoded (`user_id`, `item_id`, `timestamp`).
- Validate that frozen item embeddings are used in Shallow Net/BERT4Rec training.
- Expected ordering from paper: **BERT4Rec > Shallow Net > KNN**, and **MusiCNN** tends to outperform others.

---

## 3. Spotify API and Classical CD Dataset Plan

### 3.1 Why Use Spotify API
- Enrich your small classical CD dataset with metadata (composer, performer, release year, track duration).
- Retrieve **audio features** (tempo, key, mode, loudness, timbre proxies) to compare or augment pretrained embeddings.
- Create **mock user history** logs to simulate listening interactions for cold-start testing.

### 3.2 Implementation Outline
1. Use Spotify Web API or `spotipy` library to collect metadata and audio features.
2. Generate mock user-event logs (e.g., plays, skips, completions) following the paper’s implicit-feedback format.
3. Convert to the same structure expected by `preprocess/1_train_test_split.py`.
4. Integrate with your local pipeline and test small-scale runs.

---

## 4. Checklist for Tomorrow’s Meeting

### ✅ Architecture Review
- [ ] Map slide components (Stage 1 / Stage 2 / Metrics) to code files.
- [ ] Summarize each model’s internal logic (KNN, Shallow, BERT4Rec).
- [ ] Verify evaluation metrics match the slides and paper.

### ✅ Dataset Exploration
- [ ] Confirm dataset split logic (date, user/item filtering).
- [ ] Identify where user–item logs are loaded and transformed.
- [ ] Note sequence length setting (300) and possible variation for small datasets.

### ✅ Spotify API Plan
- [ ] Identify 10–20 classical CD tracks to test enrichment.
- [ ] Prepare small mock user logs (at least 3–5 users, 20–30 interactions).
- [ ] Check integration approach: metadata → embeddings → recommender.

### ✅ Deliverables
- [ ] Create or review architecture mapping table (print or slide form).
- [ ] Write one paragraph summary on how to adapt this architecture to small data.
- [ ] Prepare 1–2 questions for advisor:
  - “Should we test model performance vs. dataset size scaling?”
  - “Do we prioritize reproducibility (Music4All-Onion) or domain adaptation (classical) first?”

---

**Author:** Keita Katsumi  
**Purpose:** Preparation for Thesis Advisor Meeting  
**Date:** October 2025

