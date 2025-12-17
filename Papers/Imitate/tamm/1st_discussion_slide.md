---
marp: true
theme: default
paginate: true
---

# Two-Stage Model Pipeline (Overview)

**Goal:** Recommend music users may like by combining content and user behavior.  

- Stage 1: Use pretrained audio models to describe songs as vectors.  
- Stage 2: Use recommender models to learn user taste from those vectors.  
- Together: A hybrid system connecting sound features and listening patterns.  

---



## Research Questions (RQ1–RQ3)

- **RQ1:** Are pretrained audio representations a viable option for music recommender systems (MRS)?  
- **RQ2:** How do different backend models compare in the context of MRS?  
- **RQ3:** How does pretrained model performance in MRS correspond to performance in traditional MIR tasks?  

*Goal for discussion:* Identify which RQ your thesis emphasizes most — model comparison, methodology, or hybrid evaluation.

---

##  Methodology Overview

- **Step 1 – Dataset Selection:** Use the **Music4All-Onion** dataset containing audio clips and user listening history.  
- **Step 2 – Representation Generation:** Feed raw audio into pretrained models → obtain content embeddings.  
- **Step 3 – Recommendation Evaluation:** Combine embeddings with user data → train and compare recommendation models.  

 Evaluation Metrics: *HitRate@50, Recall@50, NDCG@50* to measure recommendation quality.

---

## What Is Transfer Learning?

- **Definition:** Reusing a model trained for a **general task** (e.g., genre tagging) to help a **new task** (music recommendation).  
- **How:** The pretrained audio model’s weights are **frozen** but used to extract meaningful features from new songs.  
- **Example:** MusiCNN learned “what guitars and moods sound like” → that knowledge helps the recommender understand new tracks.  

💡 The conversion to embeddings isn’t the transfer itself — the **reuse of learned knowledge** is.  

---
## Stage 1 – Pretrained Audio Model (Backend)

- **Input:** Raw music audio clips (30 seconds each) from the Music4All-Onion dataset.  
- **Process:** Pass each clip through pretrained models (e.g., MusiCNN, MERT) → generate **embeddings**.  
- **Output:** Numerical vectors stored in a **vectorized database**, one per backend model.  

Each model represents raw audo in vector presentation differently → 7 versions of song embeddings for comparison.  

---

## Example: Vectorized Database (Stage 1 Output)

| Track | MusiCNN | MERT | EncodecMAE | Music2Vec | MusicFM | Jukebox |
|-------|----------|------|-------------|-------------|-------------|-------------|
| **Song A** | [0.18, 0.25, …] | [0.52, 0.71, …] | [0.09, …] | […] | […] | […] |
| **Song B** | [0.12, 0.33, …] | [0.60, 0.75, …] | [0.11, …] | […] | […] | […] |
| **...** | ... | ... | ... | ... | ... | ... |

Each column = a **different embedding space** (different way to describe music).  
These vectors are later used as inputs for recommender models in Stage 2.

---

## Stage 2 – Recommender Model (Frontend)

- **Input:** User–item interaction history + pretrained embeddings from Stage 1.  
- **Process:** Train models (KNN, Shallow Net, BERT4Rec) to learn user preferences.  
- **Output:** Ranked list of recommended tracks tailored to each listener.  

📈 The recommender iterates over the **vectorized database** to find songs closest to user taste.  

---

##  Stage 2 – Input, Process, Output (Detailed)

| Phase | Description |
|--------|--------------|
| **Input** | (a) User listening history (implicit events like play, skip, complete) <br> (b) Song embeddings from Stage 1 |
| **Process** | Recommender compares user history and item vectors to learn preference patterns using 3 architectures (KNN, Shallow Net, BERT4Rec) |
| **Output** | Top-N ranked song recommendations and performance metrics (HitRate@50, Recall@50, NDCG@50) |

💡 Each model type tests a different balance between content-based and collaborative filtering.



---

## Questoins: How we get or set up User History?

- **Option 1 – Mock Data:** Simulated logs that mimic real user behavior (plays, skips, completions).  Acceptable for thesis-scale experiments? Like public privacy issue?
- **Option 2 – Public Dataset:** Borrow real anonymized data such as **Music4All-Onion** for reproducibility and comparison.  Maybe it will not align our iPlapiti dataset?
- **Option 3 – Hybrid:** Start with mock data, then validate using a small subset of a public dataset.  



