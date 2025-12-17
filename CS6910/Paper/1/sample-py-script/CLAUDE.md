# Architecture Guide: Pretrained Audio Representations for Music Recommendation

**Created for:** Graduate Thesis Meeting Preparation
**Project:** Comparative Analysis of Pretrained Audio Representations in Music Recommender Systems (RecSys 2024)
**Repository:** Darel13712/pretrained-audio-representations

---

## Executive Summary

This codebase implements a **two-stage music recommendation pipeline** that evaluates how well different pretrained audio embeddings work for recommendation tasks. The key insight: **freeze audio representations** (no fine-tuning) to isolate the quality of embeddings themselves.

**Core Question:** Which pretrained audio model produces the best embeddings for music recommendation?

**Answer:** MusicNN, MERT, and Music2Vec perform best; BERT4Rec architecture consistently outperforms simpler models.

---

## System Architecture Overview

### Two-Stage Pipeline

```
┌─────────────────────────────────────────────────────────────────┐
│ STAGE 1: EMBEDDING EXTRACTION (Frozen, No Fine-tuning)         │
├─────────────────────────────────────────────────────────────────┤
│ Input: 56,512 audio tracks (Music4All dataset)                 │
│ Process: Extract embeddings using 6 pretrained models          │
│ Output: embeddings/{model_name}.npy (56512 × embedding_dim)    │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│ STAGE 2A: DATA PREPARATION (Music4All-Onion Split)             │
├─────────────────────────────────────────────────────────────────┤
│ Strategy: Time-based split (1 year train, split last month)    │
│ Process: Filter cold-start users/items, cap sequences at 300   │
│ Output: data/{train,val,test}.pqt                              │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│ STAGE 2B: RECOMMENDATION MODELS (3 Architectures)              │
├─────────────────────────────────────────────────────────────────┤
│ KNN: Zero-parameter baseline (average embeddings + FAISS)      │
│ Shallow Net: Frozen items + learnable users + cosine + hinge   │
│ BERT4Rec: Masked transformer with frozen item projections      │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│ STAGE 2C: EVALUATION (Top-50 Metrics)                          │
├─────────────────────────────────────────────────────────────────┤
│ Metrics: HitRate@50, Recall@50, NDCG@50, MRR@50, Precision@50  │
│ Output: {cosine,shallow,bert}.csv comparison tables            │
└─────────────────────────────────────────────────────────────────┘
```

---

## Stage 1: Pretrained Audio Embeddings

### Six Embedding Models

| Model | Dimension | File | Description |
|-------|-----------|------|-------------|
| **MusicNN** | 200 | `extract_item_embeddings/musicnn.py` | Tagging-focused CNN |
| **MERT** | 1024 | `extract_item_embeddings/mert.py` | Music understanding transformer (v1) |
| **Music2Vec** | 768 | `extract_item_embeddings/m2v.py` | Self-supervised contrastive learning |
| **EnCodec-MAE** | 768 | `extract_item_embeddings/emae.py` | Masked autoencoder on EnCodec tokens |
| **JukeMiR** | 4800 | `extract_item_embeddings/jb.py` | Very high-dimensional (joint embedding) |
| **MusicFM** | 750 | `extract_item_embeddings/mfm.py` | Foundation model for music |

### Key Implementation Detail

**All embeddings are frozen** (not fine-tuned during recommendation training). This isolates the quality of pretrained representations from recommendation model capacity.

**Code Pattern:**
```python
# Each extraction script follows this pattern:
embeddings = extract_embeddings(audio_files)  # Shape: (56512, embedding_dim)
np.save(f'embeddings/{model_name}.npy', embeddings)
```

**Output:**
- `embeddings/musicnn.npy` (56512 × 200)
- `embeddings/mert.npy` (56512 × 1024)
- ...and 4 more files

---

## Stage 2A: Data Preparation

### Music4All-Onion Temporal Split

**Philosophy:** Time-based split (not random) to simulate real-world deployment.

| Split | Time Range | Strategy |
|-------|------------|----------|
| Train | 2019-02-20 to 2020-02-19 | Full year of listening history |
| Validation | 2020-02-20 to 2020-03-20 (first 50%) | Split last month in half |
| Test | 2020-02-20 to 2020-03-20 (last 50%) | Other half of last month |

### Data Processing Pipeline

**File: `preprocess/0_get_plays_pqt.py`**
- Maps original track IDs → sorted indices (0-56511)
- Creates `trackid_sorted.csv` for embedding alignment

**File: `preprocess/1_train_test_split.py`**
- Time-based split with Music4All-Onion strategy
- Removes cold-start users (not in train)
- Removes cold-start items (not in train)
- Caps sequence length at 300 items
- Outputs: `data/train.pqt`, `data/val.pqt`, `data/test.pqt`

**Data Format (Parquet files):**
```
user_id | item_sequence | timestamp_sequence
--------|---------------|-------------------
  123   | [45, 12, ...]  | [t1, t2, ...]
```

---

## Stage 2B: Three Recommendation Models

### 1. KNN Baseline (Zero Parameters)

**File:** `knn.py`
**Architecture:** Non-parametric nearest-neighbor search

**Algorithm:**
1. For each user, compute user embedding = mean(listened item embeddings)
2. Use FAISS to find 50 nearest items to user embedding
3. Remove items user already listened to
4. Return top-50 recommendations

**Code Signature:**
```python
python knn.py
# Runs for all 6 embedding models automatically
# Output: metrics/{model}_cosine_{val,test}.csv
```

**Why This Baseline Matters:**
- Zero learnable parameters → pure embedding quality test
- Fast inference (FAISS optimized)
- Expected HitRate@50: 15-25%

---

### 2. Shallow Net (Learnable User Embeddings)

**File:** `model.py` (class), `train.py` (training script)

**Architecture:**
```
Item Path:              User Path:
Frozen Item Embeddings  Learnable User Embeddings
       ↓                       ↓
  Linear(768 → hidden)    Direct lookup
       ↓                       ↓
   L2 Normalize            L2 Normalize
       ↓                       ↓
       └─────→ Cosine Similarity ←─────┘
                      ↓
              Hinge Loss (margin=1)
```

**Key Hyperparameters:**

| Parameter | Default | Description |
|-----------|---------|-------------|
| `neg_samples` | 20 | Negative samples per positive |
| `batch_size` | 10000 | Very large batch for contrastive learning |
| `hidden_dim` | 0 | If >0, adds linear projection (else direct cosine) |
| `item_freeze` | 1 | Always freeze items (0=unfreeze) |
| `user_freeze` | 0 | 0=learnable users, 1=freeze |
| `user_init` | 1 | Initialize users from item averages |
| `l2` | 0.0 | L2 regularization weight |

**Training Command:**
```bash
python train.py musicnn item 20 10000 100 1 0 "baseline" 1 0 0 0 0.0
#              model  sample neg batch epochs freeze comment init dynamic hidden conf l2
```

**Loss Function (from `train.py`):**
```python
def hinge_loss(pos_scores, neg_scores, margin=1.0):
    """
    pos_scores: (batch, 1) - cosine similarity with positive items
    neg_scores: (batch, neg_samples) - cosine similarity with negatives
    """
    return torch.mean(torch.clamp(margin - pos_scores + neg_scores, min=0))
```

**Expected Performance:**
- HitRate@50: 25-35%
- Better than KNN due to learnable user representations

---

### 3. BERT4Rec (Masked Sequential Transformer)

**File:** `bert4rec.py` (class), `train_bert.py` (training script)

**Architecture:**
```
Input: User item sequence (e.g., [item_1, item_2, ..., item_n])
         ↓
Randomly mask 15% of items (replace with [MASK] token)
         ↓
Add positional embeddings (learned)
         ↓
Transformer Encoder (2 layers, 2 heads, hidden=1024)
         ↓
Project to item embedding space (Linear)
         ↓
Frozen Item Embeddings Lookup
         ↓
Predict masked items (Cross-Entropy Loss)
```

**Key Implementation Details:**

1. **Masking Strategy (from `dataset.py`):**
   - 15% of items randomly masked
   - Replaced with special `[MASK]` token
   - Predict original item ID

2. **Frozen Item Integration:**
   - Item embeddings frozen (from Stage 1)
   - Transformer output projected to embedding space
   - Similarity computed with frozen embeddings for prediction

3. **Inference:**
   - Append `[MASK]` to user sequence
   - Forward pass through transformer
   - Rank items by similarity to predicted embedding

**Training Command:**
```bash
python train_bert.py mert item 100 128 200 1 0 "baseline" 1 0 1024 0 0.0
#                   model  sample max_seq batch epochs freeze _ comment _ _ hidden _ _
```

**Key Hyperparameters:**

| Parameter | Default | Description |
|-----------|---------|-------------|
| `max_seq_len` | 100 | Maximum sequence length for transformer |
| `batch_size` | 128 | Smaller than Shallow Net (GPU memory) |
| `hidden_dim` | 1024 | Transformer hidden dimension |
| `num_heads` | 2 | Multi-head attention heads |
| `num_layers` | 2 | Transformer encoder layers |

**Expected Performance:**
- HitRate@50: 30-40%
- Best model (captures sequential patterns)

---

## Stage 2C: Evaluation & Results Aggregation

### Metrics Computation

**File:** `metrics.py`
Uses `rs_metrics` library for standardized recommendation metrics.

**Five Metrics @ K=50:**

| Metric | Formula | Interpretation |
|--------|---------|----------------|
| **HitRate@50** | % users with ≥1 relevant item in top-50 | Coverage |
| **Recall@50** | % of relevant items retrieved | Sensitivity |
| **Precision@50** | % of top-50 that are relevant | Specificity |
| **NDCG@50** | Discounted cumulative gain (rank-aware) | Ranking quality |
| **MRR@50** | 1/rank of first relevant item | Early precision |

### Results Aggregation

**File:** `table.py`

**Purpose:** Aggregate metrics across all models and configurations into comparison tables.

**Outputs:**
- `cosine.csv` - KNN baseline results (all 6 embeddings)
- `shallow.csv` - Shallow Net results (all 6 embeddings)
- `bert.csv` - BERT4Rec results (all 6 embeddings)

**Usage:**
```bash
python table.py
# Reads from metrics/ directory
# Outputs CSV comparison tables
```

---

## Slide-to-Code Map

For your thesis presentation, map each slide topic to specific code files:

| Slide Topic | Code Files | Line References |
|-------------|------------|-----------------|
| **Pipeline Overview** | `INDEX.md`, `PIPELINE_DIAGRAM.txt` | — |
| **Stage 1: Embeddings** | `extract_item_embeddings/*.py` | Each script ~100-200 lines |
| **Data Split Strategy** | `preprocess/1_train_test_split.py` | Lines defining time ranges |
| **KNN Baseline** | `knn.py` | FAISS search logic |
| **Shallow Net Architecture** | `model.py:1-150` | Class definition |
| **Shallow Net Training** | `train.py:1-300` | Training loop + hinge loss |
| **BERT4Rec Architecture** | `bert4rec.py:1-200` | Transformer + masking |
| **BERT4Rec Training** | `train_bert.py:1-250` | Training loop + masking strategy |
| **Evaluation Metrics** | `metrics.py`, `table.py` | Metric computation |
| **Results Comparison** | `{cosine,shallow,bert}.csv` | Output tables |

---

## Expected Results & Sanity Checks

### Performance Hierarchy (from Paper)

```
BERT4Rec ≥ Shallow Net ≥ KNN
```

### Typical HitRate@50 Ranges

| Model | Expected HitRate@50 |
|-------|---------------------|
| KNN | 15-25% |
| Shallow Net | 25-35% |
| BERT4Rec | 30-40% |

### Embedding Quality Ranking

```
MusicNN ≈ MERT ≈ Music2Vec > EnCodec-MAE > MusicFM >> JukeMiR
```

**Why JukeMiR Underperforms:**
- Very high dimensionality (4800D) → curse of dimensionality
- Requires different normalization or distance metric

---

## Extending to Your Dataset (Small Classical CD Collection)

### Steps to Substitute Your Data

1. **Replace Audio Files:**
   - Place your classical CD audio files in a directory
   - Create a track ID mapping (similar to `trackid_sorted.csv`)

2. **Extract Embeddings:**
   - Run `extract_item_embeddings/musicnn.py` (or other models) on your audio
   - Output: `embeddings/your_model.npy` (shape: num_tracks × embedding_dim)

3. **Generate Mock User Logs:**
   - Create synthetic listening history with timestamp-based structure
   - Format: `user_id, item_id, timestamp`
   - Follow Music4All-Onion temporal split strategy

4. **Preprocess:**
   - Modify `preprocess/1_train_test_split.py` to use your data
   - Adjust time ranges for train/val/test splits

5. **Train Models:**
   - Run `knn.py`, `train.py`, `train_bert.py` with your embeddings
   - Use same hyperparameters as baseline

6. **Enrich with Spotify Metadata:**
   - If available, join Spotify features (tempo, valence, etc.)
   - Could create hybrid embeddings: [audio_embedding, metadata_features]

---

## Quick Reference: Common Commands

### Full Reproduction Pipeline

```bash
# 1. Extract embeddings (choose one model)
cd extract_item_embeddings
python musicnn.py  # or mert.py, m2v.py, etc.

# 2. Preprocess data
cd ../preprocess
python 0_get_plays_pqt.py
python 1_train_test_split.py

# 3. Run KNN baseline
cd ..
python knn.py

# 4. Train Shallow Net
python train.py musicnn item 20 10000 100 1 0 "baseline" 1 0 0 0 0.0

# 5. Train BERT4Rec
python train_bert.py musicnn item 100 128 200 1 0 "baseline" 1 0 1024 0 0.0

# 6. Aggregate results
python table.py
```

---

## Architecture Deep-Dive Checklist

For your advisor meeting, ensure you can explain:

- [ ] **Why freeze embeddings?** → Isolate embedding quality from model capacity
- [ ] **Why time-based split?** → Simulate real-world deployment (not random)
- [ ] **Why hinge loss in Shallow Net?** → Margin-based contrastive learning
- [ ] **Why BERT4Rec masks items?** → Self-supervised sequential learning
- [ ] **Why KNN uses FAISS?** → Efficient approximate nearest-neighbor search
- [ ] **Why large batch size in Shallow Net?** → More negatives per positive sample
- [ ] **Why expected BERT4Rec > Shallow Net > KNN?** → More complex model captures more patterns
- [ ] **Why normalize embeddings?** → Cosine similarity requires unit vectors
- [ ] **Why remove cold-start users/items?** → Can't evaluate what's not in training

---

## Key Findings Summary (1 Slide)

**Research Question:**
Which pretrained audio representation works best for music recommendation?

**Method:**
Two-stage pipeline with frozen embeddings across 3 recommendation architectures

**Results:**
1. **Model Ranking:** BERT4Rec > Shallow Net > KNN (as expected)
2. **Embedding Ranking:** MusicNN ≈ MERT ≈ Music2Vec (top performers)
3. **Key Insight:** Embedding quality matters more than model complexity (gap between KNN variants larger than gap between models)

**Implications:**
- Pretrained audio models transfer well to recommendation tasks
- No fine-tuning needed (frozen embeddings sufficient)
- Sequential modeling (BERT4Rec) captures temporal patterns best

---

## Additional Documentation Files

For more detailed information:

- **[QUICK_REFERENCE.md](QUICK_REFERENCE.md)** - 2-page cheat sheet with commands and hyperparameters
- **[ARCHITECTURE.md](ARCHITECTURE.md)** - 639-line comprehensive technical reference
- **[PIPELINE_DIAGRAM.txt](PIPELINE_DIAGRAM.txt)** - ASCII visual diagram of entire pipeline
- **[INDEX.md](INDEX.md)** - Navigation hub and quick start guide

---

## Contact & References

**Original Paper:** "Comparative Analysis of Pretrained Audio Representations in Music Recommender Systems" (RecSys 2024)
**Repository:** Darel13712/pretrained-audio-representations
**Dataset:** Music4All-Onion (56,512 tracks, listening logs)

---

*This architecture guide was generated to support graduate thesis meeting preparation. Focus on understanding the two-stage pipeline flow: audio → embeddings → recommendations → evaluation.*
