# Quick Reference Guide

## One-Page Overview

**Repository:** Comparative Analysis of Pretrained Audio Representations in Music Recommender Systems (RecSys '24)

**Goal:** Evaluate 6+ audio embedding models (MusicNN, MERT, Music2Vec, EnCodec-MAE, JukeMiR, MusicFM) across 3 recommendation architectures (KNN, Shallow Net, BERT4Rec).

---

## File Locations Quick Map

```
extract_item_embeddings/     → Audio → Embeddings (Stage 1)
preprocess/                  → Raw logs → Train/Val/Test (Stage 2A)
model.py                     → Shallow Net architecture
bert4rec.py                  → BERT4Rec architecture
train.py                     → Shallow Net training
train_bert.py                → BERT4Rec training
knn.py                       → KNN baseline evaluation
metrics.py, table.py         → Results aggregation
```

---

## Stage 1: Embedding Extraction

| Model | File | Dim | SR | Framework |
|-------|------|-----|----|----|
| MusicNN | musicnn.py | 200 | auto | musicnn lib |
| MERT | mert.py | 1024 | 24kHz | HF transformers |
| Music2Vec | m2v.py | 768 | 24kHz | HF transformers |
| EnCodec-MAE | emae.py | 768 | 24kHz | encodecmae |
| JukeMiR | jb.py | 4800 | 44kHz | jukemirlib |
| MusicFM | mfm.py | 750 | 24kHz | musicfm |

**Output:** `embeddings/{model}.npy` shape (56512, D)

---

## Stage 2A: Data Pipeline

```
Raw Logs (Music4All)
    ↓ [0_get_plays_pqt.py]
plays.pqt (all interactions, mapped)
    ↓ [1_train_test_split.py]
    ├─ train.pqt (1 year)
    ├─ val.pqt (50% last month)
    └─ test.pqt (50% last month)
```

**Dates:** Train: 2019-02-20 to 2020-02-19 | Test: 2020-02-20 to 2020-03-20

---

## Stage 2B: Three Training Paths

### Path A: KNN (Zero Parameters)
```bash
python knn.py
# user_emb = mean(item_embs[history])
# No training, just inference
```
**Time:** Minutes | **Space:** Minimal

### Path B: Shallow Net
```bash
python train.py [model] [args...]
# Architecture: Frozen Items → Linear → Cosine Similarity (Learnable Users)
# Loss: Hinge with negative sampling
# Parameters: ~user_count * hidden_dim
```
**Time:** Hours | **Space:** GB (small)

### Path C: BERT4Rec
```bash
python train_bert.py [model] [args...]
# Architecture: Masked transformer with frozen item embeddings
# Loss: CrossEntropyLoss on masked positions
# Parameters: ~transformer weights (2 layers, 2 heads)
```
**Time:** Hours-Days | **Space:** GB (medium)

---

## Recommended Hyperparameters

### For Starting Point (Shallow Net)
```python
model_name = "musicnn"      # Strong baseline
sample_type = "item"        # Item-centric sampling
neg_samples = 20            # Hinge loss negatives
batch_size = 10000          # Large batches
hidden_dim = 128            # Output projection dim
item_freeze = 1             # Always freeze items
user_freeze = 0             # Learn user embeddings
use_confidence = 1          # Weight by play count
l2 = 0.01                   # Regularization
```

### For Starting Point (BERT4Rec)
```python
model_name = "musicnn"
max_seq_len = 100           # Moderate sequence length
batch_size = 128            # Batch size
item_freeze = 1             # Always freeze items
mlm_probability = 0.2       # Standard masking
```

---

## Output File Locations

After training:
```
preds/{run_name}.pqt           # User-item recommendations
metrics/{run_name}_val.csv     # Validation metrics
metrics/{run_name}_test.csv    # Test metrics
runs/{run_name}/               # TensorBoard logs
checkpoints/{model}/           # Model checkpoints
model_embeddings/{run_name}_*  # Learned embeddings
```

---

## Metric Definitions

| Metric | Formula | Interpretation |
|--------|---------|-----------------|
| **HitRate@50** | % users with ≥1 relevant in top-50 | Coverage |
| **Recall@50** | % relevant items retrieved in top-50 | Completeness |
| **Precision@50** | % of top-50 that are relevant | Exactness |
| **NDCG@50** | Discounted cumulative gain (normalized) | Ranking quality |
| **MRR@50** | Mean reciprocal rank of first hit | Ranking of first match |

---

## Common Command Examples

### Generate KNN Baseline
```bash
python knn.py
# Outputs: metrics/musicnn_cosine_test.csv, etc.
```

### Train Shallow Net (MusicNN)
```bash
python train.py musicnn item 20 10000 100 1 0 "baseline" 1 0 0 0 0.0
# Args: model sample_type neg batch epochs item_freeze user_freeze comment user_init dynamic hidden use_conf l2
```

### Train BERT4Rec (MERT)
```bash
python train_bert.py mert item 100 128 200 1 0 "bert_baseline" 1 0 1024 0 0.0
# Args: model sample_type max_seq batch epochs item_freeze _ comment _ _ hidden _ _
```

### Aggregate Results
```bash
python table.py
# Outputs: cosine.csv, shallow.csv, bert.csv
```

---

## Debugging Checklist

- [ ] Embeddings shape correct? `embeddings/{model}.npy` should be (56512, D)
- [ ] Data files exist? `data/train.pqt`, `data/val.pqt`, `data/test.pqt`
- [ ] Track ID mapping correct? Check `trackid_sorted.csv`
- [ ] CUDA available? `torch.cuda.is_available()`
- [ ] Dependencies installed? `transformers`, `torch`, `faiss`, `rs_metrics`
- [ ] Output directories exist? Create `checkpoints/`, `preds/`, `metrics/`, `model_embeddings/`

---

## Performance Expectations

**Expected Ranking:**
```
BERT4Rec (best) ≥ Shallow Net (medium) ≥ KNN (baseline)
```

**HitRate@50 Range:**
- KNN baseline: 15-25%
- Shallow Net: 25-35%
- BERT4Rec: 30-40%

(Varies significantly with embedding model quality)

---

## Architecture in 3 Sentences

1. **Stage 1:** Extract fixed-size frozen embeddings from 6 pretrained audio models (no fine-tuning).
2. **Stage 2A:** Prepare user-track interaction data with time-based train/val/test splits (Music4All-Onion).
3. **Stage 2B:** Train 3 different recommendation models on top of embeddings: KNN (zero params), Shallow Net (minimal learnable layers), BERT4Rec (transformer).

---

## Key Insight: Why This Design?

- **Frozen embeddings:** Isolate the effect of embedding quality from model architecture
- **Three models:** Range from simple (KNN) to expressive (BERT4Rec) = understand model contribution
- **Multiple embeddings:** Comprehensive evaluation of audio representation methods
- **Ablations:** Control item/user freezing to understand what can be learned

---

## For Your Thesis Meeting

**Slide 1 - Overview:**
"We evaluate 6 audio embedding methods (MusicNN, MERT, Music2Vec, EnCodec-MAE, JukeMiR, MusicFM) on music recommendation using 3 model architectures (KNN baseline, Shallow Net with negative sampling, BERT4Rec transformer)."

**Slide 2 - Pipeline:**
"Audio → Pretrained Embeddings (frozen) → Train/Val/Test split → Train recommendation models → Aggregate metrics across embedding methods."

**Slide 3 - Key Results:**
"BERT4Rec outperforms Shallow Net which outperforms KNN. MERT and Music2Vec are competitive high-dim embeddings. MusicNN is strong low-dim baseline."

---

## Important Implementation Notes

1. **Track ID Mapping:** Original Music4All IDs must be mapped to sorted indices for numpy indexing
2. **Embedding Freezing:** Item embeddings are always frozen; this is the paper's key design choice
3. **Negative Sampling:** Shallow Net uses random sampling (efficient); BERT4Rec uses full softmax (no sampling)
4. **Sequence Length:** BERT4Rec has max length cap (50-300); Shallow Net has no sequence dependency
5. **Confidence Weighting:** Optional log-scaled weighting by interaction frequency

---

## Files to Keep Untouched

- `extract_item_embeddings/*.py` - Don't modify unless adding new embedding model
- `preprocess/*.py` - Don't modify unless changing data split strategy
- `bert4rec.yaml` - Only for reference; most params overridden in train_bert.py

---

## Files You'll Likely Edit

- `train.py` - Modify loss function, add regularization
- `train_bert.py` - Change BERT architecture, add techniques
- `model.py` - Add new shallow model variants
- `metrics.py` - Add new metrics or change k values

---

## Size & Compute Estimates

| Stage | GPU Memory | CPU | Time |
|-------|-----------|-----|------|
| Embedding extraction | 2-8GB | 2-4 cores | 2-24 hours (parallel) |
| Data prep | <1GB | 1 core | ~10 minutes |
| KNN | 2GB | - | <1 minute |
| Shallow Net | 4GB | 2 cores | 2-4 hours |
| BERT4Rec | 8-16GB | 4 cores | 4-12 hours |
| Results aggregation | <1GB | 1 core | <1 minute |

---

## Next Steps for Research

1. Try different embedding models (already extracted)
2. Ablate: item_freeze on/off, user_init on/off
3. Sweep hyperparameters: neg_samples, hidden_dim, batch_size
4. Add new recommendation models in model.py
5. Modify loss functions (hinge vs. BPR vs. InfoNCE)
6. Analyze learned embeddings vs. audio embeddings

