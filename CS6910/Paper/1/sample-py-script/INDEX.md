# Documentation Index

This directory contains code for "Comparative Analysis of Pretrained Audio Representations in Music Recommender Systems" (RecSys 2024).

## Documentation Files (Generated for Your Thesis)

### Start Here
- **[QUICK_REFERENCE.md](QUICK_REFERENCE.md)** - 2-page cheat sheet with commands, hyperparameters, and expected results
- **[README.md](README.md)** - Original paper's documentation

### Deep Dives
- **[ARCHITECTURE.md](ARCHITECTURE.md)** - Comprehensive 639-line architectural guide covering:
  - Stage 1: Audio embedding extraction (6 models, 56K tracks)
  - Stage 2A: Data preparation (Music4All-Onion time-based splits)
  - Stage 2B: Three recommendation models (KNN, Shallow Net, BERT4Rec)
  - Stage 2C: Results aggregation and evaluation
  - Complete hyperparameter reference
  - Debugging guide and extension instructions

- **[PIPELINE_DIAGRAM.txt](PIPELINE_DIAGRAM.txt)** - Visual ASCII diagram of the entire pipeline from audio files to evaluation metrics

### Reference Files
- **[Analysis-procedure.md](Analysis-procedure.md)** - Notes on generating architecture documentation

---

## Code Organization

### Stage 1: Embedding Extraction
**Directory:** `extract_item_embeddings/`

Extract frozen embeddings from 6 pretrained audio models:
- `musicnn.py` - MusicNN (200D)
- `mert.py` - MERT-v1 (1024D)
- `m2v.py` - Music2Vec (768D)
- `emae.py` - EnCodec-MAE (768D)
- `jb.py` - JukeMiR (4800D)
- `mfm.py` - MusicFM (750D)

**Outputs:** `embeddings/{model_name}.npy` (shape: 56512 x embedding_dim)

### Stage 2A: Data Preparation
**Directory:** `preprocess/`

Convert raw logs to train/val/test splits:
- `0_get_plays_pqt.py` - Map original track IDs to sorted indices
- `1_train_test_split.py` - Time-based split (1 year train, split test month)

**Outputs:** `data/{train,val,test}.pqt`

### Stage 2B: Model Training
**Top level files:**

- `model.py` - Shallow embedding model class definition
- `train.py` - Training script for Shallow Net
- `bert4rec.py` - BERT4Rec model class definition
- `train_bert.py` - Training script for BERT4Rec
- `knn.py` - KNN baseline (no training needed)
- `dataset.py` - Dataset classes for PyTorch DataLoaders

**Outputs:** Predictions, metrics, checkpoints, TensorBoard logs

### Stage 2C: Results Aggregation
**Top level files:**

- `metrics.py` - Example metric computation (uses `rs_metrics`)
- `table.py` - Aggregate results across runs into comparison tables

**Outputs:** `{cosine,shallow,bert}.csv` comparison tables

---

## Quick Start Commands

### 1. Generate KNN Baseline (All Embedding Models)
```bash
cd /path/to/sample-py-script
python knn.py
# Creates metrics for musicnn, mert, music2vec, encodecmae, jukemir, musicfm
# Output: metrics/{model}_cosine_{val,test}.csv
```

### 2. Train Shallow Net (Single Embedding Model)
```bash
python train.py musicnn item 20 10000 100 1 0 "baseline" 1 0 0 0 0.0
# Args: model sample_type neg_samples batch_size epochs item_freeze user_freeze comment user_init dynamic hidden_dim use_confidence l2
```

### 3. Train BERT4Rec (Single Embedding Model)
```bash
python train_bert.py mert item 100 128 200 1 0 "baseline" 1 0 1024 0 0.0
# Args: model sample_type max_seq_len batch_size epochs item_freeze _ comment _ _ hidden_dim _ _
```

### 4. Aggregate All Results
```bash
python table.py
# Reads from metrics/ directory, outputs cosine.csv, shallow.csv, bert.csv
```

---

## Key Concepts

### Two-Stage Pipeline
1. **Stage 1:** Extract frozen embeddings from pretrained audio models
   - No fine-tuning of audio representations
   - Isolates effect of embedding quality
   
2. **Stage 2:** Train learnable recommendation models
   - KNN: Zero learnable parameters (baseline)
   - Shallow Net: Cosine similarity with learnable user embeddings
   - BERT4Rec: Masked transformer with frozen items

### Three Recommendation Architectures
- **KNN:** User embedding = average of item embeddings (FAISS search)
- **Shallow Net:** Frozen items → Linear → Cosine similarity ← Learnable users
- **BERT4Rec:** Masked LM with BERT transformer (2 layers, 2 heads)

### Dataset Strategy (Music4All-Onion)
- **Train:** Full year (2019-02-20 to 2020-02-19)
- **Validation:** 50% of last month (2020-02-20 to 2020-03-20)
- **Test:** Other 50% of last month
- **Filtering:** Remove cold-start users/items not in training

### Evaluation Metrics
- HitRate@50: % users with ≥1 relevant item in top-50
- Recall@50: % relevant items retrieved
- Precision@50: % of top-50 that are relevant
- NDCG@50: Ranking quality metric
- MRR@50: Mean reciprocal rank

---

## Directory Tree

```
sample-py-script/
├── extract_item_embeddings/       # Stage 1: Embedding extraction
│   ├── musicnn.py, mert.py, m2v.py, emae.py, jb.py, mfm.py
│   ├── *.sh                       # SLURM job scripts
│   └── trackid_sorted.csv         # Track ID mapping
│
├── preprocess/                    # Stage 2A: Data preparation
│   ├── 0_get_plays_pqt.py
│   └── 1_train_test_split.py
│
├── model.py                       # Shallow Net architecture
├── bert4rec.py                    # BERT4Rec architecture
├── dataset.py                     # Dataset classes
│
├── train.py                       # Train Shallow Net
├── train_bert.py                  # Train BERT4Rec
├── knn.py                         # KNN baseline
│
├── metrics.py                     # Stage 2C: Metric computation
├── table.py                       # Results aggregation
│
├── README.md                      # Original documentation
├── Analysis-procedure.md          # Architecture doc instructions
│
├── ARCHITECTURE.md                # (Generated) Full architecture guide
├── PIPELINE_DIAGRAM.txt           # (Generated) ASCII pipeline diagram
├── QUICK_REFERENCE.md             # (Generated) 2-page cheat sheet
├── INDEX.md                       # (This file)
│
├── bert4rec.yaml                  # BERT4Rec config (reference only)
├── train.sh                       # SLURM wrapper (optional)
└── train_bert.sh                  # SLURM wrapper (optional)
```

---

## External Dependencies

### Core ML Libraries
```
torch>=1.9.0
transformers>=4.0
sklearn
pandas
numpy
faiss-cpu  # or faiss-gpu
scipy
```

### Audio Processing
```
musicnn
soundfile
encodecmae
jukemirlib
musicfm
```

### Utilities
```
rs_metrics          # Recommendation system metrics
tqdm
tensorboard
```

---

## Common Issues & Solutions

| Problem | Solution |
|---------|----------|
| Shape mismatch with embeddings | Verify track ID mapping with `trackid_sorted.csv` |
| CUDA out of memory (BERT4Rec) | Reduce `max_seq_len` and `batch_size` |
| Poor KNN baseline | Check FAISS index type and embedding normalization |
| Missing output directories | Create `checkpoints/`, `preds/`, `metrics/`, `model_embeddings/` |
| Dependencies not found | Install with `pip install -r requirements.txt` |

---

## Expected Results (from Paper)

**Performance Ranking:**
```
BERT4Rec ≥ Shallow Net ≥ KNN
```

**Typical HitRate@50:**
- KNN: 15-25%
- Shallow Net: 25-35%
- BERT4Rec: 30-40%

**Embedding Quality:**
MusicNN ≈ MERT ≈ Music2Vec > EnCodec-MAE > MusicFM >> JukeMiR (very high-dim)

---

## For Your Thesis Meeting

### 1-Minute Summary
"We evaluate 6 audio embedding models (MusicNN, MERT, Music2Vec, EnCodec-MAE, JukeMiR, MusicFM) across 3 recommendation architectures (KNN, Shallow Net, BERT4Rec) on 56K tracks from Music4All. BERT4Rec consistently outperforms simpler models, and embedding quality significantly impacts final performance."

### 3-Minute Summary
1. Extract frozen embeddings from 6 pretrained audio models
2. Prepare Music4All dataset with time-based validation
3. Train 3 recommendation models with increasing complexity
4. Compare results across all embedding-model combinations
5. Key finding: Transformer-based BERT4Rec best captures sequential patterns

### Visual (Use PIPELINE_DIAGRAM.txt)
Shows three stages: embedding extraction, data preparation, and recommendation training

---

## Next Steps for Extending This Work

1. **Add New Embedding Models:** Create script in `extract_item_embeddings/`, output `.npy` file
2. **Try New Losses:** Modify `hinge_loss()` in `train.py` (e.g., try BPR loss)
3. **Ablation Studies:** Test `item_freeze`, `user_init`, confidence weighting
4. **Hyperparameter Sweeps:** Vary `neg_samples`, `hidden_dim`, `max_seq_len`
5. **Analyze Embeddings:** Visualize learned user/item embeddings with t-SNE or UMAP
6. **Sequence Modeling:** Extend Shallow Net with RNN/GRU layers (non-masked)

---

## References

- Original Paper: "Comparative Analysis of Pretrained Audio Representations in Music Recommender Systems" (RecSys '24)
- Dataset: Music4All-Onion
- Evaluation: `rs_metrics` library for recommendation metrics

---

## Document Generation Notes

This documentation was generated to provide comprehensive understanding of:
1. **Directory structure** - What each file does and where it fits
2. **Pipeline architecture** - How data flows from raw audio to recommendations
3. **Model details** - Exact architectures, losses, and inference procedures
4. **Configuration** - Hyperparameters, defaults, and ranges
5. **Practical guidance** - Common commands, debugging, and extension points

See ARCHITECTURE.md for the complete 600+ line detailed reference.

