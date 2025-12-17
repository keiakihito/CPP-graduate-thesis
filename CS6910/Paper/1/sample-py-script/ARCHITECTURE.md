# Music Recommendation System Architecture
## Comparative Analysis of Pretrained Audio Representations

---

## Executive Summary

This is a **two-stage music recommendation pipeline** that evaluates multiple pretrained audio representation models for recommendation tasks. The system extracts frozen embeddings from diverse pretrained audio models, then trains learnable recommendation models on top of them. The research compares 8+ different audio representation backends across 3 recommendation model architectures: KNN (baseline), Shallow Net (shallow embedding model with negative sampling), and BERT4Rec (masked sequence transformer).

---

## Directory Structure

```
sample-py-script/
├── extract_item_embeddings/          # Stage 1: Embedding extraction from audio
│   ├── musicnn.py                   # MusicNN embedding extractor
│   ├── mert.py                      # MERT-v1-330M embedding extractor
│   ├── m2v.py                       # Music2Vec embedding extractor
│   ├── emae.py                      # EnCodec-MAE embedding extractor
│   ├── jb.py                        # JukeMiR embedding extractor
│   ├── mfm.py                       # MusicFM embedding extractor
│   ├── *.sh                         # SLURM job submission scripts
│   └── trackid_sorted.csv           # Track ID mapping (sorted indices)
│
├── preprocess/                       # Data preparation
│   ├── 0_get_plays_pqt.py           # Maps original track IDs to sorted indices
│   └── 1_train_test_split.py        # Train/val/test split (Music4All-Onion strategy)
│
├── model.py                          # Shallow embedding model architectures
│   ├── ShallowEmbeddingModel         # Cosine similarity-based recommender
│   └── ShallowInteractionModel       # Hadamard product-based recommender
│
├── bert4rec.py                       # BERT4Rec model (masked transformer)
├── bert4rec.yaml                     # Default BERT4Rec hyperparameters
│
├── knn.py                            # KNN baseline: user embedding = item avg
│
├── dataset.py                        # Dataset classes for training
│   ├── InteractionDataset            # User-centric sampling
│   ├── InteractionDatasetItems       # Item-centric sampling
│   ├── MaskedLMDataset               # BERT4Rec training data
│   ├── MaskedLMPredictionDataset      # BERT4Rec inference data
│   └── PaddingCollateFn              # Batch padding for sequences
│
├── train.py                          # Training script for Shallow Net
├── train.sh                          # SLURM wrapper for train.py
│
├── train_bert.py                     # Training script for BERT4Rec
├── train_bert.sh                     # SLURM wrapper for train_bert.py
│
├── metrics.py                        # Metric computation (example usage)
├── table.py                          # Results aggregation script
│
├── README.md                         # High-level documentation
└── Analysis-procedure.md             # Instructions for architecture doc generation
```

---

## Pipeline Architecture

### Stage 1: Audio Representation Extraction

**Purpose:** Extract pretrained embeddings from multiple audio models without fine-tuning.

**Input:** Audio files (various sample rates, formats: MP3, WAV)

**Output:** Fixed-size embedding arrays saved as `.npy` files

**Embedding Models:**

| Model | File | Input SR | Output Dim | Framework | Notes |
|-------|------|----------|------------|-----------|-------|
| **MusicNN** | `musicnn.py` | - | 200 | musicnn library | MSD pre-trained CNN |
| **MERT** | `mert.py` | 24kHz | 1024 | HuggingFace (m-a-p/MERT-v1-330M) | Speech/music transformer |
| **Music2Vec** | `m2v.py` | 24kHz | 768 | HuggingFace (m-a-p/music2vec-v1) | Contrastive learning model |
| **EnCodec-MAE** | `emae.py` | 24kHz | 768 | encodecmae library | Audio-codec MAE model |
| **JukeMiR** | `jb.py` | 44kHz | varies | jukemirlib | Jukebox-based mirror (layer 36) |
| **MusicFM** | `mfm.py` | 24kHz | 750 | Custom (musicfm library) | Music Foundation Model |

**Key Design Decisions:**
- All embeddings are extracted from **penultimate/hidden layers** (pre-output), not final classification logits
- **Temporal aggregation:** Most models average across time dimension to get fixed-size track embeddings
- **No fine-tuning:** Item embeddings are frozen during recommendation training
- **Track ID mapping:** Original Music4All track IDs → sorted indices for numpy array indexing

**File Format:**
```
embeddings/{model_name}.npy  # Shape: (num_tracks, embedding_dim)
embeddings/trackid_sorted.csv # Mapping: index -> original_track_id
```

---

### Stage 2A: Data Preparation

**Purpose:** Convert raw user-track interactions into train/val/test sets with proper time-based splits.

#### 0_get_plays_pqt.py

**Input:** 
- Raw Music4All logs: `userid_trackid_timestamp.tsv`
- Track ID mapping: `trackid_sorted.csv`

**Processing:**
1. Read original track IDs from Music4All
2. Map to sorted indices using `trackid_to_idx` lookup
3. Chunk-wise processing to handle large files (1M rows/chunk)
4. Save mapped interactions

**Output:** `plays.pqt` (Parquet)
- Columns: `user_id`, `track_id` (original), `trackid_idx` (mapped), `timestamp`

#### 1_train_test_split.py

**Music4All-Onion Strategy:**
```
START_DATE = '2019-02-20'       # Data collection begins
TEST_DATE = '2020-02-20'        # 1 year after start → train/test split

TRAIN:  [2019-02-20 to 2020-02-19]  # Fixed historical period
VAL:    [2020-02-20 to 2020-03-20]  # First month of test period
TEST:   [2020-02-20 to 2020-03-20]  # Split 50/50 with validation
```

**Processing:**
1. Load interactions from `plays.pqt`
2. Aggregate by (user_id, track_id) to get play counts (`count`) and first play timestamp
3. Filter: Only keep users and items present in both train and test
4. 50/50 random split on test users (same items)
5. Output: Compressed interaction counts, not raw timestamps

**Output:** Three Parquet files
```
data/train.pqt  # user_id, track_id, timestamp (min), count
data/val.pqt    # Validation set (50% of test period)
data/test.pqt   # Test set (50% of test period)
```

---

### Stage 2B: Recommendation Model Training

#### Model 1: K-Nearest Neighbors (Baseline)

**File:** `knn.py`

**Algorithm:**
1. Load pretrained item embeddings: `embeddings/{model_name}.npy`
2. Compute user embeddings as **mean** of interacted items: `user_emb = mean(item_embs[user_history])`
3. Normalize both (cosine similarity)
4. Build FAISS index for fast nearest neighbor search
5. For each test/val user, retrieve k=100 nearest items, filter seen items

**Key Features:**
- Zero learnable parameters
- Single-pass inference
- Baseline for comparison

**Metrics Computed:** HitRate@50, MRR@50, Precision@50, Recall@50, NDCG@50

---

#### Model 2: Shallow Embedding Model

**File:** `model.py` → `ShallowEmbeddingModel`

**Architecture:**
```
Frozen Item Embeddings (from pretrained model)
          ↓
     Linear + ReLU    [emb_dim_in → emb_dim_out]
          ↓
  Learned Item Reps

Learnable User Embeddings (random init)
          ↓
     Linear + ReLU    [emb_dim_in → emb_dim_out]
          ↓
  Learned User Reps

Cosine Similarity: score = cos(user_rep, item_rep)
```

**Training Objective:**
```
Hinge Loss with Negative Sampling:
loss = max(0, Δ - score(user, pos_item) + score(user, neg_item))
     * confidence_weight

where Δ = 0.2 (margin)
confidence = (1 + 2 * log(1 + interaction_count))
```

**Key Features:**
- Item embeddings frozen (requires_grad=False)
- User embeddings learnable (learnable from scratch or initialized from avg)
- Negative sampling: random item per positive
- Confidence weighting: frequent interactions weighted higher
- L2 regularization available

**Hyperparameters (from train.py):**
```
model_name:          Embedding model to use (musicnn, mert, etc.)
sample_type:         'user' or 'item' (dimension to negate sample over)
neg_samples:         Number of negatives per positive (default: 20)
batch_size:          Training batch size (default: 10000)
num_epochs:          Max epochs (early stop @ patience=16)
item_freeze:         Freeze item embeddings (1/0)
user_freeze:         Freeze user embeddings (1/0)
hidden_dim:          Output dimension after linear layer
user_init:           Initialize user embeddings from avg (1/0)
use_confidence:      Apply confidence weighting (1/0)
l2:                  L2 regularization coefficient
```

**Inference:**
1. Extract normalized embeddings: user_embs, item_embs (apply model's linear layer)
2. Build FAISS index on item embeddings
3. Search for k=100 nearest items per user (filter seen items)

---

#### Model 3: BERT4Rec

**File:** `bert4rec.py` + `train_bert.py`

**Architecture:**
```
Sequential Item IDs (variable length, padded)
          ↓
    Randomly Mask [p=0.2]
          ↓
Item Embeddings (frozen from pretrained)
          ↓
Positional Embeddings (learned, pos_emb_size=max(200, seq_len))
          ↓
BERT Transformer Encoder
  - hidden_size: embedding_dim (varies by model)
  - num_layers: 2
  - num_heads: 2
  - intermediate_size: 256
          ↓
Linear Head (hidden_size → vocab_size)
          ↓
CrossEntropyLoss on masked positions
```

**Dataset Handling:**
```python
class MaskedLMDataset:
    - Max sequence length: configurable (50-300 items)
    - Masking probability: 20%
    - Special tokens: padding (vocab_size-1), mask (vocab_size-2)
    - Padding: pad_sequence(batch_first=True) with custom collate_fn
    - Attention mask: binary tensor (1 for valid, 0 for padding)

class MaskedLMPredictionDataset:
    - For inference: use last sequence position + mask token
    - Return: item history, target item
```

**Training Objective:**
```
CrossEntropyLoss(outputs.view(-1, vocab_size), labels.view(-1))
- Applied only to masked positions (ignore_value=-100)
- No negative sampling (full softmax over all items)
```

**Key Features:**
- Frozen item embeddings (can unfreeze with item_freeze=0)
- Weight tying: head.weight = item_embeddings.weight
- Masked language modeling (MLM) task
- Early stopping: patience=16
- LR scheduler: ReduceLROnPlateau

**Hyperparameters:**
```
max_seq_len:         Sequence length cap (default: 50 in yaml, up to 300)
mlm_probability:     Masking rate (default: 0.2)
item_freeze:         Freeze item embeddings
force_last_item_masking_prob: Prob to always mask last item
batch_size:          Typically larger (128-256)
hidden_dim:          Must match embedding dimension
```

**Inference:**
```
For each user:
  1. Get last n-1 items from history (capped at max_seq_len-1)
  2. Append mask token → create sequence
  3. Forward pass through transformer + head
  4. Extract logits at last sequence position
  5. Remove special tokens, take top-k
  6. Filter seen items → return recommendations
```

---

### Stage 2C: Results Aggregation & Evaluation

#### metrics.py

**Purpose:** Compute per-user recommendation metrics using `rs_metrics` library

**Metrics Computed:**
- **HitRate@k:** % of users with ≥1 relevant item in top-k
- **MRR@k:** Mean Reciprocal Rank (average of 1/rank for relevant items)
- **Precision@k:** % of top-k items that are relevant
- **Recall@k:** % of relevant items retrieved in top-k
- **NDCG@k:** Normalized Discounted Cumulative Gain

**Example Usage:**
```python
hitrate(test_df, predictions_dict, k=50)
# Returns: mean and per-user scores
```

**Outputs:** 
- Per-user metric dataframes (Parquet format)
- Confidence intervals (95%) via Student's t-distribution
- CSV exports with mean ± confidence bands

#### table.py

**Purpose:** Aggregate metrics from multiple training runs into comparison tables

**Workflow:**
1. Define model mappings: `{model_name: run_identifier}`
   - `cosine_map`: KNN baseline results
   - `shallow_map`: Shallow Net results
   - `bert_map`: BERT4Rec results
2. Load test metrics from `metrics/{run_id}_test.csv`
3. Extract mean scores (ignore confidence bounds)
4. Add model metadata (embedding dimension)
5. Sort by performance (e.g., HitRate@50)
6. Export to separate CSVs per model type

**Output Tables:**
```
cosine.csv      # KNN results
shallow.csv     # Shallow Net results
bert.csv        # BERT4Rec results
```

---

## Key Design Patterns

### 1. Embedding Freezing
- **Item embeddings** are always frozen (they're from pretrained models)
- **User embeddings** can be learnable (Shallow Net) or part of a learner (BERT4Rec)
- This is the **key differentiator**: how much do we fine-tune the representation layer?

### 2. Negative Sampling (Shallow Net)
- **Strategy:** Random negative samples during training
- **Why:** Makes loss computation efficient (hinge loss on single neg)
- **Contrast with BERT4Rec:** Full softmax → no explicit sampling needed

### 3. Temporal Aggregation
- Pretrained models extract frame-level or patch-level representations
- **Solution:** Average across time/sequence dimension to get single track vector
- Example: MERT outputs (seq_len, 1024) → (1024,) via .mean(-2)

### 4. Dataset Splits
- **Time-based validation:** Last month = test, previous month = val
- **Rationale:** Reflects real-world recommendation (future items)
- **Filtering:** Remove cold-start users/items not in training

### 5. Baseline Comparison
- **KNN:** Zero-parameter baseline using only means
- **Shallow Net:** Minimal learnable layers (good for analysis)
- **BERT4Rec:** Expressive sequential model (higher compute)

---

## Data Formats & Conventions

### Embedding Arrays
```
Shape: (num_tracks, embedding_dim)
Type: numpy float32 arrays (.npy)
Storage: embeddings/{model_name}.npy

Example sizes:
- musicnn: (56512, 200)
- mert: (56512, 1024)
- jukemir: (56512, 4800)
- random: (56512, hidden_dim)
```

### Interaction DataFrames (Parquet)
```
Columns:
- user_id: integer (LabelEncoded in training)
- track_id: original track ID (from Music4All)
- item_id: LabelEncoded track ID during training
- timestamp: interaction timestamp
- count: number of plays (aggregated)

Shape: train (N_inter, 4), val/test (M_inter, 4)
```

### Prediction Format
```
user_id | item_id
--------|--------
   0    |  [1, 5, 12, ...]  # top-100 items (exploded in CSV)
   1    |  [3, 7, 22, ...]
```

### Checkpoint Format
```
{
  'epoch': int,
  'model_state_dict': nn.Module.state_dict(),
  'optimizer_state_dict': optimizer.state_dict(),
  'loss': float
}
```

---

## Training Workflow Example

### Shallow Net
```bash
python train.py \
  musicnn          # Embedding model
  item             # Negative sampling dimension
  20               # Negatives per positive
  10000            # Batch size
  100              # Max epochs
  1                # Freeze item embeddings (always true)
  0                # Don't freeze user embeddings
  "baseline_run"   # Experiment comment
  1                # Initialize users from averages
  0                # No dynamic unfreezing
  0                # Output dimension = input dimension
  0                # No confidence weighting
  0.0              # No L2 regularization
```

**Output:**
- TensorBoard logs: `runs/musicnn-Nov01_15:30_0_20_baseline_run/`
- Predictions: `preds/{run_name}.pqt`
- Metrics: `metrics/{run_name}_val.csv`, `metrics/{run_name}_test.csv`
- Embeddings: `model_embeddings/{run_name}_users.npy`, `model_embeddings/{run_name}_items.npy`

### BERT4Rec
```bash
python train_bert.py \
  musicnn          # Embedding model
  item             # Sampling dimension (not used in BERT)
  300              # Max sequence length
  128              # Batch size
  200              # Max epochs
  1                # Freeze item embeddings
  0                # Not used
  "bert_baseline"  # Comment
  1                # Not used
  0                # Not used
  1024             # Hidden dimension (overridden by embedding dim)
  0                # Not used
  0.0              # Not used
```

**Output:** Same structure as Shallow Net

---

## Expected Results (from paper)

**Performance Hierarchy (typical):**
```
BERT4Rec ≥ Shallow Net ≥ KNN
```

**Strong Embeddings:**
- MusicNN tends to be strong baseline
- MERT and Music2Vec are competitive
- JukeMiR very high-dimensional (4800) → compute cost

**Shallow Net vs BERT4Rec:**
- Shallow Net: simpler, interpretable
- BERT4Rec: captures sequential patterns
- Gain depends on how sequential user behavior is

---

## Configuration Files

### bert4rec.yaml
```yaml
cuda_visible_devices: 0
data_path: ../data/ml-1m.txt

dataset:
  max_length: 50              # Sequence length cap
  mlm_probability: 0.2        # Masking probability
  force_last_item_masking_prob: 0

dataloader:
  batch_size: 128
  test_batch_size: 256
  num_workers: 8
  validation_size: 10000

model: BERT4Rec
model_params:
  vocab_size: 2               # vocab_size = num_items + 2 (special)
  max_position_embeddings: 200
  hidden_size: 64             # Overridden by embedding_dim
  num_hidden_layers: 2
  num_attention_heads: 2
  intermediate_size: 256

seqrec_module:
  lr: 0.001
  predict_top_k: 10           # Validation evaluation k
  filter_seen: True

trainer_params:
  max_epochs: 200

patience: 20
sampled_metrics: False
top_k_metrics: [10, 100]
```

---

## External Dependencies

### Pretrained Model Libraries
```
musicnn              # MusicNN: pip install musicnn
transformers         # BERT + Music2Vec + MERT: pip install transformers
encodecmae           # EnCodec-MAE: pip install encodecmae
jukemirlib           # JukeMiR: pip install jukemirlib
musicfm              # MusicFM: custom library
```

### ML/Data Libraries
```
torch                # PyTorch (deep learning)
torch.nn             # Neural networks
torch.optim          # Optimizers
torch.utils.data     # DataLoader
sklearn              # LabelEncoder
pandas               # DataFrames
numpy                # Arrays
faiss                # Fast similarity search (for inference)
scipy                # Statistics (confidence intervals)
```

### Utility Libraries
```
rs_metrics           # Recommendation metrics (hitrate, recall, ndcg, etc.)
tqdm                 # Progress bars
soundfile            # Audio I/O
tensorboard          # Logging
```

---

## Common Issues & Debugging

### Issue 1: Track ID Mismatch
**Symptom:** Shape mismatch when indexing embeddings
**Cause:** Not using sorted track IDs correctly
**Solution:** Always use mapping from `trackid_sorted.csv`

### Issue 2: Memory Overflow in BERT4Rec
**Symptom:** CUDA OOM with long sequences
**Cause:** max_seq_len too large, batch_size too large
**Solution:** Reduce max_seq_len (try 50-100), reduce batch_size

### Issue 3: Poor KNN Performance
**Symptom:** KNN baseline has 0% hitrate
**Cause:** Embedding dimension mismatch or wrong index type
**Solution:** Check embeddings shape, ensure FAISS index is cosine (IP on normalized vecs)

### Issue 4: BERT4Rec Overfitting
**Symptom:** Training loss decreases, val loss increases
**Cause:** Insufficient regularization, item embedding not frozen
**Solution:** Freeze items (item_freeze=1), increase l2 regularization

---

## Extending the System

### Adding New Embedding Model
1. Create `extract_item_embeddings/newmodel.py`
2. Load pretrained model from HuggingFace or library
3. For each track:
   - Load audio (adapt sample rate as needed)
   - Forward pass through model
   - Extract penultimate layer
   - Average across time (if needed)
4. Save as `embeddings/newmodel.npy`
5. Add to `table.py` mappings for evaluation

### Adding New Recommendation Model
1. Subclass `nn.Module` or define custom class
2. Accept frozen item embeddings in `__init__`
3. Implement forward pass (can use cosine, bilinear, etc.)
4. Create dataset class if needed (MaskedLMDataset for sequential)
5. Create training script with standard workflow:
   - Load embeddings & data
   - Create model, optimizer, scheduler
   - Training loop with validation
   - Save best checkpoint, extract predictions
   - Compute metrics with rs_metrics

### Modifying Loss Function
- **For Shallow Net:** Edit `hinge_loss()` function in train.py
- **For BERT4Rec:** Edit criterion (e.g., from CrossEntropyLoss to BPR)
- **Add regularization:** L2, dropout, batch norm (if applicable)

---

## Summary Table: File → Responsibility

| File | Purpose | Input | Output |
|------|---------|-------|--------|
| `extract_item_embeddings/*.py` | Extract frozen audio embeddings | Audio files | `.npy` arrays |
| `preprocess/0_get_plays_pqt.py` | Map original track IDs | Raw logs + mapping | `plays.pqt` |
| `preprocess/1_train_test_split.py` | Time-based splits | `plays.pqt` | `train/val/test.pqt` |
| `model.py` | Shallow Net architecture | - | Module definitions |
| `bert4rec.py` | BERT4Rec architecture | - | Module definitions |
| `dataset.py` | Dataset classes | DataFrames | PyTorch Tensors |
| `train.py` | Train Shallow Net | Embeddings + data | Checkpoints + metrics |
| `train_bert.py` | Train BERT4Rec | Embeddings + data | Checkpoints + metrics |
| `knn.py` | Compute KNN baseline | Embeddings + data | Metrics + predictions |
| `metrics.py` | Example metric computation | Predictions + ground truth | Metric CSVs |
| `table.py` | Aggregate results | Metric CSVs | Comparison tables |

