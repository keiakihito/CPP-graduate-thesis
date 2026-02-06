# Prototype Pipeline Plan

(File Boundary: Python Research Job + Node.js API)

## Goal

Build an end-to-end prototype that:

* Extracts **one embedding vector per track (whole track)** using **one pretrained model**
* Stores artifacts **as files** (local first, then S3)
* Runs a **Composer Retrieval proxy task** over **all tracks across all albums**
* Reports metrics:

  * **Primary:** Precision@5
  * **Auxiliary:** HitRate@5
  * **Ranking:** NDCG@5
* Enforces two critical rules:

  * **Exclude the query track itself** (`track_id` match) from search results
  * Use **cosine similarity** for nearest-neighbor search

---

## Key Design Decision

To avoid mixing runtimes, responsibilities are separated by a **file-based interface**:

* **Python (Research Job)**

  * Embedding extraction
  * ANN index build
  * Evaluation (proxy task + metrics)

* **Node.js / TypeScript (Backend API)**

  * Load artifacts
  * Serve search / demo endpoints
  * Integrate with UI and AWS services

### Runtime Boundary = Versioned Artifacts

* `embeddings.f32` (or `.npy`)
* `track_ids.json`
* `index` (`faiss.index` or `hnsw.index`)
* `eval_results.json`

Node **never recomputes embeddings**.
Python **never serves APIs**.

---

## Phase 1 — Local Prototype (Docker-friendly)

### Data Sources (Already Available)

* **S3**: audio files (`.m4a`, `.mp3`, `.wav`)
* **RDS**: metadata table
  (`album_id`, `track_id`, `composer`, `audio_s3_key`, `duration`, …)

---

### Local Storage / Artifacts Layout

Recommended directory structure:

```
artifacts/
  v1_model_<MODEL_NAME>/
    metadata/
      tracks_metadata.csv        # export from RDS (or sqlite)
    audio_cache/                 # downloaded from S3
      <album_id>/<track_id>.m4a
    audio_wav/                   # standardized wav (optional)
      <track_id>.wav
    embeddings/
      embeddings.f32             # Float32 binary (N × D)
      track_ids.json             # row index → track_id
      embeddings_meta.json       # model, dim, normalization, timestamp
    index/
      faiss.index | hnsw.index   # ANN index file
      index_meta.json            # metric, index type, build params
    eval/
      eval_results.json          # Precision@5 / HitRate@5 / NDCG@5
```

**Notes**

* `tracks_metadata.csv` is the **ground-truth label source** (composer).
* `embeddings.f32 + track_ids.json` fully define the vector space.
* Cosine similarity is implemented via **L2-normalization + inner product**.

---

## Phase 1 Checklist (Incremental Steps)

### [ ] Step 1 — Export metadata from RDS

Create `tracks_metadata.csv` with:

* `track_id`, `album_id`, `track_no`, `title`
* `composer`
* `audio_s3_key`
* `duration_sec`

Normalization:

* Ensure `track_id` uniqueness
* Standardize `composer` strings

**Deliverable**

* `artifacts/.../metadata/tracks_metadata.csv`

---

### [ ] Step 2 — Download audio from S3

For each track:

* Download `audio_s3_key`
* Save to `audio_cache/<album_id>/<track_id>.m4a`

**Deliverable**

* Local audio cache populated

---

### [ ] Step 3 — Standardize audio (recommended)

Convert to model-required format:

* WAV / mono / 16kHz (or model-specific)

**Deliverable**

* `audio_wav/<track_id>.wav`

---

## Phase 1A — Python Research Job

### [ ] Step 4 — Generate track-level embeddings

For each track:

1. Load full-track audio
2. Run **pretrained embedding model**
3. If chunk-level output → average over time
4. L2-normalize embedding
5. Append to matrix `E (N × D)`

Write:

* `embeddings.f32`
* `track_ids.json`
* `embeddings_meta.json`

---

### [ ] Step 5 — Build ANN index (cosine)

Choose one:

**FAISS**

* Inner-product index on normalized vectors
* Save as `faiss.index`

**HNSW**

* Cosine or inner-product mode
* Save as `hnsw.index`

Record:

* index type
* parameters
* similarity metric

---

### [ ] Step 6 — Proxy Task: Composer Retrieval

For each query track `q`:

* Search Top-5 across **all tracks**
* **Exclude self** (`track_id == q.track_id`)
* Relevant if `composer == q_composer`

Metrics:

* **Precision@5**
* **HitRate@5**
* **NDCG@5**

Aggregate over all tracks.

**Deliverable**

* `eval_results.json`

---

## Phase 1B — Node.js / TypeScript Backend

### [ ] Step 7 — Load artifacts and expose APIs

Node responsibilities:

* Load metadata
* Load ANN index + `track_ids.json`
* Serve retrieval endpoints

Suggested APIs:

* `GET /health`
* `GET /models`
* `GET /tracks/:trackId/nearest?k=5`
* `GET /tracks/:trackId/composer-eval?k=5`
* `GET /eval/summary`

**Guarantees**

* Always exclude self
* Cosine similarity consistent with Python job

---

## Phase 2 — AWS Integration

### [ ] Step 8 — Upload artifacts to S3

Upload:

```
s3://<bucket>/vector-artifacts/v1_model_<MODEL_NAME>/
```

---

### [ ] Step 9 — Run Node container on AWS

* ECS / Fargate (recommended)
* Download artifacts at startup
* Load into memory
* Serve APIs

---

## Phase 3 — Production-like Vector DB (Optional)

* Pinecone / Qdrant
* OpenSearch k-NN
* Aurora PostgreSQL + pgvector

Node switches from local index → managed vector DB.
Python pipeline remains unchanged.

---

## Planned Extensions

* [ ] 30s segment embeddings
* [ ] Movement-level segmentation
* [ ] Multiple model comparison
* [ ] Fine-tuning experiments (Python only)

---

## Output Artifacts (Progress Report)

* [ ] `tracks_metadata.csv`
* [ ] `embeddings.f32` / `.npy`
* [ ] `track_ids.json`
* [ ] ANN index file
* [ ] `eval_results.json`
* [ ] Sample Top-5 retrieval outputs
