# Thesis Implementation & Drafting Sprint Plan

This roadmap outlines the incremental steps to transition from the current **Thesis Proposal** to the **Final Thesis Draft**. It integrates the technical implementation (AWS Web App, Models) with the writing process, ensuring that the thesis document evolves alongside the software.

## Phase 1: Infrastructure & Data Ingestion (Weeks 1-3)
**Goal:** Establish the AWS pipeline to ingest and segment audio from the iPalpiti archive.

- [x] **Sprint 1.1: AWS Environment Setup**
    - [x] Set up AWS Organization/Account structure.
    - [x] Configure IAM roles and permissions for Lambda, S3, and RDS.
    - [x] Provision S3 buckets for `raw-audio` and `segmented-audio`.
    - [x] Provision RDS (PostgreSQL) instance for metadata.

- [ ] **Sprint 1.2: Audio Ingestion Pipeline**
    - [x] Develop Lambda function triggered by S3 upload to valid audio formats.
    - [ ] Implement audio segmentation logic (e.g., slicing into 30s clips) using `ffmpeg` layer in Lambda or Fargate.
    - [ ] Store segmented clips in `segmented-audio` bucket.
    - [ ] Update RDS with track metadata (ID, original file, segment timestamp).

- [ ] **Sprint 1.3: Thesis Writing - Methodology (System Architecture)**
    - [ ] Document the AWS architecture decisions in `chapters/chapter03.tex`.
    - [ ] Describe the segmentation strategy and parameters (window size, hop length).
    - [ ] Create detailed diagrams of the ingestion pipeline.

## Phase 2: Feature Extraction & Vector Database (Weeks 4-6)
**Goal:** Transform audio segments into vector embeddings using pretrained models.

- [ ] **Sprint 2.1: Model Serving Infrastructure**
    - [ ] Select compute environment for inference (Lambda vs. SageMaker vs. Fargate).
    - [ ] Containerize the feature extraction worker (Python + PyTorch/TensorFlow).
    - [ ] Integrate **MusicNN** and **MERT** models into the container.

- [ ] **Sprint 2.2: Vector Database Setup**
    - [ ] Set up Vector Store (e.g., Pinecone, Milvus, or `pgvector` on RDS).
    - [ ] Define vector schema (embedding, track_id, model_version).

- [ ] **Sprint 2.3: Extraction Pipeline Execution**
    - [ ] Run the full iPalpiti archive through the extraction pipeline.
    - [ ] Verify vectors are correctly stored and indexed.
    - [ ] Implement a basic "Find Similar" API endpoint (KNN) to test retrieval.

- [ ] **Sprint 2.4: Thesis Writing - Methodology (Feature Extraction)**
    - [ ] Document the specific models used (versions, layers extracted).
    - [ ] Describe the vector database schema and indexing strategy (HNSW, IVFFlat).
    - [ ] Update `chapters/chapter03.tex` with implementation details.

## Phase 3: Mock Data & Recommendation Models (Weeks 7-9)
**Goal:** Simulate user behavior and implement the core recommendation algorithms.

- [ ] **Sprint 3.1: Synthetic Data Generation**
    - [ ] Define "User Archetypes" (e.g., "Violin Lover", "High Tempo Fan").
    - [ ] Write scripts to generate synthetic interaction logs (listening sessions) based on these archetypes.
    - [ ] Populate RDS with mock user history.

- [ ] **Sprint 3.2: Recommendation Model Implementation**
    - [ ] **Baseline:** Finalize KNN (pure content retrieval).
    - [ ] **Shallow Net:** Train a simple MLP to predict user preference vectors.
    - [ ] **Sequential:** Implement/Fine-tune BERT4Rec on the synthetic session data.

- [ ] **Sprint 3.3: Thesis Writing - Methodology (Algorithms)**
    - [ ] mathematically define the three models in `chapters/chapter03.tex`.
    - [ ] Describe the synthetic data generation process (algorithms, distribution).

## Phase 4: Evaluation & Benchmarking (Weeks 10-12)
**Goal:** Measure performance and gather results.

- [ ] **Sprint 4.1: Accuracy Metrics**
    - [ ] Implement evaluation scripts for HitRate@K, NDCG@K, and MRR.
    - [ ] Run benchmarks for all three models across different dataset sizes (RQ1, RQ3).

- [ ] **Sprint 4.2: System Metrics**
    - [ ] Measure latency (P95, P99) of the recommendation API on AWS Lambda.
    - [ ] Estimate cost per 1000 requests.

- [ ] **Sprint 4.3: Thesis Writing - Results (Evaluation)**
    - [ ] Rename `chapters/chapter04.tex` to "Experimental Results".
    - [ ] Generate tables comparing Model Accuracy (RQ1, RQ3).
    - [ ] Generate graphs for System Latency and Cost.
    - [ ] Analyze Overfitting/Generalization results (RQ2).

## Phase 5: Discussion & Final Polish (Weeks 13-14)
**Goal:** Interpret results and finalize the document.

- [ ] **Sprint 5.1: Discussion & Conclusion**
    - [ ] Create `chapters/chapter05.tex` (Discussion).
    - [ ] Interpret why certain models performed better (e.g., "MusicNN captured timbre better than MERT...").
    - [ ] Address RQ4 (Robustness) with qualitative analysis.
    - [ ] Create `chapters/chapter06.tex` (Conclusion & Future Work).

- [ ] **Sprint 5.2: Final Review**
    - [ ] Switch `main.tex` back to the official Thesis Title Page (uncomment lines).
    - [ ] Ensure formatting meets Cal Poly guidelines.
    - [ ] Final proofread.

## Phase 6: Defense Preparation
**Goal:** Prepare presentation materials.

- [ ] **Sprint 6.1: Slide Deck**
    - [ ] Create slides summarizing Intro, Architecture, Results, and Demo.
- [ ] **Sprint 6.2: Demo Polish**
    - [ ] Ensure the Web App UI is presentable for a live demo (optional but recommended).
