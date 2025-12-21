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
    - [ ] Implement extraction of Chroma Matrices (for harmonic content) and Short-Time RMS (for dynamic range) alongside MusicNN/MERT embeddings [Shi].
    - [ ] Develop an "Audio-to-Feature" Wrapper:
        - [ ] Create a preprocessing layer that accepts raw audio and outputs Spectrograms (for MusicNN) and Raw Waveforms (for MusicNet/Custom CNN) dynamically.
        - [ ] Ensure this layer runs efficiently on AWS Lambda/Fargate to minimize inference latency [Reddy].
    - [ ] Verify vectors are correctly stored and indexed.
    - [ ] Implement a basic "Find Similar" API endpoint (KNN) to test retrieval.

- [ ] **Sprint 2.4: Thesis Writing - Methodology (Feature Extraction)**
    - [ ] Document the specific models used (versions, layers extracted).
    - [ ] Describe the vector database schema and indexing strategy (HNSW, IVFFlat).
    - [ ] Update `chapters/chapter03.tex` with implementation details.

## Phase 3: Mock Data & Recommendation Models (Weeks 7-9)
**Goal:** Simulate user behavior and implement the core recommendation algorithms.

- [ ] **Sprint 3.1: Archetypal User Simulation & Sequence Generation**
    - [ ] Implement Schedl et al.’s clustering logic to generate "Archetypal Users" (e.g., Cluster 0: Folk/Indie vs. Cluster 6: High-Energy) rather than random users.
    - [ ] Implement Abbattista’s "Personalized Popularity" logic (simulate "repeated consumption" behavior).
    - [ ] Implement "Graph-Walk" Transition Logic:
        - [ ] Build a similarity graph where edges connect songs with similar Chroma (Harmony) or BPM (Tempo).
        - [ ] Generate sessions by performing random walks on this graph to create coherent sequences for BERT4Rec learning [Abbattista].
    - [ ] Populate RDS with mock user history.

- [ ] **Sprint 3.2: Recommendation Model Implementation**
    - [ ] **Baseline:** Finalize KNN (pure content retrieval).
    - [ ] **Shallow Net:**
        - [ ] Implement Feature Fusion & Normalization:
            - [ ] Normalize Chroma vectors (0-1 range) and MusicNN embeddings (z-score) to a common scale before concatenation.
        - [ ] Implement the "Comprehensive" feature averaging strategy (Zhang) to map user history to a single vector.
    - [ ] **Sequential:** Fine-tune BERT4Rec with "Personalized Popularity Scores" (PPS) as a feature (Abbattista).

- [ ] **Sprint 3.3: Thesis Writing - Methodology (Algorithms)**
    - [ ] mathematically define the three models in `chapters/chapter03.tex`.
    - [ ] Describe the synthetic data generation process (algorithms, distribution).

## Phase 4: Evaluation & Benchmarking (Weeks 10-12)
**Goal:** Measure performance and gather results.

- [ ] **Sprint 4.1: Accuracy Metrics**
    - [ ] Implement evaluation scripts for HitRate@K, NDCG@K, and MRR.
    - [ ] Implement Intra-List Diversity (ILD) Metric:
        - [ ] Calculate the average Cosine Distance between all pairs of tracks in a recommended list to quantify "openness" [Porcaro].
    - [ ] Implement "Cold Start" Simulation, specifically testing "New Item Cold Start" (Tamm).
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
    - [ ] Discuss the "Piano vs. Orchestra" gap (Shi’s Maestro findings vs. iPalpiti dataset).
    - [ ] Analyze the impact of dataset size and whether Reddy’s "Compact" model theory holds true for iPalpiti.
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
