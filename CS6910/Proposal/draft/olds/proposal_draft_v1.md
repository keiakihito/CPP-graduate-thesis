# Comparative Analysis of Pretrained Audio Representations for Small-Scale Music Archives: A Case Study on the iPalpiti Collection

## ABSTRACT
This thesis investigates the application of content-based music recommendation techniques to small-scale, specialized music archives, specifically the iPalpiti Music Archive. While large streaming platforms leverage massive user interaction datasets for collaborative filtering, niche archives often face the "cold start" problem and lack sufficient user data. This research adapts the methodology of Tamm et al. (2024) [1]—who compared pretrained audio representations like MusicNN, MERT, and Jukebox—to evaluate their effectiveness in a small-domain context. By integrating these models into a modern AWS serverless architecture, this study aims to develop a cost-effective, scalable recommendation system. The project utilizes mock user data to train and compare backend models (KNN, Shallow Networks, and BERT4Rec) on frozen audio embeddings, ultimately determining the optimal configuration for enhancing music discovery in specialized libraries.

## INTRODUCTION
The digital preservation of music archives presents a unique challenge: making content discoverable without the massive user interaction data that powers commercial giants like Spotify or Apple Music. The iPalpiti Music Archive, featuring recordings by award-winning musicians from the iPalpiti Festival, is a prime example of such a specialized domain. In these contexts, traditional collaborative filtering fails due to data sparsity.

Recent advancements in Music Information Retrieval (MIR) have shown that deep learning models can extract rich semantic information directly from audio signals. The work of Tamm et al. [1] demonstrates that "frozen" embeddings from pretrained models (e.g., MusicNN, MERT) can serve as powerful features for recommendation systems, even with limited data. 

This thesis proposes to adapt these findings to the iPalpiti archive, building a hybrid recommendation engine on AWS. By leveraging serverless infrastructure (Lambda, S3, RDS), the system aims to provide real-time, content-aware recommendations that help users navigate the emotional and stylistic landscape of the archive.

## LITERATURE REVIEW

### 1. Pretrained Audio Representations
Modern MIR relies heavily on transfer learning. Models like MusicNN [2] (convolutional) and MERT (transformer-based) are pretrained on vast datasets to capture timbral, rhythmic, and harmonic features. Tamm et al. (2024) [1] conducted a comprehensive comparative analysis, showing that while large models like Jukebox capture detailed semantics, lighter models like MusicNN often suffice for recommendation tasks when computational resources are constrained. This project will evaluate a subset of these models (e.g., MusicNN, MERT, EnCodec-MAE) specifically for classical and orchestral music performance.

### 2. Deep Content-Based Recommendation using Embeddings
Recommendation systems typically fall into collaborative, content-based, or hybrid categories. For small archives with limited user interaction data, pure content-based methods are essential. Unlike traditional systems that rely on manual tags (e.g., genre, tempo), this research focuses on Deep Content-Based Recommendation. By utilizing frozen audio embeddings from large pretrained models, the system extracts rich, high-dimensional representations of the music itself. This allows for nuanced similarity comparisons that go beyond surface-level metadata, enabling the system to recommend tracks that are acoustically and semantically similar to a user's interests [1] [3].

### 3. Serverless Audio Processing
Deploying deep learning models in production requires robust infrastructure. AWS Serverless patterns (Lambda, Fargate) offer a cost-effective way to handle sporadic workloads common in academic or archive settings. This review includes architectural patterns for decoupling audio processing (feature extraction) from the user-facing API (recommendation serving).

## THEORETICAL FOUNDATIONS

### 1. The Cold Start Problem in Niche Archives
Standard recommendation algorithms, particularly Collaborative Filtering, depend on dense user-item interaction matrices to infer preferences. In specialized archives with limited traffic, these matrices are sparse, making it impossible to derive meaningful patterns—a challenge known as the Cold Start Problem. The theoretical premise of this work is that the audio signal itself provides a rich source of feature data. By projecting raw audio into a high-dimensional vector space using deep neural networks, we can quantify musical similarity mathematically (e.g., via Cosine Similarity) and generate recommendations without requiring prior user history.

### 2. Vector-Based Audio Representation and Retrieval
The core technical foundation of this system is the transformation of raw audio into high-dimensional vector embeddings. The iPalpiti music dataset is first segmented into consistent audio clips and processed through pretrained models (e.g., MusicNN, MERT) to generate dense feature vectors. These embeddings are stored in a vectorized database, enabling efficient similarity search and retrieval. Backend models leverage this latent vector space to identify semantic and acoustic relationships between tracks, while the frontend recommendation logic utilizes mock user interaction data mapped to these vectors to simulate and predict user preferences.

### 3. Sequential Audio Modeling (Content-Based)
Beyond simple similarity, this research explores the temporal aspect of music consumption. By employing sequential models like BERT4Rec [3] on top of audio embeddings, the system can model "listening sessions" as sequences of acoustic events. This allows for predicting the next most suitable track based on the flow of audio features in a session (simulated via mock data), effectively treating music recommendation as a content-driven sequence modeling task rather than just a static retrieval problem.

## SYSTEM ARCHITECTURE

The proposed system is built on Amazon Web Services (AWS), integrating with an existing audio segmentation backend.

### High-Level Architecture Diagram
![Architecture Diagram](architecture_diagram.png)

### 1. Infrastructure Layer
- Storage: Amazon S3 stores the raw audio (WAV/MP3) and the pre-computed feature vectors (embeddings).
- Database: Amazon RDS (PostgreSQL) manages metadata (Albums, Tracks) and user interaction logs.
- Compute: 
    - AWS Lambda: Handles API requests and lightweight business logic.
    - AWS SageMaker / Fargate: Hosts the heavier inference tasks for extracting embeddings from new audio tracks.

### 2. Application Layer
- RecommendationService: Acts as a Facade, providing a clean API (recommendForUser, recommendSimilar) to the frontend.
- Strategy Implementation:
    - KNN Baseline: Performs similarity search directly on the frozen audio embeddings to find nearest neighbors.
    - Shallow Network: A lightweight neural network that learns to map simulated user preferences to the embedding space.
    - Sequential Model: Deploys BERT4Rec to predict the next track in a sequence, effectively modeling the "flow" of a listening session.

### 3. Data Pipeline
1. Ingestion: Audio uploaded to S3 triggers an extraction event.
2. Feature Extraction: A worker (container) downloads the audio, runs a pretrained model (e.g., MusicNN), and saves the embedding vector.
3. Indexing: Vectors are indexed (e.g., in a vector store or FAISS index) for fast retrieval.

## RESEARCH GOAL

### Problem Statement
Small-scale music institutions possess valuable cultural assets but lack the technical resources to build personalized discovery tools. Existing state-of-the-art models are often evaluated on generic pop datasets (e.g., MTG-Jamendo) and require expensive infrastructure.

### Objective
This research aims to:

1.  Adapt the Tamm et al. methodology to the specific domain of the iPalpiti archive.

2.  Compare the performance of different audio backends (MusicNN vs. MERT) in this classical/performance-focused domain.

3.  Implement a production-ready, serverless recommendation API on AWS that demonstrates cost-efficiency and scalability.

### Research Questions
- RQ1: How do pretrained audio representations perform in recommendation tasks when trained and evaluated on small-scale or domain-specific datasets (e.g., classical music)?
- RQ2: To what extent do different pretrained models overfit or generalize when the dataset size is reduced?
- RQ3: How does dataset size influence the relative performance ranking among pretrained audio models?
- RQ4(Optional): Which aspects of audio embeddings (genre, timbre, dynamics) remain robust or degrade under small-sample fine-tuning or transfer?

## METHODOLOGY

### Phase 1: Dataset Preparation
- Audio Data: Use the iPalpiti Music Archive (digitized performances).
- Preprocessing: Convert to mono, 16kHz (or model-specific sample rate).
- Mock Data Generation: Since real user history is absent, we will generate synthetic user sessions. Virtual users will be simulated to have "preferences" for specific audio clusters (e.g., users who like high-tempo violin pieces), creating a ground truth for training.

### Phase 2: Model Implementation
We will implement three tiers of recommendation logic, all fundamentally driven by content-based embeddings:

1.  Baseline (KNN): A pure content retrieval approach using K-Nearest Neighbors on raw audio embeddings to find acoustically similar tracks.

2.  Shallow Neural Net: A lightweight model that learns to map user preferences to specific regions of the audio embedding space.

3.  Sequential (BERT4Rec): A transformer model that treats the sequence of listened tracks as a language modeling problem, using the underlying content embeddings to understand the musical progression of a session.

### Phase 3: Evaluation Strategy
The specific evaluation metrics and experimental protocols are currently under development and will be finalized in the upcoming semester. The research will focus on establishing a robust framework for assessing both the recommendation quality (using the synthetic datasets) and the system efficiency (latency and cost) within the AWS environment.

## EVALUATION ROADMAP

The detailed evaluation plan is a key objective for the next semester. The current research direction identifies the following areas for development:

1.  Algorithmic Accuracy:
    - Investigation into appropriate metrics for small-scale, specialized archives (e.g., adapting standard ranking metrics like HitRate or NDCG).
    - Development of a validation strategy to assess how well the models leverage the vector embeddings to solve the cold-start problem.

2.  System Performance:
    - Definition of benchmarks for the serverless infrastructure.
    - Planning for scalability tests to evaluate the cost-effectiveness of the proposed AWS architecture.

## REFERENCES

[1] R. Tamm, M. Sachdeva, and S. Lind, "Comparative Analysis of Pretrained Audio Representations in Music Recommender Systems," in *Proceedings of the 18th ACM Conference on Recommender Systems (RecSys '24)*, 2024.

[2] J. Pons and X. Serra, "musicnn: Pre-trained convolutional neural networks for music audio tagging," in *Proceedings of the 20th International Society for Music Information Retrieval Conference (ISMIR)*, 2019, pp. 1-7.

[3] F. Sun, J. Liu, J. Wu, C. Pei, X. Lin, W. Ou, and P. Jiang, "BERT4Rec: Sequential Recommendation with Bidirectional Encoder Representations from Transformer," in *Proceedings of the 28th ACM International Conference on Information and Knowledge Management (CIKM)*, 2019, pp. 1471–1480.
