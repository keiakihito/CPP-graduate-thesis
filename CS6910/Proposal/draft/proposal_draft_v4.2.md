Thesis Proposal


Topics

Comparative Analysis of Pretrained Audio Representations for Small-Scale Music Archives: A Case Study on the iPalpiti Collection


|                          | Printed | Signed | Date |
|--------------------------|---------|--------|------|
| Thesis Advisor           |         |        |      |
| Possible Committee Member|         |        |      |
| Possible Committee Member|         |        |      |


# ABSTRACT
This thesis investigates the application of content-based music recommendation techniques to small-scale, specialized music archives, specifically the iPalpiti Music Archive. While large streaming platforms leverage massive user interaction datasets for collaborative filtering, niche archives often face the "cold start" problem and lack sufficient user data. This research adapts the methodology of Tamm et al. (2024) [1]—who compared pretrained audio representations like MusicNN, MERT, and Jukebox—to evaluate their effectiveness in a small-domain context. By integrating these models into a modern AWS serverless architecture, this study aims to develop a cost-effective, scalable recommendation system. The project utilizes mock user data to train and compare backend models (KNN, Shallow Networks, and BERT4Rec) on frozen audio embeddings, ultimately determining the optimal configuration for enhancing music discovery in specialized libraries.

# INTRODUCTION
The digital preservation of music archives presents a unique challenge: making content discoverable without the massive user interaction data that powers commercial giants like Spotify or Apple Music. The iPalpiti Music Archive, featuring recordings by award-winning musicians from the iPalpiti Festival, is a prime example of such a specialized domain. In these contexts, traditional collaborative filtering fails due to data sparsity.

Recent advancements in Music Information Retrieval (MIR) have shown that deep learning models can extract rich semantic information directly from audio signals. The work of Tamm et al. [1] demonstrates that "frozen" embeddings from pretrained models (e.g., MusicNN, MERT) can serve as powerful features for recommendation systems, even with limited data.

This thesis proposes to adapt these findings to the iPalpiti archive, building a hybrid recommendation engine on AWS. By leveraging serverless infrastructure (Lambda, S3, RDS), the system aims to provide real-time, content-aware recommendations that help users navigate the emotional and stylistic landscape of the archive.

# LITERATURE REVIEW

The domain of music recommendation has historically been dominated by Collaborative Filtering (CF) approaches, which rely on dense user-interaction matrices. However, specialized archives such as the iPalpiti Music Archive typically lack the massive volume of user data required for these methods, leading to the "Cold Start" problem. This literature review synthesizes twelve key research contributions that collectively argue for a Deep Content-Based approach. By leveraging pretrained audio representations, sequential modeling, and efficient neural architectures, it is possible to build high-performance recommendation systems that rely on the audio signal itself rather than historical usage data.

Introduction: Thesis Objectives and Research Questions

This thesis addresses four research questions that guide the development and evaluation of content-based music recommendation systems for small-scale, specialized archives:

- RQ1: How do pretrained audio representations perform in recommendation tasks when trained and evaluated on small-scale or domain-specific datasets (e.g., classical music)?
- RQ2: To what extent do different pretrained models overfit or generalize when the dataset size is reduced?
- RQ3: How does dataset size influence the relative performance ranking among pretrained audio models?
- RQ4 (Optional): Which aspects of audio embeddings (genre, timbre, dynamics) remain robust or degrade under small-sample fine-tuning or transfer?

To answer these questions, the methodology consists of three phases:

1.	Dataset Preparation: Process the iPalpiti Music Archive and generate synthetic user sessions with simulated preferences (e.g., "Fast Tempo Violin Enthusiast" vs. "Orchestral Purist") to create ground truth for training without real user data.
2.	Model Implementation: Compare three recommendation approaches—KNN baseline (pure content retrieval), Shallow Neural Network (user preference mapping to embedding space), and Sequential Model (BERT4Rec for session modeling)—all fundamentally driven by frozen audio embeddings.
3.	Evaluation: Develop metrics to assess both algorithmic accuracy (HitRate, NDCG adapted for small-scale archives) and system performance (AWS serverless infrastructure cost-efficiency and scalability), with emphasis on understanding how dataset size affects model performance and generalization.

## 1. Pretrained Audio Representations and Feature Extraction
The core technical foundation of this thesis rests on the ability to extract meaningful semantic features directly from raw audio waveforms or spectrograms. Tamm et al. (2024) [1] provide the primary methodological framework for this research. Their comparative analysis of six pretrained models—MusiCNN, MERT, Jukebox, MusicFM, Music2Vec, and EncodecMAE—demonstrates that "frozen" embeddings from these models can effectively drive recommendation tasks. Crucially, MusiCNN (a lighter, supervised discriminative model trained on auto-tagging) achieved the best performance across all recommendation methods (HitRate@50: 0.385), significantly outperforming larger generative models like Jukebox (4800-dimensional embeddings). This validates the choice of efficient, discriminative backends for an AWS Lambda-based architecture.

### 1.1 Compact Architectures for Resource-Constrained Deployment
Pourmoazemi and Maleki (2024) [2] address the "Continuity Problem" in music streaming by proposing a Compact Convolutional Transformer (CCT) architecture for genre-based recommendation. Their hybrid model combines six convolutional layers (filters: 32, 64, 128) for local feature extraction from mel-spectrograms, followed by two transformer encoders with multi-head attention (2 heads, 128-dimensional). The CCT achieves 93.75% test accuracy on the GTZAN dataset while containing only 454,187 parameters—significantly fewer than state-of-the-art CRNN models. For recommendation, they use cosine similarity between feature maps, demonstrating that learned representations effectively capture genre-specific patterns for content-based music retrieval.

### 1.2 Transfer Learning and Cross-Validation (Pei-Chun Lin et al., 2024)
P.-C. Lin, Yu, and Odle (2024) [3] demonstrate the efficacy of repurposing vision models for audio tasks. In their work on the GTZAN dataset, they utilized a TL-CNN-GRU architecture, freezing the pre-trained weights of MobileNetV2 for spatial feature extraction while using a bidirectional GRU to capture temporal dependencies. Their application of 10-Fold Cross-Validation (10-FCV) was crucial for small datasets, raising accuracy from 55% to 71% and significantly improving F1-scores for specific genres like classical (0.92). This validates the use of transfer learning to mitigate data scarcity in specialized archives.

### 1.3 Self-Supervised Learning for Unlabeled Data
Ramos et al. [4] explored Self-Supervised Learning (SSL) using the Audio Spectrogram Transformer (AST) within a SimCLR framework. Training on the Free Music Archive (FMA) dataset with InfoNCE contrastive loss, they demonstrated that music embeddings can be learned without explicit labels, organizing tracks by composition, timbre, and flow rather than conventional genre classifications. Their qualitative evaluation showed that the trained model achieved 48% satisfactory recommendations compared to only 10% for the untrained baseline, with learned representations capturing "subtle elements of musical structure" beyond obvious metadata.

### 1.4 Edge Deployment and In-Model Featurization
Reddy et al. [5] developed MusicNet, a compact CNN for real-time background music detection optimized for edge deployment. MusicNet achieves 81.3% TPR at 0.1% FPR while being only 0.2 MB in size—10x smaller than competing models—with 11.1ms inference time (4x faster than best-performing alternatives). MusicNet incorporates in-model featurization, processing raw audio directly without requiring external feature extraction, simplifying deployment and maintenance in production systems. Together with Pourmoazemi and Maleki's [2] CCT, these papers contribute to the project's goal of cost-efficiency and scalability (RQ3).

### 1.5 Domain-Specific Challenges in Classical Music
While generic music tagging models are often trained on pop/rock datasets like GTZAN, classical music requires capturing complex harmonic and dynamic structures. Shi (2025) [9] validated a CNN-based approach using the Maestro dataset (piano performances), achieving 93% accuracy in emotion recognition by combining MFCCs with Chroma matrices and Short-Time RMS energy.

Although the Maestro dataset focuses exclusively on piano, this thesis proposes that Shi's feature extraction strategy is transferable to the iPalpiti archive's string and orchestral collections. Just as Shi demonstrated that ST-RMS captures the emotional intensity of a piano sonata, we posit that these energy features are equally critical for modeling the dynamic range of violin and orchestral performances, which standard pop-music models often compress or ignore. This validates the efficacy of spectral-based CNN architectures for extracting semantic and emotional information from classical performance data.

## 2. Sequential User Modeling and Recommendation Logic
While audio features describe what a track sounds like, recommendation logic must understand how users consume music over time. This section examines strategies for modeling temporal listening patterns and user preferences.

### 2.1 Personalized Popularity Awareness
Abbattista et al. [6] offer a critical counter-perspective. Their study on Personalized Popularity Awareness revealed that complex transformer models often underperform compared to simple baselines because they fail to account for "repeated consumption" (users re-listening to favorites). While the iPalpiti archive focuses on discovery, this insight suggests that the recommendation engine should perhaps include a "Personalized Most Popular" signal or a mechanism to handle repeat listening, preventing the model from over-optimizing for novelty.

### 2.2 Hybrid Emotion and Sequential Modeling (Jing Lin et al., 2025)
While P.-C. Lin focused on genre classification, J. Lin, Huang, and Zhang (2025) [7] propose a system centered on user emotional states. Their architecture integrates a Deep CNN for extracting audio features (mapped to the Valence-Arousal-Dominance model) with a Self-Attention Mechanism to model the temporal dynamics of user preferences. Unlike recurrent models (LSTM/GRU) which process sequentially, their use of Self-Attention allows for capturing long-term dependencies in user behavior. This approach achieved 82% emotion matching accuracy, supporting the proposed thesis strategy of fusing static audio embeddings with dynamic user context vectors.

### 2.3 User Archetypes and Clustering-Based Modeling
Schedl et al. [8] further refine user modeling by identifying Country Archetypes based on geographic listening behavior and unsupervised clustering. Using t-SNE and OPTICS on 369 million listening events from 70 countries, they identified 9 distinct country clusters reflecting shared music preferences at the track level. Their "geo-aware" VAE architecture extends standard collaborative filtering by incorporating geographic context through a gating mechanism, testing four user models (country ID, cluster ID, cluster distances, country distances). Results demonstrated that all context-aware models significantly outperformed baseline VAE, with relative improvements of 4.9-7.4% across precision, recall, and NDCG metrics. For our project, this contributes to the design of the Mock Data Generation phase (Methodology Phase 1), suggesting that synthetic users should be modeled not just randomly, but as distinct "listener archetypes" (e.g., "Fast Tempo Violin Enthusiast" vs. "Orchestral Purist").

## 3. End-to-End System Architectures and Deployment Strategies
Beyond theoretical model design, practical recommendation systems require careful architectural decisions regarding feature aggregation, user representation, and deployment infrastructure. The following papers demonstrate strategies for operationalizing deep learning models in production environments.

### 3.1 User Preference Aggregation via Feature Averaging
Zhang [10] proposes a CNN-based system that constructs user preference vectors by aggregating the classification features of their listening history. Using MFCC and mel spectrogram features extracted from 400 digital piano pieces (100 per genre) across four genres (classical, pop, rock, pure music), they compared two user modeling approaches: "Comprehensive" (single averaged feature vector, achieving 50.35% accuracy) vs. "Multicategory" (distinct category-specific vectors, achieving 42.89% accuracy overall but performing better for multicategory users). The Comprehensive approach achieved higher overall accuracy, while the Multicategory approach was more effective for users with diverse genre preferences. This comparison directly informs the Shallow Network tier of our methodology (Methodology Phase 2), specifically demonstrating how to map a user's listening history to a single point in the embedding space through feature averaging—a computationally efficient approach suitable for serverless deployment.

### 3.2 API Integration and Microservice Deployment
P.-C. Lin et al. [3] demonstrate practical deployment integration by connecting their TL-CNN-GRU model to external music platforms. Their prototype system integrates the trained model with YouTube and Spotify APIs to provide real-time genre-based recommendations, bridging the gap between offline model training and online serving. This validates that deep learning recommendation models can be deployed as microservices that interface with existing music streaming infrastructure, supporting our AWS Lambda-based architecture where feature extraction and recommendation logic exist as separate, scalable services.

### 3.3 Modular Architecture and Separation of Concerns
P.-C. Lin et al. (2024) [3] provide a practical framework for integrating deep learning models into a recommendation pipeline. Their "Novel Recommendation System" architecture separates the user interface (uploading audio) from the backend processing (spectrogram generation and model inference), validating the architectural decision to decouple feature extraction services from the recommendation serving layer.

### 3.4 Addressing the Metadata Bottleneck
(This section has been removed in favor of Section 1.5, which addresses domain specificity more directly).

## 4. Beyond Accuracy: Diversity and Long-Tail Discovery
Traditional recommendation evaluation focuses on accuracy metrics like HitRate or Precision@K, which measure how often the system correctly predicts user preferences. However, for specialized archives whose mission is educational and exploratory, diversity and novelty become equally important success criteria.

Porcaro et al. [11] conducted a 12-week longitudinal study with 110 participants on the Impact of Diversity in music recommendations. Focusing on Electronic Music exposure, they found that high-diversity recommendations significantly increased users' openness to unfamiliar genres, fueled curiosity, and helped deconstruct genre stereotypes. Specifically, they measured both implicit attitudes (via Single Category IAT) and explicit openness (via Guttman scale), demonstrating that exposure diversity positively impacts listeners' willingness to explore new music. This is particularly relevant for the iPalpiti Music Archive, whose mission is to expose listeners to specialized, potentially unfamiliar classical performances. It suggests that our evaluation metrics (Evaluation Phase 3) should look beyond simple accuracy (HitRate) and consider Diversity or Novelty metrics to ensure the system is effectively surfacing the "long tail" of the archive.

## 5. Synthesis and Contributions to the Thesis
These twelve papers collectively provide the theoretical foundation, technical methodologies, and evaluation frameworks necessary to address the four research questions guiding this thesis on small-scale, domain-specific music recommendation.

Addressing RQ1 (Small-Scale Dataset Performance): Papers [1-5] demonstrate that pretrained audio representations can perform effectively even with limited training data. Tamm et al. [1] establish that frozen embeddings from models like MusiCNN achieve HitRate@50 of 0.385, validating that audio signals alone can drive recommendations without requiring massive datasets. Critically, Pourmoazemi and Maleki [2] and P.-C. Lin et al. [3] both evaluate on GTZAN—a relatively small dataset (1000 tracks, 100 per genre)—achieving 93.75% and 71% accuracy respectively, demonstrating viability for domain-specific archives. Ramos et al. [4] show that self-supervised learning on the Free Music Archive can capture "subtle elements of musical structure" beyond obvious metadata, suggesting pretrained models generalize well to specialized domains like classical music. Reddy et al. [5] prove that compact models (0.2 MB) can achieve production-grade performance, addressing resource constraints typical of small-scale archives.

Addressing RQ2 (Overfitting and Generalization): P.-C. Lin et al. [3] directly address overfitting through 10-Fold Cross-Validation (10-FCV), showing that careful validation strategies improve generalization even when dataset size is limited. Their genre-specific F1-scores improved substantially with 10-FCV: blues (0.49→0.74), classical (0.87→0.92), metal (0.61→0.84), reggae (0.35→0.69). Tamm et al. [1] demonstrate that using frozen embeddings (without fine-tuning) prevents overfitting on small datasets by leveraging knowledge learned from massive pretraining corpora. This approach is particularly relevant for RQ2, as it suggests that transfer learning with frozen representations may generalize better than fine-tuned models when data is scarce. Ramos et al. [4] validate self-supervised learning as another strategy for improving generalization without labeled data.

Addressing RQ3 (Dataset Size Impact on Model Ranking): Tamm et al. [1] provide crucial evidence that model ranking changes based on dataset characteristics: MusiCNN (a lighter, supervised discriminative model with fewer parameters) outperformed larger generative models like Jukebox (4800-dimensional embeddings), achieving the best HitRate@50 across all recommendation methods. This finding directly supports RQ3 by demonstrating that larger models do not automatically perform better on domain-specific tasks. Pourmoazemi and Maleki [2] reinforce this with their Compact Convolutional Transformer (454,187 parameters) achieving 93.75% accuracy while being "significantly fewer than state-of-the-art CRNN models." These results suggest that when dataset size is limited, compact models may be more efficient and equally effective, challenging the assumption that larger models always rank higher.

Addressing RQ4 (Robustness of Embedding Aspects): Ramos et al. [4] provide qualitative evidence that embeddings capture composition, timbre, and flow even without explicit training on these attributes, suggesting these aspects remain robust under self-supervised learning. Zhang [10] demonstrates that MFCC and mel spectrogram features extracted from classical music (400 digital piano pieces) retain discriminative power for genre classification (50.35% accuracy with Comprehensive approach), indicating that spectral and timbral features are robust to small-sample conditions. J. Lin et al. [7] show that emotional features can be extracted from audio spectrograms with 82% emotion matching accuracy, suggesting that high-level semantic attributes remain accessible even with limited data. However, P.-C. Lin et al. [3] reveal that genre-specific performance varies: classical music achieved high F1-scores (0.92) while rock struggled (0.50), suggesting that some musical genres may be more challenging to represent with embeddings under small-sample conditions.

Supporting Methodologies: Papers [6, 8] inform the mock data generation strategy (Methodology Phase 1) by demonstrating that user archetypes (Schedl et al. [8]: country clusters with 4.9-7.4% performance gains) and popularity-aware modeling (Abbattista et al. [6]) improve recommendation quality beyond pure content-based approaches. Papers [3, 10] validate the architectural decisions for AWS serverless deployment, with Zhang [10] demonstrating efficient user preference aggregation (Comprehensive approach) and P.-C. Lin et al. [3] showing API integration patterns (YouTube/Spotify) and modular design principles. Porcaro et al. [11] establish that diversity metrics are critical for evaluating specialized archives, shaping the evaluation framework (Methodology Phase 3).

In summary, these papers collectively validate that pretrained audio embeddings can drive effective recommendation systems for small-scale archives (RQ1), frozen representations and cross-validation strategies mitigate overfitting (RQ2), compact models may outperform larger models on domain-specific tasks (RQ3), and timbral, spectral, and compositional features remain robust while genre-specific and high-level semantic attributes show variable performance under small-sample conditions (RQ4). This synthesis provides a comprehensive roadmap for adapting the Tamm et al. methodology to the iPalpiti Music Archive context.

# THEORETICAL FOUNDATIONS

## 1. The Cold Start Problem in Niche Archives
Standard recommendation algorithms, particularly Collaborative Filtering, depend on dense user-item interaction matrices to infer preferences. In specialized archives with limited traffic, these matrices are sparse, making it impossible to derive meaningful patterns—a challenge known as the Cold Start Problem. The theoretical premise of this work is that the audio signal itself provides a rich source of feature data. By projecting raw audio into a high-dimensional vector space using deep neural networks, we can quantify musical similarity mathematically (e.g., via Cosine Similarity) and generate recommendations without requiring prior user history.

## 2. Vector-Based Audio Representation and Retrieval
The core technical foundation of this system is the transformation of raw audio into high-dimensional vector embeddings. The iPalpiti music dataset is first segmented into consistent audio clips and processed through pretrained models (e.g., MusicNN, MERT) to generate dense feature vectors. These embeddings are stored in a vectorized database, enabling efficient similarity search and retrieval. Backend models leverage this latent vector space to identify semantic and acoustic relationships between tracks, while the frontend recommendation logic utilizes mock user interaction data mapped to these vectors to simulate and predict user preferences.

## 3. Sequential Audio Modeling (Content-Based)
Beyond simple similarity, this research explores the temporal aspect of music consumption. By employing sequential models like BERT4Rec [3] on top of audio embeddings, the system can model "listening sessions" as sequences of acoustic events. This allows for predicting the next most suitable track based on the flow of audio features in a session (simulated via mock data), effectively treating music recommendation as a content-driven sequence modeling task rather than just a static retrieval problem.

# SYSTEM ARCHITECTURE

The proposed system is built on Amazon Web Services (AWS), integrating with an existing audio segmentation backend.

## High-Level Architecture Diagram
![Architecture Diagram](architecture_diagram.png)

## 1. Infrastructure Layer
- Storage: Amazon S3 stores the raw audio (WAV/MP3) and the pre-computed feature vectors (embeddings).
- Database: Amazon RDS (PostgreSQL) manages metadata (Albums, Tracks) and user interaction logs.
- Compute:
    - AWS Lambda: Handles API requests and lightweight business logic.
    - AWS SageMaker / Fargate: Hosts the heavier inference tasks for extracting embeddings from new audio tracks.

## 2. Application Layer
- RecommendationService: Acts as a Facade, providing a clean API (recommendForUser, recommendSimilar) to the frontend.
- Strategy Implementation:
    - KNN Baseline: Performs similarity search directly on the frozen audio embeddings to find nearest neighbors.
    - Shallow Network: A lightweight neural network that learns to map simulated user preferences to the embedding space.
    - Sequential Model: Deploys BERT4Rec to predict the next track in a sequence, effectively modeling the "flow" of a listening session.

## 3. Data Pipeline
1. Ingestion: Audio uploaded to S3 triggers an extraction event.
2. Feature Extraction: A worker (container) downloads the audio, runs a pretrained model (e.g., MusicNN), and saves the embedding vector.
3. Indexing: Vectors are indexed (e.g., in a vector store or FAISS index) for fast retrieval.

# RESEARCH GOAL

	Problem Statement
Small-scale music institutions possess valuable cultural assets but lack the technical resources to build personalized discovery tools. Existing state-of-the-art models are often evaluated on generic pop datasets (e.g., MTG-Jamendo) and require expensive infrastructure.

	Objective
This research aims to:

1.	Adapt the Tamm et al. methodology to the specific domain of the iPalpiti archive.

2.	Compare the performance of different audio backends (MusicNN vs. MERT) in this classical/performance-focused domain.

3.	Implement a production-ready, serverless recommendation API on AWS that demonstrates cost-efficiency and scalability.

	Research Questions
•	RQ1: How do pretrained audio representations perform in recommendation tasks when trained and evaluated on small-scale or domain-specific datasets (e.g., classical music)?
•	RQ2: To what extent do different pretrained models overfit or generalize when the dataset size is reduced?
•	RQ3: How does dataset size influence the relative performance ranking among pretrained audio models?
•	RQ4 (Optional): Which aspects of audio embeddings (genre, timbre, dynamics) remain robust or degrade under small-sample fine-tuning or transfer?

# METHODOLOGY

Phase 1: Dataset Preparation
- Audio Data: Use the iPalpiti Music Archive (digitized performances).
- Preprocessing: Convert to mono, 16kHz (or model-specific sample rate). Following the success of Shi (2025) [9] in using multi-dimensional features (MFCC, Chroma, Energy) for classical music emotion classification, we will ensure our selected pretrained models (e.g., MusiCNN) sufficiently capture these harmonic and spectral characteristics. While Shi's work was limited to piano (Maestro dataset), we hypothesize that these features—specifically Chroma for harmony and RMS for dynamic range—are instrument-agnostic and will generalize effectively to the string and orchestral performances in the iPalpiti archive.
- Mock Data Generation: To simulate realistic user behavior, we will generate "Archetypal Users" based on the clustering methodology of Schedl et al. (2021) [8]. Instead of random interaction data, synthetic users will be modeled as distinct "listener archetypes" (e.g., a "Cluster 0" user preferring Indie/Alternative styles vs. a "Cluster 6" user preferring High-Energy tracks). This provides a theoretically grounded "ground truth" for training and evaluation.

Phase 2: Model Implementation
We will implement three tiers of recommendation logic, drawing on distinct architectural precedents:

1.	Baseline (KNN): A pure content retrieval approach using K-Nearest Neighbors on frozen audio embeddings.

2.	Shallow Network (User Mapping): A lightweight neural network that maps user preferences to the embedding space. This mirrors the feature aggregation strategy used by Zhang (2022), who demonstrated that averaging features for "Comprehensive" user modeling outperforms multi-category splits for general accuracy.

3.	Sequential Context (Self-Attention/BERT4Rec): Adapting the findings of J. Lin et al. (2025), who utilized Self-Attention to capture dynamic user states, this tier will deploy BERT4Rec. While P.-C. Lin et al. (2024) successfully used GRUs for temporal audio features, this project favors Transformer-based attention mechanisms to better model the non-linear "flow" of a listening session using synthetic user data. Furthermore, acknowledging Abbattista et al. (2024) [6] who noted that Transformers can underperform due to the "repeat consumption" phenomenon, we will integrate a "Personalized Popularity" signal to balance discovery with user preference stability.

Phase 3: Evaluation Strategy
The specific evaluation metrics and experimental protocols are currently under development and will be finalized in the upcoming semester. The research will focus on establishing a robust framework for assessing both the recommendation quality (using the synthetic datasets) and the system efficiency (latency and cost) within the AWS environment.

# EVALUATION ROADMAP

The detailed evaluation plan is a key objective for the next semester. The current research direction identifies the following areas for development:

1.	Algorithmic Accuracy:
    - Investigation into appropriate metrics for small-scale, specialized archives (e.g., adapting standard ranking metrics like HitRate or NDCG).
    - Development of a validation strategy to assess how well the models leverage the vector embeddings to solve the cold-start problem.

2.	System Performance:
    - Definition of benchmarks for the serverless infrastructure.
    - Planning for scalability tests to evaluate the cost-effectiveness of the proposed AWS architecture.

# References
[1] R. Tamm, M. Sachdeva, and S. Lind, "Comparative Analysis of Pretrained Audio Representations in Music Recommender Systems," in *Proceedings of the 18th ACM Conference on Recommender Systems (RecSys '24)*, Bari, Italy, Oct. 2024.

[2] N. Pourmoazemi and S. Maleki, "A music recommender system based on compact convolutional transformers," *Expert Systems With Applications*, vol. 255, Art. no. 124473, 2024.

[3] P.-C. Lin, C.-Y. Yu, and E. Odle, "Prototype for a Personalized Music Recommendation System Based on TL-CNN-GRU Model with 10-Fold Cross-Validation," in *Proceedings of the 7th Artificial Intelligence and Cloud Computing Conference (AICCC)*, Tokyo, Japan, Dec. 14–16, 2024, pp. 87–94. ACM. https://doi.org/10.1145/3719384.3719396

[4] A. C. Ramos and B. A. T. de Freitas, "Self-Supervised Learning of Music Representations for Recommendation Systems," Course Project, Institute of Computing, University of Campinas (UNICAMP).

[5] C. K. A. Reddy, V. Gopal, H. Dubey, S. Matusevych, R. Cutler, and R. Aichner, "MusicNet: Compact Convolutional Neural Network for Real-time Background Music Detection," in *Proceedings of Interspeech 2022*, Incheon, Korea, Sept. 2022.

[6] D. Abbattista, V. W. Anelli, T. Di Noia, C. Macdonald, and A. Petrov, "Enhancing Sequential Music Recommendation with Personalized Popularity Awareness," in *Proceedings of the 18th ACM Conference on Recommender Systems (RecSys '24)*, Bari, Italy, Oct. 2024.

[7] J. Lin, S. Huang, and Y. Zhang, "Deep neural network-based music user preference modeling, accurate recommendation, and IoT-enabled personalization," *Alexandria Engineering Journal*, vol. 125, pp. 232–244, 2025. https://doi.org/10.1016/j.aej.2025.03.057

[8] M. Schedl, C. Bauer, W. Reisinger, D. Kowald, and E. Lex, "Listener Modeling and Context-Aware Music Recommendation Based on Country Archetypes," *Frontiers in Artificial Intelligence*, vol. 3, Art. no. 508725, Feb. 2021.

[9] Y. Shi, "A CNN-Based Approach for Classical Music Recognition and Style Emotion Classification," *IEEE Access*, vol. 13, 2025. https://doi.org/10.1109/ACCESS.2025.3535411

[10] Y. Zhang, "Music Recommendation System and Recommendation Model Based on Convolutional Neural Network," *Mobile Information Systems*, vol. 2022, Art. no. 3387598, May 2022.

[11] L. Porcaro, E. Gómez, and C. Castillo, "Perceptions of Diversity in Electronic Music: The Impact of Listener, Artist, and Track Characteristics," *Proceedings of the ACM on Human-Computer Interaction*, vol. 6, no. CSCW1, Article 109, Apr. 2022. https://doi.org/10.1145/3512956
