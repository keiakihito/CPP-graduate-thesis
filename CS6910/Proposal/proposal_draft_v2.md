# Comparative Analysis of Pretrained Audio Representations for Small-Scale Music Archives: A Case Study on the iPalpiti Collection

## ABSTRACT
This thesis investigates the application of content-based music recommendation techniques to small-scale, specialized music archives, specifically the iPalpiti Music Archive. While large streaming platforms leverage massive user interaction datasets for collaborative filtering, niche archives often face the "cold start" problem and lack sufficient user data. This research adapts the methodology of Tamm et al. (2024) [1]—who compared pretrained audio representations like MusicNN, MERT, and Jukebox—to evaluate their effectiveness in a small-domain context. By integrating these models into a modern AWS serverless architecture, this study aims to develop a cost-effective, scalable recommendation system. The project utilizes mock user data to train and compare backend models (KNN, Shallow Networks, and BERT4Rec) on frozen audio embeddings, ultimately determining the optimal configuration for enhancing music discovery in specialized libraries.

## INTRODUCTION
The digital preservation of music archives presents a unique challenge: making content discoverable without the massive user interaction data that powers commercial giants like Spotify or Apple Music. The iPalpiti Music Archive, featuring recordings by award-winning musicians from the iPalpiti Festival, is a prime example of such a specialized domain. In these contexts, traditional collaborative filtering fails due to data sparsity.

Recent advancements in Music Information Retrieval (MIR) have shown that deep learning models can extract rich semantic information directly from audio signals. The work of Tamm et al. [1] demonstrates that "frozen" embeddings from pretrained models (e.g., MusicNN, MERT) can serve as powerful features for recommendation systems, even with limited data. 

This thesis proposes to adapt these findings to the iPalpiti archive, building a hybrid recommendation engine on AWS. By leveraging serverless infrastructure (Lambda, S3, RDS), the system aims to provide real-time, content-aware recommendations that help users navigate the emotional and stylistic landscape of the archive.

## LITERATURE REVIEW

The domain of music recommendation has historically been dominated by Collaborative Filtering (CF) approaches, which rely on dense user-interaction matrices. However, specialized archives such as the **iPalpiti Music Archive** typically lack the massive volume of user data required for these methods, leading to the "Cold Start" problem. This literature review synthesizes thirteen key research contributions that collectively argue for a **Deep Content-Based** approach. By leveraging pretrained audio representations, sequential modeling, and efficient neural architectures, it is possible to build high-performance recommendation systems that rely on the audio signal itself rather than historical usage data.

### Thesis Objectives and Research Questions

This thesis addresses four research questions that guide the development and evaluation of content-based music recommendation systems for small-scale, specialized archives:

- **RQ1**: How do pretrained audio representations perform in recommendation tasks when trained and evaluated on small-scale or domain-specific datasets (e.g., classical music)?
- **RQ2**: To what extent do different pretrained models overfit or generalize when the dataset size is reduced?
- **RQ3**: How does dataset size influence the relative performance ranking among pretrained audio models?
- **RQ4 (Optional)**: Which aspects of audio embeddings (genre, timbre, dynamics) remain robust or degrade under small-sample fine-tuning or transfer?

To answer these questions, the methodology consists of three phases:

1. **Dataset Preparation**: Process the iPalpiti Music Archive and generate synthetic user sessions with simulated preferences (e.g., "Fast Tempo Violin Enthusiast" vs. "Orchestral Purist") to create ground truth for training without real user data.
2. **Model Implementation**: Compare three recommendation approaches—KNN baseline (pure content retrieval), Shallow Neural Network (user preference mapping to embedding space), and Sequential Model (BERT4Rec for session modeling)—all fundamentally driven by frozen audio embeddings.
3. **Evaluation**: Develop metrics to assess both algorithmic accuracy (HitRate, NDCG adapted for small-scale archives) and system performance (AWS serverless infrastructure cost-efficiency and scalability), with emphasis on understanding how dataset size affects model performance and generalization.

### 1. Pretrained Audio Representations and Feature Extraction

The core technical foundation of this thesis rests on the ability to extract meaningful semantic features directly from raw audio waveforms or spectrograms.

**Tamm et al. (2024)** [1] provide the primary methodological framework for this research. Their comparative analysis of six pretrained models—**MusiCNN**, **MERT**, **Jukebox**, **MusicFM**, **Music2Vec**, and **EncodecMAE**—demonstrates that "frozen" embeddings from these models can effectively drive recommendation tasks. Crucially, **MusiCNN** (a lighter, supervised discriminative model trained on auto-tagging) achieved the best performance across all recommendation methods (HitRate@50: 0.385), significantly outperforming larger generative models like Jukebox (4800-dimensional embeddings). This validates the choice of efficient, discriminative backends for an AWS Lambda-based architecture.

**Pourmoazemi and Maleki (2024)** [4] address the "Continuity Problem" in music streaming—maintaining a continuous flow of music that aligns with user preferences—by proposing a **Compact Convolutional Transformer (CCT)** architecture for genre-based recommendation. Their hybrid model combines six convolutional layers (with progressively increasing filters: 32, 64, 128) for local feature extraction from mel-spectrograms, followed by two transformer encoders with multi-head attention (2 heads, 128-dimensional) to capture global interdependencies among feature maps. Critically for resource-constrained deployment, the CCT achieves **93.75% test accuracy** on the GTZAN dataset while containing only **454,187 parameters**—significantly fewer than state-of-the-art CRNN models. The model outperforms previous architectures across precision (0.923), recall (0.928), and F1-score (0.923) metrics. For recommendation, they use cosine similarity between feature maps extracted from the penultimate layer, demonstrating that the learned representations effectively capture genre-specific patterns for content-based music retrieval. This validates the thesis's architectural strategy of using lightweight hybrid CNN-Transformer models for efficient feature extraction.

Complementing the CCT approach, **Lin et al. (2024)** [5] propose a **Transfer Learning + CNN + GRU (TL-CNN-GRU)** model that leverages pretrained **MobileNetV2** weights for spatial feature extraction combined with bidirectional GRU (1024 units) for capturing temporal dependencies in music sequences. Using spectrograms resized to 256×256×3 as input, their architecture applies **10-Fold Cross-Validation (10-FCV)** to mitigate overfitting and improve generalization. The model achieved **71% accuracy** on the GTZAN dataset, representing a significant improvement over the TL+CNN baseline (53% accuracy) and the TL+CNN+GRU model (55% accuracy). With **7,779,402 total parameters** (5,521,418 trainable, 2,257,984 frozen from MobileNetV2), the system demonstrates that transfer learning from vision models pretrained on ImageNet can be effectively repurposed for audio classification tasks. Notably, genre-specific F1-scores improved substantially with 10-FCV: blues (0.49→0.74), classical (0.87→0.92), metal (0.61→0.84), and reggae (0.35→0.69), though rock remained challenging (0.26→0.50). Their prototype platform integrates the trained model with YouTube and Spotify APIs to provide real-time genre-based recommendations, validating the practical deployment of deep learning models for personalized music discovery. This work reinforces the thesis's emphasis on transfer learning and sequential modeling for audio recommendation systems.

Expanding on the potential of transformer-based architectures for audio, **Ramos et al.** [6] explored **Self-Supervised Learning (SSL)** using the **Audio Spectrogram Transformer (AST)** within a SimCLR framework. Training on the Free Music Archive (FMA) dataset with InfoNCE contrastive loss, they demonstrated that music embeddings can be learned without explicit labels, organizing tracks by composition, timbre, and flow rather than conventional genre classifications. Notably, their qualitative evaluation showed that the trained model achieved 48% satisfactory recommendations compared to only 10% for the untrained baseline, with the learned representations capturing "subtle elements of musical structure" beyond obvious metadata. This supports our thesis that unsupervised embeddings can uncover the latent "musicality" of the iPalpiti archive without manual annotation.

**Reddy et al.** [7] developed **MusicNet**, a compact CNN for real-time background music detection optimized for edge deployment. MusicNet achieves 81.3% TPR at 0.1% FPR while being only 0.2 MB in size—10x smaller than competing models—with 11.1ms inference time (4x faster than best-performing alternatives). Crucially, MusicNet incorporates **in-model featurization**, processing raw audio directly without requiring external feature extraction, simplifying deployment and maintenance in production systems. Together with Pourmoazemi and Maleki's CCT, these papers contribute to the project's goal of **cost-efficiency and scalability** (RQ3), proving that high-performance audio analysis does not require prohibitive computational resources.

Having established the technical foundation for extracting meaningful audio features from raw signals, the next challenge is to model how users interact with music over time. While audio embeddings capture *what* a track sounds like, recommendation systems must also understand *how* users consume music sequentially to generate personalized suggestions.

### 2. Sequential User Modeling and Recommendation Logic

While audio features describe *what* a track sounds like, recommendation logic must understand *how* users consume music over time.

**Abbattista et al.** [8] offer a critical counter-perspective. Their study on **Personalized Popularity Awareness** revealed that complex transformer models often underperform compared to simple baselines because they fail to account for "repeated consumption" (users re-listening to favorites). While the iPalpiti archive focuses on discovery, this insight suggests that the recommendation engine should perhaps include a "Personalized Most Popular" signal or a mechanism to handle repeat listening, preventing the model from over-optimizing for novelty.

**Lin et al.** [9] propose a **hybrid architecture** that combines a Deep CNN for audio emotion modeling with a Self-Attention mechanism for user emotion modeling. Their system integrates three components: (1) Deep Convolutional Neural Network (DCNN) for extracting emotional features from audio signals using spectrograms, (2) Self-Attention Mechanism (specifically Scaled Dot-Product Attention with Multi-Head Attention) for capturing temporal dynamics in user emotional states, and (3) collaborative filtering enhanced with Neural Collaborative Filtering (NCF) and SVD++. Achieving 82% emotion matching accuracy and 83% recommendation accuracy (Precision@10), their hybrid approach significantly outperforms traditional content-based filtering (63% emotion matching, 62% Precision@10) and collaborative filtering (66% emotion matching, 68% Precision@10). Although this project scopes out explicit emotion recognition, Lin's architectural pattern—fusing a static content vector (Audio CNN) with a dynamic context vector (Self-Attention)—directly informs our "Hybrid Strategy" (RQ2), where we combine frozen embeddings with sequential user state.

**Schedl et al.** [10] further refine user modeling by identifying **Country Archetypes** based on geographic listening behavior and unsupervised clustering. Using t-SNE and OPTICS on 369 million listening events from 70 countries, they identified 9 distinct country clusters reflecting shared music preferences at the track level. Their "geo-aware" VAE architecture extends standard collaborative filtering by incorporating geographic context through a gating mechanism, testing four user models (country ID, cluster ID, cluster distances, country distances). Results demonstrated that all context-aware models significantly outperformed baseline VAE, with relative improvements of 4.9-7.4% across precision, recall, and NDCG metrics. For our project, this contributes to the design of the **Mock Data Generation** phase (Methodology Phase 1), suggesting that synthetic users should be modeled not just randomly, but as distinct "listener archetypes" (e.g., "Fast Tempo Violin Enthusiast" vs. "Orchestral Purist") to train the backend models effectively.

With audio feature extraction methods (Section 2) and user modeling strategies (Section 3) established, the next step is to examine complete end-to-end systems that integrate these components for practical deployment. The following papers demonstrate how CNN-based architectures can be operationalized into production recommendation systems while maintaining computational efficiency.

### 3. End-to-End System Architectures and Deployment Strategies

Beyond theoretical model design, practical recommendation systems require careful architectural decisions regarding feature aggregation, user representation, and deployment infrastructure. The following papers demonstrate strategies for operationalizing deep learning models in production environments.

**Zhang [11]** proposes a CNN-based system that constructs user preference vectors by aggregating the classification features of their listening history. Using MFCC and mel spectrogram features extracted from 400 digital piano pieces (100 per genre) across four genres (classical, pop, rock, pure music), they compared two user modeling approaches: **"Comprehensive"** (single averaged feature vector, achieving 50.35% accuracy) vs. **"Multicategory"** (distinct category-specific vectors, achieving 42.89% accuracy overall but performing better for multicategory users). The Comprehensive approach achieved higher overall accuracy, while the Multicategory approach was more effective for users with diverse genre preferences. This comparison directly informs the **Shallow Network** tier of our methodology (Methodology Phase 2), specifically demonstrating how to map a user's listening history to a single point in the embedding space through feature averaging—a computationally efficient approach suitable for serverless deployment.

**Lin et al.** [5] demonstrate practical deployment integration by connecting their TL-CNN-GRU model to external music platforms. Their prototype system integrates the trained model with **YouTube and Spotify APIs** to provide real-time genre-based recommendations, bridging the gap between offline model training and online serving. This validates that deep learning recommendation models can be deployed as microservices that interface with existing music streaming infrastructure, supporting our AWS Lambda-based architecture where feature extraction and recommendation logic exist as separate, scalable services.

**Prasad et al.** [12] provide a comprehensive architectural framework for **AI-Powered Recommendation Systems**, emphasizing modular design principles. Their architecture separates concerns between data collection, feature extraction, and recommendation generation, integrating multiple ML paradigms: supervised learning (decision trees, random forests, neural networks) for classification, unsupervised learning (K-Means, DBSCAN) for pattern discovery, and reinforcement learning (Deep Q-learning, Multi-Armed Bandit) for continuous improvement. This modular approach directly parallels our architectural decision to decouple feature extraction (Lambda/Fargate for audio processing) from the recommendation serving layer, enabling independent scaling and maintenance of each component.

**Dias et al.** [13] further validate the end-to-end viability of CNN-based genre classification for recommendation, achieving 76% accuracy on the GTZAN dataset. By explicitly addressing the metadata bottleneck that arises from manual genre labeling, their work reinforces the core thesis premise: deep learning can replace manual annotation pipelines, making content-based recommendation viable even for archives lacking comprehensive metadata.

While technical implementation and accuracy metrics are essential, the ultimate success of a recommendation system for specialized archives depends on its ability to fulfill the archive's mission. For iPalpiti, this means exposing listeners to unfamiliar performances and expanding musical horizons—objectives that require evaluation metrics beyond traditional accuracy measures.

### 4. Beyond Accuracy: Diversity and Long-Tail Discovery

Traditional recommendation evaluation focuses on accuracy metrics like HitRate or Precision@K, which measure how often the system correctly predicts user preferences. However, for specialized archives whose mission is educational and exploratory, diversity and novelty become equally important success criteria.

**Porcaro et al.** [14] conducted a **12-week longitudinal study with 110 participants** on the **Impact of Diversity** in music recommendations. Focusing on Electronic Music exposure, they found that high-diversity recommendations significantly increased users' openness to unfamiliar genres, fueled curiosity, and helped deconstruct genre stereotypes. Specifically, they measured both implicit attitudes (via Single Category IAT) and explicit openness (via Guttman scale), demonstrating that exposure diversity positively impacts listeners' willingness to explore new music. This is particularly relevant for the **iPalpiti Music Archive**, whose mission is to expose listeners to specialized, potentially unfamiliar classical performances. It suggests that our evaluation metrics (Evaluation Phase 3) should look beyond simple accuracy (HitRate) and consider **Diversity** or **Novelty** metrics to ensure the system is effectively surfacing the "long tail" of the archive.

### 5. Synthesis and Contributions to the Thesis

These twelve papers collectively provide the theoretical foundation, technical methodologies, and evaluation frameworks necessary to address the four research questions guiding this thesis on small-scale, domain-specific music recommendation.

**Addressing RQ1 (Small-Scale Dataset Performance)**: Papers [1, 4-7] demonstrate that pretrained audio representations can perform effectively even with limited training data. Tamm et al. [1] establish that frozen embeddings from models like MusiCNN achieve HitRate@50 of 0.385, validating that audio signals alone can drive recommendations without requiring massive datasets. Critically, **Pourmoazemi and Maleki [4]** and **Lin et al. [5]** both evaluate on **GTZAN**—a relatively small dataset (1000 tracks, 100 per genre)—achieving 93.75% and 71% accuracy respectively, demonstrating viability for domain-specific archives. Ramos et al. [6] show that self-supervised learning on the Free Music Archive can capture "subtle elements of musical structure" beyond obvious metadata, suggesting pretrained models generalize well to specialized domains like classical music. Reddy et al. [7] prove that compact models (0.2 MB) can achieve production-grade performance, addressing resource constraints typical of small-scale archives.

**Addressing RQ2 (Overfitting and Generalization)**: **Lin et al. [5]** directly address overfitting through **10-Fold Cross-Validation (10-FCV)**, showing that careful validation strategies improve generalization even when dataset size is limited. Their genre-specific F1-scores improved substantially with 10-FCV: blues (0.49→0.74), classical (0.87→0.92), metal (0.61→0.84), reggae (0.35→0.69). **Tamm et al. [1]** demonstrate that using **frozen embeddings** (without fine-tuning) prevents overfitting on small datasets by leveraging knowledge learned from massive pretraining corpora. This approach is particularly relevant for RQ2, as it suggests that transfer learning with frozen representations may generalize better than fine-tuned models when data is scarce. Ramos et al. [6] validate self-supervised learning as another strategy for improving generalization without labeled data.

**Addressing RQ3 (Dataset Size Impact on Model Ranking)**: **Tamm et al. [1]** provide crucial evidence that model ranking changes based on dataset characteristics: **MusiCNN** (a lighter, supervised discriminative model with fewer parameters) **outperformed larger generative models like Jukebox** (4800-dimensional embeddings), achieving the best HitRate@50 across all recommendation methods. This finding directly supports RQ3 by demonstrating that larger models do not automatically perform better on domain-specific tasks. **Pourmoazemi and Maleki [4]** reinforce this with their **Compact Convolutional Transformer (454,187 parameters)** achieving 93.75% accuracy while being "significantly fewer than state-of-the-art CRNN models." These results suggest that when dataset size is limited, compact models may be more efficient and equally effective, challenging the assumption that larger models always rank higher.

**Addressing RQ4 (Robustness of Embedding Aspects)**: Ramos et al. [6] provide qualitative evidence that embeddings capture **composition, timbre, and flow** even without explicit training on these attributes, suggesting these aspects remain robust under self-supervised learning. Zhang [11] demonstrates that **MFCC and mel spectrogram features** extracted from classical music (400 digital piano pieces) retain discriminative power for genre classification (50.35% accuracy with Comprehensive approach), indicating that spectral and timbral features are robust to small-sample conditions. Lin et al. [9] show that **emotional features** can be extracted from audio spectrograms with 82% emotion matching accuracy, suggesting that high-level semantic attributes remain accessible even with limited data. However, Lin et al. [5] reveal that **genre-specific performance varies**: classical music achieved high F1-scores (0.92) while rock struggled (0.50), suggesting that some musical genres may be more challenging to represent with embeddings under small-sample conditions.

**Supporting Methodologies**: Papers [8-10] inform the mock data generation strategy (Methodology Phase 1) by demonstrating that user archetypes (Schedl et al. [10]: country clusters with 4.9-7.4% performance gains) and popularity-aware modeling (Abbattista et al. [8]) improve recommendation quality beyond pure content-based approaches. Papers [5, 11-13] validate the architectural decisions for AWS serverless deployment, with Zhang [11] demonstrating efficient user preference aggregation (Comprehensive approach), Lin et al. [5] showing API integration patterns (YouTube/Spotify), and Prasad et al. [12] providing modular design principles. Porcaro et al. [14] establish that diversity metrics are critical for evaluating specialized archives, shaping the evaluation framework (Methodology Phase 3).

In summary, these papers collectively validate that **pretrained audio embeddings can drive effective recommendation systems for small-scale archives** (RQ1), **frozen representations and cross-validation strategies mitigate overfitting** (RQ2), **compact models may outperform larger models on domain-specific tasks** (RQ3), and **timbral, spectral, and compositional features remain robust while genre-specific and high-level semantic attributes show variable performance** under small-sample conditions (RQ4). This synthesis provides a comprehensive roadmap for adapting the Tamm et al. methodology to the iPalpiti Music Archive context.

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

[1] R. Tamm, M. Sachdeva, and S. Lind, "Comparative Analysis of Pretrained Audio Representations in Music Recommender Systems," in *Proceedings of the 18th ACM Conference on Recommender Systems (RecSys '24)*, Bari, Italy, Oct. 2024.

[2] J. Pons and X. Serra, "musicnn: Pre-trained convolutional neural networks for music audio tagging," in *Proceedings of the 20th International Society for Music Information Retrieval Conference (ISMIR)*, 2019, pp. 1-7.

[3] F. Sun, J. Liu, J. Wu, C. Pei, X. Lin, W. Ou, and P. Jiang, "BERT4Rec: Sequential Recommendation with Bidirectional Encoder Representations from Transformer," in *Proceedings of the 28th ACM International Conference on Information and Knowledge Management (CIKM)*, 2019, pp. 1471–1480.

[4] N. Pourmoazemi and S. Maleki, "A music recommender system based on compact convolutional transformers," *Expert Systems With Applications*, vol. 255, Art. no. 124473, 2024.

[5] P.-C. Lin, C.-Y. Yu, and E. Odle, "Prototype for a Personalized Music Recommendation System Based on TL-CNN-GRU Model with 10-Fold Cross-Validation," in *Proceedings of the 7th Artificial Intelligence and Cloud Computing Conference (AICCC 2024)*, Tokyo, Japan, Dec. 2024, pp. 87-94.

[6] A. C. Ramos and B. A. T. de Freitas, "Self-Supervised Learning of Music Representations for Recommendation Systems," Course Project, Institute of Computing, University of Campinas (UNICAMP).

[7] C. K. A. Reddy, V. Gopal, H. Dubey, S. Matusevych, R. Cutler, and R. Aichner, "MusicNet: Compact Convolutional Neural Network for Real-time Background Music Detection," in *Proceedings of Interspeech 2022*, Incheon, Korea, Sept. 2022.

[8] D. Abbattista, V. W. Anelli, T. Di Noia, C. Macdonald, and A. Petrov, "Enhancing Sequential Music Recommendation with Personalized Popularity Awareness," in *Proceedings of the 18th ACM Conference on Recommender Systems (RecSys '24)*, Bari, Italy, Oct. 2024.

[9] J. Lin, S. Huang, and Y. Zhang, "Deep neural network-based music user preference modeling, accurate recommendation, and IoT-enabled personalization," *Alexandria Engineering Journal*, vol. 125, pp. 232-244, 2025.

[10] M. Schedl, C. Bauer, W. Reisinger, D. Kowald, and E. Lex, "Listener Modeling and Context-Aware Music Recommendation Based on Country Archetypes," *Frontiers in Artificial Intelligence*, vol. 3, Art. no. 508725, Feb. 2021.

[11] Y. Zhang, "Music Recommendation System and Recommendation Model Based on Convolutional Neural Network," *Mobile Information Systems*, vol. 2022, Art. no. 3387598, May 2022.

[12] M. S. V. Prasad and G. Sharma, "Music Recommendation System," Parul University, Vadodara, Gujarat, India.

[13] J. Dias, H. Deshmukh, V. Pillai, and A. Shah, "Music Genre Classification & Recommendation System using CNN," St. John College of Engineering and Management, Palghar, India.

[14] L. Porcaro, E. Gómez, and C. Castillo, "Assessing the Impact of Music Recommendation Diversity on Listeners: A Longitudinal Study," Manuscript, 2022.
