# Comprehensive Literature Review

## 1. Introduction

The domain of music recommendation has historically been dominated by Collaborative Filtering (CF) approaches, which rely on dense user-interaction matrices. However, specialized archives such as the **iPalpiti Music Archive** typically lack the massive volume of user data required for these methods, leading to the "Cold Start" problem. This literature review synthesizes twelve research contributions that collectively argue for a **Deep Content-Based** approach. By leveraging pretrained audio representations and efficient neural architectures, it is possible to build recommendation systems that rely on the audio signal itself rather than historical usage data.

This thesis focuses on: (1) **backend audio embedding model families** (e.g., CNN-, RNN-, and transformer-based pretrained models) for content-based retrieval; (2) **ranking-based evaluation** using metrics such as NDCG, Precision@K, and Recall@K; and (3) **classical music archives** in settings where user interaction data is absent. The review is organized around the technical foundation of audio embeddings (Section 2), followed by related work on user modeling and deployment (Sections 3–4), evaluation beyond accuracy (Section 5), and a synthesis aligned with these objectives (Section 6).

## 2. Pretrained Audio Representations and Feature Extraction

The core technical foundation of this thesis rests on the ability to extract meaningful semantic features directly from raw audio waveforms or spectrograms.

**Tamm et al. (2024)** [1] provide the primary methodological framework for this research. Their comparative analysis of six pretrained models—**MusiCNN**, **MERT**, **Jukebox**, **MusicFM**, **Music2Vec**, and **EncodecMAE**—demonstrates that "frozen" embeddings from these models can effectively drive recommendation tasks. Crucially, **MusiCNN** (a lighter, supervised discriminative model trained on auto-tagging) achieved the best performance across all recommendation methods (HitRate@50: 0.385), significantly outperforming larger generative models like Jukebox (4800-dimensional embeddings). This validates the choice of efficient, discriminative backends for content-based ranking when user data is unavailable.

**Pourmoazemi and Maleki (2024)** [2] propose a **Compact Convolutional Transformer (CCT)** architecture for genre-based recommendation. Their hybrid model combines six convolutional layers (with progressively increasing filters: 32, 64, 128) for local feature extraction from mel-spectrograms, followed by two transformer encoders with multi-head attention (2 heads, 128-dimensional) to capture global interdependencies among feature maps. The CCT achieves **93.75% test accuracy** on the GTZAN dataset while containing only **454,187 parameters**—significantly fewer than state-of-the-art CRNN models. The model outperforms previous architectures across precision (0.923), recall (0.928), and F1-score (0.923) metrics. For recommendation, they use cosine similarity between feature maps extracted from the penultimate layer, demonstrating that the learned representations effectively capture genre-specific patterns for content-based music retrieval.

Complementing the CCT approach, **Lin et al. (2024)** [3] propose a **Transfer Learning + CNN + GRU (TL-CNN-GRU)** model that leverages pretrained **MobileNetV2** weights for spatial feature extraction combined with bidirectional GRU (1024 units) for capturing temporal dependencies in music sequences. Using spectrograms resized to 256×256×3 as input, their architecture applies **10-Fold Cross-Validation (10-FCV)** to mitigate overfitting and improve generalization. The model achieved **71% accuracy** on the GTZAN dataset, with genre-specific F1-scores improving substantially with 10-FCV: blues (0.49→0.74), classical (0.87→0.92), metal (0.61→0.84), and reggae (0.35→0.69), though rock remained challenging (0.26→0.50). This work reinforces the relevance of transfer learning and sequential feature modeling for audio representation in recommendation contexts.

Expanding on the potential of transformer-based architectures for audio, **Ramos et al.** [4] explored **Self-Supervised Learning (SSL)** using the **Audio Spectrogram Transformer (AST)** within a SimCLR framework. Training on the Free Music Archive (FMA) dataset with InfoNCE contrastive loss, they demonstrated that music embeddings can be learned without explicit labels, organizing tracks by composition, timbre, and flow rather than conventional genre classifications. Their qualitative evaluation showed that the trained model achieved 48% satisfactory recommendations compared to only 10% for the untrained baseline, with the learned representations capturing "subtle elements of musical structure" beyond obvious metadata. This supports the thesis that unsupervised embeddings can uncover latent structure in specialized archives such as classical music without manual annotation.

**Reddy et al.** [5] developed **MusicNet**, a compact CNN for real-time background music detection. MusicNet achieves 81.3% TPR at 0.1% FPR while being only 0.2 MB in size—10x smaller than competing models—with 11.1ms inference time. Crucially, MusicNet incorporates **in-model featurization**, processing raw audio directly without requiring external feature extraction. Together with the CCT of Pourmoazemi and Maleki [2], these papers demonstrate that high-performance audio embedding models need not be large, supporting the use of compact model families for backend embedding extraction in content-based systems.

## 3. Related Work: User Modeling and Collaborative Filtering

In settings where user interaction data exists, sequential user modeling and collaborative filtering have been extensively studied. The following work is treated as background; the present thesis does not rely on user data or session modeling.

**Abbattista et al.** [6] study **Personalized Popularity Awareness** and show that complex transformer models can underperform simple baselines when they fail to account for "repeated consumption" (users re-listening to favorites). **Lin et al.** [7] propose a hybrid architecture combining a Deep CNN for audio emotion modeling with a Self-Attention mechanism for user emotion modeling, achieving 82% emotion matching accuracy and 83% recommendation accuracy (Precision@10); the pattern of fusing a static content vector (Audio CNN) with a dynamic context vector (Self-Attention) is relevant when user state is available. **Schedl et al.** [8] identify **Country Archetypes** via geographic listening behavior and unsupervised clustering (t-SNE, OPTICS on 369 million listening events from 70 countries), and their "geo-aware" VAE shows that context-aware models can outperform baseline VAE (4.9–7.4% relative gains in precision, recall, and NDCG). For classical archives without user data, these contributions illustrate the gap that content-based embedding and ranking approaches aim to fill.

## 4. Related Work: Content-Based Systems and Feature Aggregation

Beyond pretrained embedding models, several works address how content-based systems aggregate features and structure recommendation pipelines. These are summarized here as related design context.

**Zhang [10]** proposes a CNN-based system that constructs user preference vectors by aggregating the classification features of listening history. Using MFCC and mel spectrogram features from 400 digital piano pieces (100 per genre) across four genres (classical, pop, rock, pure music), they compare **"Comprehensive"** (single averaged feature vector, 50.35% accuracy) vs. **"Multicategory"** (category-specific vectors, 42.89% overall). The Comprehensive approach achieved higher overall accuracy, illustrating how embedding-space aggregation can support content-based retrieval when preference signals are available; in our setting, analogous aggregation can be applied to item-side embeddings for ranking.

**Lin et al.** [3] demonstrate integration of their TL-CNN-GRU model with external music platforms (YouTube and Spotify APIs) for real-time genre-based recommendations. **Prasad et al.** [11] provide a modular framework for **AI-Powered Recommendation Systems**, separating data collection, feature extraction, and recommendation generation and integrating supervised, unsupervised, and reinforcement learning paradigms. **Dias et al.** [9] validate CNN-based genre classification for recommendation, achieving 76% accuracy on GTZAN and reinforcing that deep learning can replace manual annotation pipelines, making content-based recommendation viable for archives lacking comprehensive metadata.

## 5. Beyond Accuracy: Diversity and Long-Tail Discovery

Traditional recommendation evaluation focuses on accuracy metrics like Precision@K or Recall@K; ranking quality is also captured by NDCG. For specialized archives whose mission is exploratory, diversity and novelty are additional success criteria.

**Porcaro et al.** [12] conducted a **12-week longitudinal study with 110 participants** on the **Impact of Diversity** in music recommendations. Focusing on Electronic Music exposure, they found that high-diversity recommendations significantly increased users' openness to unfamiliar genres, fueled curiosity, and helped deconstruct genre stereotypes. They measured both implicit attitudes (Single Category IAT) and explicit openness (Guttman scale). This is particularly relevant for the **iPalpiti Music Archive**, whose mission is to expose listeners to specialized, potentially unfamiliar classical performances. Evaluation frameworks for such archives may therefore combine ranking metrics (NDCG, Precision@K, Recall@K) with diversity or novelty measures to ensure the system surfaces the "long tail" of the archive.

## 6. Synthesis and Contributions to the Thesis

The twelve papers above provide the theoretical and empirical basis for a V1 thesis focused on backend audio embedding model families, ranking-based evaluation (NDCG, Precision@K, Recall@K), and classical music archives without user interaction data.

**Audio embedding model families**: Papers [1]–[5] establish that pretrained audio representations can drive content-based recommendation. **Tamm et al. [1]** show that frozen embeddings from models such as MusiCNN achieve HitRate@50 of 0.385 and that lighter discriminative models can outperform larger generative ones (e.g., Jukebox). **Pourmoazemi and Maleki [2]** and **Lin et al. [3]** evaluate on **GTZAN** (1000 tracks, 100 per genre), achieving 93.75% and 71% accuracy respectively, demonstrating viability for domain-sized or genre-specific corpora. **Ramos et al. [4]** show that self-supervised learning can capture "subtle elements of musical structure" beyond metadata. **Reddy et al. [5]** show that compact models (0.2 MB) can deliver strong performance, supporting the use of efficient embedding backends.

**Ranking and evaluation**: **Tamm et al. [1]** evaluate recommendation quality with HitRate; the thesis extends this to NDCG, Precision@K, and Recall@K for ranking-based assessment. **Schedl et al. [8]** report gains in precision, recall, and NDCG for context-aware models, illustrating the relevance of these metrics. **Lin et al. [7]** report Precision@10 for hybrid emotion-based recommendation. **Porcaro et al. [12]** argue for diversity and novelty as complementary criteria when evaluating systems for exploratory archives such as iPalpiti.

**Classical music and archives without user data**: **Zhang [10]** use classical (and other) digital piano pieces and show that MFCC and mel spectrogram features retain discriminative power for genre classification. **Lin et al. [3]** report strong classical F1 (0.92) in their genre-specific results. **Dias et al. [9]** and **Prasad et al. [11]** support content-based, metadata-light design. **Ramos et al. [4]** suggest that unsupervised embeddings can capture latent structure in music without labels, aligning with the absence of user interaction data in classical archives.

**Related work (user modeling and deployment)**: **Abbattista et al. [6]**, **Lin et al. [7]**, and **Schedl et al. [8]** are retained as background on user modeling and CF; the thesis does not adopt mock users or session-based training. Deployment-oriented aspects of **Lin et al. [3]** and **Prasad et al. [11]** are treated as related system design rather than as evaluation targets.

In summary, the literature supports **pretrained audio embeddings as the backbone for content-based recommendation** in classical archives, **ranking-based evaluation (NDCG, Precision@K, Recall@K)** as the primary evaluation framework, and **diversity/long-tail discovery** as an additional dimension for exploratory archives. This synthesis provides a roadmap for applying the comparative embedding methodology of **Tamm et al. [1]** to the iPalpiti Music Archive in a V1 scope centered on backend embedding families and ranking metrics.

## References

[1] R. Tamm, M. Sachdeva, and S. Lind, "Comparative Analysis of Pretrained Audio Representations in Music Recommender Systems," in *Proceedings of the 18th ACM Conference on Recommender Systems (RecSys '24)*, Bari, Italy, Oct. 2024.

[2] N. Pourmoazemi and S. Maleki, "A music recommender system based on compact convolutional transformers," *Expert Systems With Applications*, vol. 255, Art. no. 124473, 2024.

[3] P.-C. Lin, C.-Y. Yu, and E. Odle, "Prototype for a Personalized Music Recommendation System Based on TL-CNN-GRU Model with 10-Fold Cross-Validation," in *Proceedings of the 7th Artificial Intelligence and Cloud Computing Conference (AICCC 2024)*, Tokyo, Japan, Dec. 2024, pp. 87-94.

[4] A. C. Ramos and B. A. T. de Freitas, "Self-Supervised Learning of Music Representations for Recommendation Systems," Course Project, Institute of Computing, University of Campinas (UNICAMP).

[5] C. K. A. Reddy, V. Gopal, H. Dubey, S. Matusevych, R. Cutler, and R. Aichner, "MusicNet: Compact Convolutional Neural Network for Real-time Background Music Detection," in *Proceedings of Interspeech 2022*, Incheon, Korea, Sept. 2022.

[6] D. Abbattista, V. W. Anelli, T. Di Noia, C. Macdonald, and A. Petrov, "Enhancing Sequential Music Recommendation with Personalized Popularity Awareness," in *Proceedings of the 18th ACM Conference on Recommender Systems (RecSys '24)*, Bari, Italy, Oct. 2024.

[7] J. Lin, S. Huang, and Y. Zhang, "Deep neural network-based music user preference modeling, accurate recommendation, and IoT-enabled personalization," *Alexandria Engineering Journal*, vol. 125, pp. 232-244, 2025.

[8] M. Schedl, C. Bauer, W. Reisinger, D. Kowald, and E. Lex, "Listener Modeling and Context-Aware Music Recommendation Based on Country Archetypes," *Frontiers in Artificial Intelligence*, vol. 3, Art. no. 508725, Feb. 2021.

[9] J. Dias, H. Deshmukh, V. Pillai, and A. Shah, "Music Genre Classification & Recommendation System using CNN," St. John College of Engineering and Management, Palghar, India.

[10] Y. Zhang, "Music Recommendation System and Recommendation Model Based on Convolutional Neural Network," *Mobile Information Systems*, vol. 2022, Art. no. 3387598, May 2022.

[11] M. S. V. Prasad and G. Sharma, "Music Recommendation System," Parul University, Vadodara, Gujarat, India.

[12] L. Porcaro, E. Gómez, and C. Castillo, "Assessing the Impact of Music Recommendation Diversity on Listeners: A Longitudinal Study," Manuscript, 2022.
