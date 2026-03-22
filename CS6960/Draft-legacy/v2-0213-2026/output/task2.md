### 1. Parts that directly support backend audio embedding comparison (keep, maybe re‑order)

- **Section 2 – Pretrained Audio Representations and Feature Extraction (core of V1)**  
  - **Tamm et al. [1]**: direct benchmark of multiple pretrained audio models; perfect anchor for “model families” and frozen embeddings.  
  - **Pourmoazemi & Maleki [2] (CCT)**: CNN+Transformer hybrid; strong example of lightweight CNN/Transformer family.  
  - **Lin et al. [3] (TL‑CNN‑GRU)**: hybrid CNN+RNN; good for “RNN‑style / temporal modeling” family.  
  - **Ramos et al. [4] (AST + SSL)**: self‑supervised transformer embeddings; good for “other architectures / SSL” family.  
  - **Reddy et al. [5] (MusicNet)**: compact CNN; relevant for “efficient/compact CNN backends”.  
  - **Action**:  
    - Keep all of these.  
    - Reframe this section explicitly as **“Backend Audio Embedding Model Families for Music Recommendation”** and group papers by architecture (CNN, CNN+RNN, CNN+Transformer, SSL/Transformer, compact models).

- **Pieces of Section 4 – End‑to‑End System Architectures and Deployment Strategies**  
  - **Zhang [10]**: CNN features aggregated into user preference vectors → supports “how embeddings are used for retrieval / ranking”.  
  - **Dias et al. [9]**: CNN genre classification → additional CNN evidence.  
  - **Action**:  
    - Keep these, but move/merge them under the **CNN family / backend usage** subsection instead of a full “deployment” section.

- **Pieces of Section 5 – Beyond Accuracy: Diversity and Long‑Tail Discovery**  
  - **Porcaro et al. [12]**: motivates **diversity/novelty metrics**; still relevant to how you evaluate embeddings (complementary to NDCG, Precision@K, Recall@K).  
  - **Action**:  
    - Keep as a **short evaluation‑metrics subsection** (“Beyond ranking accuracy”) that justifies considering diversity/novelty for classical archives.

---

### 2. Parts to de‑emphasize, reframe, or move to background/related work

- **Section 1 – Introduction & Thesis Objectives**  
  - Heavy emphasis on **collaborative filtering, synthetic user sessions, sequential models, AWS cost‑efficiency, dataset size questions (old RQs)**.  
  - **Action**:  
    - Keep only a **short motivation**: CF dominates, but small classical archives lack user data → need content‑based, embedding‑driven methods.  
    - Remove or sharply compress: simulated user sessions, KNN vs shallow NN vs sequential models, AWS serverless cost metrics, dataset‑size framing.  
    - Replace RQ1–RQ4 text with your new **V1 RQ1/RQ2**.

- **Section 3 – Sequential User Modeling and Recommendation Logic**  
  - **Abbattista et al. [6]**, **Lin et al. [7]**, **Schedl et al. [8]** focus on **user modeling, emotion modeling, popularity, country archetypes, CF/NCF**.  
  - These are no longer central because V1 is **backend‑only embeddings + offline proxy tasks**, not end‑to‑end user sequence modeling.  
  - **Action**:  た
    - Move this to a **very short “Related Work: User and Context Modeling”** subsection near the end.  
    - Summarize at a high level: “There is a parallel line of work that models users, emotions, and context; this thesis instead holds the user side fixed and focuses on backend embeddings.”  
    - De‑emphasize any details about mock user generation, popularity‑awareness, or geo‑aware modeling; they are now background.

- **Most of Section 4 – System Architectures and Deployment Strategies**  
  - **Lin et al. [3] deployment**, **Prasad et al. [11] architecture**, AWS Lambda/Fargate framing → strongly tied to V0 “production‑ready API” objective.  
  - **Action**:  
    - Either drop these details or compress them into a tiny **“Implementation Considerations (Optional)”** subsection that says: similar systems deploy these models as microservices; in this thesis deployment is out of scope / only a minimal pipeline.  
    - Emphasize only what informs **how embeddings are computed/stored/queried**, not cloud cost or full microservice architecture.

6. Synthesis and Positioning of This Thesis

6.1 Summary of Evidence Across Audio Embedding Model Families
- CNN-based models
- Hybrid CNN-RNN models
- Transformer / SSL-based models
- Compact vs large models

6.2 Evaluation Gaps in Existing Literature
- Heavy reliance on classification accuracy
- Limited use of ranking-based proxy tasks
- Lack of focus on classical music archives

6.3 Positioning of This Thesis
- Focus on backend-only embedding comparison
- Use of ranking metrics (NDCG, Precision@K, Recall@K)
- Classical music as a domain-specific testbed

---

### 3. Proposed revised literature review structure (titles + rationale)

You can keep “Section 1: Introduction” and then re‑structure Sections 2–6 like this:

- **1. Introduction and Motivation**  
  - **Purpose**: Motivate content‑based recommendation for small classical archives and briefly contrast with CF.  
  - **Reuse**: First part of current Sec. 1 (CF dominance, cold‑start for iPalpiti), trimmed.  
  - **Rationale**: Sets up *why* backend audio embeddings matter for iPalpiti.

- **2. Backend Audio Embedding Model Families for Music Recommendation**  
  - **2.1 CNN‑based and Compact Models**  
    - Use Tamm’s MusiCNN results, MusicNet [5], Dias [9], Zhang [10].  
    - **Rationale**: Establish CNN and compact CNNs as a core, efficient family for audio embeddings.  
  - **2.2 Hybrid CNN‑RNN and Temporal Modeling**  
    - Use TL‑CNN‑GRU [3].  
    - **Rationale**: Represent “RNN‑style / temporal” family and show how temporal modeling is layered atop spectral CNN front‑ends.  
  - **2.3 CNN‑Transformer Hybrids and Pure Transformers**  
    - Use CCT [2], AST/SSL [4].  
    - **Rationale**: Capture transformer‑based or hybrid architectures as another family, including self‑supervised approaches.  
  - **2.4 Self‑Supervised and Transfer‑Learning Approaches**  
    - Highlight transfer from ImageNet (TL‑CNN‑GRU) and self‑supervised AST [4].  
    - **Rationale**: Show how pretraining/SSL change the embedding properties, relevant when you compare families in V1.  

- **3. Evaluation Strategies for Music Embeddings**  
  - **3.1 Classification‑Based Evaluation (Accuracy, F1, etc.)**  
    - Briefly summarize how many works evaluate embeddings via genre/emotion classification (GTZAN accuracies, emotion matching).  
    - **Rationale**: Show the dominant evaluation style and its limitations for ranking‑based recommendation.  
  - **3.2 Ranking‑Based Evaluation for Recommendation (NDCG, Precision@K, Recall@K)**  
    - Use Tamm et al. (HitRate, NDCG) and any others that approximate ranking.  
    - **Rationale**: Bridge from classification to the ranking metrics you will actually use in V1.  
  - **3.3 Beyond Accuracy: Diversity and Long‑Tail Discovery**  
    - Place Porcaro et al. [12] here.  
    - **Rationale**: Argue why classical archives care about diversity/novelty in addition to ranking accuracy.

- **4. System‑Level and User‑Modeling Work (Background / Related Work)**  
  - **4.1 User and Context Modeling**  
    - Highly compressed view of Abbattista [6], Lin emotion+NCF [7], Schedl country archetypes [8].  
    - **Rationale**: Acknowledge broader ecosystem of user modeling, while stating that V1 isolates the backend embeddings.  
  - **4.2 Deployment and System Architectures (Optional Background)**  
    - Very short summary of Prasad [11], Lin deployment [3], AWS‑style designs.  
    - **Rationale**: Shows that embeddings can be integrated into real systems; in V1 this is secondary / possible future work.

- **5. Synthesis and Positioning of This Thesis**  
  - **5.1 Summary of Evidence Across Model Families**  
    - Synthesize which families seem strong/weak in prior work (CNN vs hybrid vs transformer vs compact).  
  - **5.2 Gaps in Current Literature for Classical Archives**  
    - Emphasize:  
      - lack of **systematic comparison** of backend families on **classical** music;  
      - limited use of **ranking‑based proxy tasks** for archives without user data.  
  - **5.3 Link to Research Questions and Objectives**  
    - Tie directly to V1 RQ1/RQ2 and objectives (backend comparison, ranking metrics, classical domain).

This structure lets you:

- Reuse almost all citations.  
- Shift emphasis away from **CF / sequential user modeling / AWS**.  
- Make the lit review clearly support **“we compare backend audio embedding families on ranking‑based tasks for a classical archive”**.