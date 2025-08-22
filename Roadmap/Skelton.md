# Thesis Proposal Skeleton

## Title (Working)
**Learning Classical Music Similarity with Neural Audio Embeddings:  
A Comparative Study of CNN, CRNN, and Pretrained Models with Hybrid Re-ranking**

---

## Research Question
What model and system design choices lead to better classical music recommendations?

---

## Hypotheses

### H1: Pretrained embeddings > training from scratch
*Audio features learned by large, general-purpose models trained on millions of recordings (pretrained embeddings) will produce better classical music recommendations than models trained only on my smaller dataset from scratch.*

- Training from scratch = start with random weights, learn only from my dataset.  
- My dataset is relatively small → risk of underfitting/overfitting.  
- Pretrained models like **BEATs** or **OpenL3** already capture broad audio knowledge (timbre, harmony, rhythm).  
- By reusing these embeddings, I can transfer that knowledge to classical music similarity.  
- **Hypothesis:** recommendations will be more accurate and musically meaningful when using pretrained embeddings.  

---

### H2: CRNN > CNN for capturing temporal structure in classical recordings
*A model that combines convolution (CNN) with recurrent layers (CRNN) will better capture the temporal flow of music and therefore outperform a plain CNN for classical music recommendations.*

- CNNs are good at spotting short-time patterns (like chords or timbres) in spectrograms.  
- But classical music is highly **temporal**: phrasing, long melodies, gradual dynamics.  
- CRNNs add a recurrent layer (GRU/LSTM) on top of CNN features → they remember what came before, modeling time dependencies.  
- **Hypothesis:** because classical pieces rely heavily on structure over time, CRNNs will provide more musically appropriate similarities than CNNs.  

---

### H5: Hybrid re-ranking improves perceived quality for listeners
*After generating candidate recommendations using audio similarity, re-ranking the results with simple rules (e.g., ensure diversity across composers/works, allow metadata filters like “piano only”) will make the recommendations feel more satisfying to listeners, even if raw accuracy is similar.*

- Pure audio similarity may cluster too narrowly (e.g., 10 Beethoven piano sonatas in a row).  
- Listeners often prefer variety and control → different composers, eras, or instruments.  
- A hybrid re-rank step can balance **similarity + diversity**, or enforce metadata constraints.  
- **Hypothesis:** users in a small study will rate hybrid lists as **more enjoyable/interesting** than raw audio-only lists.  

---

## Potential Models (Backbones)

- **CNN (Convolutional Neural Network):**  
  Learns local time–frequency patterns from spectrograms; simple but limited temporal context.  

- **CRNN (Convolutional + Recurrent Neural Network):**  
  Combines CNN feature extraction with recurrent layers (GRU/LSTM) to capture longer-term phrasing and dynamics.  

- **Transformer (self-attention):**  
  Uses attention over spectrogram patches to model long-range dependencies; powerful but computationally heavy.  

- **Pretrained (e.g., BEATs, OpenL3):**  
  Embeddings learned from massive audio datasets; transferred to classical music similarity without (or with minimal) fine-tuning.  

---

## Method

1. **Data & Setup**  
   - Build database with metadata (composer, piece, instrumentation, style, etc.).  
   - Preprocess audio into spectrograms (mel, CQT).  

2. **Model Comparison**  
   - Train/evaluate CNN and CRNN; optionally include Transformer.  
   - Extract pretrained embeddings (BEATs, OpenL3).  
   - Measure similarity using an ANN index (FAISS).  

3. **Hybrid Re-ranking**  
   - Apply diversity-aware and metadata-aware rules to top-K results.  
   - Compare lists before vs. after re-ranking.  

4. **Evaluation**  
   - **Objective:** Precision/Recall@K, NDCG@K, MAP@K (using “same work/composer/instrumentation” rules).  
   - **Subjective:** Small user study (pairwise preference test).  
   - **Efficiency:** Model size, embedding time, query latency.  

---

## Expected Results & Conclusion
- Identify which hypotheses hold true (e.g., CRNN > CNN, pretrained > scratch).  
- Demonstrate trade-offs between accuracy and computation (CRNN vs Transformer).  
- Show that hybrid re-ranking increases listener satisfaction without large accuracy loss.  

---

## Contribution
- A reproducible **benchmark pipeline** for classical music similarity.  
- Comparative analysis of CNN, CRNN, and pretrained embeddings on classical repertoire.  
- Evidence that **hybrid re-ranking** enhances user-perceived quality.  
- Practical system design insights for recommendation in niche music domains.
