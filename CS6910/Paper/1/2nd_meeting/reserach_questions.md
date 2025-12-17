# Thesis Discussion Note 2 – Refining Research Questions

## 1. Background
The base paper, *Comparative Analysis of Pretrained Audio Representations in Music Recommender Systems* (Tamm et al., RecSys 2024), evaluates six pretrained audio models (MusiCNN, MERT, EncodecMAE, Music2Vec, MusicFM, Jukebox) with three recommender architectures (KNN, Shallow Net, BERT4Rec) using the **Music4All-Onion** dataset. Their work demonstrates that pretrained audio embeddings can enhance recommendation performance but exhibit significant variability across models and tasks.

Our project extends this methodology to a **small, domain-specific dataset** — a collection of **classical music CDs** — to examine whether pretrained representations remain effective under data scarcity. This direction directly responds to our advisor’s suggestion to explore **model robustness and overfitting behavior** under limited data conditions.

---

## 2. Comparison of Research Focus
| Aspect | Tamm et al. (2024) | Proposed Extension (Our Work) |
|--------|---------------------|-------------------------------|
| Dataset | Large-scale, multi-genre (Music4All-Onion) | Small-scale, single-genre (Classical CDs) |
| Research Emphasis | Benchmarking pretrained models across architectures | Studying the effect of **dataset size** and **domain constraints** |
| Core Question | Are pretrained audio representations viable for MRS? | Are pretrained audio representations **robust** and **generalizable** with small data? |
| Evaluation Scope | 6 pretrained backends × 3 recommenders | Subset of top-performing models (e.g., MusiCNN, MERT, EncodecMAE) × scaled dataset variations |
| Expected Challenge | Performance ranking across models | Overfitting, reduced variance, and domain homogeneity |

---

## 3. Revised and Extended Research Questions

### **RQ1 (Revised from Tamm RQ1)**  
**How do pretrained audio representations perform in recommendation tasks when trained and evaluated on *small-scale* or *domain-specific* datasets (e.g., classical music)?**  
→ Focus: transfer learning robustness under limited user–item interactions.

---

### **RQ2 (New – Overfitting and Generalization)**  
**To what extent do different pretrained models overfit or generalize when the dataset size is reduced?**  
→ Experiment: progressively subsample user–item data (e.g., 100%, 50%, 10%) and track performance degradation.

---

### **RQ3 (New – Scaling Sensitivity)**  
**How does dataset size influence the relative performance ranking among pretrained audio models?**  
→ Investigation: determine if simpler or more expressive models adapt better to low-resource conditions.

---

### **RQ4 (Optional – Qualitative Extension)**  
**Which aspects of audio embeddings (genre, timbre, dynamics) remain robust or degrade under small-sample fine-tuning or transfer?**  
→ Supplementary analysis: visualize embedding spaces using PCA/t-SNE to assess stability across dataset scales.

---

## 4. Expected Contributions
1. **Empirical insight** into the *scaling limits* of pretrained audio representations for recommender systems.  
2. **Evaluation framework** for testing overfitting and generalization in small-domain music datasets.  
3. **Guidelines** for applying transfer learning in low-resource music recommendation settings.

---

## 5. Next Steps
- Finalize the dataset schema (CD collection + mock user logs).
- Select 2–3 pretrained models for initial replication (likely MusiCNN, MERT, EncodecMAE).  
- Prepare simulation pipeline to vary dataset size.  
- Draft experiment outline for testing overfitting and stability.

---
**Prepared for:** First thesis follow-up discussion  
**Author:** Keita Katsumi  
**Date:** October 2025

