# Research Question Comparison – Paper vs. Our Version

## 1. Focus Shift
- **Paper:** Investigates whether pretrained audio representations *work effectively* for music recommender systems in general (broad validation across models and architectures).  
- **Our Work:** Examines whether those representations *remain robust and generalizable* when applied to a **small, domain-specific classical music dataset**, focusing on limited data and domain bias.

---

## 2. Analytical Emphasis
- **Paper:** Benchmarks performance across **six pretrained audio models × three recommender models**, emphasizing *relative accuracy and ranking*.  
- **Our Work:** Studies **scaling behavior and overfitting**, emphasizing *stability, generalization, and sensitivity* when user–item data is reduced (e.g., 100%, 50%, 10%).

---

## 3. Research Question Evolution
- **Paper RQ1–RQ3:**  
  1. Are pretrained audio representations viable for MRS?  
  2. How do different backend models compare?  
  3. How does pretrained performance relate to MIR tasks?  

- **Our RQ1–RQ3:**  
  1. How do pretrained audio representations perform under *small-scale or domain-specific* conditions?  
  2. To what extent do models *overfit or generalize* when data is limited?  
  3. How does *dataset size* influence performance ranking among pretrained models?

---

**Summary:**  
→ The paper focuses on *breadth* (cross-model comparison).  
→ Our project focuses on *depth* (robustness and generalization under data scarcity).

---

## Model Architectures (2–3 sentences each)

**KNN (Content-KNN over Item Embeddings).**  
Treat each track as a point in the pretrained embedding space. For a user, aggregate vectors of positive items (e.g., average or weighted sum) and retrieve the **k** nearest items by cosine similarity; no gradient training, strong baseline for small data.

---

**Shallow Net (Frozen Items, Learnable Users).**  
Keep **item embeddings frozen** (from the audio backend) and learn a **user embedding** per user. Score by **cosine similarity** and train with a **hinge-margin loss** using **in-batch or sampled negatives**; this gives a light‑weight collaborative layer that adapts users to the fixed content space and resists overfitting on tiny catalogs.

---

**BERT4Rec (Masked Sequential Transformer with Frozen Projection).**  
Model each user’s listening history as a sequence and apply **bidirectional masking** like BERT to predict hidden items. We keep the **audio→item projection frozen** so the transformer learns sequence patterns without overfitting the content geometry; stronger, but compute‑heavier than Shallow/KNN.

---