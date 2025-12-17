### What HitRate@50 Measures
- Evaluates if a user got **at least one correct song** among the Top-50 recommendations.  
- Computed *offline* using the user’s known future listens (validation/test).  
- **Higher HitRate@50 ⇒ model predicted future preferences more accurately.**

---

### KNN Logic (+ Where Sorting Happens)
- **Step 1:** Build each user’s *taste vector* = average of embeddings of songs from the training period. (`mean(axis=0)`)
- **Step 2:** Normalize embeddings → inner product becomes cosine similarity.  
  ```python
  index = faiss.IndexFlatIP(item_embs.shape[1])
  index.add(item_embs)
  distances, indices = index.search(np.array([user_vector]),
                                   k + len(history))
  ```
  → **FAISS** computes cosine(user, song) for all songs and **sorts** them by similarity (highest first).  
- **Step 3:** Remove songs the user already listened to (`if idx not in history`) and take Top-50 new songs → final recommendations.

---

### Concrete Example (Hit = 1)
**Training:** user listened → [A, B, C]  
**Catalog:** [A, B, C, D, E, F, G, H, I]  
**user_history = {A, B, C}**

**FAISS raw ranking:** [C, B, D, E, A, F, G, H, I]   
→ Filter out {A, B, C} = [D, E, F, G, H, I] = recommendations  

**Future listens (test):** [E, K]  
✅ Because E ∈ recommendations → HitRate@50 = 1 for this user.  
If no overlap → HitRate@50 = 0.

