# Section 2 — Related Work Script
**Target time: ~3 minutes**

---

## Slide 8 — [Section Cover: Related Work]
*(no script — pause briefly, let the slide breathe)*

---

## Slide 9 — Related Work: Why Embeddings Matter Here
*(~1.5 min)*

The literature is clear on one thing: when interaction data is sparse, collaborative filtering breaks down. Schedl and colleagues established this in 2018, and it's still the baseline assumption in cold-start research. The fallback is content-based retrieval — and for music, that means audio embeddings.

Deldjoo and colleagues in 2024 surveyed content-driven music recommendation and confirmed that pretrained embeddings are now the dominant approach for cold-start candidate generation.

But here's the gap that motivated our work: Tamm and Aljanaki in 2024 showed that pretrained embeddings are effective in RecSys pipelines — but their behavior in small, single-domain settings is largely unstudied. And critically, strong MIR benchmark accuracy does not automatically translate to well-structured retrieval neighborhoods. A model can win on a classification benchmark and still produce embeddings that rank irrelevant items above relevant ones.

---

## Slide 10 — Related Work: CNN vs. Transformer
*(~1.5 min)*

On the architecture side, we're comparing two families.

The CNN family, represented here by PANNs — Pretrained Audio Neural Networks — was introduced by Kong and colleagues in 2020. These models are strong at audio pattern recognition and scale reasonably with depth. Cnn6, Cnn10, and Cnn14 represent small, medium, and large capacity within that family.

The Transformer family, represented by MERT, uses self-attention over spectrograms to capture long-range dependencies. MERT-95M and MERT-330M give us medium and large capacity within that family.

Prior benchmark work has compared these architectures on large, heterogeneous datasets. The gap in the literature is exactly our setting: a small, single-domain archival collection where stylistic homogeneity may fundamentally change how these models behave.
