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

But here's the gap that motivated our work: Tamm and Aljanaki in 2024 showed specifically that pretrained representations vary substantially in how well they structure embedding neighborhoods for retrieval — and that this variation has not been benchmarked across model capacity tiers. Critically, strong MIR benchmark accuracy does not automatically translate to well-structured retrieval neighborhoods. A model can win on a classification benchmark and still produce embeddings that rank irrelevant items above relevant ones.

---

## Slide 10 — Related Work: CNN vs. Transformer
*(~1.5 min)*

On the architecture side, we're comparing two families, and the concrete differences in how they scale matter for interpreting the results.

The CNN family is represented by PANNs — Pretrained Audio Neural Networks — from Kong and colleagues in 2020. These models are pretrained on AudioSet and scale via depth: Cnn6 has 6 convolutional layers and 4.8 million parameters, Cnn10 has 10 layers and 5.2 million, and Cnn14 has 14 layers and 80.8 million. That's a 17-times parameter spread within a single shared training regime — which is unusually large and makes within-family comparison well-controlled.

The Transformer family is represented by MERT, from Li and colleagues in 2023, trained via self-supervised learning on 160,000 hours of music. MERT-95M and MERT-330M give us a 3.5-times parameter increase. One important constraint: there is no small-capacity MERT checkpoint publicly available, so the Transformer family starts at 95 million parameters — we can't compare small versus large the way we can for CNN.

AST — the Audio Spectrogram Transformer — applies self-attention directly to spectrogram patches without any convolutional inductive bias. Because attention scales quadratically with the number of patches, AST variants are substantially more expensive than CNN counterparts at equivalent depth.

Prior scaling studies on these architectures report classification accuracy on heterogeneous benchmarks — AudioSet, ESC-50. That's not retrieval quality in a constrained domain. Translating those conclusions directly to archival candidate generation is unreliable, which is exactly the gap we address.
