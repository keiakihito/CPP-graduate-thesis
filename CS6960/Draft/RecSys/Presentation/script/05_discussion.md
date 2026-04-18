# Section 5 — Discussion Script
**Target time: ~5 minutes**

---

## Slide 23 — [Section Cover: Discussion]
*(no script — pause briefly)*

---

## Slide 24 — Discussion: What This Means
*(~1.5 min)*

So what do we take from these results?

The first finding is that task structure matters more than model size. On the Composer task — a structured, categorical retrieval problem — there is variation across models and the Transformer-Medium performs best. On the Character task — an abstract, affective retrieval problem — all models converge to the same performance level. The embedding model's capacity simply doesn't change what it can capture about emotional character in a 203-track classical archive.

The second finding is that the largest model never achieves the best ranking quality on either task. Not once, in either evaluation. That's a strong result.

This aligns with what Tamm and Aljanaki found in 2024: MIR benchmark accuracy does not predict retrieval quality in RecSys settings. The iPalpiti archive is stylistically homogeneous — almost all classical chamber and orchestral music. We think this homogeneity compresses the embedding space, leaving very little room for additional capacity to provide a meaningful signal.

---

## Slide 25 — System-Level Implications
*(~1.5 min)*

For anyone designing a real cold-start recommendation system, these findings have direct operational meaning.

In cold-start settings, the embedding model is the ranking system. There is no collaborative signal to compensate for poor candidates. If your embeddings don't rank well, the user gets bad recommendations, period. That makes model selection a direct operational decision, not just a research hyperparameter.

And the practical recommendation from our data is clear: mid-sized models offer the best cost-quality profile. CNN-Small and CNN-Medium are within 0.04 NDCG points of Transformer-Medium on the Composer task, at roughly ten times lower extraction latency.

For a system like iPalpiti — no GPU infrastructure, catalog updates whenever new recordings are digitized — this difference is not academic. A 25-times extraction overhead is a real operational cost that needs to be justified by a real ranking improvement. We found no such justification.

---

## Slide 26 — Limitations
*(~1 min)*

A few honest caveats.

The dataset is small at 203 tracks. This is intentional — it's a controlled stress test — and we argue that if capacity scaling fails in a small setting, it likely fails in larger homogeneous archives too. But we haven't tested that claim directly.

We evaluated the candidate generation stage in isolation. Downstream re-ranking was not evaluated. It's possible that small NDCG differences here compound or cancel at the system level.

The character labels come from a pretrained model and carry whatever biases Music2Emo has. However, because the labels are fixed before evaluation and reused identically across all embedding models, the bias is shared equally — relative comparisons between models remain valid.

And mean-pooling may dilute temporal information in longer recordings. More sophisticated aggregation might change the results.

---

## Slide 27 — Future Work
*(~1 min)*

There are four directions we find most interesting.

Scaling up to larger, heterogeneous catalogs is the most important next step. Does the non-monotonic pattern persist when the embedding space is more spread out?

Richer labels — either expert annotations or listener-derived similarity — would improve the Character task's sensitivity and reduce dependence on pseudo-labels.

Sequence-aware aggregation beyond mean-pooling would address the temporal limitation directly.

And end-to-end evaluation — connecting candidate generation to downstream re-ranking — would tell us whether these NDCG differences actually matter at the system level for users.
