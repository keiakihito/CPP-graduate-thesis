# Section 1 — Introduction Script
**Target time: ~6 minutes**
*Slides 1–3 are improvised (narrative bridge from real-world project to research). Script begins at Slide 4.*

---

## Slide 4 — The Cold-Start Problem in Music RecSys
*(~1.5 min)*

So, when we talk about building a recommendation system for iPalpiti, the very first obstacle we run into is what's called the cold-start problem.

There is no user history. No listening logs, no ratings, no "people who liked this also liked that." Collaborative filtering, which powers most modern recommendation systems, simply has nothing to work with.

What we do have is the audio itself — over 200 tracks, each one a rich, full recording. And that means candidate generation has to rely entirely on content. We extract an embedding from each track, and the system ranks everything by similarity to that embedding.

The implication is stark: embedding quality equals ranking quality. There is no fallback. If the embedding doesn't capture musical similarity well, the whole recommendation pipeline fails.

---

## Slide 5 — "Bigger Models = Better Ranking?"
*(~1 min)*

This brings us to a natural question — and honestly, a natural assumption.

If you look at Music Information Retrieval benchmarks, larger models almost always win. More parameters, better features, better accuracy. The natural instinct is to take the biggest model you can afford and put it in your pipeline.

But our setting is different. We have 203 tracks, not a million. We're running on CPU, not a dedicated GPU cluster. And the pipeline re-ingests the catalog regularly.

So the question we set out to answer is: does that assumption actually hold here?

---

## Slide 6 — Research Questions
*(~1.5 min)*

We formalized that question into three research questions.

RQ1 asks how encoder capacity affects retrieval quality under cold-start conditions — specifically whether the effect is consistent across within-family parameter scaling, across categorical versus semantic retrieval tasks, and across CNN and Transformer architecture families. That last part matters because architecture and pretraining objectives co-vary, so we want to know if the pattern holds regardless.

RQ2 asks how metric choice changes our conclusions. When the number of relevant items per query is large relative to the evaluation cutoff, position-insensitive metrics like Recall@K get structurally compressed — and that can hide genuine ranking differences. We want to understand when that's happening.

RQ3 is the operational one: at what point does the marginal improvement in ranking quality drop below the marginal cost of extraction? This is where the cost-quality trade-off becomes a decision rule.

Our hypothesis going in was that capacity scaling would be non-monotonic, and that performance differences would be too small to justify the extraction overhead.

---

## Slide 7 — Four Contributions
*(~1 min)*

Before moving to related work, let me briefly flag the four things this paper contributes.

First, the cold-start scaling study itself — the first systematic evaluation of audio encoder capacity for candidate generation where no interaction signals are available to compensate for embedding quality. Prior scaling work focuses on heterogeneous benchmark performance; we isolate the regime where embedding quality alone determines ranking.

Second, the non-monotonic, task-dependent scaling finding. Across both families, the largest model never achieves the best ranking quality on either task — and the strength of capacity effects differs sharply between structured and abstract retrieval.

Third, the metric mismatch characterization. When the relevant set is large relative to K, set-based metrics like Recall@K are bounded above by K divided by the number of relevant items — they compress genuine ranking differences. NDCG@K recovers those differences.

And fourth, a concrete cost-quality trade-off result: extraction latency scales near-linearly with parameter count while ranking quality does not, and mid-capacity encoders Pareto-dominate larger variants under realistic latency budgets.
