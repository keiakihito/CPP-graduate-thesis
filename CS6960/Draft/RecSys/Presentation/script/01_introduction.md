# Section 1 — Introduction Script
**Target time: ~6 minutes**
*Slides 1–2 are improvised (narrative bridge from real-world project to research). Script begins at Slide 3.*

---

## Slide 3 — The Real-World Constraint
*(~1 min)*

So imagine a user finishes listening to a Beethoven piano sonata on iPalpiti. What should play next?

This is actually the core value a recommendation system delivers — what Spotify calls serendipity. Not just playing something the user already knows, but surfacing a piece they have never heard before yet will almost certainly love. That moment of discovery is why recommendation systems matter.

And there's neuroscience behind this. Research on music and the brain shows that dopamine release — the reward response — peaks not at the familiar and not at the random, but at the sweet spot in between: something that feels connected to what you just heard, but still surprises you. That's the target. That's what we're trying to engineer.

To hit that sweet spot, the system needs to know what is musically similar to what the user just heard. And that is exactly where our problem starts.

---

## Slide 4 — The Cold-Start Problem in Music RecSys
*(~1.5 min)*

Computing musical similarity requires user data in most systems. Collaborative filtering says: people who listened to this also listened to that. But for iPalpiti, there is no user history. No listening logs, no ratings, no behavioral signal of any kind.

What we do have is the audio itself — over 200 tracks, each one a rich, full recording. And that means candidate generation has to rely entirely on content. We extract an embedding from each track, and the system ranks everything by similarity to that embedding.

The implication is stark: embedding quality equals ranking quality. There is no fallback. If the embedding doesn't capture musical similarity well, the whole recommendation pipeline fails — and that serendipitous discovery moment never happens.

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
