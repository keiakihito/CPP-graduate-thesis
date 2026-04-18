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

We formalized that question into two research questions.

RQ1 asks whether increasing model capacity consistently improves ranking quality in this cold-start setting. The word "consistently" is doing a lot of work there — we're not asking if a bigger model ever does better, we're asking if it reliably does better across tasks and metric types.

RQ2 asks under what conditions the computational cost of a higher-capacity model actually justifies deploying it. This is the operational framing — if you're building a real system, you need to know when spending ten times more on extraction is worth it.

Our hypothesis going in was that capacity scaling would be non-monotonic, and that the extraction overhead would be impossible to justify given the small dataset.

---

## Slide 7 — Three Contributions
*(~1 min)*

Before moving to related work, let me briefly flag the three things this paper contributes.

First, a systematic evaluation of pretrained audio embeddings as a cost-quality trade-off in cold-start candidate generation — something that, to our knowledge, hasn't been done in this exact framing.

Second, a concrete empirical finding: capacity scaling is non-monotonic and task-dependent. Structured retrieval and abstract retrieval behave quite differently.

And third, a metric mismatch warning that I think is practically important. Recall@K and F1@K can be structurally suppressed in settings where the relevant set per query is large — and ignoring that leads to misleading conclusions about model quality.
