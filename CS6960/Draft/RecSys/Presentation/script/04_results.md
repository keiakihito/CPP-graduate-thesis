# Section 4 — Results Script
**Target time: ~7 minutes**

---

## Slide 18 — [Section Cover: Results]
*(no script — pause briefly)*

---

## Slide 19 — Results: Composer Retrieval (Structured Task)
*(~2 min)*

Let's look at the results for the Sanity Proxy — the structured task where relevance means same composer.

The first thing to notice is the CNN family. Going from Small to Medium, NDCG@5 actually drops slightly — from 0.548 to 0.545. Then at Large, it recovers to 0.585. That's a non-monotonic pattern. More capacity doesn't reliably help, even within the same architecture family.

The Transformer family is more striking. Transformer-Medium achieves the best NDCG@5 across all models at 0.642. But Transformer-Large — which has three times more parameters — drops to 0.588. More parameters made it worse on structured retrieval.

And here's the punchline: CNN-Small, with just 4.8 million parameters, is only 0.04 NDCG points behind Transformer-Large at 330 million parameters — nearly 70 times fewer parameters for essentially the same ranking quality. The capacity investment buys you nothing.

---

## Slide 20 — Results: Character Retrieval (Abstract Task)
*(~1.5 min)*

Now the Musical Character Proxy — the abstract task where relevance means sharing an affective label.

The picture here is very different. The NDCG@5 spread across all five models is less than 0.025. That's essentially negligible — all models perform about the same on this task. The ranking quality is flat across the entire capacity range.

Hit@5 is high for all models, at 0.749 or above. But Recall@5 is uniformly very low — 0.039 or below.

That gap between Hit@5 and Recall@5 is not a model problem. It's a structural metric problem, and I want to explain what's happening there.

---

## Slide 21 — Why Is Recall@5 So Low? (Metric Mismatch)
*(~1.5 min)*

In the Character task, many tracks share at least one affective label. The relevant set per query can be quite large — tens of tracks. When you only retrieve K equals 5, Recall@5 is structurally capped at a small fraction of the relevant items, no matter how good the ranking is.

Think of it this way: if there are 40 relevant tracks for a given query, and you retrieve 5, perfect ranking gives you Recall@5 of 5/40, which is 0.125. Our models average 0.039, which means they're capturing fewer than 2 of those 40. Recall and F1 look terrible.

But Hit@5 at 0.749 tells us that models are, in fact, ranking at least one relevant item in the top 5 three quarters of the time. The models are working — the metric is misleading.

This is why we treat Recall@5 and F1@5 as secondary signals with caution in this setting. NDCG@5 and Hit@5 are the reliable signals here.

---

## Slide 22 — ROI Collapses at High Capacity
*(~2 min)*

Now the cost side. This is where the argument becomes very concrete.

Look at the table on the left. CNN-Small extracts an embedding in about 2,179 milliseconds per track. Transformer-Large takes 55,724 milliseconds — that's approximately 25 times slower.

And what do you get for that 25-fold increase in extraction time? On the Composer task, Transformer-Large scores 0.588 — worse than Transformer-Medium at 0.642. On the Character task, it scores 0.631 — essentially tied with CNN-Small at 0.653.

The scatter plot on the right makes this visually clear. As you move right along the X-axis — increasing latency — the NDCG@5 on the Y-axis doesn't go up. For the Composer task, it actually peaks at Transformer-Medium and then falls. For the Character task, it's flat the whole way.

CNN-Small sits at the bottom left: lowest cost, competitive quality on both tasks.

This is the ROI collapse. The investment in extraction compute is not converting into ranking quality.
