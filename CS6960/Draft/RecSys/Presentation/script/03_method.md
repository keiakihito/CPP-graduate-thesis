# Section 3 — Method Script
**Target time: ~8 minutes**

---

## Slide 11 — [Section Cover: Method]
*(no script — pause briefly)*

---

## Slide 12 — Evaluation Pipeline Overview
*(~1 min)*

The pipeline is intentionally minimal. You can see the diagram here — there is no personalization layer, no collaborative filtering signal, no learned re-ranking stage. The only variable across all experiments is the embedding model.

The dataset is the full iPalpiti catalog: 203 tracks, 24.9 hours of classical music. Ranking is done by cosine similarity. Query track goes in, the system returns the top-K most similar tracks from the remaining 202.

The design is deliberate — we want to isolate the embedding model as the single variable. Any difference in ranking quality is attributable to the embedding, nothing else.

---

## Slide 12b — Audio Preprocessing: Track-Level Embedding
*(~1 min)*

Before we can rank anything, we need to turn each full recording into a single fixed-size vector. That requires two steps.

First, segmentation. Audio models expect fixed-length input — you cannot feed a 45-minute symphony directly. We split each track into non-overlapping 30-second windows. Thirty seconds is a standard window length in Music Information Retrieval research, long enough to capture musical character, short enough to stay within every model's expected input range.

Each 30-second segment is then passed through the embedding model independently, producing one vector per segment.

Second, mean-pooling. We average all the segment vectors to get a single track-level embedding. This aggregation is identical for all five models — the same code path, the same formula — so the comparison stays fair.

The trade-off is that temporal structure within a recording is not preserved. Whether a piece starts quietly and builds to a climax, or the reverse, looks the same after pooling. That's a known limitation, and I'll come back to it in the Limitations section.

---

## Slide 13 — Formal Problem Setup
*(~1 min)*

For those who want the formal framing: we have a catalog D of 203 tracks. An embedding model maps each track to a vector in R^d. We rank by cosine similarity between the query embedding and all candidate embeddings.

Relevance is binary — either a track is relevant to the query or it isn't — defined per proxy task. For the Sanity task, relevance means same composer. For the Character task, relevance means sharing at least one affective label. I'll explain both in a moment.

---

## Slide 14 — Models Evaluated
*(~1 min)*

We evaluate five models across two architecture families and three capacity tiers.

On the CNN side: Cnn6 at 4.8 million parameters, Cnn10 at 5.2 million, and Cnn14 at 80.8 million. On the Transformer side: MERT-95M at 95 million parameters and MERT-330M at 330 million.

The primary comparison is within-family — CNN-Small versus CNN-Medium versus CNN-Large, and Transformer-Medium versus Transformer-Large. This lets us ask directly: does more capacity within the same architecture improve ranking?

Cross-family comparison is descriptive only. CNN and Transformer models differ in architecture and pretraining objective, so a fair head-to-head isn't the point.

---

## Slide 14a — CNN Family: PANNs Architecture
*(~1 min)*

Let me briefly walk through how each family works, because the architecture difference matters for interpreting the results.

PANNs — Pretrained Audio Neural Networks — are CNN-based models trained on AudioSet, a large-scale audio classification benchmark with 527 sound categories. The pipeline is straightforward: raw audio is converted to a log-mel spectrogram, then a stack of CNN blocks extracts hierarchical local patterns across time and frequency. Global pooling collapses that into a fixed-size vector — that is the embedding we use for retrieval.

Capacity scales by adding more CNN blocks and filters. Cnn6, Cnn10, and Cnn14 differ in depth and filter count, giving us that 4.8 to 80.8 million parameter range — a 17× spread within a single training regime.

---

## Slide 14b — Transformer Family: MERT Architecture
*(~1 min)*

MERT takes a fundamentally different approach. It is self-supervised — trained on 160 thousand hours of music with no human labels. The training objective is masked audio modeling: the model learns to predict masked segments of audio using two teacher signals, one acoustic and one based on musical structure.

The encoder is a Transformer, which means it operates on the full input sequence simultaneously through self-attention, rather than scanning locally like a CNN. This gives it a much longer effective receptive field.

MERT-95M and MERT-330M differ in the number of Transformer layers and attention heads. One important practical note: there is no small MERT checkpoint — the minimum entry point is already 95 million parameters. So unlike PANNs where we have three capacity tiers, MERT gives us only two, and the gap between them is 3.5×.

---

## Slide 15 — Two Proxy Retrieval Tasks
*(~1.5 min)*

Since we have no ground-truth human relevance judgments, we design two proxy tasks using metadata as the relevance signal.

Task 1 is the Sanity Proxy — a structured task. A track is relevant if it shares the same composer. These labels come directly from editorial metadata and are clean and categorical. This is the easiest case for a retrieval system: same composer means similar stylistic DNA.

Task 2 is the Musical Character Proxy — an abstract task. Relevance is defined by shared affective label: Energetic, Calm, Tense, or Lyrical. These labels are generated by Music2Emo, a pretrained music emotion recognition model, and then quantized via the arousal-valence framework. This task is harder because the relevant set per query is much larger — many tracks can share at least one label.

The two tasks let us ask different questions: can the models rank structurally similar tracks? And can they rank emotionally similar ones?

---

## Slide 16 — How Character Labels Were Built
*(~1.5 min)*

Let me walk through how the character labels were constructed, because the design choices here matter for interpreting the results.

Music2Emo outputs a valence score and an arousal score for each track, on a normalized zero-to-one scale. We then threshold those scores to assign binary tags.

We started with a principled, simple choice: split each dimension into even thirds using the 33rd and 67th percentiles. Energetic gets high arousal, Calm gets low arousal, Tense gets low valence.

After plotting the actual VA scores for our 203 tracks — you can see the scatter on the right — we confirmed this was the right call. The data clusters tightly in a narrow mid-range band, roughly 0.25 to 0.55 on both axes. Fixed absolute thresholds would have cut through the densest region and produced highly imbalanced or near-empty label classes. The tertile boundaries land right at the edges of the dense cluster, not through the middle of it.

Lyrical is the special case: it targets the inner bounding box — the 40th to 60th percentile band on both axes — capturing the dense, emotionally restrained center of the distribution. Without it, the most common VA profile in our dataset would go unlabeled entirely.

---

## Slide 17 — Evaluation Protocol
*(~1 min)*

The evaluation uses leave-one-out cross-validation over all 203 tracks. Each track serves as the query once, excluded from the candidate pool, and the remaining 202 are ranked. Metrics are averaged across all 203 queries.

We use four metrics. NDCG@5 and Hit@5 are our primary signals — they're rank-aware and capture whether relevant items appear near the top of the list. Recall@5 and F1@5 are secondary, and I'll explain shortly why they need to be interpreted with caution in this setting.

Ties are handled with averaged ranks to avoid order-dependence artifacts.
