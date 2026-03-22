# Chapter 4: Methodology

This chapter describes the experimental design for comparing backend audio embedding model families on classical music similarity and retrieval. The methodology is structured to support reproducibility and to isolate the embedding model as the primary variable while holding preprocessing, indexing, and evaluation protocols constant. The design is intended to be executed at modest scale; specific choices regarding dataset size, model instances, and hyperparameters will be documented at the time of execution.

---

## 4.1 Overall Pipeline

An overview of this pipeline is illustrated in Figure X.

The evaluation follows a single pipeline applied identically for each candidate embedding model:

1. **Audio preprocessing**: Raw audio from the iPalpiti archive is converted into the input representation required by each model (or into a common intermediate format where feasible), with consistent segmentation and normalization rules as described below.
2. **Embedding generation**: Each pretrained model consumes the preprocessed input and produces one or more fixed-dimensional embedding vectors per track.
3. **Indexing**: Embeddings for all tracks in the evaluation set are stored in a similarity-search index. The same index type and configuration are used for every model to avoid confounding retrieval performance with index design.
4. **Ranking**: For each query track, the system retrieves the top-K nearest neighbors by embedding similarity (e.g., cosine or L2) and produces an ordered list of candidate tracks.
5. **Evaluation**: Retrieved rankings are compared against metadata-defined ground truth for the chosen proxy tasks, and ranking metrics are computed.

No component downstream of embedding extraction (indexing, ranking, or evaluation) depends on the model family; the only variable is the embedding model itself.

---

## 4.2 Candidate Model Families

Models are grouped into families to support analysis of architectural influence (e.g., for the exploratory research question on which characteristics most affect ranking performance). The following families are in scope; the exact set of model instances within each family will be fixed at execution time based on availability and computational constraints.

- **CNN-based**: Models whose core is convolutional layers over spectral or waveform input (e.g., mel-spectrogram or raw audio). Includes both supervised discriminative models trained on tagging or classification and compact models designed for efficiency.
- **RNN-based / hybrid CNN–RNN**: Models that combine convolutional feature extraction with recurrent layers (e.g., GRU or LSTM) for temporal modeling. Represent the family that explicitly models sequential structure over time.
- **Transformer / SSL**: Models based on self-attention or trained with self-supervised objectives (e.g., contrastive or masked prediction) on audio. Includes transformer encoders over spectrograms or learned representations.
- **Compact / efficient**: A cross-cutting category for models deliberately designed for low parameter count, fast inference, or small footprint. These may overlap with CNN-based or hybrid families but are treated as a distinct group when analyzing trade-offs between accuracy and resource use.

At least one representative from each family that is feasible to run on the available hardware will be included. The final list of model names, versions, and provenance (e.g., pretrained checkpoint sources) will be reported in the results chapter.

---

## 4.3 Embedding Generation

**Input representation**: Each model expects a specific input format (raw waveform, mel-spectrogram, or other time–frequency representation). Where models differ, preprocessing is performed per model according to its published or reference implementation (e.g., sample rate, window size, hop length, number of mel bins). Where a common representation can be used across multiple models without violating their intended usage, a single preprocessing path may be adopted to reduce variability. The choice (per-model vs common preprocessing) will be stated explicitly so that results are interpretable.

**Segmentation**: Full tracks may exceed the maximum input length of some models. The methodology allows for either (a) segmenting each track into fixed-length or variable-length chunks, extracting embeddings per segment, and aggregating to a single vector per track (e.g., by averaging or max-pooling), or (b) using a single representative segment per track (e.g., a fixed-duration clip from the middle or from multiple non-overlapping windows). The chosen strategy will be applied consistently across all models that require segmentation, and will be documented (including segment length and aggregation method) so that the procedure can be replicated.

**Normalization**: Input normalization (e.g., mean–variance scaling or min–max scaling of spectrograms, or waveform gain normalization) follows the conventions of each model’s reference implementation. Embedding vectors may be L2-normalized before indexing and similarity computation so that cosine similarity and L2 distance are equivalent where appropriate; any such post-processing will be applied uniformly to all models.

No specific hyperparameter values (e.g., exact segment length in seconds, number of mel bins) are fixed here; they will be recorded in the experimental configuration at execution time.

---

## 4.4 Similarity Search and Ranking Procedure

- **Index**: A single type of approximate or exact nearest-neighbor index (e.g., flat exhaustive search or a scalable approximate method) is used for all models. Index build and query parameters will be chosen so that retrieval is consistent and, where applicable, exact enough that differences in reported metrics reflect embedding quality rather than index approximation error.
- **Similarity measure**: Either cosine similarity or L2 distance is used for all models, with embedding normalization applied if necessary so that the two are aligned. The choice will be stated in the results.
- **Query protocol**: For each track in the evaluation query set, the system retrieves the top-K candidates from the index, excluding the query track itself when it appears in the collection. K is chosen to cover the range of cutoffs needed for the evaluation metrics (e.g., K ≥ max of the Precision@K and Recall@K cutoffs used).
- **Ground truth**: Relevance for each query is defined by the proxy task. For example, for a “same work” task, relevant items are all other tracks that share the same work identifier in the metadata. For “same performer,” relevant items share the same performer or ensemble. The exact metadata fields and matching rules will be specified per proxy task when the evaluation set is finalized.

This procedure yields one ranked list per query per model; these lists are the inputs to the evaluation step.

---

## 4.5 Evaluation Metrics

Ranking quality is measured with standard information retrieval metrics:

- **NDCG (Normalized Discounted Cumulative Gain)**: Measures the quality of the ranking with position discount. Relevance grades can be binary (relevant vs not) or graded if multiple levels of relevance are defined. The reported metric will specify the cutoff (e.g., NDCG@K) and how relevance is defined.
- **Precision@K**: Fraction of the top-K retrieved items that are relevant. Reported for one or more values of K.
- **Recall@K**: Fraction of all relevant items that appear in the top-K. Reported for one or more values of K.

Aggregation across queries will be done by averaging (e.g., mean NDCG@K, mean Precision@K, mean Recall@K across all queries in the evaluation set). If the evaluation set is split (e.g., by work or by performer), results may be reported overall and, where informative, per subset.

**Optional — diversity**: If the ground truth or proxy task design allows, a simple diversity or novelty measure (e.g., diversity of retrieved works or performers at K) may be computed to complement the primary ranking metrics. This is optional for V1 and will be included only if feasible without expanding the scope of the thesis.

---

## 4.6 Scope for V1: Fixed vs To Be Expanded

**Fixed in V1 (design commitments):**

- Pipeline structure: preprocessing → embedding extraction → indexing → ranking → evaluation, with the embedding model as the only variable.
- Use of metadata-based proxy tasks for relevance.
- Evaluation metrics: NDCG, Precision@K, Recall@K as the primary metrics; diversity optional.
- Model grouping into the four families above (CNN-based, RNN-based/hybrid, transformer/SSL, compact).
- Consistency rules: same index type and similarity measure for all models; same aggregation and reporting conventions.

**To be expanded or finalized at execution time:**

- Exact list of model names and checkpoints per family.
- Dataset: subset of the iPalpiti archive (number of tracks, metadata coverage, train/validation/test or query/corpus splits).
- Preprocessing: concrete segment length, aggregation method, and per-model parameters where they differ.
- Proxy tasks: final set of tasks (e.g., same work, same performer) and exact metadata definitions.
- Index: library, configuration, and K values used for retrieval.
- Metric cutoffs: specific K values for NDCG@K, Precision@K, Recall@K.
- Computational environment and runtime constraints that determine which models are included.

All such choices will be documented in the thesis (methodology or results chapter) so that the study can be reproduced or extended.
