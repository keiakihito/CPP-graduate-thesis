1. Refined Problem Statement (V1)

Problem Statement

Learned audio embeddings are widely used to support music recommendation and similarity search in modern streaming platforms. However, much of the existing research and many industrial systems are optimized for large, heterogeneous, and often pop-centric music catalogs. As a result, it remains unclear how different audio embedding model families behave when applied to more homogeneous yet structurally complex domains such as classical music.

This limitation is particularly relevant for curated classical music archives such as the iPalpiti collection, where reliable similarity search and ranking among works, performances, and artists must rely primarily on audio content rather than large-scale user interaction data. Understanding how backend audio representations perform in this setting is essential for designing effective content-based retrieval and recommendation systems for classical music.

This thesis investigates the performance of alternative backend audio embedding models on classical music similarity and retrieval tasks. Using offline proxy tasks and standard ranking metrics (e.g., NDCG, Precision@K, Recall@K), we systematically compare multiple embedding model families on the same classical music corpus. The goal is to identify which architectural and representational choices most strongly influence ranking performance in this domain and to provide practical guidance for building content-based systems for classical music archives.

2. Research Questions

RQ1 (Primary Research Question)
For a classical music archive (iPalpiti), how do different backend audio embedding model families (e.g., CNN-based, RNN-based, and alternative architectures) compare in terms of ranking performance on proxy similarity and retrieval tasks, as measured by NDCG, Precision@K, and Recall@K?

RQ2 (Optional, Exploratory)
Which characteristics of the embedding models (e.g., temporal modeling strategy, input representation, or embedding dimensionality) are most associated with improved ranking performance on these classical music proxy tasks?

Note: RQ1 alone defines a complete MVP-scale master’s thesis. RQ2 provides an additional analytical layer and may be treated as exploratory.


3. Objectives (V1)

This research aims to:

1. Evaluate multiple backend audio embedding model families
Apply and assess several audio embedding architectures (e.g., CNN-based, RNN-based, and alternative models) on the iPalpiti classical music archive using a consistent evaluation pipeline.

2.  Compare models using ranking-based proxy tasks
Measure and compare model performance on classical music similarity and retrieval tasks using ranking metrics such as NDCG, Precision@K, and Recall@K.

3. Derive practical recommendations for classical music archives
Analyze experimental results to provide actionable guidance on selecting and deploying audio embeddings for content-based recommendation and search in classical music collections, optionally demonstrating a minimal backend pipeline for similarity search.