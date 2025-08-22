# Abstract
This thesis investigates how different neural network architectures and system design choices affect the quality of classical music recommendation. The central research question is: What model and system design choices lead to better classical music recommendations? To answer this, I compare three approaches: <b>
```
- (1) convolutional neural networks (CNNs) trained from scratch, 
- (2) convolutional–recurrent neural networks (CRNNs) designed to capture the temporal flow of music
- (3) state-of-the-art pretrained audio embeddings (e.g., BEATs, OpenL3). 
```
<b>In addition, I explore a hybrid re-ranking method that applies simple diversity and metadata constraints to refine the recommendation list. Using a dataset of classical recordings with curated metadata, I evaluate these models through both objective retrieval metrics (precision, recall, NDCG) and a small user study measuring listener preferences. 

<b>The hypotheses guiding this work are that pretrained embeddings will outperform models trained from scratch, that CRNNs will outperform plain CNNs due to their ability to model long-term structure, and that hybrid re-ranking will improve user satisfaction. The expected contribution is a reproducible benchmark pipeline for classical music similarity and practical insights into how architecture and design choices shape recommendation quality in a niche domain.

## Potential Thesis titile (Working)
**Learning Classical Music Similarity with Neural Audio Embeddings:  
A Comparative Study of CNN, CRNN, and Pretrained Models with Hybrid Re-ranking**
