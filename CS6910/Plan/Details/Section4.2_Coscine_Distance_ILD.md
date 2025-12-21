Based on your Sprint Plan 4.1 and the provided sources (specifically Porcaro et al.), here is the explanation of Intra-List Diversity (ILD) using Cosine Distance, followed by a guide on how to describe and implement it in your thesis.
1. What is "Cosine Distance ILD"?
Intra-List Diversity (ILD) measures how different the items inside a single recommendation list are from each other.
• The Context: If your system recommends 10 songs, and they are all "Slow Violin Sonatas by Bach," the list is accurate but has low diversity. If the list contains a mix of "Violin Sonatas," "Fast Orchestral Symphonies," and "Modern Cello Solos," it has high diversity.
• The Metric: You calculate this using Cosine Distance on the audio embeddings (MusicNN/MERT vectors).
    ◦ Cosine Similarity: How alike two vectors are (1.0 = identical, 0.0 = orthogonal).
    ◦ Cosine Distance: How different they are (1−Similarity).
• The Goal: You want a score that tells you if your model is trapping users in a "filter bubble" (low distance) or encouraging discovery (high distance).

--------------------------------------------------------------------------------
2. How to Describe it in Your Thesis (Drafting Guide)
In your Evaluation section (Chapter 4), you should frame this as "Quantifying Discovery and Openness." You are using ILD as a mathematical proxy for the psychological concepts discussed by Porcaro et al.
Here is a structured way to write this section:
A. Justification (Why diversity matters)
Explain that accuracy (HitRate) isn't enough; you need to prove your system exposes users to the breadth of the archive.
• Drafting Text: "While accuracy metrics such as HitRate assess the system's ability to match user preferences, they do not measure the system's ability to facilitate discovery. Porcaro et al. (2022) demonstrate that exposure to diverse music recommendations significantly increases listeners' 'openness' to unfamiliar genres and helps deconstruct stereotypes associated with specific musical styles
. To ensure the iPalpiti recommendation engine fosters this exploration rather than reinforcing 'filter bubbles'
, we evaluated the Intra-List Diversity (ILD) of the generated playlists."
B. The Metric Definition (How you measured it)
Define ILD mathematically using the embeddings you generated in Phase 2.
• Drafting Text: "We defined Intra-List Diversity as the average pairwise dissimilarity between all tracks in a recommended list L. We utilized Cosine Distance in the latent embedding space (extracted via MusicNN/MERT) as the dissimilarity measure. For a list of size N, the ILD is calculated as: ILD(L)=N(N−1)1​i∈L∑​j∈L,j=i∑​(1−CosineSimilarity(vi​,vj​)) where vi​ and vj​ are the normalized feature vectors of tracks i and j. A higher ILD score indicates a recommendation list that covers a broader semantic range of the archive."
C. Interpreting the Result (What it means)
Explain how this connects back to the "Cold Start" and "Discovery" goals.
• Drafting Text: "By optimizing for ILD alongside accuracy, the system addresses the 'concentration effect' often observed in sequential recommenders
. This ensures that users are exposed to the 'long tail' of the iPalpiti archive, aligning with Porcaro et al.'s findings that diversity generates curiosity and willingness to explore unfamiliar content
."

--------------------------------------------------------------------------------
3. Implementation Logic for Sprint 4.1
To implement this in Sprint 4.1, you can use scikit-learn or scipy. You don't need to write the distance formula from scratch.
1. Generate Recommendations: Your model outputs a list of Top-K track IDs for a user.
2. Retrieve Vectors: Fetch the embeddings corresponding to those IDs.
3. Calculate ILD:

import numpy as np
from scipy.spatial.distance import pdist

def calculate_ild(recommendation_vectors):
    """
    Calculates Intra-List Diversity (ILD) for a list of track vectors.
    
    Args:
        recommendation_vectors (np.array): Matrix of shape (N_tracks, Embedding_Dim)
                                           e.g., (10, 200) for MusicNN
    Returns:
        float: Average Cosine Distance (0.0 to 1.0)
    """
    # 1. Calculate pairwise cosine distances between all tracks in the list
    # 'pdist' computes distances between all pairs: (0,1), (0,2)... (N, N-1)
    # metric='cosine' returns 1 - cosine_similarity
    distances = pdist(recommendation_vectors, metric='cosine')
    
    # 2. Average the distances
    if len(distances) == 0:
        return 0.0
    
    avg_distance = np.mean(distances)
    
    return avg_distance

# Example Usage in your Evaluation Loop
# vectors = [embedding_track_A, embedding_track_B, ..., embedding_track_J]
# diversity_score = calculate_ild(vectors)

Why this satisfies the requirement:
• It solves the "Abstract Metric Issue" by converting the abstract concept of "Diversity/Openness" (from Porcaro) into a concrete number (avg_distance).
• It is computationally efficient (using pdist), fitting your AWS Lambda constraints.
• It directly uses the embeddings you already validated in Phase 2.