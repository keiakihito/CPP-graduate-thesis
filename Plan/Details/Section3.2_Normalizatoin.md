
Based on your Sprint Plan 3.2 and the provided sources, here is an explanation of Multi-Modal Feature Normalization in this context, followed by a guide on how to describe it academically in your final thesis draft.
1. What is "Normalization" in this context?
Normalization is a data preprocessing technique used to scale different features to a common range (usually 0 to 1, or a standard deviation of 1). In your specific case, you are fusing three very different types of audio data identified in your literature review:
1. Deep Embeddings (MusiCNN/MERT): High-dimensional vectors (e.g., 200 dimensions) representing timbre. These values might range from −10.5 to +10.5.
2. Chroma Vectors (Shi, 2025): 12-dimensional vectors representing harmony (pitch classes C, C#, D...). These are usually probabilities between 0.0 and 1.0.
3. RMS Energy (Shi, 2025): A single scalar value representing loudness/dynamics. This is often very small, e.g., 0.05.
The Problem: If you simply concatenate these into one long vector and feed it into a neural network (your Shallow Net), the model will likely ignore the RMS and Chroma because their numbers are numerically tiny compared to the Embeddings. The model perceives larger numbers as "more important."
The Solution: You apply Normalization (e.g., Z-score or Min-Max scaling) to each feature set independently before fusing them. This ensures that a "loudness" change of 0.1 is treated with the same mathematical weight as a "timbre" change of 0.1.

--------------------------------------------------------------------------------
2. How to Describe it in Your Thesis (Drafting Guide)
In your methodology section (Chapter 3), you should frame this as "Multi-Modal Feature Fusion and Standardization." You are combining Shi's feature selection with Zhang's user modeling strategy.
Here is a structured way to write this section:
A. Justification (Why you chose these features)
Explain that you are combining deep representations with explicit musical features to capture the nuances of classical music.
• Drafting Text: "To comprehensively model the stylistic and emotional content of the iPalpiti archive, we extracted a multi-modal feature set. Following the methodology of Shi (2025), who demonstrated that classical music recognition requires capturing intricate harmonic and dynamic patterns, we extracted Chroma matrices (to represent harmonic progression) and Short-Time RMS energy (to represent dynamic intensity) [Shi: 536, 542]. These explicit features were complemented by latent semantic embeddings extracted using MusiCNN [Tamm: 946], ensuring both high-level timbre and low-level musical structure are represented."
B. The Normalization Process (The Math)
Explain that you normalized these features to prevent numerical instability.
• Drafting Text: "Given the heterogeneous nature of these feature sets—where deep embeddings, Chroma vectors, and RMS values occupy vastly different numerical ranges—direct concatenation would result in gradient dominance by features with larger magnitudes. To mitigate this, we applied Z-score normalization (StandardScaler) to each feature modality independently prior to fusion. This transformation centers each feature distribution around a mean of zero with a unit standard deviation, ensuring that the Shallow Network optimizes weights based on feature variance rather than arbitrary scale."
C. User Modeling via Feature Averaging (Zhang's Strategy)
Explain how you turn these song vectors into a user vector. This is where you cite Zhang.
• Drafting Text: "To generate a unified representation of user preference from the normalized item vectors, we adopted the 'Comprehensive' feature averaging strategy proposed by Zhang (2022). Zhang demonstrated that aggregating feature vectors from a user's listening history into a single mean centroid effectively captures the user's global preferences [Zhang: 258, 273]. By averaging the normalized multi-modal vectors of all tracks in a user's synthetic history, we constructed a fixed-length User Embedding that serves as the input for the Shallow Network recommender."

--------------------------------------------------------------------------------
3. Implementation Logic for Sprint 3.2
To implement this in Sprint 3.2, you will use Python's scikit-learn and numpy libraries.
1. Extract & Store: You have arrays for your tracks:
    ◦ emb (Shape: N x 200)
    ◦ chroma (Shape: N x 12)
    ◦ rms (Shape: N x 1)
2. Normalize (The Step added in v2):
3. Fuse (Concatenate):
4. User Aggregation (Zhang's Logic):
By describing it this way, you show you are not just "stacking data," but carefully engineering the input to respect the specific acoustic properties of classical music (Shi) while using a validated method for user profiling (Zhang).