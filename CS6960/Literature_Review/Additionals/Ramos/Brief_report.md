Self-Supervised Learning for Music Recommendation Systems: A Synthesis

Executive Summary

This document synthesizes findings from a project exploring self-supervised learning (SSL) for music recommendation systems. The core of the work involves training an Audio Spectrogram Transformer (AST) model with a SimCLR framework to generate robust music embeddings without relying on labeled data. The primary objective was to evaluate how effectively these learned representations capture human-perceived musical similarity.

Qualitative evaluations demonstrate the system's success, with recommendations from the trained model achieving a 48% "Satisfactory" rating, a marked improvement over the 10% rating for an untrained baseline model. A key insight from the analysis is that the model learned to group music based on subtle structural features rather than explicit metadata like genre. The embeddings formed distinct clusters that were independent of genre labels, suggesting that genre alone may not be the optimal criterion for measuring musical similarity. The project successfully validates a promising, label-free approach to creating more nuanced and effective music recommendation engines.

1. Project Overview and Objectives

The project addresses the challenges faced by traditional music recommendation systems, which often rely on collaborative filtering or metadata-based approaches and encounter scalability issues with large music databases. Self-supervised learning presents a powerful alternative, enabling the generation of generalized embeddings directly from audio data without the need for manual labels.

Primary Objectives:

* To explore the application of the Audio Spectrogram Transformer (AST) model, trained from scratch using the SimCLR contrastive learning framework, to derive music embeddings.
* To evaluate how effectively these self-supervised representations capture the concept of musical similarity as perceived by humans.
* To assess the quality of recommendations produced by a simple system leveraging these embeddings.

2. Technical Methodology

The methodology combines a state-of-the-art audio model with a contrastive learning framework, applied to a standard music dataset. Special attention was given to data pre-processing and augmentation to ensure the model learns robust features.

2.1 Model Architecture and Framework

* Core Model: The Audio Spectrogram Transformer (AST), sourced from Hugging Face, serves as the primary encoder.
* Learning Framework: The SimCLR framework is used for contrastive learning. A SimCLR-inspired projection head was implemented in PyTorch, consisting of a 2-layer MLP network with 512 units per layer and ReLU activation.
* Process Flow: The AST model processes audio inputs as mel-spectrograms, generating 512-dimensional embeddings. The projection head then maps these embeddings into a 128-dimensional latent space where the contrastive loss is calculated.

2.2 Dataset and Pre-processing

* Dataset: The project utilized the Free Music Archive (FMA) small subset, which contains 8,000 30-second audio tracks.
* Experimental Subset: To accommodate limited computational resources, the experiment was conducted on a smaller partition:
  * Training: 1,000 tracks
  * Validation: 200 tracks
  * Test: 200 tracks
* Data Truncation: All tracks were truncated to their first 10 seconds.
* Audio Representation: The raw .mp3 files were converted into mel-spectrograms. This representation was chosen over raw waveforms or standard spectrograms because the mel scale better approximates human auditory perception, which is more sensitive to variations in lower frequencies.

Representation Type	Description
Waveform	A one-dimensional array representing sampled sound amplitudes over time.
Normal Spectrogram	A two-dimensional plot of frequency vs. time, with intensity representing amplitude.
Mel Spectrogram	A spectrogram where the frequency axis is scaled according to the mel scale, mimicking human hearing.

2.3 Training Protocol

The model was trained from scratch to purely evaluate the capabilities of self-supervision.

* Training Parameters:
  * Epochs: 10
  * Optimizer: Adam with a learning rate of 3e-5
  * Batch Size: 8 tracks
  * Hardware: Google Colaboratory T4 GPUs
* Loss Function: The InfoNCE loss function was used. This contrastive metric encourages the model to maximize the similarity between augmented views of the same audio track (positive pairs) while minimizing their similarity to all other tracks in the batch (negative pairs). The training and validation loss curves converged after the 4th epoch, indicating the model learned effectively without overfitting.
* Data Augmentation: Augmentations were applied directly to the audio before spectrogram generation to create positive pairs for contrastive learning. The techniques included:
  * Noise Addition: Adding random values from a normal distribution.
  * Time Stretch: Stretching or compressing track length by up to 20%.
  * Pitch Change: Adjusting pitch by -2, -1, 0, 1, or 2 semitones.

3. Key Results and Analysis

The project's evaluation was performed through both qualitative visualization of the embedding space and a practical test of a simple recommendation system built upon these embeddings.

3.1 Embedding Distribution and Visualization

To understand the structure of the learned representations, the t-SNE dimensionality reduction technique was applied to the embeddings from the validation dataset.

* Trained vs. Untrained: The visualizations showed a stark contrast. Untrained embeddings appeared as a single, randomly distributed cloud of points. In contrast, the trained embeddings organized into a distinct web-like pattern composed of multiple filaments of grouped tracks.
* Genre Independence: Despite not being trained on genre information, the embeddings were colored using the FMA dataset's genre labels for analysis. The resulting plots showed that the model's groupings were not dependent on genre. This suggests the AST model learned to cluster tracks based on more subtle elements of musical structure rather than high-level categories. The paper posits that "genre labels may not be the best information for understanding music similarity."

3.2 Recommendation System Performance

A simple recommendation system was built to test the practical utility of the embeddings. For a given query song, the system recommended the track with the highest cosine similarity in the embedding space. A qualitative test was conducted on 50 songs.

* Evaluation Categories: Recommendations were manually classified into three groups:
  * Satisfactory: Clear and evident similarity between the query and recommended song.
  * Partially Satisfactory: Some similarities could be identified, but the match was imperfect.
  * Unsatisfactory: The recommended song was not similar to the query.
* Comparative Results: The performance of the trained model was compared against the untrained (random) model.

Model	Satisfactory	Partially Satisfactory	Unsatisfactory
Trained Model	48% (24 songs)	26% (13 songs)	26% (13 songs)
Untrained Model	10% (5 songs)	20% (10 songs)	70% (35 songs)

3.3 Analytical Insights

The qualitative tests provided several key insights into the model's behavior:

* Significant Performance Gain: The trained model dramatically outperformed the random baseline, increasing satisfactory recommendations from 10% to 48% and reducing unsatisfactory ones from 70% to 26%.
* Genre-Specific Performance: The vast majority of "Satisfactory" recommendations were for songs in the pop and hip-hop genres. This hypothesizes that the model may more effectively capture the intrinsic characteristics of these styles, possibly because prominent beats and basslines are well-represented in mel-spectrograms.
* Beyond Genre Similarity: The analysis reinforces the idea that genre is not a definitive measure of similarity. In 7 cases, recommended songs shared the same genre as the query but were still classified as unsatisfactory or partially satisfactory due to a lack of musical resemblance. This supports the conclusion that the system learns to recommend based on features like composition, timbre, and flow.

4. Conclusion and Future Directions

The work successfully presents a self-supervised framework for music recommendation that learns meaningful representations directly from audio. The system provides recommendations that significantly outperform an untrained model, demonstrating its ability to capture nuanced musical features that align with human perception.

Summary of Findings:

* The AST model fine-tuned with the SimCLR framework is effective for learning music representations in a self-supervised manner.
* The analysis of embeddings and test results suggests that genre alone is an insufficient criterion for grouping similar songs and that subtler features are critical.

Potential Improvements and Future Work:

* Combining the self-supervised approach with supervised fine-tuning methods.
* Refining the recommendation algorithms, potentially by integrating track metadata (e.g., genre, artist) with embedding similarity.
* Expanding experiments to use the full duration of tracks and a larger portion of the FMA dataset.
