Music Genre Classification and Recommendation Using Convolutional Neural Networks

Executive Summary

This document provides a comprehensive analysis of a system designed for the automatic classification of music genres and the subsequent recommendation of music, as detailed in the research by Jessica Dias et al. The core challenge addressed is the escalating need for accurate metadata to organize, manage, and search the vast and rapidly growing digital music libraries on platforms like Spotify. The project's central thesis is that a combination of Machine Learning and Neural Networking, specifically utilizing a Convolutional Neural Network (CNN), can effectively automate this task, which is currently performed manually.

The proposed system successfully develops a classifier that predicts the genre of an audio file from ten categories: Blues, Classical, Country, Disco, HipHop, Jazz, Metal, Pop, Reggae, and Rock. This model is trained and tested on the widely-used GTZAN dataset. The research demonstrates that the CNN model outperforms other machine learning algorithms such as Support Vector Machine (SVM), K-Nearest Neighbors (KNN), and Feed-forward Neural Networks, achieving a test accuracy of up to 82% when data processing techniques are applied.

In addition to classification, the system incorporates a multi-faceted recommendation engine. This engine employs three distinct algorithms—Popularity Filtering, Content-Based Filtering, and Collaborative Filtering—to provide users with music suggestions tailored to their listening habits and preferences. The project concludes with the successful creation of the classifier and recommendation engine, with future scope identified for its implementation within a graphical user interface (GUI) to create a fully functional website or application.

1. Project Overview and Objectives

The proliferation of digital music platforms has created an immense need for efficient and accurate music information retrieval systems. The manual categorization of music genres is a time-consuming task that struggles to keep pace with the volume of new music being released daily. This project aims to address this challenge by creating an automated system that can classify music genres and provide recommendations.

The primary objectives of the project are:

* To build a robust music genre classifier using a Convolutional Neural Network (CNN) trained on audio features.
* To develop a comprehensive music recommendation system that leverages multiple filtering techniques to suggest songs to users.
* To extract and process relevant features from raw audio signals to enable accurate classification and analysis.
* To evaluate the performance of the CNN model against other common machine learning classifiers.

2. Technical Approach and System Architecture

The system is architected around two primary functions: Music Genre Classification and Music Recommendation. The foundation of the classification model is a Convolutional Neural Network, selected for its proven efficacy in pattern recognition tasks.

Core Model: Convolutional Neural Networks (CNN)

A CNN is a type of deep learning neural network specifically designed to analyze structured arrays of data, such as images or, in this case, visual representations of audio like spectrograms. CNNs are highly effective at detecting hierarchical patterns (e.g., lines, gradients, and more complex structures), which makes them well-suited for identifying the intricate features that define a music genre. The model is a feed-forward neural network composed of multiple stacked convolutional layers, capable of learning progressively more abstract features from the input data without requiring extensive pre-processing.

Dataset: GTZAN

The model was trained and evaluated using the GTZAN dataset, a standard benchmark for music genre recognition research. Key characteristics of the dataset include:

* Content: 1000 audio tracks, each 30 seconds in duration.
* Structure: The tracks are evenly distributed across 10 distinct genres, with 100 files per genre.
* Genres: Blues, Classical, Country, Disco, HipHop, Jazz, Metal, Pop, Reggae, and Rock.
* Format: All audio files are in the .wav format.
* Origin: The files were collected from various sources, including personal CDs, radio, and microphone recordings, to ensure a diversity of recording conditions.

System Workflows

The system operates via two distinct, user-initiated workflows.

Music Genre Classification Process

As illustrated in the system's block diagram, the classification process follows these steps:

1. User Authentication: The user enters credentials which are verified against an authentication database.
2. File Upload: The authenticated user uploads a raw audio file.
3. Preprocessing: The raw input undergoes a series of preprocessing steps to extract meaningful features. This includes:
  * MFCC Calculation: Mel-Frequency Cepstral Coefficients are computed to represent the short-term power spectrum of the sound.
  * Sampling: The audio is sampled to create a standardized digital representation.
  * Vector Formation: The extracted features are formed into a numerical vector suitable for the neural network.
4. Classification: The processed input vector is fed into the trained CNN model.
5. Output: The model predicts the genre and outputs the result to the user. An implementation example shows the model correctly classifying a file as "pop".

Music Recommendation System Process

The recommendation workflow is designed to suggest new music based on user input:

1. User Authentication: The process begins with user verification.
2. Song Input: The user enters a song name as input.
3. Recommendation Engine: The input is passed to the recommendation engine, which utilizes a combination of algorithms to generate a list of suggestions.
4. Filtering: The engine applies Popularity Based Filtering and Similarity Based Filtering to curate the recommendations.
5. Output: The system presents a ranked list of recommended songs. An implementation example shows that given "Fleet Foxes" and "The End" as input, the system recommends similar artists and songs like "Quiet Houses - Fleet Foxes" and "St. Elmo's Fire - Dave Grusin".

3. Recommendation Engine Algorithms

The recommendation engine is a hybrid system that integrates three distinct algorithms to generate suggestions.

* Popularity Filtering: This is a straightforward, non-personalized method that recommends the most popular songs from the training dataset based on an aggregate ranking (e.g., total listen counts). It does not consider individual user preferences.
* Content-Based Filtering: This method provides recommendations based on the attributes of an item and a profile of the user's tastes. The system creates a user-specific classifier for likes and dislikes by analyzing product attributes and the user's historical interactions.
* Collaborative Filtering: This algorithm operates on the principle that users who agreed in the past will likely agree in the future. It generates recommendations by finding "peer" users or items with similar rating histories. The system implements an item-based collaborative filtering model, using listen counts as a form of implicit feedback to calculate the similarity between two items.

4. Performance and Results

The performance of the CNN model was rigorously evaluated through training and testing, and its accuracy was compared against other machine learning algorithms.

Model Training and Evaluation

The model's training process was monitored via accuracy and loss curves over approximately 70 epochs. These plots show that while the training accuracy steadily increased to nearly 70% and loss decreased, the test accuracy and loss were more volatile. A direct evaluation of the model on the test set yielded a specific accuracy score.

* Test Accuracy: An evaluation script run on the test set reported a Test accuracy of 0.6252 (62.52%).

Comparative Analysis of Classifiers

A key contribution of the research is the comparative analysis of the CNN against other models. The results demonstrate the superior performance of the CNN, especially when data processing is applied.

| Model | With Data Processing | Without Data Processing | | :--- | :---: | :---: | :---: | :---: | :---: | :---: | | | Train | CV | Test | Train | CV | Test | | Support Vector Machine | .97 | .60 | .60 | .75 | .32 | .28 | | K-Nearest Neighbors | 1.00 | .52 | .54 | 1.00 | .21 | .21 | | Feed-forward Neural Network | .96 | .55 | .54 | .64 | .26 | .25 | | Convolutional Neural Network | .95 | .84 | .82 | .85 | .59 | .53 |

As shown in the table, the Convolutional Neural Network with data processing achieves a test accuracy of 82%, significantly outperforming all other tested models. A related bar chart visualizes this superiority, with accompanying text stating that "the accuracy of CNN is 76%." The source presents these varying accuracy figures (62.5%, 76%, and 82%) across different evaluation contexts.

5. Review of Relevant Literature

The project is situated within a body of existing research on music information retrieval. The literature review highlights several key approaches and findings that inform the current work:

* Lee et al. (2015): Proposed the "MusicRecom" system, which used a combination of musical feature extraction (MFCCs, DFB) and user usage history (skips, repeats) for recommendations.
* Panchwagh et al. (2016): Found that single classifiers could outperform ensemble methods for genre classification, with the SMO classifier achieving 100% accuracy on the MFCC feature set.
* Elbir et al. (2018): Used digital signal processing on the GTZAN dataset and found that an SVM approach yielded the best results in their study.
* Panwar et al. (2018): Employed a hybrid Convolutional Recurrent Neural Network (CRNN) to predict genre and mood tags from mel-spectrograms of raw audio.
* Fessahaye et al. (2019): Developed "T-RECSYS," a hybrid recommendation system using content-based and collaborative filtering that achieved precision scores up to 88% on Spotify data.
* Ndou et al. (2021): Showed that using shorter three-second audio segments for feature extraction could improve CNN accuracy by increasing the volume of training data.

6. Conclusion and Future Development

The project successfully demonstrates the viability of using a Convolutional Neural Network to create an effective classifier for predicting the genre of music audio files across ten distinct categories. The developed system also includes a robust recommendation algorithm capable of suggesting music similar to what a user is listening to.

The primary avenue for future development is the implementation of the system in a user-facing format. The research outlines a clear path forward to transition the underlying models into a fully functional website or application, complete with a graphical user interface (GUI), examples of which were presented in the project documentation.
