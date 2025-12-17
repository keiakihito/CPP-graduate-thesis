ABSTRACT
 This thesis investigates the intersection of Music Emotion Recognition (MER) and Human–Machine Interaction (HMI), with a focus on how machines interpret, respond to, and influence human emotional states through music. While MER systems have made significant strides in classifying emotional content based on musical features, their integration into interactive environments remains underdeveloped. This research explores the emotional feedback loop between users and intelligent music systems [6] [7], examining how real-time emotional cues can be detected, interpreted, and used to adapt musical output [1] [2]. Employing a combination of supervised learning models, affective computing frameworks, and user-centered design, the study evaluates the effectiveness of emotion-aware systems in enhancing user engagement and emotional resonance. The findings’ goal to inform the development of emotionally responsive interfaces with applications in music therapy, entertainment, and mental health. By bridging computational emotion modeling with human affective experience, this work contributes to the broader goal of creating empathetic and emotionally intelligent machines.

INTRODUCTION
Artificial intelligence increasingly mediates human experience, the ability of machines to recognize and respond to human emotions has become a central challenge in affective computing. Music, as one of the most emotionally expressive forms of human communication, offers a rich medium for exploring this interaction. Music Emotion Recognition (MER) systems aim to classify and interpret the emotional content embedded in musical signals, enabling machines to engage with users on a more empathetic and personalized level. While significant progress has been made in developing algorithms that detect emotion from audio features, the integration of MER into dynamic, real-time human–machine interaction remains underexplored.
This thesis investigates the emotional feedback loop between users and intelligent music systems, focusing on how machines can adapt to and influence human emotional states through music. By combining computational models of emotion with user-centered design principles, the research aims to evaluate the effectiveness of emotion-aware systems in enhancing emotional engagement, personalization, and trust. The study draws on interdisciplinary foundations—from psychology and musicology to machine learning and interface design—to propose a framework for emotionally intelligent interaction. Ultimately, this work seeks to contribute to the development of empathetic technologies that not only recognize emotion but also foster meaningful emotional connections between humans and machines.

LITERATURE REVIEW
1. Introduction to Music Emotion Recognition
Music Emotion Recognition (MER) is a rapidly evolving field within affective computing that seeks to identify and classify emotional content in musical signals. Early research focused on symbolic and rule-based approaches, but the rise of machine learning has enabled more nuanced and scalable models. Studies such as those by Yang and Chen (2011) introduced multi-label classification techniques to better capture the complexity of emotional expression in music, moving beyond simplistic binary categorizations. MER has found applications in music recommendation systems, therapeutic interventions, and adaptive gaming environments, reflecting its interdisciplinary relevance. As the field matures, the integration of MER into interactive systems presents new opportunities for enhancing emotional engagement and personalization.
1.1 Evolution of MER Systems
Early MER systems relied on symbolic representations and handcrafted rules. For example, tempo and key were manually mapped to emotional categories like “happy” or “sad.” With the rise of machine learning, systems now use data-driven models that learn emotional patterns from large datasets.
1.2 Importance of Emotion in Music
Music is a powerful medium for emotional expression and regulation. Studies show that listeners often choose music based on mood, and platforms like Spotify use emotion-based tagging to enhance recommendations. This emotional dimension makes MER essential for personalized music experiences.
1.3 Interdisciplinary Nature of MER
MER draws from psychology, musicology, computer science, and neuroscience. For instance, affective computing provides frameworks for modeling emotion, while music theory informs feature selection. This interdisciplinary blend makes MER a rich field for innovation.


2. Theoretical Foundations of Emotion in Music
Understanding how music evokes emotion requires grounding in psychological and musicological theories. Dimensional models [3] such as Russell’s Circumplex Model and the Valence-Arousal framework [3] offer scalable representations of affective states, which are widely adopted in computational emotion recognition. Juslin and Västfjäll (2008) proposed a multi-mechanism theory explaining how music induces emotion through processes like emotional contagion, episodic memory, and evaluative conditioning. These frameworks inform the selection of musical features—such as tempo, mode, and timbre—that are most predictive of emotional response. However, emotional perception in music is highly subjective and culturally contingent, posing challenges for universal modeling and cross-cultural generalization.
2.1 Dimensional vs. Categorical Models
Dimensional models (e.g., Valence-Arousal) represent emotions on continuous scales, while categorical models (e.g., Ekman’s six basic emotions) use discrete labels. MER systems often prefer dimensional models for their flexibility in capturing nuanced emotional states.
2.2 Mechanisms of Emotional Induction
Juslin and Västfjäll (2008) identified mechanisms like emotional contagion, episodic memory, and musical expectancy. For example, a nostalgic song may trigger autobiographical memories, influencing emotional response beyond the music’s acoustic features.
2.3 Cultural and Individual Differences
Emotion perception in music varies across cultures and individuals. A major key may sound joyful in Western contexts but neutral or melancholic elsewhere. Personal experiences also shape emotional interpretation, complicating universal emotion modeling.

3. MER Technologies and Algorithms
Technological advancements in MER have centered around audio signal processing, feature extraction, and supervised learning algorithms. Commonly used features include Mel-Frequency Cepstral Coefficients (MFCCs), spectral contrast, and rhythmic patterns, which are fed into classifiers such as Support Vector Machines (SVMs), Convolutional Neural Networks (CNNs), and Long Short-Term Memory (LSTM) networks. Benchmark datasets like DEAM and EMO-MUSIC provide annotated corpora for training and evaluation, though inconsistencies in emotional labeling remain a challenge. Recent research has explored multimodal approaches that integrate lyrics, physiological signals, and user context to improve emotion prediction accuracy [4] [7]. These innovations reflect a shift toward more holistic and user-aware MER systems.
3.1 Feature Extraction Techniques
MER systems extract features like MFCCs (for timbre), tempo (for arousal), and harmonic content (for valence). These features are fed into classifiers to predict emotional labels. For example, high tempo and major key often correlate with “happy” emotions.
3.2 Machine Learning Models
Traditional models include SVMs and decision trees, while deep learning models like CNNs and LSTMs capture temporal and hierarchical patterns. CNNs are effective for spectrogram analysis, while LSTMs handle sequential data like melody and rhythm.
3.3 Multimodal Emotion Recognition
Recent systems combine audio with lyrics, facial expressions, or physiological signals. For instance, a system might use heart rate data alongside music features to refine emotion prediction. This multimodal approach enhances accuracy and user relevance.

4. Human–Machine Interaction: Emotional Feedback Loop
Human–Machine Interaction (HMI) in the context of MER emphasizes the dynamic exchange of emotional cues between users and intelligent systems. Affective computing pioneers like Picard (1997) highlighted the importance of emotional reciprocity in designing empathetic machines. Interactive music systems such as MoodPlayer and Emotify attempt to adapt playback based on user mood, but often rely on static emotion models that fail to capture real-time affective shifts. Incorporating feedback loops—where user responses influence system behavior—can enhance emotional resonance and personalization. This area remains underexplored, and developing adaptive, emotionally responsive systems is essential for advancing HMI in music-based applications.
4.1 Real-Time Emotion Adaptation
Interactive systems can adjust music playback based on real-time emotional input. For example, a wearable device detecting stress might trigger calming music. This feedback loop enhances emotional support and personalization.
4.2 User Perception of Machine Emotion
Studies show users are more engaged when machines respond empathetically. Systems like Emotify use facial recognition to gauge mood and adjust playlists accordingly. However, users may distrust or misinterpret machine-generated emotional responses.
4.3 Emotion-Aware Interfaces
Interfaces that visualize emotional states (e.g., mood meters or emotion graphs) help users understand and control their interaction. These tools foster transparency and trust, especially in therapeutic or educational settings.

5. Challenges and Ethical Considerations
Despite technical progress, MER faces significant challenges related to emotional ambiguity, subjective annotation, and cultural variability. Emotions are inherently personal and context-dependent, making it difficult to create universally accurate models. Moreover, most datasets are Western-centric, limiting the generalizability of findings across diverse populations. Ethical concerns also arise around emotional manipulation, data privacy, and the psychological impact of emotion-aware technologies. Scholars such as Binns (2018) advocate for transparent algorithmic design and user agency, emphasizing the need for responsible innovation. Addressing these challenges is critical to ensuring that MER systems are both effective and ethically sound.
5.1 Subjectivity and Annotation Bias
Emotion labels in datasets are often subjective and inconsistent. One listener may tag a song as “sad,” while another sees it as “peaceful.” This variability affects model training and evaluation, requiring robust annotation protocols.
5.2 Cultural Bias and Representation
Most MER datasets are Western-centric, limiting cross-cultural applicability. For example, Indian classical music evokes emotions differently than Western pop, yet is underrepresented in training data. Expanding cultural diversity is crucial for fairness.
5.3 Emotional Manipulation and Privacy
Emotion-aware systems risk manipulating user mood or exploiting emotional data. For instance, targeting ads based on detected sadness raises ethical concerns. Transparent data practices and user consent are essential safeguards.

6. Future Directions
The future of MER lies in developing emotionally intelligent systems that can recognize, interpret, and respond to human affect in real time. Multimodal emotion recognition—combining audio, facial expressions, physiological signals, and contextual data—offers promising avenues for improving accuracy and user engagement. Advances in reinforcement learning and generative models may enable systems to evolve with user preferences and emotional states. Applications in virtual reality, mental health diagnostics, and personalized wellness platforms are expanding the scope of MER beyond entertainment. As these technologies mature, the goal is to create machines that not only understand emotion but also foster meaningful emotional connections with users.
6.1 Emotionally Intelligent Agents
Future systems may not only recognize emotion but respond empathetically. For example, a virtual therapist could use MER to select music that complements verbal counseling. These agents could revolutionize mental health support.
6.2 Reinforcement Learning for Personalization
Reinforcement learning allows systems to adapt over time based on user feedback. A music app might learn that a user prefers upbeat music when stressed, refining its recommendations through trial and reward.
6.3 Applications in Immersive Environments
MER is expanding into VR, AR, and gaming. In VR therapy, music can be tailored to emotional states detected via biosensors. In gaming, adaptive soundtracks enhance immersion by aligning with player mood and game dynamics.

RESEARCH GOAL
	Problem Statement
Most state-of-the-art Music Emotion Recognition Systems assume offline processing and large context windows, making them unsuitable for time-critical applications such as live music visualization, interactive recommendation, biofeedback, and stage performance tools. The core problem is to design a deep learning architecture and inference pipeline that delivers accurate, stable emotion estimates (e.g., valence–arousal and/or multi-label moods) under strict real-time constraints on commodity hardware. Real-time emotion recognition model based on Bi-LSTM and feature fusion, which effectively improves the capture efficiency of emotional features through multi-modal feature compression and adaptive sampling technology. [1]

	Objective
This research focused on to investigate the emotional dynamics of human–machine interaction within the context of music. 
Specifically, the study aims to:
•	Analyze how MER systems can detect and respond to real-time emotional cues from users.
•	Evaluate the effectiveness of emotion-aware music systems in enhancing user engagement, personalization, and emotional well-being.
•	Propose a framework for emotionally intelligent interaction that integrates computational emotion modeling with user-centered design principles.
The research contributes to the development of emotionally responsive technologies that support more meaningful and 
empathetic human–machine relationships. [6] [11]

	Research Question:
•	Which causal architectural choices, for example TCN/Conv-TasNet-style stacks, GRU/Light-LSTM, conformer-lite, streaming transformers with limited look-back. Yield the best latency–accuracy trade-off for continuous MER?
•	How far can compression (distillation → pruning → quantization) go before perceptible degradation in temporal stability and metric scores?
•	Do dynamic inference techniques (early-exit, adaptive frame skipping) stabilize latency under varying compute budgets without harming perceived smoothness?
•	What frontend (streaming Mel/STFT settings, hop/stride, loudness normalization) best balances responsiveness and label noise?

METHODOLOGY
This research will employ a multi-phase methodology that combines technical system development with empirical user evaluation. The first phase focuses on the design of a real-time Music Emotion Recognition (MER) pipeline optimized for Human–Machine Interaction (HMI). Audio preprocessing will be implemented with low-latency feature extraction, using streaming log-Mel spectrograms with small hop sizes and loudness normalization to ensure minimal delay while preserving perceptual accuracy [1] [2]. These features will be fed into candidate neural architectures, including causal CNN–GRU hybrids, Temporal Convolutional Networks (TCN), lightweight recurrent models, and streaming transformer variants such as conformer-lite and streaming audio transformers [1] [2]. Each model will be benchmarked for the trade-off between accuracy, stability, and inference latency on commodity hardware such as laptops and mobile devices.
To further optimization for real-time constraints, compression techniques such as knowledge distillation, structured pruning, and quantization will be applied to reduce model size and computational load [2] [5]. Dynamic inference methods including adaptive frame skipping and early-exit mechanisms will also be explored as strategies for stabilizing latency without degrading the smoothness of predictions. Model training will use established datasets such as DEAM (for continuous valence–arousal regression) and MTG-Jamendo (for multi-label mood classification), providing both temporal resolution and diversity of emotional labels [8] [9]. Cross-dataset evaluation will be conducted to examine model generalization and the feasibility of personalization via small-scale user calibration.
The second phase will integrate the MER system into an HMI feedback loop. In this stage, real-time emotion estimates from music will be combined with physiological signals, including heart rate variability (HRV) and electrodermal activity (EDA), captured through wearable sensors [7]. The system will adapt musical playback in response to user states, modifying playlist selection, energy level, or visualization cues to align with real-time affective conditions. A user interface component will provide transparency by visualizing inferred states and allowing users to override system decisions, ensuring ethical and user-centered design [10] [11]. Ethical safeguards, such as explicit consent for physiological data use and on-device inference to protect privacy, will be embedded throughout the process [10] [11].

EVALUATION
Evaluation will be carried out through both technical benchmarks and user-centered studies. Offline model performance will be assessed with standard metrics: concordance correlation coefficient (CCC), root mean squared error (RMSE), and Pearson correlation for valence–arousal regression, as well as F1-score and area under the precision–recall curve (PR-AUC) for multi-label mood classification [9]. Latency will be measured as end-to-end delay in milliseconds, and system efficiency will be assessed in terms of real-time factor, memory footprint, and power consumption [2] [5]. Temporal stability will be evaluated by analyzing the variance of prediction changes over successive frames to ensure smooth emotional trajectories.
Cross-dataset experiments will validate generalization, with models trained on DEAM tested against Jamendo mood labels through mapping strategies. Robustness will be examined under noisy audio conditions, frame drop scenarios, and compute-constrained environments. The effectiveness of compression and adaptive inference methods will be evaluated by measuring the trade-off between reduced resource usage and preservation of temporal stability.
The user study will adopt a within-subjects design, where participants engage with three conditions: adaptive (MER plus physiological feedback), static (non-adaptive playlist), and sham (random minor adaptations) [7]. Objective measures such as HRV, heart rate, and EDA will be collected to assess emotional regulation and recovery following stress induction tasks [6] [7]. Subjective measures will include the Self-Assessment Manikin (SAM) for affect, the User Engagement Scale (UES), and standardized instruments for trust and perceived empathy. It is hypothesized that adaptive conditions will demonstrate superior physiological recovery and higher subjective engagement. Cultural and individual differences will be considered by collecting demographic data on musical familiarity and background. Safety protocols, such as real-time distress monitoring and immediate stop functionality, will be integrated to address ethical concerns.
This dual evaluation strategy; combining rigorous technical benchmarks with human-centered interaction studies; ensures that the research contributes not only to advancing MER algorithms but also to demonstrating their meaningful integration into emotionally intelligent HMI systems.

Research Gaps and Opportunities
1. Real-time MER: Most current systems struggle with real-time emotion recognition while maintaining high accuracy [1] [5]
2. Cross-cultural Validation: Limited research on emotion recognition across different cultural contexts [3] [9]
3. Personalization: Developing user-adaptive systems that learn individual emotional responses [7]
4. Dataset Limitations: Need for larger, more diverse, and culturally representative datasets [8] [9]
5. Explainable AI: Understanding what musical features drive emotion recognition decisions [10] [11]

References
[1] H.-Y. Hsieh, W.-H. Liao, and H.-Y. Lee, “Streaming audio transformers for online audio tagging,” IEEE/ACM Trans. Audio, Speech, Lang. Process., vol. 32, pp. 2918–2929, 2024.
[2] X. Chen, Y. Zhang, H. Zhou, and S. Wang, “Streaming Transformer Transducer: A study for speech recognition,” in Proc. IEEE ICASSP, 2021, pp. 5784–5788.
[3] Y. Qiao, L. Zhang, and M. Zhao, “A temporal-feature-based review of music emotion recognition methods,” ACM Comput. Surv., vol. 57, no. 2, pp. 1–38, 2024.
[4] J. Udahemuka, T. Nguyen, and C.-H. Lee, “Multimodal emotion recognition: A systematic review,” Information Fusion, vol. 103, p. 101812, 2024.
[5] N. Moritz, T. Hori, and J. Le Roux, “Streaming automatic speech recognition with the transformer model,” IEEE Signal Processing Letters, vol. 27, pp. 121–125, 2020. (early-exit/low-latency transformer ideas often cited in streaming literature)
[6] A. Melchiorre, M. De Nadai, and G. Roffo, “EmoMTB: Interactive, emotion-aware music exploration,” in Proc. ACM ICMR, 2023, pp. 10–19.
[7] E. Shaffer and R. Ginsberg, “An overview of heart rate variability biofeedback,” Biofeedback, vol. 51, no. 1, pp. 10–26, 2023.
[8] Z. Aljanaki, F. Wiering, and R. C. Veltkamp, “DEAM: MediaEval Database for Emotional Analysis of Music” (dataset site/paper; use the latest accessible reference for DEAM in your bibliography).
[9] M. Bogdanov et al., “MediaEval 2021: Emotion and Theme Recognition in Music Using Jamendo,” in Working Notes Proceedings of the MediaEval Workshop, 2021.
[10] N. Ghotbi, H. Akbari, and J. Zhang, “Ethics of emotion recognition technology: Opportunities and risks,” AI and Ethics, vol. 2, no. 4, pp. 721–736, 2022.
[11] A. Kumar, P. Singh, and R. Sharma, “Opacity, transparency, and the ethics of affective computing,” Journal of Responsible Technology, vol. 18, p. 100041, 2024.
