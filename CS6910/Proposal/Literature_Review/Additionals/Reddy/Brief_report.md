MusicNet: A Compact, High-Performance Neural Network for Real-time Background Music Detection

Executive Summary

This document synthesizes the findings of a research paper detailing MusicNet, a compact convolutional neural network (CNN) designed for the accurate, real-time detection of background music in communication pipelines. Developed to address the challenges of remote work and online meetings where music often co-occurs with speech and noise, MusicNet provides a highly efficient and effective solution.

The model significantly outperforms 20 state-of-the-art (SOTA) benchmarks, achieving a True Positive Rate (TPR) of 81.3% at an extremely low False Positive Rate (FPR) of 0.1%. This operating point is critical for user experience, translating to less than one false detection every three hours of continuous use. Beyond its superior accuracy, MusicNet is engineered for practical deployment on edge devices; it is 10 times smaller and has 4 times faster inference than the best-performing models benchmarked. A key innovation is its "in-model featurization," which processes raw audio directly, simplifying product integration and maintenance. These characteristics establish MusicNet as a best-in-class, production-ready solution for enhancing audio quality in modern communication systems.

I. The Challenge: Music in Real-Time Communications

The proliferation of online meetings has introduced complex audio environments where background music can interfere with clear communication. The accurate detection of such music is essential for enabling systems to apply appropriate audio processing, such as prompting a user to switch to a speech enhancement mode that preserves music or using a music-optimized codec.

A solution for a mainstream communication system must meet several stringent requirements:

* High Accuracy at Low False Positives: To avoid disrupting users with incorrect notifications, the system requires a high detection rate for music (TPR) with a minimal rate of false alarms (FPR). The target metrics are:
  * FPR ≤ 0.1%: This ensures a user experiences a false positive no more than once every three hours of continuous evaluation.
  * TPR ≥ 80%: This ensures the detection feature is useful and reliable.
* Real-Time Performance: The model must have low latency and fast inference to operate on live audio streams without noticeable delay. An average latency of 4.5 seconds is considered acceptable.
* Efficiency on Edge Devices: The solution must be compact in size and have low CPU usage to be deployed across a wide range of devices, from PCs (Windows, Mac, Linux) to mobile phones (Android/iOS).
* Ease of Integration and Maintenance: The model should be easy to integrate into existing product stacks and update over time without requiring significant code changes.

II. MusicNet: Proposed Architecture and Methodology

MusicNet is a binary classifier specifically designed to meet these challenges. It identifies the presence of music within 9-second audio clips.

Core Design and Architecture

* Model Type: A compact CNN-based model.
* Input: 9 seconds of raw audio waveform sampled at 16kHz (wideband audio).
* Output: A binary classification indicating the presence or absence of music.
* Key Architectural Choices: The network deliberately avoids batch normalization layers, as experiments showed this technique worsened classification accuracy by making the model insensitive to level variations crucial for music detection. The architecture is optimized with appropriate dropout, filter lengths, and filter counts to ensure generalizability.

In-Model Featurization

A primary contribution of MusicNet is the integration of audio featurization directly into the model architecture.

* Process: The model's first layer is a 1-D convolutional layer that takes the raw audio waveform as input and learns to produce a LogMel spectrogram. It converts the waveform into real and imaginary spectrum parts, calculates the absolute magnitude, and applies fixed Mel-weights to generate a 120-band Mel spectrum.
* Benefits: This approach provides significant practical advantages for deployment:
  * Simplified Integration: The model can be trained and run on raw audio, eliminating the need for product teams to implement and maintain a separate featurization pipeline in C++.
  * Portability and Consistency: It resolves common code portability issues and numerical mismatches between Python-based training environments (e.g., librosa, TensorFlow) and C++ production code.
  * Easy Updates: New models with different input features can be deployed without requiring source code changes in the host application.

Training and Validation Datasets

* Training Data: MusicNet was trained on the balanced subset of Google's Audio Set, a large corpus of labeled 10-second YouTube video fragments. Audio clips containing a "music" label were treated as positive examples, and all others as negative examples.
* Test Data: To ensure performance on realistic use cases, a custom, strongly-labeled test set of 1000 real-world clips was created. This dataset was meticulously designed to represent challenging scenarios in video calls and was labeled by two expert listeners. Its composition includes:
  * Clips from the 3rd Deep Noise Suppression Challenge and the Freesound dataset.
  * A wide variety of musical instruments (e.g., Piano, Guitar, Violin, Drums, Flute) and genres (e.g., Rock, Pop, Jazz, Classical).
  * Synthesized scenarios mixing clean speech with instrumental music at various signal-to-music ratios (SMRs).
  * Negative examples including only clean speech, noisy speech, and background noise.
  * Audio captured through various devices in both headset and speaker modes.

III. Performance Benchmarking and Results

MusicNet's performance was rigorously compared against 20 state-of-the-art Pretrained Audio Neural Networks (PANNs). The analysis confirmed its superior suitability for the target application.

Comparative Analysis

The evaluation prioritized performance at the strict FPR=0.1% operating point, as overall metrics like Area Under the Curve (AUC) can be misleading for this specific use case. For example, the Cnn14 DecisionLevelMax model has an excellent AUC of 0.99 but is unusable in production with a TPR of only 7.3% at the required 0.1% FPR.

At this critical threshold, MusicNet achieves a TPR of 81.3%, the best of any model evaluated.

Key Performance Metrics

The following table highlights MusicNet's advantages in accuracy, speed, and size compared to other notable SOTA models. Inference time was measured on an Intel i7-1065G7 CPU.

Model	Inference Time (ms)	Size (MB)	TPR at 0.1% FPR	AUC
Proposed Model (MusicNet)	11.1	0.2	81.3%	0.97
Cnn6	140.7	18.5	80.7%	0.98
ResNet54	340.8	398.2	75.3%	0.99
Cnn14 emb512	242.4	293.0	67.6%	0.98
MobileNetV1	15.1	18.4	52.4%	0.98
Wavegram Cnn14	285.2	308.9	61.8%	0.95
Cnn14 DecisionLevelMax	249.4	308.1	7.3%	0.99

Analysis of Superiority

The paper attributes MusicNet's superior performance to three key factors:

1. Optimized Input Representation: The use of longer 9-second audio clips and a higher-resolution 120-band Mel spectrogram provides richer features for detecting music compared to methods that use shorter clips or fewer Mel bands.
2. Avoidance of Batch Normalization: This design choice helps the model remain sensitive to variations in audio levels, which is an important cue for music detection, and improves its ability to generalize to challenging test cases.
3. Compact and Effective Architecture: The model is significantly smaller and more efficient than competitors like MobileNet V1 and V2, yet achieves better accuracy through careful design choices regarding dropout, filter lengths, and the number of filters.

IV. Practical Implementation and Deployment

MusicNet is designed for seamless deployment in production environments.

* Cross-Platform Portability: The model is converted into the ONNX format, allowing it to leverage the ONNX Runtime C++ library for efficient, client-side inference across diverse operating systems including Windows, macOS, Linux, Android, and iOS.
* Production Suitability: Its fast inference time (11.1 ms), low CPU overhead, and extremely small file size (0.2 MB) make it exceptionally well-suited for deployment on resource-constrained edge devices, especially mobile phones where network bandwidth for updates can be limited.

V. Conclusions and Future Directions

Summary of Contributions

MusicNet presents a significant advancement in real-time audio event detection for communication systems. Its main contributions are:

1. Best-in-Class Performance: It provides superior detection accuracy (81.3% TPR at 0.1% FPR) for real-world usage compared to an extensive list of 20 SOTA models.
2. Exceptional Efficiency: It features the lowest complexity and smallest model size of any high-performing music detector evaluated, making it ideal for deployment on edge devices.
3. Simplified Integration: Its in-model featurization design facilitates easy integration and maintenance, a critical factor for production environments.

Future Work

The research outlines several avenues for future improvement:

* Performance Enhancement: Optimize the model to achieve a TPR of ≥ 95% at the 0.1% FPR target.
* Efficiency Gains: Further reduce the model's inference time.
* Expanded Validation: Collect a more diverse and extensive real-world test set containing over 5,000 clips to continue improving model robustness.
