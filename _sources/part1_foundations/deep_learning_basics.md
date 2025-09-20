# Deep Learning Basics Refresher

## Neural Networks: Data → Representation → Prediction

At the heart of deep learning are **neural networks**, which can be thought of as layered systems that transform raw input data into useful representations and predictions. When we feed an input—say a music spectrogram—into a network, the first layers capture low-level patterns such as frequency content or energy. As information flows through deeper layers, the network extracts progressively more abstract concepts, like rhythmic structure, timbral qualities, or harmonic relationships. Finally, the output layer translates these internal representations into predictions for a given task, such as classifying the genre of a track or identifying the instrument being played.  

Training a neural network requires optimizing its parameters so that the predictions align with expected outcomes. This is done by defining a **loss function**—for example, cross-entropy for classification or mean squared error for regression—that measures the discrepancy between predicted and true values. Through the process of **backpropagation**, the network iteratively adjusts its weights to reduce this error, gradually improving its ability to model the input-output relationship.  

![Placeholder: Diagram of a neural network from input → hidden layers → output](path/to/placeholder_neural_network.png)

---

## Supervised Learning: Definitions, Tasks, and Limitations

Most early applications of deep learning in Music Information Retrieval (MIR) relied on **supervised learning**, where models are trained using input–label pairs. For example, a dataset might contain thousands of audio clips annotated with genres, instruments, or chord labels, and the goal of the network is to learn a mapping from the audio signal to these categorical outputs. In this setting, the presence of labeled data is essential: without it, the model cannot learn what “counts” as a correct prediction.  

Supervised approaches have led to notable advances in MIR, enabling systems to classify songs by genre, recognize chord sequences, or identify instruments. However, they also reveal important limitations. Because each model is trained for a specific task, the learned representations are often **narrow and task-dependent**, making it difficult to transfer them to new settings. More importantly, the requirement for large amounts of labeled data creates a bottleneck: collecting and annotating such datasets is expensive, time-consuming, and often impractical in domains as complex as music {cite}`Humphrey2013MIRDL,Choi2017Transfer`.  

![Placeholder: Diagram of supervised learning pipeline with ground truth labels](path/to/placeholder_supervised_pipeline.png)

---

## The Issues of Labeling in Music

These limitations become even more apparent when considering the particular challenges of annotation in music. Creating large, high-quality labeled datasets requires significant expertise: for example, annotating chord progressions or transcribing polyphonic passages is a task that demands trained musicians. Even simpler labeling efforts, such as genre classification, are fraught with ambiguity—does a track belong to *trip-hop* or *lo-fi hip-hop*, and who decides where the boundary lies?  

Beyond cost and ambiguity, labels in music are also subject to **subjectivity**. Descriptions of mood, emotion, or aesthetic quality are not only difficult to formalize but can vary drastically depending on the annotator’s background or cultural context. This introduces biases and inconsistencies into datasets, further limiting their usefulness for supervised training. As a result, while supervised learning has driven many successes in MIR, it struggles to scale in a way that reflects the richness and diversity of music {cite}`Flexer2016Subjectivity,Kim2019Mood`.  

![Placeholder: Illustration of annotation challenges in music (genre ambiguity, mood subjectivity, annotation cost)](path/to/placeholder_labeling_challenges.png)

---

*Next: [Core Ideas in SSL](core_ideas_ssl.md)*
