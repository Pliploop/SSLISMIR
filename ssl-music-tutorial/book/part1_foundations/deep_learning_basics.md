# Deep Learning Basics Refresher

## Neural Networks: Data → Representation → Prediction

At the heart of deep learning are **neural networks**, which can be thought of as layered systems that transform raw input data into useful representations and predictions. Given an input \(x \in \mathcal{X}\) (for example, a music spectrogram), the network applies a sequence of transformations parameterized by weights \(\theta\):  

\[
h = f_\theta(x) \quad \in \mathbb{R}^d
\]

where \(h\) is a learned **representation** in a latent space of dimension \(d\). A task-specific **head** then maps this representation to a prediction \(\hat{y}\):  

\[
\hat{y} = g_\theta(h).
\]

Training adjusts \(\theta\) to minimize a loss function \( \mathcal{L}(\hat{y}, y) \) that measures the discrepancy between predictions and ground-truth labels. For instance, classification tasks use the cross-entropy loss  

\[
\mathcal{L}_{\text{CE}} = - \sum_{c=1}^C y_c \log \hat{y}_c,
\]

while regression tasks may rely on mean squared error (MSE). Through **backpropagation**, gradients of the loss with respect to parameters are computed and used to update the model, gradually improving predictions.  

![Placeholder: Diagram of a neural network from input → hidden layers → output](path/to/placeholder_neural_network.png)

---

## Supervised Learning: Definitions, Tasks, and Limitations

In **supervised learning**, models learn from labeled data \(\{(x_i, y_i)\}_{i=1}^N\), where \(x_i\) is the input and \(y_i\) is the ground-truth label. The objective is to minimize the expected prediction error:  

\[
\min_\theta \; \mathbb{E}_{(x,y) \sim \mathcal{D}} \; \mathcal{L}(g_\theta(f_\theta(x)), y).
\]

For example:  
- **Genre classification**: \(x\) = audio clip, \(y\) = discrete genre label.  
- **Chord recognition**: \(x\) = spectrogram frame, \(y\) = chord label.  
- **Instrument recognition**: \(x\) = waveform segment, \(y\) = instrument category.  

Supervised learning has driven many successes in MIR, but suffers from a key limitation: the learned representation \(h = f_\theta(x)\) is optimized narrowly for the supervised task at hand. These task-specific embeddings may not transfer well to other applications. More importantly, supervised learning requires large, annotated datasets, which are costly to produce in music domains {cite}`Humphrey2013MIRDL,Choi2017Transfer`.  

![Placeholder: Diagram of supervised learning pipeline with ground truth labels](path/to/placeholder_supervised_pipeline.png)

---

## The Issues of Labeling in Music

Labeling music data exposes several challenges:  

1. **Cost and expertise**: Precise tasks such as chord transcription or onset detection demand expert annotators.  
2. **Ambiguity**: Many labels, such as *genre* or *style*, lack objective definitions—should a given track be labeled *trip-hop* or *lo-fi hip hop*?  
3. **Subjectivity**: Emotion or mood labels vary across listeners, introducing inconsistency.  
4. **Bias**: Different datasets adopt different annotation practices, leading to noisy supervision.  

As a result, labels can be expensive, ambiguous, and sometimes ill-defined. This makes supervised learning less practical for large-scale MIR, and motivates the turn toward **self-supervised approaches**, where the data itself provides the training signal {cite}`Flexer2016Subjectivity,Kim2019Mood`.  

![Placeholder: Illustration of annotation challenges in music (genre ambiguity, mood subjectivity, annotation cost)](path/to/placeholder_labeling_challenges.png)

---

*Next: [Core Ideas in SSL](core_ideas_ssl.md)*
