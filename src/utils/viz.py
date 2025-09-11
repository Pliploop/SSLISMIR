import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Optional, List, Tuple, Union
import plotly.graph_objects as go
import plotly.express as px
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import umap
import torch
from torchaudio.transforms import MelSpectrogram
import matplotlib.pyplot as plt
import librosa.display



def compute_tsne(embeddings: np.ndarray, n_components: int = 2, 
                 perplexity: float = 30.0, random_state: int = 42) -> np.ndarray:
    """
    Compute T-SNE reduction of embeddings.
    
    Args:
        embeddings: Input embeddings array of shape (n_samples, n_features)
        n_components: Number of components for reduction
        perplexity: T-SNE perplexity parameter
        random_state: Random seed for reproducibility
        
    Returns:
        Reduced embeddings array of shape (n_samples, n_components)
    """
    tsne = TSNE(n_components=n_components, perplexity=perplexity, 
                random_state=random_state, n_jobs=-1)
    return tsne.fit_transform(embeddings)


def compute_umap(embeddings: np.ndarray, n_components: int = 2,
                 n_neighbors: int = 15, min_dist: float = 0.1,
                 random_state: int = 42) -> np.ndarray:
    """
    Compute UMAP reduction of embeddings.
    
    Args:
        embeddings: Input embeddings array of shape (n_samples, n_features)
        n_components: Number of components for reduction
        n_neighbors: Number of neighbors for UMAP
        min_dist: Minimum distance between points
        random_state: Random seed for reproducibility
        
    Returns:
        Reduced embeddings array of shape (n_samples, n_components)
    """
    reducer = umap.UMAP(n_components=n_components, n_neighbors=n_neighbors,
                       min_dist=min_dist, random_state=random_state)
    return reducer.fit_transform(embeddings)


def create_matplotlib_scatter(reduced_embeddings: np.ndarray, 
                            labels: Optional[np.ndarray] = None,
                            title: str = "Embedding Visualization",
                            figsize: Tuple[int, int] = (10, 8)) -> plt.Figure:
    """
    Create a matplotlib scatter plot of reduced embeddings.
    
    Args:
        reduced_embeddings: 2D embeddings array of shape (n_samples, 2)
        labels: Optional labels for coloring points
        title: Plot title
        figsize: Figure size as (width, height)
        
    Returns:
        matplotlib Figure object
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    if labels is not None:
        # Create scatter plot with different colors for each label
        unique_labels = np.unique(labels)
        colors = plt.cm.Set3(np.linspace(0, 1, len(unique_labels)))
        
        for i, label in enumerate(unique_labels):
            mask = labels == label
            ax.scatter(reduced_embeddings[mask, 0], reduced_embeddings[mask, 1],
                      c=[colors[i]], label=str(label), alpha=0.7, s=30)
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    else:
        ax.scatter(reduced_embeddings[:, 0], reduced_embeddings[:, 1], 
                  alpha=0.7, s=30)
    
    ax.set_xlabel('Component 1')
    ax.set_ylabel('Component 2')
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig


def create_plotly_scatter(reduced_embeddings: np.ndarray,
                         labels: Optional[np.ndarray] = None,
                         title: str = "Embedding Visualization") -> go.Figure:
    """
    Create an interactive plotly scatter plot of reduced embeddings.
    
    Args:
        reduced_embeddings: 2D embeddings array of shape (n_samples, 2)
        labels: Optional labels for coloring points
        title: Plot title
        
    Returns:
        plotly Figure object
    """
    if labels is not None:
        # Create plotly express scatter plot with labels
        fig = px.scatter(
            x=reduced_embeddings[:, 0],
            y=reduced_embeddings[:, 1],
            color=labels,
            title=title,
            labels={'x': 'Component 1', 'y': 'Component 2'},
            color_discrete_sequence=px.colors.qualitative.Set3
        )
    else:
        # Create basic scatter plot without labels
        fig = px.scatter(
            x=reduced_embeddings[:, 0],
            y=reduced_embeddings[:, 1],
            title=title,
            labels={'x': 'Component 1', 'y': 'Component 2'}
        )
    
    fig.update_traces(marker=dict(size=8, opacity=0.7))
    fig.update_layout(
        width=800,
        height=600,
        showlegend=True,
        hovermode='closest'
    )
    
    return fig

def show_audio_and_spectrogram(audio: torch.Tensor, sr: int, n_fft: int = 400, hop_length: int = 160, n_mels: int = 128, f_min: float = 0.0, f_max: float = None, win_length: int = None, power: float = 2.0):
    # Create MelSpectrogram transform with correct parameter names
    mel_transform = MelSpectrogram(
        sample_rate=sr,
        n_fft=n_fft,
        hop_length=hop_length,
        n_mels=n_mels,
        f_min=f_min,
        f_max=f_max,
        win_length=win_length,
        power=power
    )
    
    # Apply the transform
    mel_spec = mel_transform(audio)
    
    # Convert to log scale for better visualization
    mel_spec_db = torch.log(mel_spec + 1e-8)
    
    # Create figure with two subplots, height ratio 1 spectrogram to 0.2 waveform
    fig, ax = plt.subplots(2, 1, figsize=(10, 4), height_ratios=[0.2, 1])
    
    # Plot waveform using the newer waveshow function
    librosa.display.waveshow(audio.numpy(), sr=sr, ax=ax[0], color = "black")
    ax[0].set_title('Waveform')
    ax[0].set_ylabel('Amplitude')
    
    # Plot mel spectrogram
    librosa.display.specshow(
        mel_spec_db.numpy(), 
        sr=sr, 
        hop_length=hop_length,
        x_axis='time', 
        y_axis='mel', 
        ax=ax[1],

    )
    ax[1].set_title('Mel Spectrogram')
    ax[1].set_ylabel('Mel Frequency')
    ax[1].set_xlabel('Time')

    ## grid on for both axes
    ax[0].grid(True)
    ax[1].grid(True)
    
    plt.tight_layout()
    return fig

def embeddings_to_numpy(embeddings: List[torch.Tensor]) -> np.ndarray:
    """
    Convert list of embeddings to numpy array.
    """
    return np.concatenate(embeddings, axis=0)

def labels_to_numpy(labels: List[torch.Tensor]) -> np.ndarray:
    """
    Convert list of labels to numpy array.
    """
    return np.concatenate(labels, axis=0)