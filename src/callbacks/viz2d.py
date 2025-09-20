import os
import torch
import numpy as np
from typing import Optional, List, Dict, Any
from lightning.pytorch import Callback
import wandb
import matplotlib.pyplot as plt
import librosa
import librosa.display
import seaborn as sns
from sklearn.manifold import TSNE
from umap import UMAP
import torch.nn.functional as F

try:
    from utils.viz import (
            create_matplotlib_scatter, 
            create_plotly_scatter, embeddings_to_numpy, labels_to_numpy
        )
except ImportError: # notebook
    from src.utils.viz import (
        create_matplotlib_scatter, 
        create_plotly_scatter, embeddings_to_numpy, labels_to_numpy
    )


class Embedding2DVisualizationCallback(Callback):
    """
    PyTorch Lightning callback for 2D visualization of embeddings.
    
    This callback extracts embeddings from the model, computes dimensionality reduction
    (T-SNE or UMAP), and logs visualizations to Weights & Biases.
    """
    
    def __init__(
        self,
        every_n_epochs: int = 1,    
        reduction_method: str = "umap",
        tsne_perplexity: float = 30.0,
        umap_n_neighbors: int = 15,
        umap_min_dist: float = 0.1,
        random_state: int = 42,
        **kwargs
    ):
        """
        Initialize the callback.
        
        Args:
            every_n_steps: Compute reduction every N steps
            reduction_method: Dimensionality reduction method ("umap" or "tsne")
            save_embeddings: Whether to save embeddings locally
            save_dir: Directory to save embeddings
            tsne_perplexity: T-SNE perplexity parameter
            umap_n_neighbors: UMAP n_neighbors parameter
            umap_min_dist: UMAP min_dist parameter
            random_state: Random seed for reproducibility
        """
        ## init super without kwargs (Callback doesn't accept them)
        super().__init__()

        if reduction_method not in ["umap", "tsne"]:
            raise ValueError(f"reduction_method must be 'umap' or 'tsne', got {reduction_method}")
        
        self.every_n_epochs = every_n_epochs
        self.reduction_method = reduction_method
        self.tsne_perplexity = tsne_perplexity
        self.umap_n_neighbors = umap_n_neighbors
        self.umap_min_dist = umap_min_dist
        self.random_state = random_state

        if self.reduction_method == "tsne":
            self.reducer = TSNE(
                perplexity=self.tsne_perplexity,
                random_state=self.random_state
            )
        else:
            self.reducer = UMAP(
                n_neighbors=self.umap_n_neighbors,
                min_dist=self.umap_min_dist,
                random_state=self.random_state
            )

        self.reducer_fitted = False
        
        # Initialize embedding and label caches
        self.embedding_cache = []
        self.label_cache = []
        
    
    def on_validation_batch_end(
        self, 
        trainer, 
        pl_module, 
        outputs, 
        batch: Any, 
        batch_idx: int
    ) -> None:
        """Extract embeddings from validation batch."""
        if trainer.current_epoch % self.every_n_epochs == 0:
            self._extract_embeddings(pl_module, batch, "val")
    
    def on_validation_epoch_end(self, trainer, pl_module) -> None:
        """Compute reduction and log at end of validation epoch."""

        
        if len(self.embedding_cache) > 0 and trainer.current_epoch % self.every_n_epochs == 0:
            self._compute_reduction_and_log(trainer, "val", trainer.current_epoch)
            self._reset_cache()
    
    def _extract_embeddings(self, pl_module, batch: Any, stage: str) -> None:
        """
        Extract embeddings from batch using the model's extract_features method.
        
        Args:
            pl_module: PyTorch Lightning module
            batch: Input batch
            stage: Current stage (train/val/test)
        """
        
        # Extract clean data
        clean_data = batch["audio"]
        
        # Extract labels if available
        labels = batch.get("name", None)
        
        # Set model to eval mode and disable gradients
        with torch.no_grad():
            embeddings = pl_module.backbone(clean_data)["z"]
            embeddings = pl_module.projection_head(embeddings)
        # normalize the embeddings
        embeddings = F.normalize(embeddings, p=2, dim=1)
        
        # Add to cache
        self.embedding_cache.append(embeddings.detach().cpu())
        if labels is not None:
            self.label_cache += labels[:embeddings.shape[0]]
        
    def _compute_reduction_and_log(self, trainer, stage: str, step_or_epoch: int) -> None:
        """
        Compute dimensionality reduction and log visualizations.
        
        Args:
            trainer: PyTorch Lightning trainer
            stage: Current stage (train/val/test)
            step_or_epoch: Current step or epoch number
        """
        if len(self.embedding_cache) == 0:
            return
        
        # Convert embeddings and labels to numpy
        embeddings_np = embeddings_to_numpy(self.embedding_cache)
        labels_np = self.label_cache

        reduced_embeddings = self.reducer.fit_transform(embeddings_np)
        
        
        # Create visualizations
        title = f"{stage.capitalize()} Embeddings - {self.reduction_method.upper()}"
        
        # # Matplotlib scatter plot
        # fig_matplotlib = create_matplotlib_scatter(
        #     reduced_embeddings, 
        #     labels_np, 
        #     title=title
        # )
        
        # Plotly interactive plot
        fig_plotly = create_plotly_scatter(
            reduced_embeddings, 
            labels_np, 
            title=title
        )
        
        # Log to wandb
        if trainer.logger and hasattr(trainer.logger, 'experiment'):
            # Log matplotlib figure
            trainer.logger.experiment.log({
                # f"{stage}/embeddings_2d_{self.reduction_method}_matplotlib": wandb.Image(fig_matplotlib),
                f"{stage}/embeddings_2d_{self.reduction_method}_plotly": fig_plotly,
                # f"{stage}/embedding_count": len(embeddings_np),
                # f"{stage}/embedding_dimension": embeddings_np.shape[1],
                # f"{stage}/reduced_dimension": reduced_embeddings.shape[1]
            })
        
        
        # Close matplotlib figure to free memory
        # plt.close(fig_matplotlib)
        
        # Reset cache after computing reduction
        self._reset_cache()
    
    
    def _reset_cache(self) -> None:
        """Reset the embedding and label caches."""
        self.embedding_cache.clear()
        self.label_cache.clear()
    
    def on_fit_end(self, trainer, pl_module) -> None:
        """Clean up at the end of training."""
        self._reset_cache()
    
    def on_test_end(self, trainer, pl_module) -> None:
        """Clean up at the end of testing."""
        self._reset_cache()


        # INSERT_YOUR_CODE

import os
import torch

class SaveEmbeddingsCallback(Callback):
    """
    Callback to save embeddings for each sample in the validation set.
    Saves one .pt file per sample, containing the embedding, label, and file_path.
    """
    def __init__(self, save_dir: str = "embeddings", every_n_epochs: int = 1):
        super().__init__()
        self.save_dir = save_dir
        os.makedirs(self.save_dir, exist_ok=True)
        self.every_n_epochs = every_n_epochs
        self._reset_cache()

    def _reset_cache(self):
        """Reset the embedding and label caches."""
        self.z_cache = []
        self.g_cache = []
        self.label_cache = []
        self.file_path_cache = []

    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0):
        # Assume batch contains: "file_path", "label", and the model returns embeddings in outputs
        # outputs: the output of pl_module.validation_step
        # batch: the batch dict from the dataloader

        # Get embeddings from outputs or batch
        # Try to support both dict and tensor outputs
        if trainer.current_epoch % self.every_n_epochs != 0:
            return
        clean_data = batch["audio"]
        with torch.no_grad():
            z = pl_module.backbone(clean_data)["z"]
            g = pl_module.projection_head(z)
            z = z.detach().cpu()
            g = g.detach().cpu()

        # Get file paths and labels
        file_paths = batch.get("file_path", None)
        labels = batch.get("name", None)

        # Convert to list if not already
        if isinstance(file_paths, str):
            file_paths = [file_paths]
        if labels is not None and not isinstance(labels, (list, tuple)):
            labels = labels.tolist() if hasattr(labels, "tolist") else [labels]

        # Detach embeddings and move to cpu
        
        # Save to cache for epoch end
        self.z_cache.append(z)
        self.g_cache.append(g)
        if labels is not None:
            self.label_cache.append(labels)
        self.file_path_cache.extend(file_paths)

    def on_validation_epoch_end(self, trainer, pl_module):
        # Concatenate all cached embeddings and labels
        if trainer.current_epoch % self.every_n_epochs != 0:
            return
        z = torch.cat(self.z_cache, dim=0)
        g = torch.cat(self.g_cache, dim=0)
        file_paths = self.file_path_cache
        labels = None
        if self.label_cache:
            # Flatten list of lists
            labels = [item for sublist in self.label_cache for item in (sublist if isinstance(sublist, (list, tuple)) else [sublist])]

        # Save each embedding individually
        for idx, file_path in enumerate(file_paths):
            # Clean file_path to use as filename
            base_name = os.path.basename(file_path)
            name, _ = os.path.splitext(base_name)
            current_epoch = trainer.current_epoch
            os.makedirs(os.path.join(self.save_dir, f"epoch_{current_epoch}"), exist_ok=True)
            save_name = f"epoch_{current_epoch}/{name}.pt"
            save_path = os.path.join(self.save_dir, save_name)
            data = {
                "z": z[idx],
                "g": g[idx]
            }
            if labels is not None:
                data["label"] = labels[idx]
            data["file_path"] = file_path
            torch.save(data, save_path)

        # Reset cache after saving
        self._reset_cache()


class LinearProbeCallback(Callback):
    """
    Callback to fit a linear multiclass logistic regression on the embeddings.
    """
    def __init__(self, model_type: str = "logistic", every_n_epochs: int = 10, **kwargs):
        super().__init__()
        self.kwargs = kwargs
        self.model_type = model_type
        self.every_n_epochs = every_n_epochs
        self.reset_model()
        self._reset_cache()

    def reset_model(self):

        from sklearn.linear_model import LogisticRegression
        from sklearn.neural_network import MLPClassifier
        self.model = LogisticRegression() if self.model_type == "logistic" else MLPClassifier(**self.kwargs)

    def _reset_cache(self):
        """Reset the embedding and label caches."""
        self.embedding_cache = []
        self.label_cache = []

    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0):
        # Assume batch contains: "file_path", "label", and the model returns embeddings in outputs
        # outputs: the output of pl_module.validation_step
        # batch: the batch dict from the dataloader

        # Get embeddings from outputs or batch
        # Try to support both dict and tensor outputs
        # Extract clean data
        if trainer.current_epoch % self.every_n_epochs != 0:
            return
        
        clean_data = batch["audio"]
        
        # Extract labels if available
        labels = batch.get("name", None)
        
        # Set model to eval mode and disable gradients
        embeddings = pl_module.backbone(clean_data)["z"]
        embeddings = pl_module.projection_head(embeddings)

        # Detach embeddings and move to cpu
        embeddings = embeddings.detach().cpu()

        # Save to cache for epoch end
        self.embedding_cache.append(embeddings)
        if labels is not None:
            self.label_cache.append(labels)

    def on_validation_epoch_end(self, trainer, pl_module):
        # Concatenate all cached embeddings and labels
        if trainer.current_epoch % self.every_n_epochs != 0:
            return
        
        embeddings = torch.cat(self.embedding_cache, dim=0)
        labels = None
        if self.label_cache:
            # Flatten list of lists
            labels = [item for sublist in self.label_cache for item in (sublist if isinstance(sublist, (list, tuple)) else [sublist])]
        
        # Fit the model
        print(embeddings[0])
        print(labels)
        print(f"Fitting model with {embeddings.shape[0]} embeddings and {len(labels)} labels")
        self.model.fit(embeddings, labels)
        
        # score the model
        score = self.model.score(embeddings, labels)
        
        # Log the score
        trainer.logger.experiment.log({
            "linear_probe_score": score
        })

        # reset the model
        self.reset_model()
        
        # Reset cache after fitting
        self._reset_cache()

