import os
import torch
import numpy as np
from typing import Optional, List, Dict, Any
from pytorch_lightning import Callback
from pytorch_lightning.utilities.types import STEP_OUTPUT
import wandb

from ..utils.viz import (
    compute_tsne, compute_umap, create_matplotlib_scatter, 
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
        every_n_steps: int = 100,
        reduction_method: str = "umap",
        save_embeddings: bool = False,
        save_dir: str = "embeddings",
        tsne_perplexity: float = 30.0,
        umap_n_neighbors: int = 15,
        umap_min_dist: float = 0.1,
        random_state: int = 42
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
        super().__init__()
        
        if reduction_method not in ["umap", "tsne"]:
            raise ValueError(f"reduction_method must be 'umap' or 'tsne', got {reduction_method}")
        
        self.every_n_steps = every_n_steps
        self.reduction_method = reduction_method
        self.save_embeddings = save_embeddings
        self.save_dir = save_dir
        self.tsne_perplexity = tsne_perplexity
        self.umap_n_neighbors = umap_n_neighbors
        self.umap_min_dist = umap_min_dist
        self.random_state = random_state
        
        # Initialize embedding and label caches
        self.embedding_cache: List[torch.Tensor] = []
        self.label_cache: List[torch.Tensor] = []
        self.step_count = 0
        
        # Create save directory if needed
        if self.save_embeddings:
            os.makedirs(self.save_dir, exist_ok=True)
    
    def on_train_batch_end(
        self, 
        trainer, 
        pl_module, 
        outputs: STEP_OUTPUT, 
        batch: Any, 
        batch_idx: int
    ) -> None:
        """Extract embeddings from training batch."""
        self._extract_embeddings(pl_module, batch, "train")
        self.step_count += 1
        
        # Compute reduction every N steps
        if self.step_count % self.every_n_steps == 0:
            self._compute_reduction_and_log(trainer, "train", self.step_count)
    
    def on_validation_batch_end(
        self, 
        trainer, 
        pl_module, 
        outputs: STEP_OUTPUT, 
        batch: Any, 
        batch_idx: int
    ) -> None:
        """Extract embeddings from validation batch."""
        self._extract_embeddings(pl_module, batch, "val")
    
    def on_test_batch_end(
        self, 
        trainer, 
        pl_module, 
        outputs: STEP_OUTPUT, 
        batch: Any, 
        batch_idx: int
    ) -> None:
        """Extract embeddings from test batch."""
        self._extract_embeddings(pl_module, batch, "test")
    
    def on_train_epoch_end(self, trainer, pl_module) -> None:
        """Compute reduction and log at end of training epoch."""
        if len(self.embedding_cache) > 0:
            self._compute_reduction_and_log(trainer, "train", trainer.current_epoch)
            self._reset_cache()
    
    def on_validation_epoch_end(self, trainer, pl_module) -> None:
        """Compute reduction and log at end of validation epoch."""
        if len(self.embedding_cache) > 0:
            self._compute_reduction_and_log(trainer, "val", trainer.current_epoch)
            self._reset_cache()
    
    def on_test_epoch_end(self, trainer, pl_module) -> None:
        """Compute reduction and log at end of test epoch."""
        if len(self.embedding_cache) > 0:
            self._compute_reduction_and_log(trainer, "test", trainer.current_epoch)
            self._reset_cache()
    
    def _extract_embeddings(self, pl_module, batch: Any, stage: str) -> None:
        """
        Extract embeddings from batch using the model's extract_features method.
        
        Args:
            pl_module: PyTorch Lightning module
            batch: Input batch
            stage: Current stage (train/val/test)
        """
        if "clean" not in batch:
            return
        
        # Extract clean data
        clean_data = batch["clean"]
        
        # Extract labels if available
        labels = batch.get("labels", None)
        
        # Set model to eval mode and disable gradients
        with torch.no_grad():
            pl_module.eval()
            try:
                # Extract features using the model's extract_features method
                embeddings = pl_module.model.extract_features(clean_data)
                
                # Add to cache
                self.embedding_cache.append(embeddings.detach())
                if labels is not None:
                    self.label_cache.append(labels.detach())
                
            except Exception as e:
                print(f"Warning: Failed to extract embeddings in {stage} stage: {e}")
            finally:
                # Restore training mode if we were training
                if pl_module.training:
                    pl_module.train()
    
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
        labels_np = labels_to_numpy(self.label_cache) if self.label_cache else None
        
        # Compute dimensionality reduction
        if self.reduction_method == "tsne":
            reduced_embeddings = compute_tsne(
                embeddings_np, 
                perplexity=self.tsne_perplexity,
                random_state=self.random_state
            )
        else:  # umap
            reduced_embeddings = compute_umap(
                embeddings_np,
                n_neighbors=self.umap_n_neighbors,
                min_dist=self.umap_min_dist,
                random_state=self.random_state
            )
        
        # Create visualizations
        title = f"{stage.capitalize()} Embeddings - {self.reduction_method.upper()}"
        
        # Matplotlib scatter plot
        fig_matplotlib = create_matplotlib_scatter(
            reduced_embeddings, 
            labels_np, 
            title=title
        )
        
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
                f"{stage}/embeddings_2d_{self.reduction_method}_matplotlib": wandb.Image(fig_matplotlib),
                f"{stage}/embeddings_2d_{self.reduction_method}_plotly": fig_plotly,
                f"{stage}/embedding_count": len(embeddings_np),
                f"{stage}/embedding_dimension": embeddings_np.shape[1],
                f"{stage}/reduced_dimension": reduced_embeddings.shape[1]
            })
        
        # Save embeddings locally if requested
        if self.save_embeddings:
            self._save_embeddings(
                embeddings_np, 
                reduced_embeddings, 
                labels_np, 
                stage, 
                step_or_epoch
            )
        
        # Close matplotlib figure to free memory
        plt.close(fig_matplotlib)
        
        # Reset cache after computing reduction
        self._reset_cache()
    
    def _save_embeddings(
        self, 
        embeddings: np.ndarray, 
        reduced_embeddings: np.ndarray, 
        labels: Optional[np.ndarray], 
        stage: str, 
        step_or_epoch: int
    ) -> None:
        """
        Save embeddings and reduced embeddings locally.
        
        Args:
            embeddings: Original embeddings
            reduced_embeddings: Reduced 2D embeddings
            labels: Optional labels
            stage: Current stage
            step_or_epoch: Current step or epoch
        """
        # Create filename
        filename = f"{stage}_embeddings_{step_or_epoch}.npz"
        filepath = os.path.join(self.save_dir, filename)
        
        # Save data
        save_dict = {
            'embeddings': embeddings,
            'reduced_embeddings': reduced_embeddings,
            'reduction_method': self.reduction_method,
            'stage': stage,
            'step_or_epoch': step_or_epoch
        }
        
        if labels is not None:
            save_dict['labels'] = labels
        
        np.savez_compressed(filepath, **save_dict)
        print(f"Saved embeddings to {filepath}")
    
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
