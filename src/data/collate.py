import torch
from torch.utils.data import default_collate

def multiview_collate(batch):
    """
    Collate function for multiview contrastive learning.
    
    Handles two cases:
    1. Multiview data: Each item has 'view_1' and 'view_2' keys
    2. Single view data: Each item has 'audio' key
    
    For multiview data, concatenates views and creates similarity matrix.
    """
    # Check if this is multiview data
    new_batch = {}
    if "view_1" in batch[0] and "view_2" in batch[0]:
        # Extract views from each item
        view_1_list = [item.pop("view_1") for item in batch]
        view_2_list = [item.pop("view_2") for item in batch]
        
        # Stack and concatenate views
        view_1_tensor = torch.stack(view_1_list, dim=0)
        view_2_tensor = torch.stack(view_2_list, dim=0)
        views = torch.cat([view_1_tensor, view_2_tensor], dim=0)
        
        new_batch["views"] = views
        
        # Handle labels if they exist
        if "label_" in batch[0]:
            label_1_list = [item.pop("label_") for item in batch]
            # Duplicate labels for both views (same sample, different views)
            labels = label_1_list + label_1_list
            new_batch["label_"] = labels
    
    
    # Use default collate for the remaining items
    remaining_items = [item for item in batch if item]  # Remove empty dicts
    if remaining_items:
        remaining_batch = default_collate(remaining_items)
        new_batch.update(remaining_batch)
    
    # Create similarity matrix for contrastive learning
    if "label_" in new_batch and new_batch["label_"] is not None:
        labels = new_batch["label_"]
        # Create similarity matrix: 1 if same label, 0 if different
        labels = torch.Tensor(labels)
        sims = (labels.unsqueeze(1) == labels.unsqueeze(0)).float()
        new_batch["target_sims"] = sims
        
    return new_batch
