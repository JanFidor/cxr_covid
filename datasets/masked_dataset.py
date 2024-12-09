import os
from pathlib import Path
import torch
import numpy as np
from datasets.cxrdataset import CXRDataset, mean, std
from utils import denormalize
import torchvision.transforms as v2

#!/usr/bin/env python3

class MaskedDataset(torch.utils.data.Dataset):
    """
    A wrapper dataset that applies segmentation masks to images from another dataset.
    """
    
    def __init__(self, 
        ds: CXRDataset, 
        split: str, 
        chosen_masks: torch.tensor, 
        is_inverted: bool = False, 
        is_binary: bool = False
    ):
        """
        Args:
            base_dataset: The original dataset to wrap
            mask_index: The index in the label array to use for masking (default=0)
        """
        self.ds = ds
        self.filter = chosen_masks
        self.mask_dir = os.path.join("data/segmentations", ds.dataset_name, split)

        self.is_inverted = is_inverted
        self.is_binary = is_binary

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        image, label, name, meta = self.ds[idx]
        
        # Load corresponding mask
        mask_path = Path(self.mask_dir, str(idx)).with_suffix('.pt')
        mask = torch.load(mask_path, weights_only=True)
        mask = mask.permute(1, 2, 0) * self.filter
        mask = mask.permute(2, 0, 1).sum(dim=0).clip(0, 1)

        if self.is_inverted:
            mask = 1 - mask
        
        if self.is_binary:
            mask = v2.Normalize(mean, std)(torch.stack([mask, mask, mask]))
            return mask, label, name, meta

        # Apply mask to image
        image = denormalize(image)
        image = image * mask
        image = v2.Normalize(mean, std)(image)
            
        return image, label, name, meta

    def get_all_labels(self):
        """
        Pass through the base dataset's labels
        """
        return self.ds.get_all_labels()
    
    @property
    def labels(self):
        return self.ds.labels