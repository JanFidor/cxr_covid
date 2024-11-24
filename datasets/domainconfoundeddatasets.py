#!/usr/bin/env python
import numpy
import torch

class DomainConfoundedDataset(torch.utils.data.Dataset):
    def __init__(self, datasetclass1, datasetclass2, flip_indices=None):
        self.ds1 = datasetclass1
        self.ds2 = datasetclass2
        self.len1 = len(self.ds1)
        self.len2 = len(self.ds2)
        self.flip_indices = set(flip_indices or [])  # Convert to set for O(1) lookup
        self.labels = self.ds1.labels

    def __getitem__(self, idx):
        if idx < self.len1:
            item = self.ds1[idx]
        else:
            item = self.ds2[idx-self.len1]
            
        image, label, path, meta = item
        
        # Flip last label if index is in flip_indices
        if idx in self.flip_indices:
            label = label.copy()  # Create copy to avoid modifying original
            label[-1] = 1 - label[-1]  # Flip last bit (0->1 or 1->0)
            
        return image, label, path, meta

    def __len__(self):
        return self.len1 + self.len2

    def get_all_labels(self):
        labels1 = self.ds1.get_all_labels()
        labels2 = self.ds2.get_all_labels()
        
        # Apply flips to combined labels
        all_labels = numpy.vstack((labels1, labels2))
        for idx in self.flip_indices:
            all_labels[idx, -1] = 1 - all_labels[idx, -1]
            
        return all_labels
