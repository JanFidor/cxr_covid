#!/usr/bin/env python3
import os 

import numpy
import torch
from pathlib import Path
from abc import ABC, abstractmethod

# use imagenet mean,std for normalization
mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]

class CXRDataset(torch.utils.data.Dataset, ABC):
    '''
    Base class for chest radiograph datasets.
    '''

    def __init__(self, augments):
        self.tensor_dir = os.path.join("data/tensors", self.dataset_name)
        self.label_dir = os.path.join("data/labels", self.dataset_name)
        self.augments = augments

    @property
    @abstractmethod
    def dataset_name(self):
        raise NotImplementedError

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        image = self._raw_image_from_disk(self.df.index[idx])
        label = self._get_label(idx)

        return (image, label, self.df.index[idx], ['None'])

    def _raw_image_from_disk(self, name):
        tensor_name = Path(name).with_suffix('.pt')
        tensor_path = os.path.join(self.tensor_dir, tensor_name)
        tensor = torch.load(tensor_path, weights_only=True)
        if self.augments: 
            tensor = self.augments(tensor)
        return tensor.float()

    def _get_label(self, idx):
        label = numpy.zeros(len(self.labels), dtype=int)
        for i in range(0, len(self.labels)):
            if self.labels[i] != "N/A":
                if(self.df[self.labels[i].strip()].iloc[idx].astype('int') > 0):
                    label[i] = self.df[self.labels[i].strip()
                                       ].iloc[idx].astype('int')
        return label

    def get_all_labels(self):
        '''
        Return a numpy array of shape (n_samples, n_dimensions) that includes 
        the ground-truth labels for all samples.
        '''
        ndim = len(self.labels)
        nsamples = len(self)
        output = numpy.zeros((nsamples, ndim))
        for isample in range(len(self)):
            output[isample] = self._get_label(isample)
        return output
