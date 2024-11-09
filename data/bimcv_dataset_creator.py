#!/usr/bin/env python

import numpy as np
import cv2
import pandas as pd
from abc import abstractmethod, ABC
from torchvision import transforms

from PIL import Image

# use imagenet mean,std for normalization
mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]


UINT8_MAX = 255 
SKIP_WINDOWING = [
    "sub-S03169_ses-E07193_run-1_bp-chest_vp-ap_dx.png",
    "sub-S03169_ses-E07878_run-1_bp-chest_vp-ap_dx.png",
    "sub-S03067_ses-E07350_run-1_bp-chest_vp-ap_dx.png",
    "sub-S03067_ses-E07908_run-1_bp-chest_vp-ap_dx.png",
    "sub-S03240_ses-E07784_run-1_bp-chest_vp-ap_dx.png",
    "sub-S03218_ses-E07970_run-1_bp-chest_vp-ap_dx.png",
    "sub-S03240_ses-E07626_run-1_bp-chest_vp-ap_dx.png",
    "sub-S03214_ses-E07979_run-1_bp-chest_vp-ap_dx.png"
]
FLIP = [
    "sub-S03936_ses-E08636_run-1_bp-chest_vp-ap_dx.png",
    "sub-S04275_ses-E08736_run-1_bp-chest_vp-ap_dx.png",
    "sub-S04190_ses-E08730_run-1_bp-chest_vp-ap_dx.png",
    "sub-S03068_ses-E06592_run-1_bp-chest_vp-ap_cr.png"
]

class BimcvDatasetABC(ABC):
    def __init__(self):
        self.df = pd.read_csv(self.df_path)

        self.df = self.df.query('window_center == window_center | lut == lut') # remove NaN
        self.df.lut = self.df.lut.apply(lambda x: eval(x) if isinstance(x,str) else x) # strings to LUT lists      

        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])  

    @property
    @abstractmethod
    def df_path(self):
        return NotImplementedError
    
    def process_image_by_index(self, idx):
        image_path = self.df.path.iloc[idx]
        imagename = image_path.split('/')[-1]
        image = self._load_image(image_path).astype(float)
        
        if imagename not in SKIP_WINDOWING:
            image = self._process_windowing(image, imagename, idx)
        try:
            image = self.transform(image)
        except:
            image = self.transform(image.copy())

        return image, imagename
    
    def _process_windowing(self, image, imagename, idx):
        image = np.array(image, dtype=np.int64)
        lut = self.df.lut.iloc[idx]

        if isinstance(lut, list):   # Use LUT if we have it 
            image = self._preprocess_with_lut(image, idx)
        else:   #  use window data
            image = self._preprocess_with_window(image, idx)

        # clip
        image[image<0] = 0
        image[image>1] = 1
        image = (image * UINT8_MAX).astype(np.uint8)

        image = self._photometric_processing(image, idx)

        if imagename in FLIP:
            image = np.flipud(image)
        return image

    def _preprocess_with_lut(self, image, idx):
        lut = self.df.lut.iloc[idx]
        lut_min = int(self.df.lut_min.iloc[idx])
        lut = np.array(lut)

        # magic
        lut = np.concatenate(
            (np.ones(lut_min)*lut[0], lut, np.ones(65536-lut_min-len(lut))*lut[-1]), axis=0
        )
        image = lut[image]
        if self.df.rescale_slope.iloc[idx]:
            image *= self.df.rescale_slope.iloc[idx] + self.df.rescale_intercept.iloc[idx]
        max_luminosity = 2**self.df.bits_stored.iloc[idx] - 1  
        image = image.astype(np.float64) / max_luminosity

        return image

    def _preprocess_with_window(self, image, idx):
        window_center = self.df.window_center.iloc[idx]
        window_width = self.df.window_width.iloc[idx]
        window_min = int(window_center - window_width/2)
        image -= window_min
        image = image.astype(np.float64) / window_width

        return image
    
    def _photometric_processing(self, image, idx):
        photometric_interpretation = self.df.photometric_interpretation.iloc[idx]
        if photometric_interpretation == 'MONOCHROME1':
            return UINT8_MAX - image
        elif photometric_interpretation == 'MONOCHROME2':
            return image
        raise ValueError('unknown photometric interpretation: {:s}'.format(photometric_interpretation))

    def _load_image(self, image_path):
        image_array = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
        imagenet_size = (224, 224)
        image_array = cv2.resize(image_array, imagenet_size, interpolation=cv2.INTER_AREA)

        if len(image_array.shape) != 3: # add channel dimension
            image_array = np.expand_dims(image_array, -1) 

        if image_array.shape[-1] == 1: # repeat channel dimension for normalization
            image_array = np.repeat(image_array, 3, 2)
        if image_array.shape[-1] == 4: # repeat channel dimension for normalization
            image_array = cv2.cvtColor(image_array, cv2.COLOR_RGBA2RGB)

        
        return image_array
    
class BimcvDatasetPositive(BimcvDatasetABC):
    @property
    def df_path(self):
        return 'bimcv+/bimcv+.csv'
    
class BimcvDatasetNegative(BimcvDatasetABC):
    @property
    def df_path(self):
        return 'bimcv-/bimcv-.csv'
