from torchvision.transforms import v2
import torchvision.transforms.functional as F
from torchvision.models.densenet import DenseNet
import torch
import numpy as np
import math
from  monai.transforms import RandSpatialCrop, SpatialPad, RandAffined
from torch import nn

MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]
NORMALIZED_BLACK = [-0.485/0.229, -0.456/0.224, -0.406/0.225]

class RandomCenterCrop(nn.Module):
    """
    Randomly crops the center of the image with size between min_scale and max_scale of original size,
    then pads back to original size with normalized black pixels.
    Args:
        min_scale (float): minimum scale of the crop (between 0 and 1)
        max_scale (float): maximum scale of the crop (between 0 and 1)
    """
    def __init__(self, min_scale=0.8, max_scale=1.0):
        super().__init__()
        self.min_scale = min_scale
        self.max_scale = max_scale

    def forward(self, img):
        # Store original size
        _, orig_h, orig_w = img.shape
        
        # Generate random scale between min_scale and max_scale
        scale = torch.empty(1).uniform_(self.min_scale, self.max_scale).item()
        
        # Calculate crop size
        new_h = int(orig_h * scale)
        new_w = int(orig_w * scale)
        
        # Center crop
        cropped = F.center_crop(img, [new_h, new_w])
        
        # Pad back to original size with normalized black
        cropped = denormalize(cropped)
        padded = SpatialPad(spatial_size=(224, 224), mode="constant", constant_values=0)(cropped)
        normal = v2.Normalize(MEAN, STD)(padded)
        return normal

    def __repr__(self):
        return f"{self.__class__.__name__}(min_scale={self.min_scale}, max_scale={self.max_scale})" 

def get_transforms(name, abstract_intensity = None):
    if name == 'none': return v2.Identity()

    if name == 'color':
        intensity = 0
        if abstract_intensity == 'weak': intensity = 0.2
        if abstract_intensity == 'medium': intensity = 0.3
        if abstract_intensity == 'strong': intensity = 0.4
        return v2.Compose([
            v2.Lambda(lambda x: denormalize(x)),
            v2.ColorJitter(intensity, intensity, 0, 0),
            v2.Normalize(MEAN, STD),
        ])
    if name == 'affine':
        rot, trans, scale = 0, 0, 1
        if abstract_intensity == 'weak': 
            rot, trans, scale = 0, 0.05, 0
        if abstract_intensity == 'strong':
            rot, trans, scale = 5, 0.1, 0.1
        if abstract_intensity == 'extreme':
            rot, trans, scale = 5, 0.15, 0.2
        return v2.RandomAffine(degrees=rot, translate=(trans, trans), scale=(1-scale, 1+scale), fill=NORMALIZED_BLACK)
    if name == 'crop':
        right = 0
        intensity = 0
        if abstract_intensity == 'weak': intensity = 0.15
        if abstract_intensity == 'strong': intensity = 0.25
        if abstract_intensity == 'weak-forced': 
            intensity = 0.15
            right = 0.1
        if abstract_intensity == 'strong-forced': 
            intensity = 0.25
            right = 0.15

        return v2.Compose([
            v2.Lambda(lambda x: denormalize(x)),
            RandSpatialCrop(roi_size=224*(1 - intensity), max_roi_size=224*(1 - right), random_center=False,random_size=True),
            SpatialPad(spatial_size=(224, 224), mode="constant", constant_values=0),
            v2.Normalize(MEAN, STD),
        ])
    if name == "center_crop":
        if abstract_intensity == 'strong-forced': 
            intensity = 0.25
            right = 0.15
        if abstract_intensity == 'extreme-forced': 
            intensity = 0.35
            right = 0.1 
        return RandomCenterCrop(min_scale=1-intensity, max_scale=1-right)
    if name == 'flip':
        return v2.RandomHorizontalFlip(p=0.5)
    
    
    raise KeyError("incorrect augmentation type")

def denormalize(x):
    _mean = torch.tensor(MEAN)
    _std = torch.tensor(STD)
    
    x = x.permute(1, 2, 0)
    x = _std * x + _mean
    return x.permute(2, 0, 1)

"""
Color Augmentations:
    1. ColorJitter
Geom Augmentations:
    1. RandomHorizontalFlip
    2. RandomRotation
    3. RandomResize
    4. RandomTranslation
    5. RandomCrop
"""
# RandomCrop by translation -> CropPad

def random_crop_by_translation(intensity, preprocessing):
    return v2.Compose([
        v2.RandomAffine(degrees=0, translate=(intensity, intensity), fill=NORMALIZED_BLACK),
        v2.CenterCrop(int(224 * (1-preprocessing))),
        v2.Pad(math.ceil(224 * preprocessing), fill=NORMALIZED_BLACK)
    ])

def color_augmentation(intensity):
    return v2.Compose([
        v2.ColorJitter(brightness=intensity, contrast=intensity),
    ])

def color_visualize(brightness, saturation):
    return v2.Compose([
        v2.Lambda(lambda x: F.adjust_brightness(x, brightness)),
        v2.Lambda(lambda x: F.adjust_contrast(x, saturation)),
    ])

def translate_visualize(translate):
    return v2.Compose([
        v2.Lambda(lambda x: F.affine(x, 0, (translate * 224, translate * 224), 1, 0, 0, fill=0)),
    ])

def scale_visualize(zoom):
    return v2.Compose([
        RandAffined(prob=1.0, scale_range=(0.5, 0.5)),  # Scale randomly
        SpatialPad(spatial_size=(224, 224))  # Ensure fixed size
    ])

def random_compose(augmentations, p=1):
    return v2.Compose([
        v2.RandomApply([auh], p=p) for auh in augmentations
    ])

AUGMENTATION_SETUP = {
    "no_augments": v2.Identity(),
    "color-jitter": v2.Compose([
        get_transforms('color', 'medium'),
    ]),
    "affine-weak": v2.Compose([
        get_transforms('affine', 'weak'),
    ]),
    "flip": v2.Compose([
        get_transforms('flip'),
    ]),

    "baseline-augments": v2.Compose([
        get_transforms('affine', 'weak'),
        get_transforms('color', 'medium'),
        get_transforms('flip'),
    ]),
    "random_crop-weak": v2.Compose([
        get_transforms('crop', 'weak'),
        get_transforms('affine', 'weak'),
        get_transforms('color', 'medium'),
        get_transforms('flip'),
    ]),
    "random_crop-strong": v2.Compose([
        get_transforms('crop', 'strong'),
        get_transforms('affine', 'weak'),
        get_transforms('color', 'medium'),
        get_transforms('flip'),
    ]),
    "random_crop-weak-forced": v2.Compose([
        get_transforms('crop', 'weak-forced'),
        get_transforms('affine', 'weak'),
        get_transforms('color', 'medium'),
        get_transforms('flip'),
    ]),
    "random_crop-strong-forced": v2.Compose([
        get_transforms('crop', 'strong-forced'),
        get_transforms('affine', 'weak'),
        get_transforms('color', 'medium'),
        get_transforms('flip'),
    ]),
    "random_center_crop-strong-forced": v2.Compose([
        get_transforms('center_crop', 'strong-forced'),
        get_transforms('affine', 'weak'),
        get_transforms('color', 'medium'),
        get_transforms('flip'),
    ]),
    "random_crop-strong-affine-forced": random_compose([
        get_transforms('crop', 'strong-forced'),
        get_transforms('affine', 'strong'),
        get_transforms('color', 'medium'),
        get_transforms('flip'),
    ], 1),
    "random_center_crop-strong-affine": random_compose([
        get_transforms('crop', 'strong'),
        get_transforms('affine', 'strong'),
        get_transforms('color', 'medium'),
        get_transforms('flip'),
    ], 1),
    "random_crop-extreme-affine-forced": random_compose([
        get_transforms('crop', 'strong-forced'),
        get_transforms('affine', 'extreme'),
        get_transforms('color', 'strong'),
        get_transforms('flip'),
    ], 1),
    "extreme_random_crop-affine": random_compose([
        get_transforms('center_crop', 'extreme-forced'),
        get_transforms('affine', 'weak'),
        get_transforms('color', 'medium'),
        get_transforms('flip'),
    ], 1),
}


VISUALIZATION_SETUP = {
    "NoAugmentations": v2.Identity(),
    "ColorWeakMax": color_visualize(1.2, 1.2),
    "ColorWeakMin": color_visualize(0.8, 0.8),
    "ColorStrongMax": color_visualize(1.4, 1.4),
    "ColorStrongMin": color_visualize(0.6, 0.6),
}

