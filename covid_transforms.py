from torchvision.transforms import v2
import torchvision.transforms.functional as F
from torchvision.models.densenet import DenseNet
import torch
import numpy as np
import math

CROP_PAD_INTENSITY = 0.25

MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]
NORMALIZED_BLACK = [-0.485/0.229, -0.456/0.224, -0.406/0.225]

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
        intensity = 0
        if abstract_intensity == 'weak': intensity = 0.05
        if abstract_intensity == 'medium': intensity = 0.1
        if abstract_intensity == 'strong': intensity = 0.15
        return v2.RandomAffine(degrees=10, translate=(intensity, intensity), scale=(1-intensity, 1+intensity), fill=NORMALIZED_BLACK)
    if name == 'crop':
        intensity = 0
        if abstract_intensity == 'weak': intensity = 0.05
        if abstract_intensity == 'medium': intensity = 0.1
        if abstract_intensity == 'strong': intensity = 0.15
        return random_crop_by_translation(intensity, CROP_PAD_INTENSITY)
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

def random_compose(augmentations, p=1):
    return v2.Compose([
        v2.RandomApply([auh], p=p) for auh in augmentations
    ])

AUGMENTATION_SETUP = {
    "no_augments": v2.Identity(),
    "color-weak": get_transforms('color', 'weak'),
    "color-strong": get_transforms('color', 'strong'),
    "geom-weak": get_transforms('affine', 'weak'),
    "geom-strong": get_transforms('affine', 'strong'),
    "combined-weak-rare": random_compose([
        get_transforms('flip'),
        get_transforms('crop', 'weak'),
        get_transforms('affine', 'weak'),
        get_transforms('color', 'weak'),
    ], 0.5),
    "combined-medium-rare": random_compose([
        get_transforms('flip'),
        get_transforms('crop', 'medium'),
        get_transforms('affine', 'medium'),
        get_transforms('color', 'medium'),
    ], 0.5),
    "combined-strong-rare": random_compose([
        get_transforms('flip'),
        get_transforms('crop', 'strong'),
        get_transforms('affine', 'strong'),
        get_transforms('color', 'strong'),
    ], 0.5),
    "combined-weak-always": random_compose([
        get_transforms('flip'),
        get_transforms('crop', 'weak'),
        get_transforms('affine', 'weak'),
        get_transforms('color', 'weak'),
    ], 1),
    "combined-medium-always": random_compose([
        get_transforms('flip'),
        get_transforms('crop', 'medium'),
        get_transforms('affine', 'medium'),
        get_transforms('color', 'medium'),
    ], 1),
    "combined-strong-always": random_compose([
        get_transforms('flip'),
        get_transforms('crop', 'strong'),
        get_transforms('affine', 'strong'),
        get_transforms('color', 'strong'),
    ], 1),
}


VISUALIZATION_SETUP = {
    "NoAugmentations": v2.Identity(),
    "ColorWeakMax": color_visualize(1.2, 1.2),
    "ColorWeakMin": color_visualize(0.8, 0.8),
    "ColorStrongMax": color_visualize(1.4, 1.4),
    "ColorStrongMin": color_visualize(0.6, 0.6),
}