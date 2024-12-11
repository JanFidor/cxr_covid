from torchvision.transforms import v2
import torchvision.transforms.functional as F
import torch
import numpy as np
import math

from covid_transforms import AUGMENTATION_SETUP


MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]

NORMALIZED_BLACK = (-np.array(MEAN) / np.array(STD)).tolist()

def get_train_augmentations(name):
    return {
        "weak": v2.Compose([
            v2.CenterCrop(int(224 * 0.95)),
            v2.Resize(224),
            v2.RandomRotation(5),
            v2.RandomHorizontalFlip(0.5)
        ]),
        "medium": v2.Compose([
            v2.CenterCrop(int(224 * 0.85)),
            v2.Resize(224),
            v2.RandomRotation(10),
            v2.RandomHorizontalFlip(0.5)
        ]),
        "strong": v2.Compose([
            v2.CenterCrop(int(224 * 0.75)),
            v2.Resize(224),
            v2.RandomRotation(10),
            v2.RandomHorizontalFlip(0.5)
        ]),
        "strongXL": v2.Compose([
            v2.CenterCrop(int(224 * 0.70)),
            v2.Resize(224),
            v2.RandomRotation(12.5),
            v2.RandomHorizontalFlip(0.5)
        ]),
        "strongXXL": v2.Compose([
            v2.CenterCrop(int(224 * 0.65)),
            v2.Resize(224),
            v2.RandomRotation(15),
            v2.RandomHorizontalFlip(0.5)
        ]),
        "cropXXXL": v2.Compose([
            v2.CenterCrop(int(224 * 0.55)),
            v2.Resize(224),
            v2.RandomRotation(5),
            v2.RandomHorizontalFlip(0.5)
        ]),
        "weak-random": v2.Compose([
            v2.RandomResizedCrop(224, [0.95, 1]),
            v2.RandomRotation(5),
            v2.RandomHorizontalFlip(0.5)
        ]),
        "medium-random": v2.Compose([
            v2.RandomResizedCrop(224, [0.85, 1]),
            v2.RandomRotation(10),
            v2.RandomHorizontalFlip(0.5)
        ]),
        "strong-random": v2.Compose([
            v2.RandomResizedCrop(224, [0.75, 1]),
            v2.RandomRotation(10),
            v2.RandomHorizontalFlip(0.5)
        ]),
        "none": v2.Identity()
    }[name]

def denormalize(x: torch.Tensor):
    is_batched = len(x.shape) == 4

    _mean = torch.tensor(MEAN)
    _std = torch.tensor(STD)
    
    if is_batched:
        x = x.permute(0, 2, 3, 1)
    else:
        x = x.permute(1, 2, 0)

    x = _std * x + _mean

    if is_batched:
        x = x.permute(0, 3, 1, 2)
    else:
        x = x.permute(2, 0, 1)

    return x

#color-0.5|rot-10|flip-0.5
def get_augmentations(name):
    if name in AUGMENTATION_SETUP:
        return AUGMENTATION_SETUP[name]
    if name == 'none': return v2.Identity()

    all_augments = name.split("|")
    if len(all_augments) != 1:
        return v2.Compose([
            get_augmentations(n) for n in all_augments
        ])
    
    aug_type, intensity = all_augments[0].split("-")
    intensity = float(intensity)
    if aug_type == "color":
        return v2.Compose([
            v2.Lambda(lambda x: denormalize(x)),
            v2.ColorJitter(intensity, intensity, 0, 0),
            v2.Normalize(MEAN, STD),
        ])
    elif aug_type == "rot":
        return v2.RandomRotation(intensity)
    elif aug_type == "flip":
        return v2.RandomHorizontalFlip(intensity)

    raise KeyError("incorrect augmentation")

#crop_cent-0.5|crop_rand-0.5|rot-10
def get_preprocessing(name):
    if name == 'none': return v2.Identity()

    all_prepro = name.split("|")
    if len(all_prepro) != 1:
        return v2.Compose([
            get_preprocessing(n) for n in all_prepro
        ])
    
    prepro_type, intensity = all_prepro[0].split("-")
    intensity = float(intensity)
    if prepro_type == "crop_cent":
        return v2.Compose([
            v2.CenterCrop(int(224 * intensity)),
            v2.Resize(224),
        ])
    elif prepro_type == "crop_pad":
        return v2.Compose([
            v2.CenterCrop(int(224 * intensity)),
            v2.Pad((
                math.ceil(224 * (1 - intensity) / 2),
                math.ceil(224 * (1 - intensity) / 2),
                math.ceil(224 * (1 - intensity) / 2),
                math.ceil(224 * (1 - intensity) / 2)
            ), fill=NORMALIZED_BLACK),
            v2.Resize(224)
        ])
    elif prepro_type == "inv_crop_pad":
        return v2.Lambda(lambda x: outer_crop(x, intensity))
    elif prepro_type == "rot":
        return v2.Lambda(lambda x: F.rotate(x, intensity))

    raise KeyError("incorrect augmentation")

def outer_crop(tensor, intensity):
    border = int(224 * (1 - intensity) / 2)
    ord1 = (0, 2, 3, 1) if len(tensor.shape) == 4 else (1, 2, 0)
    ord2 = (0, 3, 1, 2) if len(tensor.shape) == 4 else (2, 0, 1)
    tensor = tensor.permute(ord1)
    tensor[border:224-border, border:224-border] =  torch.tensor(NORMALIZED_BLACK)
    return tensor.permute(ord2)


def denormalize_image(image):
    img = denormalize(image).numpy().transpose((1, 2, 0))
    return img.clip(0, 1)