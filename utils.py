from torchvision.transforms import v2
import torchvision.transforms.functional as F
from torchvision.models.densenet import DenseNet
import torch
import numpy as np
import math

from models.cxrclassifier import AlexNet, CXRClassifier
from pytorch_grad_cam import GradCAM, EigenCAM, GradCAMPlusPlus, EigenGradCAM


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

def denormalize(x):
    _mean = torch.tensor(MEAN)
    _std = torch.tensor(STD)
    
    x = x.permute(1, 2, 0)
    x = _std * x + _mean
    return x.permute(2, 0, 1)

#color-0.5|rot-10|flip-0.5
def get_augmentations(name):
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
            v2.Pad(math.ceil(224 * (1 - intensity)), fill=NORMALIZED_BLACK),
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


def get_gradcam(gradcam_name, model_path):
    model = load_model(model_path)
    gradcam_layers = get_gradcam_layers(model)

    return {
        "grad_cam": GradCAM(model=model, target_layers=gradcam_layers),
        "grad++_cam": GradCAMPlusPlus(model=model, target_layers=gradcam_layers),
        "eigen_cam": EigenCAM(model=model, target_layers=gradcam_layers),
        "eigengrad_cam": EigenGradCAM(model=model, target_layers=gradcam_layers),
    }.get(gradcam_name)


def get_gradcam_layers(model):
    if isinstance(model, AlexNet):
        return [model.features[-3]]
    elif isinstance(model, DenseNet):
        return [model.features[-2].denselayer16.conv2]
    raise KeyError


def load_model(model_path, model_name="densenet121-pretrain", n_labels=15):
    model = CXRClassifier()
    if model_name == 'alexnet':
        model.build_model_scratch(n_labels)
    else:
        pretrained = model_name == 'logistic' or model_name.split("-")[1] == 'pretrain'
        model.build_model(n_labels, pretrained)
    model.load_checkpoint(model_path)
    return model.model


def denormalize_image(image):
    img = denormalize(image).numpy().transpose((1, 2, 0))
    return img.clip(0, 1)