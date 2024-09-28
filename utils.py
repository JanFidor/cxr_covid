from torchvision.transforms import v2
import torchvision.transforms.functional as F
from torchvision.models.densenet import DenseNet
import torch
import numpy as np

from models.cxrclassifier import AlexNet
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

def get_preprocessing(name):
    return {
        "weak": v2.Compose([
            v2.CenterCrop(int(224 * 0.95)),
            v2.Resize(224),
        ]),
        "weak-padded": v2.Compose([
            v2.RandomCrop(224, int(224 * 0.05), fill=NORMALIZED_BLACK),
        ]),
        "medium": v2.Compose([
            v2.CenterCrop(int(224 * 0.85)),
            v2.Resize(224),
        ]),
        "medium-padded": v2.Compose([
            v2.RandomCrop(224, int(224 * 0.15), fill=NORMALIZED_BLACK),
        ]),
        "strong": v2.Compose([
            v2.CenterCrop(int(224 * 0.75)),
            v2.Resize(224),
        ]),
        "strong-padded": v2.Compose([
            v2.RandomCrop(224, int(224 * 0.25), fill=NORMALIZED_BLACK),
        ]),
        "strong-rot": v2.Compose([
            v2.CenterCrop(int(224 * 0.75)),
            v2.Resize(224),
            v2.Lambda(lambda x: F.rotate(x, 10))
        ]),
        "strongXL": v2.Compose([
            v2.CenterCrop(int(224 * 0.70)),
            v2.Resize(224)
        ]),
        "strongXL-padded": v2.Compose([
            v2.RandomCrop(224, int(224 * 0.30), fill=NORMALIZED_BLACK),
        ]),
        "strongXL-rot": v2.Compose([
            v2.CenterCrop(int(224 * 0.70)),
            v2.Resize(224),
            v2.Lambda(lambda x: F.rotate(x, 12.5)),
        ]),
        "strongXXL": v2.Compose([
            v2.CenterCrop(int(224 * 0.65)),
            v2.Resize(224),
        ]),
        "strongXXL-padded": v2.Compose([
            v2.RandomCrop(224, int(224 * 0.35), fill=NORMALIZED_BLACK),
        ]),
        "strongXXL-rot": v2.Compose([
            v2.CenterCrop(int(224 * 0.65)),
            v2.Resize(224),
            v2.Lambda(lambda x: F.rotate(x, 15)),
        ]),
        "cropXXXL": v2.Compose([
            v2.CenterCrop(int(224 * 0.55)),
            v2.Resize(224)
        ]),
        "cropXXXL-padded": v2.Compose([
            v2.RandomCrop(224, int(224 * 0.45), fill=NORMALIZED_BLACK),
        ]),
        "cropXXXL-rot": v2.Compose([
            v2.CenterCrop(int(224 * 0.55)),
            v2.Resize(224),
            v2.Lambda(lambda x: F.rotate(x, 5)),
        ]),
        "none": v2.Identity()
    }[name]


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


def load_model(model_path):
    cpt = torch.load(model_path, weights_only=False)
    return cpt["model"]


def denormalize_image(image):
    img = image.numpy().transpose((1, 2, 0))  # numpy is [h, w, c] 
    _mean = np.array(MEAN)  # mean of your dataset
    _std = np.array(STD)  # std of your dataset
    img = _std * img + _mean
    return img.clip(0, 1)