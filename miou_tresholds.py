#!/usr/bin/env python
#
# train_covid.py
#
# Run ``python train_covid.py -h'' for information on using this script.
#
import json
import argparse
import torch
import seaborn as sn
from tqdm import tqdm

from models.cxrclassifier import AlexNet
from torchvision.models.densenet import DenseNet
from pytorch_grad_cam import GradCAM, EigenCAM, GradCAMPlusPlus, EigenGradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

from torchmetrics import JaccardIndex
from torchmetrics import MeanMetric

def load_model(model_path):
    cpt = torch.load(model_path, weights_only=False)
    return cpt["model"]

def get_gradcam_layers(model):
    if isinstance(model, AlexNet):
        return [model.features[-3]]
    elif isinstance(model, DenseNet):
        return [model.features[-2].denselayer16.conv2]
    raise KeyError
        
def get_gradcams(gradcam_names, model_path):
    model = load_model(model_path)
    gradcam_layers = get_gradcam_layers(model)

    gradcams = {
        "grad_cam": GradCAM(model=model, target_layers=gradcam_layers),
        "grad++_cam": GradCAMPlusPlus(model=model, target_layers=gradcam_layers),
        "eigen_cam": EigenCAM(model=model, target_layers=gradcam_layers),
        "eigengrad_cam": EigenGradCAM(model=model, target_layers=gradcam_layers),
    }
    is_not_none = lambda x: x is not None
    return list(filter(is_not_none, [gradcams.get(n, None) for n in gradcam_names]))

def model_name_from_path(model_path):
    return model_path

def generate_localisation(threshold, gradcam, path):
    tensor = torch.load(path, weights_only=True)
    
    saliency_map = gradcam(
        input_tensor=tensor.unsqueeze(0),
        targets=[ClassifierOutputTarget(1)]
    )

    return torch.tensor(saliency_map > threshold)

def calculate_saliency_jaccard(threshold, gradcams1, gradcams2, image_paths):
    jaccard = JaccardIndex(task='multiclass', num_classes=2)
    mean_jaccard = MeanMetric()
    for path in image_paths:
        for cam1, cam2 in zip(gradcams1, gradcams2):
            att1 = generate_localisation(threshold, cam1, path)
            att2 = generate_localisation(threshold, cam2, path)

            iou = jaccard(att1, att2)
            mean_jaccard.update(iou)
    return mean_jaccard.compute().item()

def create_miou_matrix(threshold, gradcam_names, model_paths, image_paths):
    all_gradcams = [
        get_gradcams(gradcam_names, path) for path in model_paths
    ]

    d = len(model_paths)
    matrix = [[0]*d for _ in range(d)]

    for i in tqdm(range(d)):
        for ii in range(d):
            cam1 = all_gradcams[i]
            cam2 = all_gradcams[ii]
            matrix[i][ii] = calculate_saliency_jaccard(threshold, cam1, cam2, image_paths)
    print(matrix)
    return matrix

def main(args):
    threshold = 0.7
    gradcam_names = ["grad++_cam", "eigen_cam"]
    model_paths = [
        "checkpoints/dataset3.densenet121.30497.pkl.best_auroc",
        "checkpoints/dataset3.densenet121.30496.pkl.best_auroc",
        "checkpoints/dataset3.densenet121.30495.pkl.best_auroc",
        "checkpoints/dataset3.densenet121.30494.pkl.best_auroc",
        "checkpoints/dataset3.densenet121.30493.pkl.best_auroc"
    ]

    image_names = [
        "sub-S03066_ses-E07113_run-1_bp-chest_vp-pa_dx.pt",
        "sub-S03082_ses-E07936_run-1_bp-chest_vp-ap_dx.pt",
        "sub-S03349_ses-E06615_run-1_bp-chest_vp-pa_dx.pt",
        "sub-S04179_ses-E08410_run-1_bp-chest_vp-pa_cr.pt",
        "sub-S04401_ses-E08746_run-1_bp-chest_vp-pa_cr.pt"
    ]
    image_paths = [
        f"data/tensors/bimcv+/{name}" for name in image_names
    ]
    
    matrix = create_miou_matrix(threshold, gradcam_names, model_paths, image_paths)
    
    plot = sn.heatmap(matrix)
    fig = plot.get_figure()
    fig.savefig("miou_matrix.png") 

if __name__ == "__main__":
    # parser = argparse.ArgumentParser(
    #     description='Script for generating miou matrix of localizations between models'
    # )
    # parser.add_argument('--config_path', dest='config_path', type=str, required=True,
    #                     help='Path to script config')
    # parser.add_argument('--output_path', dest='output_path', type=str, required=True,
    #                     help='Output path for generated image')
    # args = parser.parse_args()
    main(None)
