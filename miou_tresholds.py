#!/usr/bin/env python
#
# train_covid.py
#
# Run ``python train_covid.py -h'' for information on using this script.
#
import json
import pandas as pd
import torch
import seaborn as sn
from tqdm import tqdm
from pathlib import Path

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
    return matrix

def creat_heatmap(save_path, threshold, gradcam_names, model_paths, image_paths, legend=False):
    matrix = create_miou_matrix(threshold, gradcam_names, model_paths, image_paths)
    
    plot = sn.heatmap(matrix, annot=True, cbar=legend)
    fig = plot.get_figure()
    fig.savefig(save_path) 
    sn.reset_defaults()

def miou_tresholds(split_path, heatmap_name, threshold, gradcam_names, model_paths):
    # threshold = 0.7
    # gradcam_names = ["grad++_cam", "eigen_cam"]
    # heatmap_name = ""
    # model_paths = [
    #     "checkpoints/experiment_name.dataset3.densenet121-pretrain.42.pkl.best_auroc",
    # ]

    # split_path = "42/dataset3"
    rootdir = Path("splits", split_path)

    traindfs = {}
    valdfs = {}
    for path in rootdir.rglob("*"):
        image_names = pd.read_csv(path).path
        dataset_dir = "bimcv+" if path.stem.split('-')[0] == 'positive' else 'bimcv-'
        image_paths = [
            f"data/tensors/{dataset_dir}/{Path(name).stem}.pt" for name in image_names
        ]
        if path.stem.endswith("val"):
            valdfs[path.stem] = image_paths[:2]
        else:
            traindfs[path.stem] = image_paths[:2]

    save_dir = f"examples/heatmaps/{split_path}/{heatmap_name}"
    Path(save_dir).mkdir(parents=True, exist_ok=True)

    paths = []
    for curr in traindfs.values():
        paths += curr
    save_path = f"{save_dir}/combined-train.png"
    creat_heatmap(save_path, threshold, gradcam_names, model_paths, paths, True)

    paths = []
    for curr in valdfs.values():
        paths += curr
    save_path = f"{save_dir}/combined-val.png"
    creat_heatmap(save_path, threshold, gradcam_names, model_paths, paths)

    for name, paths in traindfs.items():
        save_path = f"{save_dir}/{name}.png"
        creat_heatmap(save_path, threshold, gradcam_names, model_paths, paths)
    for name, paths in valdfs.items():
        save_path = f"{save_dir}/{name}.png"
        creat_heatmap(save_path, threshold, gradcam_names, model_paths, paths)

if __name__ == "__main__":
    # parser = argparse.ArgumentParser(
    #     description='Script for generating miou matrix of localizations between models'
    # )
    # parser.add_argument('--config_path', dest='config_path', type=str, required=True,
    #                     help='Path to script config')
    # parser.add_argument('--output_path', dest='output_path', type=str, required=True,
    #                     help='Output path for generated image')
    # args = parser.parse_args()

    threshold = 0.7
    gradcam_names = ["grad++_cam", "eigen_cam"]
    heatmap_name = "name"
    model_paths = [
        "checkpoints/experiment_name.dataset3.densenet121-pretrain.42.pkl.best_auroc",
    ]

    split_path = "42/dataset3"
    miou_tresholds(split_path, heatmap_name, threshold, gradcam_names, model_paths)
