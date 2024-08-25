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
from torchvision.transforms import v2
from torch.utils.data import DataLoader
from datasets import (
    GitHubCOVIDDataset,
    BIMCVCOVIDDataset,
    ChestXray14Dataset,
    PadChestDataset,
    BIMCVNegativeDataset, 
    DomainConfoundedDataset
)

BATCH_SIZE = 1

def load_model(model_path):
    cpt = torch.load(model_path, weights_only=False)
    return cpt["model"]

def get_gradcam_layers(model):
    if isinstance(model, AlexNet):
        return [model.features[-3]]
    elif isinstance(model, DenseNet):
        return [model.features[-2].denselayer16.conv2]
    raise KeyError

def get_preprocessing(name):
    return {
        "weak": v2.Compose([
            v2.CenterCrop(int(224 * 0.95)),
            v2.Resize(224),
        ]),
        "medium": v2.Compose([
            v2.CenterCrop(int(224 * 0.85)),
            v2.Resize(224),
        ]),
        "strong": v2.Compose([
            v2.CenterCrop(int(224 * 0.75)),
            v2.Resize(224),
        ])
    }.get(name, v2.Identity())

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

def generate_localisation(threshold, gradcam, tensors):
    saliency_map = gradcam(
        input_tensor=tensors,
        targets=[ClassifierOutputTarget(1)]
    )

    return torch.tensor(saliency_map > threshold)

def calculate_saliency_jaccard(threshold, gradcams1, gradcams2, dataloader):
    jaccard = JaccardIndex(task='multiclass', num_classes=2)
    mean_jaccard = MeanMetric()
    for tensors, _, _, _ in dataloader:
        for cam1, cam2 in zip(gradcams1, gradcams2):
            att1 = generate_localisation(threshold, cam1, tensors)
            att2 = generate_localisation(threshold, cam2, tensors)

            iou = jaccard(att1, att2)
            mean_jaccard.update(iou)
    return mean_jaccard.compute().item()

def create_miou_matrix(threshold, gradcam_names, model_paths, dataset):
    dataloader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
    )
    all_gradcams = [
        get_gradcams(gradcam_names, path) for path in model_paths
    ]

    d = len(model_paths)
    matrix = [[0]*d for _ in range(d)]

    for i in tqdm(range(d)):
        for ii in range(d):
            cam1 = all_gradcams[i]
            cam2 = all_gradcams[ii]
            matrix[i][ii] = calculate_saliency_jaccard(threshold, cam1, cam2, dataloader)
    return matrix

def create_heatmap(save_path, threshold, gradcam_names, model_paths, dataloader, legend=False):
    matrix = create_miou_matrix(threshold, gradcam_names, model_paths, dataloader)
    
    plot = sn.heatmap(matrix, annot=True, cbar=legend)
    fig = plot.get_figure()
    fig.savefig(save_path) 
    sn.reset_defaults()

def miou_tresholds(preprocess, split_path, heatmap_name, threshold, gradcam_names, model_paths):
    # threshold = 0.7
    # gradcam_names = ["grad++_cam", "eigen_cam"]
    # heatmap_name = ""
    # model_paths = [
    #     "checkpoints/experiment_name.dataset3.densenet121-pretrain.42.pkl.best_auroc",
    # ]

    # split_path = "42/dataset3"

    augments = get_preprocessing(preprocess)

    trainds = DomainConfoundedDataset(
            BIMCVNegativeDataset(fold='all', augments=augments, labels='chestx-ray14'),
            BIMCVCOVIDDataset(fold='all', augments=augments, labels='chestx-ray14')
            )
    valds = DomainConfoundedDataset(
            BIMCVNegativeDataset(fold='all', labels='chestx-ray14', augments=augments),
            BIMCVCOVIDDataset(fold='all', labels='chestx-ray14', augments=augments)
            )
    
    split_dir = f"splits/{split_path}"
    trainds.ds1.df = pd.read_csv(f"{split_dir}/negative-train.csv")
    valds.ds1.df = pd.read_csv(f"{split_dir}/negative-val.csv")

    trainds.ds2.df = pd.read_csv(f"{split_dir}/positive-train.csv")
    valds.ds2.df = pd.read_csv(f"{split_dir}/positive-val.csv")

    save_dir = f"examples/heatmaps/{split_path}/{heatmap_name}"
    Path(save_dir).mkdir(parents=True, exist_ok=True)

    # create_heatmap(f"{save_dir}/negative-train", threshold, gradcam_names, model_paths, trainds.ds1, True)
    # create_heatmap(f"{save_dir}/positive-train", threshold, gradcam_names, model_paths, trainds.ds2, True)
    create_heatmap(f"{save_dir}/negative-val", threshold, gradcam_names, model_paths, valds.ds1, True)
    create_heatmap(f"{save_dir}/positive-val", threshold, gradcam_names, model_paths, valds.ds2, True)

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
    miou_tresholds("strong", split_path, heatmap_name, threshold, gradcam_names, model_paths)
