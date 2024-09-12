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
import matplotlib.pyplot as plt

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
from utils import load_model, get_preprocessing
from torchmetrics import AUROC

BATCH_SIZE = 8

def model_name_from_path(model_path):
    return model_path

def model_augment_auroc(model, augments, dataset):
    dataloader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
    )
    auroc = AUROC('binary')

    for x, y, _, _ in dataloader:
        x = augments(x).cuda()
        y = y.cuda().float()
        y_hat = model(x)

        auroc.update(y_hat[:,-1], y.to(torch.int)[:,-1])
    value = auroc.compute().item()
    auroc.reset()
    return value

def create_miou_matrix(augments_lst, model_paths, dataset):
    n_model = len(model_paths)
    n_augments = len(augments_lst)


    matrix = [[0]*n_augments for _ in range(n_model)]

    for i, path in tqdm(enumerate(model_paths)):
        for ii, aug in enumerate(augments_lst):
            matrix[i][ii] = model_augment_auroc(load_model(path), aug, dataset)
    return matrix

def create_heatmap(augments_lst, model_paths, dataset, save_path):
    augments = [get_preprocessing(aug) for aug in augments_lst]
    matrix = create_miou_matrix(augments, model_paths, dataset)
    
    plot = sn.heatmap(
        matrix, annot=True, vmin=0.5, vmax=1
    )
    plot.set_xticklabels(labels=augments_lst, rotation=45) 
    plot.set_yticklabels(labels=[path.name.split('.')[0] for path in model_paths], rotation=45)
    plt.subplots_adjust(left=0.25, bottom=0.25)
    fig = plot.get_figure()
    # fig.update_layout(
    #     margin=dict(l=50, r=50, t=50, b=50),
    # )
    fig.savefig(save_path) 
    sn.reset_defaults()
    plt.clf()

def auroc_augments(augments_lst, split_path, model_paths, stage):
    trainds = DomainConfoundedDataset(
            BIMCVNegativeDataset(fold='all', augments=None, labels='chestx-ray14'),
            BIMCVCOVIDDataset(fold='all', augments=None, labels='chestx-ray14')
            )
    valds = DomainConfoundedDataset(
            BIMCVNegativeDataset(fold='all', labels='chestx-ray14', augments=None),
            BIMCVCOVIDDataset(fold='all', labels='chestx-ray14', augments=None)
            )
    
    split_dir = f"splits/{split_path}"

    trainds.ds1.df = pd.read_csv(f"{split_dir}/negative-train.csv")
    trainds.len1 = len(trainds.ds1.df)

    valds.ds1.df = pd.read_csv(f"{split_dir}/negative-val.csv")
    valds.len1 = len(valds.ds1.df)

    trainds.ds2.df = pd.read_csv(f"{split_dir}/positive-train.csv")
    trainds.len2 = len(trainds.ds2.df)

    valds.ds2.df = pd.read_csv(f"{split_dir}/positive-val.csv")
    valds.len2 = len(valds.ds2.df)

    root_dir = "examples/augmentation_auroc/"
    save_dir = f"{root_dir}/{stage}.png"
    Path(root_dir).mkdir(parents=True, exist_ok=True)

    dataset = trainds if stage == 'train' else valds
    create_heatmap(augments_lst, model_paths, dataset, save_dir)

if __name__ == "__main__":
    preprocess_lst = [
        "none",
        "weak",
        "medium",
        "strong", 
        "strongXL", 
        "strongXL-rot", 
        "strongXXL", 
        "strongXXL-rot", 
        "cropXXXL",
        "cropXXXL-rot",
    ]

    split_path = "42/dataset3"
    model_paths = list(Path("checkpoints/auroc_comparison").rglob("*"))

    auroc_augments(preprocess_lst, split_path, model_paths, "val")