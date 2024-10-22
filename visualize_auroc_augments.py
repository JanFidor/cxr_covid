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
import os
import argparse

from models.cxrclassifier import AlexNet, CXRClassifier
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
from torchmetrics import AUROC, MeanMetric

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

def create_miou_matrix(augments_lst, group_paths, dataset):
    n_model = len(group_paths)
    n_augments = len(augments_lst)
    auroc_mean = MeanMetric()


    matrix = [[0]*n_augments for _ in range(n_model)]

    for i, group_path in tqdm(enumerate(group_paths)):
        model_paths = list(Path(group_path).rglob("*last"))
        for ii, aug in enumerate(augments_lst):
            auroc_mean.reset()
            for path in model_paths:
                auroc_mean.update(model_augment_auroc(load_model(path), aug, dataset))
            matrix[i][ii] = auroc_mean.compute().item()
    return matrix

def create_heatmap(preprocessing_names, group_paths, dataset, save_path):
    preprocessing = [get_preprocessing(pre) for pre in preprocessing_names]
    matrix = create_miou_matrix(preprocessing, group_paths, dataset)
    
    plot = sn.heatmap(
        matrix, annot=True, vmin=0.5, vmax=1
    )
    plot.set_xticklabels(labels=preprocessing_names, rotation=45) 
    plot.set_yticklabels(labels=[path.split('/')[-1] for path in group_paths], rotation='horizontal')
    plot.set_title('/'.join(save_path.split("/")[-3:]))
    plt.subplots_adjust(left=0.25, bottom=0.25)
    fig = plot.get_figure()
    # fig.update_layout(
    #     margin=dict(l=50, r=50, t=50, b=50),
    # )
    fig.savefig(save_path) 
    sn.reset_defaults()
    plt.clf()

def auroc_augments(split_path, group_paths, stage):
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

    batch, augments, prepro  = group_paths[0].split("/")[1:4]
    prepro = prepro.split("-")[0]
    preprocess_lst = [path.split("/")[3] for path in group_paths]

    save_path = f"{root_dir}/{batch}/{augments}/{prepro}/{stage}.png"
    (Path(root_dir) / batch / augments / prepro).mkdir(parents=True, exist_ok=True)

    dataset = trainds if stage == 'train' else valds
    create_heatmap(preprocess_lst, group_paths, dataset, save_path)

def name_fits_criteria(name, args):
    return args.pad_type in name and args.batch in name and \
        ((("color" not in name) and args.n_aug == 0) or (f"color-{args.n_aug}" in name and args.n_aug != 0))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--split_path', dest='split_path', type=str, default="42/dataset3", required=False)
    parser.add_argument('--pad_type', dest='pad_type', type=str, required=True)
    parser.add_argument('--n_aug', dest='n_aug', type=float, required=True)
    parser.add_argument('--batch', dest='batch', type=str, required=True)

    args = parser.parse_args()
    group_paths = list(sorted([
        x[0] for x in os.walk("checkpoints/") if name_fits_criteria(x[0], args)
    ]))
    print(group_paths)

    auroc_augments(args.split_path, group_paths, "val")
    auroc_augments(args.split_path, group_paths, "train")