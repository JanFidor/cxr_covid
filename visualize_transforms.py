#!/usr/bin/env python
#
# train_covid.py
#
# Run ``python train_covid.py -h'' for information on using this script.
#

import os
import sys

import argparse
import numpy as np
import pandas
from torch.utils.data import DataLoader
from torchvision.transforms import v2
from pathlib import Path

from PIL import Image
from datasets import (
    GitHubCOVIDDataset,
    BIMCVCOVIDDataset,
    ChestXray14Dataset,
    PadChestDataset,
    BIMCVNegativeDataset, 
    DomainConfoundedDataset
)
from tqdm import tqdm

SEED = 42

def get_train_augmentations(name):
    return {
        "weak": v2.Compose([
            v2.RandomResizedCrop(224, [0.9, 0.9]),
            v2.Lambda(lambda x: v2.functional.rotate(x, 5))
        ]),
        "medium": v2.Compose([
            v2.RandomResizedCrop(224, [0.85, 0.85]),
            v2.Lambda(lambda x: v2.functional.rotate(x, 10))
        ]),
        "strong": v2.Compose([
            v2.RandomResizedCrop(224, [0.75, 0.75]),
            v2.Lambda(lambda x: v2.functional.rotate(x, 10))
        ]),
        "center-weak": v2.Compose([
            v2.CenterCrop(int(224 * 0.9)),
            v2.Resize(224),
            v2.Lambda(lambda x: v2.functional.rotate(x, 5))
        ]),
        "center-strong": v2.Compose([
            v2.CenterCrop(int(224 * 0.75)),
            v2.Resize(224),
            v2.Lambda(lambda x: v2.functional.rotate(x, 10))
        ]),
        "center-strong-val": v2.Compose([
            v2.CenterCrop(int(224 * 0.75)),
            v2.Resize(224),
        ]),
    }.get(name, None)

def train_dataset_1(
    experiment_name,
    alexnet=False,
    freeze_features=False,
    train_augments=None,
    split_name=None
):
    trainds = DomainConfoundedDataset(
            ChestXray14Dataset(fold='train', augments=train_augments, labels='chestx-ray14', random_state=SEED),
            GitHubCOVIDDataset(fold='train', augments=train_augments, labels='chestx-ray14', random_state=SEED)
            )

    valds = DomainConfoundedDataset(
            ChestXray14Dataset(fold='val', labels='chestx-ray14', random_state=SEED),
            GitHubCOVIDDataset(fold='val', labels='chestx-ray14', random_state=SEED)
            )
    
    split_dir = f"splits/{split_name}/dataset1"

    trainds.ds1.df = pandas.read_csv(f"{split_dir}/chestxray-train.csv")
    trainds.ds1.meta_df = pandas.read_csv(f"{split_dir}/chestxray-trainmeta.csv")

    valds.ds1.df = pandas.read_csv(f"{split_dir}/chestxray-val.csv")
    valds.ds1.meta_df = pandas.read_csv(f"{split_dir}/chestxray-valmeta.csv")

    trainds.ds2.df = pandas.read_csv(f"{split_dir}/padchest-train.csv")
    valds.ds2.df = pandas.read_csv(f"{split_dir}/padchest-val.csv")

def train_dataset_2(
    experiment_name,
    alexnet=False,
    freeze_features=False,
    train_augments=None,
    split_name=None
):
    trainds = DomainConfoundedDataset(
            PadChestDataset(fold='train', augments=train_augments, labels='chestx-ray14', random_state=SEED),
            BIMCVCOVIDDataset(fold='train', augments=train_augments, labels='chestx-ray14', random_state=SEED)
            )
    valds = DomainConfoundedDataset(
            PadChestDataset(fold='val', labels='chestx-ray14', random_state=SEED),
            BIMCVCOVIDDataset(fold='val', labels='chestx-ray14', random_state=SEED)
            )
    
    split_dir = f"splits/{split_name}/dataset2"

    trainds.ds1.df = pandas.read_csv(f"{split_dir}/padchest-train.csv")
    valds.ds1.df = pandas.read_csv(f"{split_dir}/padchest-val.csv")

    trainds.ds2.df = pandas.read_csv(f"{split_dir}/bimcv-train.csv")
    valds.ds2.df = pandas.read_csv(f"{split_dir}/bimcv-val.csv")

    # generate log and checkpoint paths

def train_dataset_3(
    split_name,
    n_images=None,
    _augments=None,
):
    # Unlike the other datasets, there is overlap in patients between the
    # BIMCV-COVID-19+ and BIMCV-COVID-19- datasets, so we have to perform the 
    # train/val/test split *after* creating the datasets.

    # Start by getting the *full* dataset - not split!
    augments = get_train_augmentations(_augments)

    trainds = DomainConfoundedDataset(
            BIMCVNegativeDataset(fold='all', augments=augments, labels='chestx-ray14', random_state=SEED),
            BIMCVCOVIDDataset(fold='all', augments=augments, labels='chestx-ray14', random_state=SEED)
            )
    valds = DomainConfoundedDataset(
            BIMCVNegativeDataset(fold='all', labels='chestx-ray14', random_state=SEED),
            BIMCVCOVIDDataset(fold='all', labels='chestx-ray14', random_state=SEED)
            )
    
    split_dir = f"splits/{split_name}/dataset3"
    
    trainds.ds1.df = pandas.read_csv(f"{split_dir}/traindf1.csv").iloc[:n_images]
    valds.ds1.df = pandas.read_csv(f"{split_dir}/valdf1.csv").iloc[:n_images]

    trainds.ds2.df = pandas.read_csv(f"{split_dir}/traindf2.csv").iloc[:n_images]
    valds.ds2.df = pandas.read_csv(f"{split_dir}/valdf2.csv").iloc[:n_images]

    trainds.len1 = len(trainds.ds1)
    trainds.len2 = len(trainds.ds2)
    valds.len1 = len(valds.ds1)
    valds.len2 = len(valds.ds2)

    root_dir = Path("examples", "transformations", split_name, "ds3-val", _augments)
    visualize_split(valds.ds1, root_dir / "bimcv-")
    visualize_split(valds.ds2, root_dir / "bimcv+")

def visualize_split(dataset, root_dir):
    root_dir.mkdir(parents=True, exist_ok=True)
    dataloader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=1,
    )

    for i, data in enumerate(tqdm(dataloader)):
        image = data[0][0]
        path = root_dir / f"{i}.jpg"
        save_image(image, path)

mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]

def save_image(image, path):
    img = image.numpy().transpose((1, 2, 0))  # numpy is [h, w, c] 
    _mean = np.array(mean)  # mean of your dataset
    _std = np.array(std)  # std of your dataset
    img = _std * img + _mean
    img = np.clip(img, 0, 1) * 255
    Image.fromarray(img.astype(np.uint8)[:, :, 0], mode="L").save(path)


def main():
    parser = argparse.ArgumentParser(description='Training script for COVID-19 '
            'classifiers. Make sure that datasets have been set up before '
            'running this script. See the README file for more information.')
    parser.add_argument('--dataset', dest='dataset', type=int, default=3, required=False,
                        help='The dataset number on which to train. Choose "1" or "2" or "3".')
    parser.add_argument('--split', dest='split', type=str, default="42", required=False,
                        help='Split name')
    parser.add_argument('--n_images', dest='n_images', type=int, default=50, required=False,
                        help='N images to log for each split')
    parser.add_argument('--augments', dest='augments', type=str, default="center-strong-val", required=False,
                        help='Augment strength')

    args = parser.parse_args()

    if args.dataset == 1:
        train_dataset_1(
            args.split,
            args.n_images,
            args.augments
        )
    if args.dataset == 2:
        train_dataset_2(
            args.split,
            args.n_images,
            args.augments
        )
    if args.dataset == 3:
        train_dataset_3(
            args.split,
            args.n_images,
            args.augments
        )

if __name__ == "__main__":
    main()
