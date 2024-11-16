#!/usr/bin/env python
#
# train_covid.py
#
# Run ``python train_covid.py -h'' for information on using this script.
#

import os
import sys
from pathlib import Path

import argparse

import pandas
import sklearn.metrics
from torchvision.transforms import v2

# seeding
import torch
import random
import numpy as np

from models import CXRClassifier
from datasets import (
    GitHubCOVIDDataset,
    BIMCVCOVIDDataset,
    ChestXray14Dataset,
    PadChestDataset,
    BIMCVNegativeDataset, 
    DomainConfoundedDataset
)
from load_data import load_dataset_1, load_dataset_2, load_dataset_3
from logger import initialize_wandb
from utils import get_augmentations, get_preprocessing
import wandb
from torchmetrics import AUROC
from tqdm import tqdm

MAX_BATCH=16

def train_dataset_1(
    experiment_name,
    seed,
    model_name,
    freeze_features=False,
    augments_name=None,
    preprocessing=None,
    split_name=None
):
    trainds = load_dataset_1(seed, is_train=True, augments_name=augments_name, preprocessing=preprocessing, split_name=split_name)
    valds = load_dataset_1(seed, is_train=False, augments_name=augments_name, preprocessing=preprocessing, split_name=split_name)

    # generate log and checkpoint paths
    logpath = f'logs/{experiment_name}.dataset1.{model_name}.{seed}.log'
    checkpointpath = f'checkpoints/{experiment_name}.dataset1.{model_name}.{seed}.pkl'

    classifier = CXRClassifier()
    classifier.train(
        trainds,
        valds,
        max_epochs=30,
        lr=0.01, 
        weight_decay=1e-4,
        logpath=logpath,
        checkpoint_path=checkpointpath,
        verbose=True,
        model_name=model_name,
        freeze_features=freeze_features,
        batch_size=8
    )
    wandb.save(checkpointpath)
    wandb.save(f"{checkpointpath}.best_auroc")
    wandb.save(f"{checkpointpath}.best_loss")

def train_dataset_2(
    experiment_name,
    seed,
    model_name,
    freeze_features=False,
    augments_name=None,
    preprocessing=None,
    split_name=None
):
    trainds = load_dataset_2(seed, is_train=True, augments_name=augments_name, preprocessing=preprocessing, split_name=split_name)
    valds = load_dataset_2(seed, is_train=False, augments_name=augments_name, preprocessing=preprocessing, split_name=split_name)

    # generate log and checkpoint paths
    logpath = f'logs/{experiment_name}.dataset2.{model_name}.{seed}.log'
    checkpointpath = f'checkpoints/{experiment_name}.dataset2.{model_name}.{seed}.pkl'

    classifier = CXRClassifier()
    classifier.train(
        trainds,
        valds,
        max_epochs=30,
        lr=0.01, 
        weight_decay=1e-4,
        logpath=logpath,
        checkpoint_path=checkpointpath,
        verbose=True,
        model_name=model_name,
        freeze_features=freeze_features,
        batch_size=8
    )
    wandb.save(checkpointpath)
    wandb.save(f"{checkpointpath}.best_auroc")
    wandb.save(f"{checkpointpath}.best_loss")

def train_dataset_3(
    experiment_name,
    seed,
    model_name,
    batch_size,
    freeze_features=False,
    preprocessing=None,
    augments_name=None,
    split_name=None,
    group_name=None,
    lr=0.01,
    weight_decay=1e-4,
    max_epochs=30,
):
    trainds = load_dataset_3(seed, is_train=True, augments_name=augments_name, preprocessing=preprocessing, split_name=split_name)
    valds = load_dataset_3(seed, is_train=False, augments_name=augments_name, preprocessing=preprocessing, split_name=split_name)

    # generate log and checkpoint paths
    logpath = f'logs/{experiment_name}.dataset3.{model_name}.{seed}.log'
    checkpointdir = f"checkpoints/{group_name or 'ungrouped'}/"

    Path(checkpointdir).mkdir(parents=True, exist_ok=True)
    checkpointpath = f"{checkpointdir}/{experiment_name}-{seed}.pkl"

    classifier = CXRClassifier(seed=seed)
    classifier.train(
        trainds,
        valds,
        max_epochs=max_epochs,
        lr=lr, 
        batch_size=batch_size,
        weight_decay=weight_decay,
        logpath=logpath,
        checkpoint_path=checkpointpath,
        verbose=True,
        model_name=model_name,
        freeze_features=freeze_features,
    )

    evaluate_dataset_1(seed, classifier.model, preprocessing, split_name, epoch=max_epochs)

    wandb.save(f"{checkpointpath}*", base_path=checkpointdir)

def evaluate_dataset_1(
    seed,
    model,
    preprocessing=None,
    split_name=None,
    epoch=None
):  
    model.eval()
    ds1 = load_dataset_1(seed, is_train=True, preprocessing=preprocessing, split_name=split_name)
    dl1 = torch.utils.data.DataLoader(
        ds1,
        batch_size=MAX_BATCH,
        shuffle=False,
        num_workers=1
    )

    ds2 = load_dataset_1(seed, is_train=False, preprocessing=preprocessing, split_name=split_name)
    dl2 = torch.utils.data.DataLoader(
        ds2,
        batch_size=MAX_BATCH,
        shuffle=False,
        num_workers=1
    )
    
    # Initialize metrics
    auroc = AUROC(task='binary').cuda()
    
    with torch.no_grad():
        # for batch in tqdm(dl1, leave=False):
        #     inputs, labels, _, _ = batch
        #     inputs = inputs.cuda()
        #     labels = labels.cuda()
        #     outputs = model(inputs)
            
        #     # Only keep COVID predictions (last column)
        #     covid_outputs = outputs[:, -1]
        #     covid_labels = labels[:, -1]
            
        #     # Update metrics
        #     auroc.update(covid_outputs, covid_labels.int())
        for batch in tqdm(dl2, leave=False):
            inputs, labels, _, _ = batch
            inputs = inputs.cuda()
            labels = labels.cuda()
            outputs = model(inputs)
            
            # Only keep COVID predictions (last column)
            covid_outputs = outputs[:, -1]
            covid_labels = labels[:, -1]
            
            # Update metrics
            auroc.update(covid_outputs, covid_labels.int())
    _auroc = float(auroc.compute().cpu())
    log_dict = {f"auroc/test_ood": _auroc}
    if epoch is not None:
        log_dict[f"epoch"] = epoch
    wandb.log(log_dict)

def main():
    parser = argparse.ArgumentParser(description='Training script for COVID-19 '
            'classifiers. Make sure that datasets have been set up before '
            'running this script. See the README file for more information.')
    parser.add_argument('--dataset', dest='dataset', type=int, default=1, required=False,
                        help='The dataset number on which to train. Choose "1" or "2" or "3".')
    parser.add_argument('--seed', dest='seed', type=int, default=42, required=False,
                        help='The random seed used to generate train/val/test splits')
    parser.add_argument('--network', dest='network', type=str, default='densenet121-pretrain', required=False,
                        help='The network type. Choose "densenet121-random", "densenet121-pretrain", "logistic", or "alexnet".')
    parser.add_argument('--split', dest='split', type=str, default="42", required=False,
                        help='Split name')
    parser.add_argument('--batch', dest='batch', type=int, default=8, required=False,
                        help='Experiment name')   
    parser.add_argument('--device-index', dest='deviceidx', type=int, default=None, required=False,
                        help='The index of the GPU device to use. If None, use the default GPU.')
    parser.add_argument('--augments', dest='augments', type=str, default="none", required=False,
                        help='Augment strength')
    parser.add_argument('--preprocessing', dest='preprocessing', type=str, default="none", required=False,
                        help='Augment strength')
    parser.add_argument('--experiment', dest='experiment', type=str, default='experiment_name', required=False,
                        help='Experiment name')
    parser.add_argument('--group', dest='group', type=str, default=None, required=False,
                        help='Experiment name')
    parser.add_argument('--lr', dest='lr', type=float, default=0.01, required=False,
                        help='Learning rate')
    parser.add_argument('--weight-decay', dest='weight_decay', type=float, default=1e-4, required=False,
                        help='Weight decay')
    parser.add_argument('--max-epochs', dest='max_epochs', type=int, default=30, required=False)
    args = parser.parse_args()

    for dirname in ['checkpoints', 'logs']:
        if not os.path.isdir(dirname):
            os.mkdir(dirname)

    if args.deviceidx is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = "{:d}".format(args.deviceidx)

    initialize_wandb(args.experiment, args.group, "cxr_covid", vars(args))

    # Set seeds
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)


    if args.dataset == 1:
        train_dataset_1(
            args.experiment,
            args.seed, 
            model_name=args.network, 
            freeze_features=(args.network.lower() == 'logistic'),
            augments_name=args.augments,
            preprocessing=args.preprocessing,
            split_name=args.split
        )
    if args.dataset == 2:
        train_dataset_2(
            args.experiment,
            args.seed, 
            model_name=args.network, 
            freeze_features=(args.network.lower() == 'logistic'),
            augments_name=args.augments,
            preprocessing=args.preprocessing,
            split_name=args.split
        )
    if args.dataset == 3:
        train_dataset_3(
            args.experiment,
            args.seed, 
            model_name=args.network, 
            freeze_features=(args.network.lower() == 'logistic'),
            augments_name=args.augments,
            preprocessing=args.preprocessing,
            split_name=args.split,
            group_name=args.group,
            batch_size=args.batch,
            lr=args.lr,
            weight_decay=args.weight_decay,
            max_epochs=args.max_epochs,
        )

if __name__ == "__main__":
    main()
