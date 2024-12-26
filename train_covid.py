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
from models.cxrclassifier import log_confusion_matrix
from load_data import load_dataset_1, load_dataset_2, load_dataset_3
from logger import initialize_wandb
from utils import get_augmentations, get_preprocessing
import wandb
from torchmetrics import AUROC, Precision, Recall, F1Score
from tqdm import tqdm
from torchmetrics.classification.confusion_matrix import ConfusionMatrix
from utils import denormalize
import torchxrayvision as xrv

MAX_BATCH=256

def train_dataset_1(
    experiment_name,
    seed,
    model_name,
    freeze_features=False,
    augments_name=None,
    preprocessing=None,
    split_name=None
):
    trainds = load_dataset_1(seed, fold='train', augments_name=augments_name, preprocessing=preprocessing, split_name=split_name)
    valds = load_dataset_1(seed, fold='val', augments_name=augments_name, preprocessing=preprocessing, split_name=split_name)

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
    batch_size,
    freeze_features=False,
    preprocessing=None,
    augments_name=None,
    split_name=None,
    group_name=None,
    lr=0.01,
    weight_decay=1e-4,
    max_epochs=30,
    flipped=0,
    is_inverted=False,
    is_binary=False
):
    trainds = load_dataset_2(seed, is_train=True, augments_name=augments_name, preprocessing=preprocessing, split_name=split_name)
    valds = load_dataset_2(seed, is_train=False, augments_name=augments_name, preprocessing=preprocessing, split_name=split_name)

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

    wandb.save(f"{checkpointpath}*", base_path=checkpointdir)

    evaluate_dataset_1(seed, classifier.model, preprocessing, split_name, max_epochs, is_best=False, is_cut=classifier.is_cut)
    evaluate_dataset_1(seed, classifier.model, preprocessing, None, max_epochs, is_best=False, is_cut=classifier.is_cut, )
    bestpath = f"{checkpointpath}.best_auroc"
    classifier.load_checkpoint(bestpath)
    evaluate_dataset_1(seed, classifier.model, preprocessing, split_name, max_epochs, is_best=True, is_cut=classifier.is_cut)
    evaluate_dataset_1(seed, classifier.model, preprocessing, None, max_epochs, is_best=True, is_cut=classifier.is_cut)
    

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
    flipped=0,
    is_inverted=False,
    is_binary=False
):
    msks = None

    trainds = load_dataset_3(seed, is_train=True, augments_name=augments_name, preprocessing=preprocessing, split_name=split_name, flipped=flipped, masks=msks, is_inverted=is_inverted, is_binary=is_binary)
    valds = load_dataset_3(seed, is_train=False, augments_name=augments_name, preprocessing=preprocessing, split_name=split_name, masks=msks, is_inverted=is_inverted, is_binary=is_binary)

    # generate log and checkpoint paths
    logpath = f'logs/{experiment_name}.dataset3.{model_name}.{seed}.log'
    checkpointdir = f"checkpoints/{group_name or 'ungrouped'}/"

    Path(checkpointdir).mkdir(parents=True, exist_ok=True)
    checkpointpath = f"{checkpointdir}/{experiment_name}-{seed}.pkl"

    classifier = CXRClassifier(0, seed=seed)
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

    wandb.save(f"{checkpointpath}*", base_path=checkpointdir)

    evaluate_dataset_1(seed, classifier.model, preprocessing, split_name, max_epochs, is_best=False, is_cut=classifier.is_cut)
    evaluate_dataset_1(seed, classifier.model, preprocessing, None, max_epochs, is_best=False, is_cut=classifier.is_cut, )
    bestpath = f"{checkpointpath}.best_auroc"
    classifier.load_checkpoint(bestpath)
    evaluate_dataset_1(seed, classifier.model, preprocessing, split_name, max_epochs, is_best=True, is_cut=classifier.is_cut)
    evaluate_dataset_1(seed, classifier.model, preprocessing, None, max_epochs, is_best=True, is_cut=classifier.is_cut)
    

def evaluate_dataset_1(
    seed,
    model,
    preprocessing=None,
    split_name=None,
    epoch=None,
    is_best=False,
    is_cut=False
):  
    model.eval()
    ds = load_dataset_1(seed, fold='test', preprocessing=preprocessing, split_name=split_name)
    dl = torch.utils.data.DataLoader(
        ds,
        batch_size=MAX_BATCH,
        shuffle=False,
        num_workers=1
    )

    aggregated_preds = []
    aggregated_labels = []
    
    # Initialize metrics
    auroc = AUROC('binary').cpu()
    precision = Precision(task='binary').cpu()
    recall = Recall(task='binary').cpu()
    f1 = F1Score(task='binary').cpu()
    confmat = ConfusionMatrix(task="binary", num_classes=2).cpu()
    
    with torch.no_grad():
        for batch in tqdm(dl, leave=False):
            inputs, labels, _, _ = batch
            if is_cut:
                inputs = denormalize(inputs)
                inputs = inputs[:, :1, :, :]
                mn, mx = inputs.amin(dim=[1, 2, 3]), inputs.amax(dim=[1, 2, 3])
                inputs = (inputs.permute(1, 2, 3, 0) - mn) / (mx - mn)
                inputs = (2 * inputs - 1.) * 1024
                inputs = inputs.permute(3, 0, 1, 2)
            inputs = inputs.cuda()
            labels = labels.cuda()
            outputs = model(inputs)
            
            # Only keep COVID predictions (last column)
            covid_outputs = outputs[:, -1].detach().cpu()
            covid_labels = labels[:, -1].detach().cpu().int()
            probs = torch.nn.functional.sigmoid(covid_outputs)
            predictions = (covid_outputs > 0).int()

            aggregated_preds.extend(predictions.numpy())
            aggregated_labels.extend(covid_labels.numpy())
            
            # Update metrics
            auroc.update(probs, covid_labels)
            precision.update(predictions, covid_labels)
            recall.update(predictions, covid_labels)
            f1.update(predictions, covid_labels)
            confmat.update(predictions, covid_labels)
            
    _auroc = float(auroc.compute().cpu())
    _precision = float(precision.compute().cpu())
    _recall = float(recall.compute().cpu())
    _f1 = float(f1.compute().cpu())
    model_name=f"test_{'full' if split_name is None else 'balanced'}/{'best_auroc' if is_best else 'last'}"
    log_dict = {
        f"auroc/{model_name}": _auroc, f"precision-binary/{model_name}": _precision, f"recall-binary/{model_name}": _recall, f"f1-binary/{model_name}": _f1}
    if epoch is not None:
        log_dict[f"epoch"] = epoch

    for metric_name, metric in metrics_weighted().items():
        log_dict[f"{metric_name}/{model_name}"] = metric(aggregated_labels, aggregated_preds)
    wandb.log(log_dict)
    
    # Log confusion matrix
    log_confusion_matrix(model_name, epoch, confmat.compute().cpu().numpy())

def metrics_weighted():
    return {
    "f1-weighted": lambda y_true, y_pred: sklearn.metrics.f1_score(y_true, y_pred, average='weighted'),
    "precision-weighted": lambda y_true, y_pred: sklearn.metrics.precision_score(y_true, y_pred, average='weighted'),
    "recall-weighted": lambda y_true, y_pred: sklearn.metrics.recall_score(y_true, y_pred, average='weighted'),
    "accuracy": sklearn.metrics.accuracy_score,
    "accuracy_balanced": sklearn.metrics.balanced_accuracy_score,
}

def main():
    parser = argparse.ArgumentParser(description='Training script for COVID-19 '
            'classifiers. Make sure that datasets have been set up before '
            'running this script. See the README file for more information.')
    parser.add_argument('--dataset', dest='dataset', type=int, default=3, required=False,
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
    parser.add_argument('--max-epochs', dest='max_epochs', type=int, default=1, required=False)
    parser.add_argument('--flipped', dest='flipped', type=float, default=0, required=False)
    parser.add_argument('--freeze', dest='freeze', type=int, default=0, required=False,
                        help='Freeze network parameters (1 to freeze, 0 to not freeze)')
    parser.add_argument('--inverted', dest='inverted', type=int, default=0, required=False)
    parser.add_argument('--binary', dest='binary', type=int, default=0, required=False)
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
            freeze_features=args.freeze==1,
            augments_name=args.augments,
            preprocessing=args.preprocessing,
            split_name=args.split
        )
    if args.dataset == 2:
        train_dataset_2(
            args.experiment,
            args.seed, 
            model_name=args.network, 
            freeze_features=args.freeze==1,
            augments_name=args.augments,
            preprocessing=args.preprocessing,
            split_name=args.split,
            group_name=args.group,
            batch_size=args.batch,
            lr=args.lr,
            weight_decay=args.weight_decay,
            max_epochs=args.max_epochs,
            flipped=args.flipped,
            is_inverted=args.inverted==1,
            is_binary=args.binary==1
        )
    if args.dataset == 3:
        train_dataset_3(
            args.experiment,
            args.seed, 
            model_name=args.network, 
            freeze_features=args.freeze==1,
            augments_name=args.augments,
            preprocessing=args.preprocessing,
            split_name=args.split,
            group_name=args.group,
            batch_size=args.batch,
            lr=args.lr,
            weight_decay=args.weight_decay,
            max_epochs=args.max_epochs,
            flipped=args.flipped,
            is_inverted=args.inverted==1,
            is_binary=args.binary==1
        )

if __name__ == "__main__":
    main()
