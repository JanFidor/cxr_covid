#!/usr/bin/env python3
import argparse
import itertools
import json
import os
from pathlib import Path

import numpy as np
import torch
import pandas as pd
from tqdm import tqdm
from torchmetrics import AUROC, Accuracy, Precision, Recall, F1Score
from torchvision.transforms import v2

from datasets import (
    GitHubCOVIDDataset,
    BIMCVCOVIDDataset,
    ChestXray14Dataset,
    PadChestDataset,
    BIMCVNegativeDataset,
    DomainConfoundedDataset
)
from utils import get_augmentations, get_preprocessing, load_model

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", type=str, default="densenet121-pretrain",
                       help="Model architecture name", required=False)
    parser.add_argument("--dataset", type=int, required=True, 
                       choices=[1, 2, 3],
                       help="Dataset to evaluate on (1, 2, or 3)")
    parser.add_argument('--pad_type', dest='pad_type', type=str, required=True)
    parser.add_argument('--n_aug', dest='n_aug', type=float, required=True)
    parser.add_argument('--batch', dest='batch', type=str, required=True)
    parser.add_argument("--output-path", type=str, default="simple_graph.json",
                       help="Path to save results JSON")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--split", type=str, default=None, help="Split name for loading specific dataset splits")
    return parser.parse_args()

def evaluate_configuration(model, dataset, batch_size):
    """Evaluate model performance for a specific configuration"""
    model.eval()
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=1
    )
    
    # Initialize metrics
    auroc = AUROC(task='binary').cuda()
    accuracy = Accuracy(task='binary').cuda()
    precision = Precision(task='binary').cuda()
    recall = Recall(task='binary').cuda()
    f1 = F1Score(task='binary').cuda()
    
    with torch.no_grad():
        for batch in tqdm(dataloader, leave=False):
            inputs, labels, _, _ = batch
            inputs = inputs.cuda()
            labels = labels.cuda()
            outputs = model(inputs)
            
            # Only keep COVID predictions (last column)
            covid_outputs = outputs[:, -1]
            covid_labels = labels[:, -1]
            
            # Update metrics
            auroc.update(covid_outputs, covid_labels.int())
            accuracy.update(covid_outputs, covid_labels.int())
            precision.update(covid_outputs, covid_labels.int())
            recall.update(covid_outputs, covid_labels.int())
            f1.update(covid_outputs, covid_labels.int())
    
    return {
        "auroc": float(auroc.compute().cpu()),
        "accuracy": float(accuracy.compute().cpu()),
        "precision": float(precision.compute().cpu()),
        "recall": float(recall.compute().cpu()),
        "f1": float(f1.compute().cpu())
    }

def get_dataset(dataset_num, transforms, seed, split_name='42'):
    """Get the appropriate test dataset based on dataset number"""
    if dataset_num == 1:
        ds = DomainConfoundedDataset(
            ChestXray14Dataset(fold='val', augments=transforms, labels='chestx-ray14', random_state=seed),
            GitHubCOVIDDataset(fold='val', augments=transforms, labels='chestx-ray14', random_state=seed)
        )

        split_dir = f"splits/{split_name}/dataset1"
        ds.ds1.df = pd.read_csv(f"{split_dir}/chestxray-val.csv", index_col=0)
        ds.ds1.meta_df = pd.read_csv(f"{split_dir}/chestxray-valmeta.csv", index_col=0)
        ds.ds2.df = pd.read_csv(f"{split_dir}/githubcovid-val.csv", index_col="filename")
        
    elif dataset_num == 2:
        ds = DomainConfoundedDataset(
            PadChestDataset(fold='val', augments=transforms, labels='chestx-ray14', random_state=seed),
            BIMCVCOVIDDataset(fold='val', augments=transforms, labels='chestx-ray14', random_state=seed, is_old=False)
        )
        split_dir = f"splits/{split_name}/dataset2"
        ds.ds1.df = pd.read_csv(f"{split_dir}/padchest-val.csv")
        ds.ds2.df = pd.read_csv(f"{split_dir}/positive-val.csv")
        
    elif dataset_num == 3:
        ds = DomainConfoundedDataset(
            BIMCVNegativeDataset(fold='val', augments=transforms, labels='chestx-ray14', random_state=seed),
            BIMCVCOVIDDataset(fold='val', augments=transforms, labels='chestx-ray14', random_state=seed)
        )
        split_dir = f"splits/{split_name}/dataset3"
        ds.ds1.df = pd.read_csv(f"{split_dir}/negative-val.csv")
        ds.ds2.df = pd.read_csv(f"{split_dir}/positive-val.csv")
    else:
        raise ValueError(f"Invalid dataset number: {dataset_num}")
    
    # Set lengths for DomainConfoundedDataset
    ds.len1 = len(ds.ds1)
    ds.len2 = len(ds.ds2)
    
    return ds

def name_fits_criteria(name, args):
    return args.pad_type in name and args.batch in name and \
        ((("color" not in name) and args.n_aug == 0) or (f"color-{args.n_aug}" in name and args.n_aug != 0))

def main():
    args = parse_args()
    
    # Set random seeds
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # Load model
    
    
    results = {}
    
    # Generate all combinations of parameters
    group_paths = list(sorted([
        x[0] for x in os.walk("checkpoints/") if name_fits_criteria(x[0], args)
    ]))
    # Create transforms
    preprocess = get_preprocessing(args.pad_type)
    
    # Get appropriate test dataset
    dataset = get_dataset(args.dataset, preprocess, args.seed, args.split)
    metric_lst = []
    for path in group_paths:
        for model_path in os.listdir(path):
            model = load_model(f"{path}/{model_path}", args.model_name)
            model.cuda()
            metric_lst.append(evaluate_configuration(model, dataset, 16))
    
    # Store results
    config_key = f"aug={args.n_aug}|prep={args.pad_type}|batch={args.batch}"
    results[config_key] = metric_lst
    
    # Save results
    output_path = Path("jsons", str(args.dataset), str(args.n_aug), args.pad_type + "_" + args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to {output_path}")

if __name__ == "__main__":
    main()