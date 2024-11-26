#

import os
import sys
from pathlib import Path

import argparse

import pandas as pd
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
from logger import initialize_wandb
from utils import get_augmentations, get_preprocessing
import wandb


def load_overlap(path="data/bimcv-/listjoin_ok.tsv"):
    neg_overlap_map = {}
    pos_overlap_map = {}
    # with open(path, 'r') as handle:
    #     handle.readline()
    #     for line in handle:
    #         idx, neg_id, pos_id = line.split()
    #         neg_overlap_map[neg_id] = idx
    #         pos_overlap_map[pos_id] = idx
    return neg_overlap_map, pos_overlap_map

def ds3_grouped_split(df1, df2, random_state=None, test_size=0.05):
    '''
    Split a dataframe such that patients are disjoint in the resulting folds.
    The dataframe must have an index that contains strings that may be processed
    by ds3_get_patient_id to return the unique patient identifiers.
    '''
    neg_overlap_map, pos_overlap_map = load_overlap()
    groups = ds3_get_unique_patient_ids(df1, df2, neg_overlap_map, pos_overlap_map)
    traingroups, testgroups = sklearn.model_selection.train_test_split(
            groups,
            random_state=random_state,
            test_size=test_size)
    traingroups = set(traingroups)
    testgroups = set(testgroups)

    traindict1 = {}
    testdict1 = {}
    traindict2 = {}
    testdict2 = {}
    for idx, row in df1.iterrows():
        patient_id = ds3_get_patient_id(df1, idx, neg_overlap_map)
        if patient_id in traingroups:
            traindict1[idx] = row
        elif patient_id in testgroups:
            testdict1[idx] = row
    for idx, row in df2.iterrows():
        patient_id = ds3_get_patient_id(df2, idx, pos_overlap_map)
        if patient_id in traingroups:
            traindict2[idx] = row
        elif patient_id in testgroups:
            testdict2[idx] = row
    traindf1 = pd.DataFrame.from_dict(
        traindict1, 
        orient='index',
        columns=df1.columns)
    testdf1 = pd.DataFrame.from_dict(
        testdict1, 
        orient='index',
        columns=df1.columns)
    traindf2 = pd.DataFrame.from_dict(
        traindict2, 
        orient='index',
        columns=df2.columns)
    testdf2 = pd.DataFrame.from_dict(
        testdict2, 
        orient='index',
        columns=df2.columns)
    return traindf1, testdf1, traindf2, testdf2

def ds3_get_patient_id(df, idx, jointlist):
    participant_id = df['participant'].loc[idx]
    try:
        val = jointlist[participant_id]
        print(val)
        return val
    except KeyError:
        return participant_id

def ds3_get_unique_patient_ids(df1, df2, neg_overlap_map, pos_overlap_map):
    # check that ids don't overlap to start
    if len(set(df1.participant).intersection(set(df2.participant))) != 0:
        print(df1.participant[:4])
        print(df2.participant[:4])
        #print(set(df1.participant).intersection(set(df2.participant)))
        raise ValueError
    neg_idxs = [ds3_get_patient_id(df1, idx, neg_overlap_map) for idx in df1.index]
    pos_idxs = [ds3_get_patient_id(df2, idx, pos_overlap_map) for idx in df2.index]
    neg_idxs = list(set(neg_idxs))
    pos_idxs = list(set(pos_idxs))
    neg_idxs.sort()
    pos_idxs.sort()
    return neg_idxs + pos_idxs


def load_dataset_1(
    seed,
    fold,
    augments_name='none',
    preprocessing='none',
    split_name=None
):
    augments = get_augmentations(augments_name)
    preprocessing = get_preprocessing(preprocessing)

    combined_transforms = v2.Compose([
        preprocessing, augments
    ])

    train_transforms = v2.Identity()
    if fold == 'train':
        train_transforms = combined_transforms
    elif fold == 'val':
        train_transforms = preprocessing

    ds = DomainConfoundedDataset(
        ChestXray14Dataset(fold='train', augments=train_transforms, labels='chestx-ray14', random_state=seed),
        GitHubCOVIDDataset(fold='train', augments=train_transforms, labels='chestx-ray14', random_state=seed)
        )
    
    split_dir = f"splits/{split_name}/dataset1"
    ds.ds1.df = pd.read_csv(f"{split_dir}/chestxray-{fold}.csv", index_col=0)
    ds.ds1.meta_df = pd.read_csv(f"{split_dir}/chestxray-{fold}meta.csv", index_col=0)
    ds.ds2.df = pd.read_csv(f"{split_dir}/githubcovid-{fold}.csv", index_col="filename")

    ds.len1 = len(ds.ds1)
    ds.len2 = len(ds.ds2)
    return ds

def load_dataset_2(
    seed,
    is_train,
    augments_name=None,
    preprocessing=None,
    split_name=None
):
    augments = get_augmentations(augments_name)
    preprocessing = get_preprocessing(preprocessing)

    train_transforms = v2.Compose([
        preprocessing, augments
    ]) if is_train else v2.Compose([preprocessing])

    fold = 'train' if is_train else 'val'
    ds = DomainConfoundedDataset(
        PadChestDataset(fold=fold, augments=train_transforms, labels='chestx-ray14', random_state=seed),
        BIMCVCOVIDDataset(fold=fold, augments=train_transforms, labels='chestx-ray14', random_state=seed, is_old=False)
    )
    
    split_dir = f"splits/{split_name}/dataset2"
    if split_name:
        ds.ds1.df = pd.read_csv(f"{split_dir}/padchest-{fold}.csv")
        ds.ds2.df = pd.read_csv(f"{split_dir}/positive-{fold}.csv")

    ds.len1 = len(ds.ds1)
    ds.len2 = len(ds.ds2)
    return ds

def load_dataset_3(
    seed,
    is_train,
    augments_name=None,
    preprocessing=None,
    split_name=None,
    flipped=0
):
    augments = get_augmentations(augments_name)
    preprocessing = get_preprocessing(preprocessing)

    train_transforms = v2.Compose([
        preprocessing, augments
    ]) if is_train else v2.Compose([preprocessing])
    fold = 'train' if is_train else 'val'

    ds = DomainConfoundedDataset(
        BIMCVNegativeDataset(fold='all', labels='chestx-ray14', augments=train_transforms, random_state=seed),
        BIMCVCOVIDDataset(fold='all', labels='chestx-ray14', augments=train_transforms, random_state=seed, is_old=True),
    )
    
    split_dir = f"splits/{split_name}/dataset3"
    if split_name:
        ds.ds1.df = pd.read_csv(f"{split_dir}/negative-{fold}.csv")
        ds.ds2.df = pd.read_csv(f"{split_dir}/positive-{fold}.csv")

        if flipped != 0:
            ds.flip_indices=pd.read_csv(f"{split_dir}/train-{flipped}.csv")["flipped_indices"].values
    else:
        trainvaldf1, testdf1, trainvaldf2, testdf2 = ds3_grouped_split(ds.ds1.df, ds.ds2.df, random_state=seed)
        traindf1, valdf1, traindf2, valdf2 = ds3_grouped_split(trainvaldf1, trainvaldf2, random_state=seed)

        if is_train:
            ds.ds1.df = traindf1
            ds.ds2.df = traindf2
        else:
            ds.ds1.df = valdf1
            ds.ds2.df = valdf2

    ds.len1 = len(ds.ds1)
    ds.len2 = len(ds.ds2)
    return ds