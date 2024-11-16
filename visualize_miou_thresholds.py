#!/usr/bin/env python
#
# train_covid.py
#
# Run ``python train_covid.py -h'' for information on using this script.
#
import numpy as np
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
from pytorch_grad_cam.utils.image import show_cam_on_image

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
from utils import get_preprocessing, get_gradcam, denormalize_image
from PIL import Image
from load_data import load_dataset_1, load_dataset_3

mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]


def create_miou_matrix(thresholded_maps):
    d = len(thresholded_maps)
    matrix = [[0]*d for _ in range(d)]

    for i in range(d):
        for ii in range(d):
            jaccard = JaccardIndex(task='binary')

            att1 = thresholded_maps[i]
            att2 = thresholded_maps[ii]
            matrix[i][ii] = jaccard(att1, att2).item()
    return matrix

def gradcam_mask(tensor, label, gradcam):
    attention = gradcam(input_tensor=tensor.clone(), targets=[ClassifierOutputTarget(label)])
    return attention[0]

def save_tensor(tensor, path):
    Image.fromarray(
        tensor,
        mode="RGB"
    ).save(path)

def gradcam_visualizations(save_path, thresholds, gradcams, dataset):
    dataloader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=0,
    )

    for img_id, (img, l, _, _) in enumerate(tqdm(dataloader)):
        if img_id == 70: break
        img_path = save_path / str(img_id)
        Path(img_path).mkdir(parents=True, exist_ok=True)

        attentions = [
            gradcam_mask(img, l[0, -1].item(), gradcam)
            for gradcam in gradcams
        ]
        numpy_img = denormalize_image(img[0])
        save_tensor((numpy_img * 255).astype(np.uint8), img_path / f"example.png")
        for i, att in enumerate(attentions):
            visualization = show_cam_on_image(numpy_img, att, use_rgb=True)
            save_tensor(visualization, img_path / f"vis_{i}.png")


        # log seed gradcams
        for thresh in thresholds:
            Path(img_path / str(thresh)).mkdir(parents=True, exist_ok=True)
            for i, att in enumerate(attentions):
                att[att < thresh] = 0 
                visualization = show_cam_on_image(numpy_img, att, use_rgb=True)
                save_tensor(visualization, img_path / str(thresh) / f"vis_{i}.png")
            
            thresholded_maps = [
                torch.tensor(att > thresh) for att in attentions
            ]
            matrix = create_miou_matrix(thresholded_maps)
        
            plot = sn.heatmap(matrix, annot=True, vmin=0, vmax=1)
            fig = plot.get_figure()
            fig.savefig(img_path / str(thresh) / "miou.png") 
            sn.reset_defaults()
            plt.clf()

def visualize_miou_tresholds(dataset, save_path, thresholds, gradcam_names, model_paths):
    for gradcam_name in gradcam_names:
        print(f"Starting visualizations for {gradcam_name}")
        path = save_path / gradcam_name
        gradcams = [
            get_gradcam(gradcam_name, path) for path in list(model_paths)
        ]
        # gradcam_visualizations(path / "negative-val", thresholds, gradcams, dataset.ds1)
        gradcam_visualizations(path / "positive-val", thresholds, gradcams, dataset.ds2)

if __name__ == "__main__":
    # parser = argparse.ArgumentParser(
    #     description='Script for generating miou matrix of localizations between models'
    # )
    # parser.add_argument('--config_path', dest='config_path', type=str, required=True,
    #                     help='Path to script config')
    # parser.add_argument('--output_path', dest='output_path', type=str, required=True,
    #                     help='Output path for generated image')
    # args = parser.parse_args()

    thresholds = [0.5, 0.6, 0.7]
    gradcam_names = ["grad++_cam"]
    split_name = f"42"
    preprocessing = 'crop_pad-0.9'
    for n in [1, 3]:
        if n == 1:
            ds = load_dataset_1(42, False, preprocessing=preprocessing, split_name=split_name)
        else:
            ds = load_dataset_3(42, False, preprocessing=preprocessing, split_name=split_name)
        
        for name in [
            "no_augments",
            "baseline-augments",
            "random_crop-strong"
        ]:
            model_paths =list(Path("checkpoints", "16", name, "crop_pad-0.9").rglob("*"))
            root_dir = Path(f"examples/gradcam_ious/{split_name}/dataset{n}/{name}")
            Path(root_dir).mkdir(parents=True, exist_ok=True)
            visualize_miou_tresholds(ds, root_dir, thresholds, gradcam_names, model_paths)
