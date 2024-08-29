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
from PIL import Image

mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]

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
        ]),
        "none": v2.Identity()
    }[name]

def get_gradcam(gradcam_name, model_path):
    model = load_model(model_path)
    gradcam_layers = get_gradcam_layers(model)

    return {
        "grad_cam": GradCAM(model=model, target_layers=gradcam_layers),
        "grad++_cam": GradCAMPlusPlus(model=model, target_layers=gradcam_layers),
        "eigen_cam": EigenCAM(model=model, target_layers=gradcam_layers),
        "eigengrad_cam": EigenGradCAM(model=model, target_layers=gradcam_layers),
    }.get(gradcam_name)

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

def denormalize_image(image):
    img = image.numpy().transpose((1, 2, 0))  # numpy is [h, w, c] 
    _mean = np.array(mean)  # mean of your dataset
    _std = np.array(std)  # std of your dataset
    img = _std * img + _mean
    return img.clip(0, 1)

def visualize_miou_tresholds(preprocess, split_path, heatmap_name, thresholds, gradcam_names, model_paths):
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

    root_dir = Path(f"examples/gradcam_ious/{split_path}/{heatmap_name}")
    Path(root_dir).mkdir(parents=True, exist_ok=True)

    for gradcam_name in gradcam_names:
        print(f"Starting visualizations for {gradcam_name}")
        path = root_dir / gradcam_name
        gradcams = [
            get_gradcam(gradcam_name, path) for path in model_paths
        ]
        gradcam_visualizations(path / "positive-train", thresholds, gradcams, trainds.ds1)
        gradcam_visualizations(path / "positive-val", thresholds, gradcams, valds.ds2)

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
    gradcam_names = ["grad++_cam", "eigen_cam", "eigengrad_cam"]
    for name in [
        "small_batch-none", "small_batch-weak", "small_batch-strong",
        "big_batch-none"
    ]:
        model_paths =list(Path("checkpoints", "strong-strong").rglob("*"))

        split_path = "42/dataset3"
        visualize_miou_tresholds(name.split("-")[1], split_path, name, thresholds, gradcam_names, model_paths)
