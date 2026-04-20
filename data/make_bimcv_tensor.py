#!/usr/bin/env python
# make_h5.py
#
from bimcv_dataset_creator import BimcvDatasetPositive, BimcvDatasetNegative, BimcvDatasetABC

import argparse
import tqdm
import torch
from pathlib import Path

from PIL import Image
import numpy as np

def save_pil(image, idx):
    img = image.numpy().astype(float).transpose((1, 2, 0))  # numpy is [h, w, c] 
    mean = np.array([0.4451, 0.4262, 0.3959])  # mean of your dataset
    std = np.array([0.2411, 0.2403, 0.2466])  # std of your dataset
    img = std * img + mean
    img = np.clip(img, 0, 1) * 255
    Image.fromarray(img.astype(np.uint8)[:, :, 0], mode="L").save(f"images/final_{idx}.png")

def create_dataset(args):
    creator: BimcvDatasetABC = BimcvDatasetPositive() if args.is_positive else BimcvDatasetNegative()
    image_indices = range(len(creator.df))

    Path(args.outpath).mkdir(parents=True, exist_ok=True)
    with open(f"errors.txt", "w") as f:
        for idx in tqdm.tqdm(image_indices):
            try:
                image, imagename = creator.process_image_by_index(idx)
                tensor = torch.tensor(image)
                tensor_path = f"{Path(args.outpath, Path(imagename).stem)}.pt"
                torch.save(tensor, tensor_path)
            except:
                image, imagename = creator.process_image_by_index(idx)
                print(idx)
                save_pil(image, 0)
                f.write(f"Broken image: {imagename}\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", dest='is_positive', default=True, type=bool)
    parser.add_argument("-o", dest='outpath', default='tensor/bimcv+')
    args = parser.parse_args()

    create_dataset(args)
