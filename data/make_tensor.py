#!/usr/bin/env python
# make_h5.py
#

import argparse
import tqdm
import torch
from pathlib import Path
import cv2

import numpy as np

from torchvision import transforms
from PIL import Image

# use imagenet mean,std for normalization
mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]

class TensorDatasetCreator():
    def __init__(self) -> None:
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])  


    def load_image(self, image_path):
        image_array = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
        imagenet_size = (224, 224)
        image_array = cv2.resize(image_array, imagenet_size, interpolation=cv2.INTER_AREA)

        if len(image_array.shape) != 3: # add channel dimension
            image_array = np.expand_dims(image_array, -1) 

        if image_array.shape[-1] == 1: # repeat channel dimension for normalization
            image_array = np.repeat(image_array, 3, 2)
        if image_array.shape[-1] == 4: # repeat channel dimension for normalization
            image_array = cv2.cvtColor(image_array, cv2.COLOR_RGBA2RGB)

        clipped = np.clip(image_array.astype(float) / 2**8, 0, 255)
        return self.transform(clipped.astype(np.uint8))

def save_pil(image, idx):
    img = image.numpy().transpose((1, 2, 0))  # numpy is [h, w, c] 
    _mean = np.array(mean)  # mean of your dataset
    _std = np.array(std)  # std of your dataset
    img = _std * img + _mean
    img = np.clip(img, 0, 1) * 255
    Image.fromarray(img.astype(np.uint8)[:, :, 0], mode="L").save(f"images/{idx}.png")

def create_dataset(args):
    creator = TensorDatasetCreator()
    image_paths = list(Path(args.dir_path).rglob("*.png"))

    Path(args.outpath).mkdir(parents=True, exist_ok=True)
    for i, path in enumerate(tqdm.tqdm(image_paths)):
        try:
            image = creator.load_image(path)
            # save_pil(image, i)

            tensor = torch.tensor(image)
            tensor_path = f"{Path(args.outpath, Path(path).stem)}.pt"
        
            torch.save(tensor, tensor_path)
        except:
            print(f"Broken image: {i + 14985}, {path}")



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", dest='dir_path', default='padchest', type=str)
    parser.add_argument("-o", dest='outpath', default='tensor/padchest')
    args = parser.parse_args()

    create_dataset(args)
