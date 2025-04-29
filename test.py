# test.py

from Code.Utils.image.PatchOperation import patchify
import torch
import torchvision
from torchvision.io import read_image, ImageReadMode

if __name__ == "__main__":
    img_path = '../../data/semantic_segmentation/VOCdevkit/VOC2012/JPEGImages/2012_004177.jpg'

    img_tensor = read_image(img_path, mode=ImageReadMode.RGB)

    img_batch = img_tensor.unsqueeze(0) #(1, C, H, W)

    print(img_batch.shape)

