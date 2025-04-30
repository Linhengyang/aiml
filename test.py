# test.py

from Code.Base.Functions.PatchOperation import patchify
import torch
import os
import torchvision
from torchvision.io import read_image, ImageReadMode
from PIL import Image

if __name__ == "__main__":
    img_path = '../../data/PascalVOC2012/VOCdevkit/VOC2012/JPEGImages/2012_004177.jpg'

    img_tensor = read_image(img_path, mode=ImageReadMode.RGB)

    img_batch = img_tensor.unsqueeze(0) #(1, C, H, W)

    print(img_batch.shape)

    patch_size = (100, 150)

    patches_tensor = patchify(img_batch, patch_size).squeeze(0) # (num_patches, num_channels, patch_height, patch_width)

    patches_arr = patches_tensor.numpy().transpose(0,2,3,1) # (num_patches, patch_height, patch_width, num_channels)

    output_dir = '../tmp/images'
    
    for i, patch_arr in enumerate(patches_arr):
        img_name = os.path.join(output_dir, f"image_{i:03d}.png")

        # torchvision.utils.save_image(patches_tensor[i], img_name)
        img = Image.fromarray(patch_arr)
        img.save(img_name)
    
    print("Image saving complete.")


