# test.py

from Code.Base.Functions.PatchOperation import patchify
import torch
import os
import torchvision
from torchvision.io import read_image, ImageReadMode
import PIL

if __name__ == "__main__":
    img_path = '../../data/PascalVOC2012/VOCdevkit/VOC2012/JPEGImages/2012_004177.jpg'

    # 基于 torch 的 torchvision：shape of an image should be (channels, height, width)
    img_tensor = read_image(img_path, mode=ImageReadMode.RGB)

    img_batch = img_tensor.unsqueeze(0) #(1, C, H, W)

    print(img_batch.shape)

    patch_size = (100, 150)

    patches_tensor = patchify(img_batch, patch_size).squeeze(0) # (num_patches, num_channels, patch_height, patch_width)

    # 基于 numpy 的 PIL: shape of a image should be (height, width, channels)
    patches_arr = patches_tensor.numpy().transpose(0,2,3,1) # (num_patches, patch_height, patch_width, num_channels)

    output_dir = '../tmp/images'
    
    for i, patch_arr in enumerate(patches_arr):
        img_name = os.path.join(output_dir, f"image_{i:03d}.png")

        # torchvision.utils.save_image(patches_tensor[i], img_name)
        # 避免使用 torchvision.utils.save_image.

        img = PIL.Image.fromarray(patch_arr)
        img.save(img_name)
    
    print("Image saving complete.")


