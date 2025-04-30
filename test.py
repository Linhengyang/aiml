# test.py

from Code.Base.Functions.PatchOperation import patchify
import torch
import os
import torchvision
from torchvision.io import read_image, ImageReadMode

if __name__ == "__main__":
    img_path = '../../data/PascalVOC2012/VOCdevkit/VOC2012/JPEGImages/2012_004177.jpg'

    img_tensor = read_image(img_path, mode=ImageReadMode.RGB)

    img_batch = img_tensor.unsqueeze(0) #(1, C, H, W)

    print(img_batch.shape)

    patch_size = (100, 100)

    patches_tensor = patchify(img_batch, patch_size).squeeze(0) # (num_patches, num_channels, patch_height, patch_width)
    print(patches_tensor.shape)

    output_dir = '../tmp/images'
    
    for i in range(patches_tensor.shape[0]):
        img_name = os.path.join(output_dir, f"image_{i:03d}.png")
        print(patches_tensor[i])

        torchvision.utils.save_image(patches_tensor[i], img_name)
    
    print("Image saving complete.")


