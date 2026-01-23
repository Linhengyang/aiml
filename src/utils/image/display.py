# Display.py

import PIL
import torch
import matplotlib.pyplot as plt
import numpy as np
import typing as t
import pandas as pd
from src.utils.math import find_closet_2factors










def display_images_with_labels(
        image_tensor: torch.Tensor,
        label_data: t.List|pd.DataFrame|None = None,
        grid_number_row_col:t.Tuple[int, int]|None = None,
        ):
    """
    以 grid_number_row_col (p 行 q 列) 的网格形式展示图片Tensor image_tensor 及其对应的标签 label_data

    Args:
        image_tensor (torch.Tensor):
            图像数据, shape 为 (N, num_channels, height, width). 值应在 [0, 1] 或 [0, 255] 范围内
        label_data (t.List|pd.DataFrame|None):
            标签数据, 如果是 List, 则 shape 为(N,); 如果是 dataframe, 则 shape 为 (N, num_labels), 每列都是 N 张图片的某个标签
        grid_number_row_col (p(int), q(int)):
            在 p 行 q 列的 网格中展示 N 张图片
    """
    N = image_tensor.shape[0]
    num_channels = image_tensor.shape[1]

    # 检查标签（如果有）长度是否与图片数量匹配
    if label_data is not None and len(label_data) != N:
        raise ValueError(
            f"the length of label data {len(label_data)} must match with number of images, \
                which is {image_tensor.shape} on image tensor dimension 0"
            )
    
    # 创建一个 p x q 的 subplot 网格
    if N == 1: # 如果只有一张图片, 那么不需要网格
        p, q = 1, 1
    elif grid_number_row_col: # 如果有多张图片, 且输入了网格 grid 的p 和 q
        p, q = grid_number_row_col
    else: # 如果有多张图片, 且没有输入网格 grid 的p 和 q
        p, q = find_closet_2factors(N)
    
    # figsize 根据网格大小调整，以便图片有合适的显示尺寸
    fig, axes = plt.subplots(p, q, figsize=(q * 3, p * 3))

    # 将 axes 展平，方便通过索引访问
    axes = axes.flatten()

    # 若 label_data 是 dataframe, 取出 label_data 的列名, 作为 标签的名字
    if isinstance(label_data, pd.DataFrame):
        label_names = label_data.columns.to_list()

    # 循环遍历图片（最多 N 张，或网格能容纳的数量）
    for i in range(min(N, p * q)):
        ax = axes[i] # 获取当前 subplot

        # 获取当前图片数据和标签
        img = image_tensor[i]

        # --- 图像处理 ---
        # Matplotlib 的 imshow 函数期望的图像形状通常是 (height, width) 或 (height, width, num_channels)
        # 而 PyTorch tensor 是 (num_channels, height, width)，所以需要调整通道顺序
        img = img.permute(1, 2, 0) # 从 (C, H, W) 变为 (H, W, C)

        # 将 PyTorch tensor 转换为 NumPy 数组，并移动到 CPU
        img = img.cpu().numpy()

        # 处理单通道（灰度）图片
        if num_channels == 1:
            img = img.squeeze(-1) # 移除最后一个维度 (H, W, 1) -> (H, W)
            cmap = 'gray' # 使用灰度 colormap
        elif num_channels == 3:
            cmap = None # 使用默认的彩色 colormap
        else:
            # 对于其他通道数的图片，matplotlib 可能无法直接显示，可以跳过或根据需要调整
            print(f"Warning: Image {i}'s number of channels {num_channels} is not 1 or 3. skip displaying")
            ax.set_title(f"not show {i}")
            ax.axis('off')
            continue

        # 确保图像数据类型和范围适合 imshow
        # 如果您的 tensor 是 [0, 255] 的整数，可以转换为 uint8: img = img.astype(np.uint8)
        # 如果您的 tensor 是 [0, 1] 的浮点数，则可以直接显示
        # 这里假设是 [0, 1] 的浮点数或 Matplotlib 能自动处理的类型
        img = img.astype(np.float32) # 确保是浮点类型，通常更安全

        # 显示图片
        ax.imshow(img, cmap=cmap)

        # 关闭坐标轴，使图片看起来更干净
        ax.axis('off')

        # 添加 标签/合并的标签 作为 subplot 的标题
        if isinstance(label_data, pd.DataFrame): # 若 label_data 是 dataframe,
            # 则取 i 行作为当前图片的所有 labels, 合并成为 title
            labels = [ f'{name}:{label}' for name, label in zip(label_names, label_data.iloc[i, :].to_list()) ]
            title = "\n".join(labels)
        elif isinstance(label_data, list): # 若 label_data 是 list
            # 则取 元素 i 作为当前图片的 title
            title = str(label_data[i])
        else: # 若 label_data 是 None
            # 则取 序号 i+1 作为当前图片的 title
            title = str(i+1)

        ax.set_title(title, fontsize=8) # 调整字体大小以适应空间

    # 如果图片数量 N 小于网格的总单元格数 p*q，隐藏剩余的空白 subplot
    for j in range(N, p * q):
        fig.delaxes(axes[j]) # 删除当前轴

    # 调整布局，防止标题或图片重叠
    plt.tight_layout()

    # 显示图形
    plt.show()
