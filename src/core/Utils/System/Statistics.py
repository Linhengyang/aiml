# Statistics.py
import math
import numpy as np


def get_compression_ratio(string: str, indices: list[int]) -> float:
    """
    给定 string 和 对应的 tokenized integers 列表，计算 压缩率 = string utf-8字节数量 / tokens 数量
    举例1:
        全英文字符 string, unicode number as token ID, 这样 len(string) = utf-8字节数量 = unicode number 数量
        故 压缩率 = 1
    举例2:
        全中文字符 string, unicode number as token ID, 这样 len(string) = utf-8字节数量 / 3 = unicode number 数量
        故 压缩率 = 3
    """
    num_bytes = len(bytes(string, encoding="utf-8"))
    num_tokens = len(indices)
    
    return num_bytes / num_tokens