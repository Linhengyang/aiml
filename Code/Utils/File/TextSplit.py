# TextSplit.py
import random
import typing as t
import os
import numpy as np

def split_textfile(input_file_path, ratios:t.List[float], output_dir, shuffle:bool=True):
    assert all([r > 0 for r in ratios]) and sum(ratios) <= 1.0, \
        f'ratios shall be positive floats which sum to 1'

    os.makedirs(output_dir, exist_ok=True)
    _, file_ext = os.path.splitext(input_file_path)

    # 读取所有行
    with open(input_file_path, 'r') as f:
        lines = f.readlines()
    
    if shuffle:
        random.shuffle(lines)

    total_line_num = len(lines)
    split_ends = [0] + ( total_line_num * np.array(ratios).cumsum() ).astype(int).tolist()

    for i in range(1, len(split_ends)):
        r = ratios[i-1]*100 # 当前文件 所占比例(百分号下)

        cur_part_path = os.path.join(output_dir, str(r).rstrip('0').rstrip('.')+file_ext) # 当前文件名

        with open(cur_part_path, 'w') as f:
            f.writelines( lines[split_ends[i-1]:split_ends[i]] ) # 把 split_ends[i-1]:split_ends[i] 这些行写入 当前文件
            print(f'{cur_part_path} created')

    return output_dir

