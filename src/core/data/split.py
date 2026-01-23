# DataSplit.py
import random
import numpy as np

def split_textfile(input_f, export_files, split_ratios=[0.8, 0.1, 0.1], shuffle=True):

    split_ratios = np.array(split_ratios, dtype=float)

    assert len(export_files) == len(split_ratios), \
        f'export_files length {len(export_files)} != split_ratios length {len(split_ratios)}'
    
    assert split_ratios.sum() == 1, \
        f'split ratios shall be ratios summed to 1. now is {split_ratios}'

    # 读取所有行
    with open(input_f, 'r') as f:
        lines = f.readlines() # lines: list of strings
    
    total_lines = len(lines) # 总行数

    # 打乱行的顺序
    if shuffle:
        random.shuffle(lines)
    
    
    # 计算每个每个输出文件在输入总文件中的行数 index
    split_line_inds = [int(total_lines*r) for r in split_ratios.cumsum()]
    split_line_inds = [0] + split_line_inds

    # 文件分割
    export_data = []
    for i in range(len(split_line_inds)-1):
        export_data.append( lines[ split_line_inds[i]: split_line_inds[i+1]] )
    
    # 写文件
    for export_fname, export_data in zip(export_files, export_data):
        # 把 export_data 写入到 export_fname
        with open( export_fname, 'w' ) as f:
            f.writelines( export_data )
        
    return export_files