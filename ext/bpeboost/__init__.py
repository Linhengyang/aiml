import sys
import os


# 将 bin/目录加入 Python 路径, 以便能 import .so 文件
_bin_dir = os.path.join(os.path.dirname(__file__), '../../bin')
_bin_dir = os.path.abspath(_bin_dir)
if _bin_dir not in sys.path:
    sys.path.insert(0, _bin_dir)


from mp_pair_count_merge import initialize_process, count_u16pair_batch, merge_u16pair_batch, merge_u16pair_batch_v2, close_process


__all__ = ['initialize_process', 'count_u16pair_batch', 'merge_u16pair_batch', 'close_process']