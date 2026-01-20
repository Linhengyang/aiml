import sys
import os


# 将 bin/目录加入 Python 路径, 以便能 import .so 文件
_bin_dir = os.path.join(os.path.dirname(__file__), '../../bin')
_bin_dir = os.path.abspath(_bin_dir)
if _bin_dir not in sys.path:
    sys.path.insert(0, _bin_dir)


from mp_pair_count_merge import initialize_process, count_u16pair_batch as process_count_u16pair_batch, merge_u16pair_batch as process_merge_u16pair_batch, close_process

from mt_pair_count_merge import initialize_thread, count_u32pair_batch as thread_count_u32pair_batch, merge_u32pair_batch as thread_merge_u32pair_batch


__all__ = ['initialize_process', 'process_count_u16pair_batch', 'process_merge_u16pair_batch', 'close_process', 'initialize_thread', 'thread_count_u32pair_batch', 'thread_merge_u32pair_batch']