import sys
import os
# .pyi是类型提示存根文件, 它的作用是为编译后的二进制模块(.pyd/.so)提供类型信息。所以 .pyi 文件必须与对应 .pyx & .so/.pyd 文件同名

# 将 bin/目录加入 Python 路径, 以便能 import .so 文件
_bin_dir = os.path.join(os.path.dirname(__file__), '../../bin')
_bin_dir = os.path.abspath(_bin_dir)
if _bin_dir not in sys.path:
    sys.path.insert(0, _bin_dir)


from mp_pair_count_merge import (
    initialize_process,
    count_u16pair_batch as process_count_u16pair_batch,
    merge_u16pair_batch as process_merge_u16pair_batch,
    close_process
    )

from mt_pair_count_merge import (
    initialize_thread,
    count_u32pair_batch as thread_count_u32pair_batch,
    merge_u32pair_batch as thread_merge_u32pair_batch
    )

from bow_counter import (
    bytes_chunk_count
)

__all__ = [
    'initialize_process',
    'process_count_u16pair_batch',
    'process_merge_u16pair_batch',
    'close_process',
    'initialize_thread',
    'thread_count_u32pair_batch',
    'thread_merge_u32pair_batch',
    'bytes_chunk_count'
    ]