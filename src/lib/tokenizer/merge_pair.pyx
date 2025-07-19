# distutils: language = c++
# cython: language_level=3, boundscheck=True, wraparound=True

import numpy as np
import sys
cimport numpy as np
from libc.stdint cimport int32_t, int64_t
import ctypes

np.import_array()

cdef extern from "merge_pair.h":

    # 声明 C++ 中的 return_bundle 结构体
    struct return_bundle:
        int32_t* output_tokens_flat_ptr
        bint* output_filter_ptr
        int64_t* output_tokens_lens_ptr
    

    # 声明 C++ 中的 c_merge_pair_batch 函数
    return_bundle c_merge_pair_batch(
        const int32_t* tokens_flat,
        const int64_t* offsets,
        const size_t num_chunks,
        const int32_t pair_L,
        const int32_t pair_R,
        const int32_t new_token
    )

    # 创建内存池
    void init_memory_pool(size_t block_size, size_t alignment)

    # 重置内存池
    void reset_memory_pool()

    # 销毁内存池
    void release_memory_pool()

    # 缩小内存池
    void shrink_memory_pool()







# 创建内存池接口给python. block_size size_t 从python侧传入, alignment设为16
# 经过测算，output needed size 的chunk_lens/fitler/tokens 分别大概是 8倍/10倍/40倍 batch_size bytes
# 为了保证tokens在同一个block而不是large alloc, block_size 至少是 40倍batch_size bytes
def allocate_memory(block_size):
    init_memory_pool(block_size, 16)





# 返回np.array of merged_tokens_flat/offsets给python
def merge_pair_batch(
    memoryview tokens_flat, # memoryview of int32
    memoryview offsets, # memoryview of int64
    pair_L, # int32
    pair_R, # int32
    new_token, # int32
    **kwargs
):
    # 先尝试 shrink 内存池，释放1个上一轮没用到的内存block. 对于刚初始化的内存池，shrink无效
    shrink_memory_pool()
    
    # 本 batch merge pair之前, 重置内存池
    reset_memory_pool()

    cdef size_t num_chunks = offsets.shape[0] - 1
    if num_chunks <= 0:
        return np.array([], dtype=np.int32), np.array([0], dtype=np.int64)
    
    cdef int64_t _LENGTH = tokens_flat.shape[0] # token_flat's total length
    if _LENGTH != offsets[num_chunks]:
        sys.exit(1)
    
    # const int32_t[::1]保证 memoryview是只读+内存连续的
    # 因为tokens_flat来自 np.array(..., dtype=..., copy=False) 共享了只读数据
    cdef const int32_t[::1] tokens_flat_view = tokens_flat
    cdef const int64_t[::1] offsets_view = offsets

    # get input ptr from memoryview input(zero-copy)
    cdef const int32_t* tokens_flat_ptr = &tokens_flat_view[0]
    cdef const int64_t* offsets_ptr = &offsets_view[0]

    # deploy cpp function
    cdef return_bundle result = c_merge_pair_batch(
        tokens_flat_ptr,
        offsets_ptr,
        num_chunks,
        pair_L,
        pair_R,
        new_token,
    )

    # build output tokens array & filter array
    tokens_ptr = ctypes.cast(<int32_t> result.output_tokens_flat_ptr, ctypes.POINTER(ctypes.c_int32))
    output_tokens_flat_np = np.ctypeslib.as_array(tokens_ptr, shape=(_LENGTH,))

    filter_ptr = ctypes.cast(<bool_t> result.output_filter_ptr, ctypes.POINTER(ctypes.c_bool))
    output_filter_np = np.ctypeslib.as_array(filter_ptr, shape=(_LENGTH,))

    # build output tokens offsets
    lens_ptr = ctypes.cast(<int64_t> result.output_tokens_lens_ptr, ctypes.POINTER(ctypes.c_int64))
    output_tokens_lens_np = np.ctypeslib.as_array(lens_ptr, shape=(num_chunks,))

    output_offsets_np = np.cumsum(np.insert(output_tokens_lens_np, 0, 0), dtype=np.int64)

    return output_tokens_flat_np[output_filter_np], output_offsets_np





# 销毁内存池接口给python
def release_memory():
    release_memory_pool()