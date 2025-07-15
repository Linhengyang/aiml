# distutils: language = c++
# cython: language_level=3, boundscheck=True, wraparound=True

import numpy as np
import sys
cimport numpy as np

np.import_array()

cdef extern from "tokenizer.h":

    # 声明 C++ 中的 return_bundle 结构体
    struct return_bundle {
        int* output_tokens_flat_ptr;
        bool* output_filter_ptr;
        long* output_tokens_lens_ptr;
    }

    # 声明 C++ 中的 c_merge_pair_batch 函数
    return_bundle c_merge_pair_batch(
        const int* tokens_flat,
        const long* offsets,
        int num_chunks,
        int pair_L,
        int pair_R,
        int new_token,
        int* output_tokens_flat,
        bool* output_filter,
        int* output_tokens_lens
    )

    # 创建内存池
    void init_memory_pool(size_t block_size, size_t alignment)

    # 重置内存池
    void reset_memory_pool()

    # 销毁内存池
    void release_memory_pool()

    # 缩小内存池
    void shrink_memory_pool()






# 返回np.array of merged_tokens_flat/offsets给python
def c_merge_pair_batch(
    memoryview tokens_flat, # memoryview of int32
    memoryview offsets, # memoryview of int64
    pair_L, # int32
    pair_R, # int32
    new_token, # int32
    block_size, # size_t, number of bytes for a memory block
    **kwargs
):
    # 确保第一次创建内存池. block_size 从python侧传入, alignment设为16
    init_memory_pool(block_size, 16)

    # 先尝试 shrink 内存池，以释放上一轮根本没用到的内存block. 对于刚初始化的内存池，shrink无效
    shrink_memory_pool()
    
    # 本 batch merge pair之前, 重置内存池
    reset_memory_pool()

    cdef size_t num_chunks = offsets.shape[0] - 1
    if num_chunks <= 0:
        return np.array([], dtype=np.int32), np.array([0], dtype=np.int64)
    
    cdef long _LENGTH = tokens_flat.shape[0] # token_flat's total length
    if _LENGTH != offsets[num_chunks+1]:
        sys.exit(1)

    cdef int[:] tokens_flat_view = <int[:_LENGTH]> tokens_flat
    cdef long[:] offsets_view = <long[:num_chunks+1]> offsets

    # get input ptr from memoryview input(zero-copy)
    cdef int* tokens_flat_ptr = &tokens_flat_view[0]
    cdef long* offsets_ptr = &offsets_view[0]

    # deploy cpp function
    cdef return_bundle = c_merge_pair_batch(
        tokens_flat_ptr,
        offsets_ptr,
        num_chunks,
        pair_L,
        pair_R,
        new_token,
    )

    # build output tokens array by filter
    output_tokens_flat_np = np.ctypeslib.as_array(
        return_bundle.output_tokens_flat_ptr, shape=(_LENGTH,))

    output_filter_np = np.ctypeslib.as_array(
        return_bundle.output_filter_ptr, shape=(_LENGTH,))

    # build output tokens offsets
    output_tokens_lens_np = np.ctypeslib.as_array(
        return_bundle.output_tokens_lens_ptr, shape=(num_chunks,))

    output_offsets_np = np.cumsum(np.insert(output_tokens_lens_np, 0, 0), dtype=np.int64)

    return output_tokens_flat_np[output_filter_np], output_offsets_np





# 销毁内存池接口给python
def release_memory():
    release_memory_pool()