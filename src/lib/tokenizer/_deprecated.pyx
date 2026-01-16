# distutils: language = c++
# cython: language_level=3, boundscheck=True, wraparound=True

import numpy as pynp
import sys
cimport numpy as np
from libc.stdint cimport uint16_t, int64_t, uint64_t, uintptr_t
import ctypes

np.import_array()

cdef extern from *:
    ctypedef bint bool



cdef extern from "mp_pair_count_merge.h":

    # 声明 C++ 中的 merged_u16token_filter_len_ptrs 结构体
    struct merged_u16token_filter_len_ptrs:
        uint16_t* output_tokens_flat_ptr
        bool* output_filter_ptr
        int64_t* output_tokens_lens_ptr

    # 声明 C++ 中的 c_merge_pair_batch 函数
    merged_u16token_filter_len_ptrs _deprecated_c_local_merge_u16pair_batch(
        const uint16_t* tokens_flat,
        const int64_t* offsets,
        const size_t num_chunks,
        const uint16_t pair_L,
        const uint16_t pair_R,
        const uint16_t new_token
    )




# with GIL 版本 且去掉 b_order 版本, 给 工作进程 使用 以绕开 GIL. 返回np.array of merged_tokens_flat/offsets给python
# 已经废弃, 不要再使用. 完全被 merge_u16pair_batch 替代
cpdef _deprecated_merge_u16pair_batch(
    object tokens_offsets,
    np.uint16_t pair_L,
    np.uint16_t pair_R,
    np.uint16_t new_token,
):
    # reset 进程单例内存池 / 基于单例内存池的计数器
    reset_process()

    # 得到 tokens flattened
    cdef np.ndarray[np.uint16_t, ndim=1, mode="c"] tokens_flat = tokens_offsets[0]
    # 得到 offsets
    cdef np.ndarray[np.int64_t, ndim=1, mode="c"] offsets = tokens_offsets[1]

    cdef size_t num_chunks = offsets.shape[0] - 1
    if num_chunks <= 0:
        return (pynp.array([], dtype=pynp.uint16), pynp.array([0], dtype=pynp.int64))
    
    cdef int64_t _LENGTH = tokens_flat.shape[0] # token_flat's total length
    if _LENGTH != offsets[num_chunks]:
        sys.exit(1)
    
    # const uint16_t[::1]保证 memoryview是只读+内存连续的
    # 因为tokens_flat来自 np.array(..., dtype=..., copy=False) 共享了只读数据
    cdef const uint16_t[:] tokens_flat_view = tokens_flat
    cdef const int64_t[:] offsets_view = offsets

    # 零拷贝获取数据地址: get input ptr from memoryview input(zero-copy)
    cdef const uint16_t* tokens_flat_ptr = &tokens_flat_view[0]
    cdef const int64_t* offsets_ptr = &offsets_view[0]

    # 在进程内部 调用 _deprecated_c_local_merge_u16pair_batch,
    cdef merged_u16token_filter_len_ptrs result = _deprecated_c_local_merge_u16pair_batch(
        tokens_flat_ptr,
        offsets_ptr,
        num_chunks,
        pair_L,
        pair_R,
        new_token
    )


    # 转换result里的C指针到numpy array, 构建output tokens array & filter array
    cdef uint16_t* raw_tokens_ptr = result.output_tokens_flat_ptr # 接受C指针
    cdef uintptr_t tokens_addr = <uintptr_t><void*> raw_tokens_ptr # C pointer ->void* 转换, 然后是平台安全的指针->地址整数转换
    py_tokens_ptr = ctypes.cast(tokens_addr, ctypes.POINTER(ctypes.c_uint16))
    output_tokens_flat_np = pynp.ctypeslib.as_array(py_tokens_ptr, shape=(_LENGTH,))

    cdef bool* raw_filter_ptr = result.output_filter_ptr # 保持C类型接受C指针
    cdef uintptr_t filter_addr = <uintptr_t><void*> raw_filter_ptr # C pointer ->void* 转换, 然后是平台安全的指针->地址整数转换
    py_filter_ptr = ctypes.cast(filter_addr, ctypes.POINTER(ctypes.c_uint8)) # 使用c_uint8来接住原始bool*内存地址
    output_filter_np = pynp.ctypeslib.as_array(py_filter_ptr, shape=(_LENGTH,)) # uint8类型的array
    output_filter_np = output_filter_np.astype(pynp.bool_) # 显式转换为布尔类型

    merged_tokens_flat = output_tokens_flat_np[output_filter_np]

    cdef int64_t* raw_lens_ptr = result.output_tokens_lens_ptr # 接受C指针
    cdef uintptr_t lens_addr = <uintptr_t><void*> raw_lens_ptr # C pointer ->void* 转换, 然后是平台安全的指针->地址整数转换
    py_lens_ptr = ctypes.cast(lens_addr, ctypes.POINTER(ctypes.c_int64))
    output_chunks_lens_np = pynp.ctypeslib.as_array(py_lens_ptr, shape=(num_chunks,))

    merged_offsets = pynp.cumsum(pynp.insert(output_chunks_lens_np, 0, 0), dtype=pynp.int64)

    # 过滤掉 chunks whose length = 1
    len1_chunks_ = pynp.where(output_chunks_lens_np == 1)[0].astype(pynp.int64) # 保证即使是空, 也能正确slice
    len1_chunks_tokens_ = merged_offsets[len1_chunks_] # invalid tokens 在 flat 中的 inds, 会被filter out
    mask = pynp.full(merged_tokens_flat.shape[0], True, dtype = pynp.bool_) # 用mask筛选出 len>1 的chunks 的tokens
    mask[len1_chunks_tokens_] = False
    valid_merged_tokens_flat = merged_tokens_flat[mask]

    valid_chunks_ = pynp.where(output_chunks_lens_np > 1)[0].astype(pynp.int64) # 保证即使是空, 也能正确slice
    valid_chunks_lens = output_chunks_lens_np[valid_chunks_]
    valid_merged_offsets = pynp.cumsum(pynp.insert(valid_chunks_lens, 0, 0), dtype=pynp.int64)

    # 打包, pack valid merged info with batch order
    return (valid_merged_tokens_flat, valid_merged_offsets)