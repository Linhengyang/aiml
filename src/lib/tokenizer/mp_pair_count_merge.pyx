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

    # 初始化进程环境: 单例内存池 / 基于单例内存池的可复用计数器
    void init_process(size_t block_size, size_t alignment, size_t capacity)

    # 重置进程环境：重置单例内存池 / 清空可复用计数器
    void reset_process()

    # 销毁进程的单例内存池 / 基于该单例内存池的可复用计数器，准备退出程序
    void release_process()

    # 声明 C++ 中的 u16token_pair_counts_ptrs 结构体
    struct u16token_pair_counts_ptrs:
        uint16_t* L_tokens_ptr
        uint16_t* R_tokens_ptr
        uint64_t* counts_ptr
        size_t size;

     # 声明 C++ 中的 c_local_count_u16pair_batch 函数
    u16token_pair_counts_ptrs c_local_count_u16pair_batch(
        const uint16_t* L_tokens,
        const uint16_t* R_tokens,
        const size_t len
    )

    # 声明 C++ 中的 merged_u16token_offset_ptrs 结构体 for merge_pair_func
    struct merged_u16token_offset_ptrs:
        uint16_t* merged_tokens_flat_ptr
        int64_t* merged_offsets_ptr
        size_t merged_num_chunks

    # 声明 C++ 中的 c_merge_pair_batch 函数
    merged_u16token_offset_ptrs c_local_merge_u16pair_batch(
        const uint16_t* tokens_flat,
        const int64_t* offsets,
        const size_t num_chunks,
        const uint16_t pair_L,
        const uint16_t pair_R,
        const uint16_t new_token,
        const bool if_filter_len1
    )







# 创建内存池/计数器接口给python. block_size size_t 从python侧传入, alignment设为64
cpdef initialize_process(size_t block_size):
    init_process(block_size, 64, 1024)





# 只是给python提供了reset进程的接口，但实际上count_pair_batch和merge_pair_batch每一次执行前都reset了
cpdef clear_process():
    reset_process()





# 关闭并清理资源接口给python
cpdef close_process():
    release_process()







# with GIL 版本 且去掉 b_order 版本, 给 工作进程 使用 以绕开 GIL
# 返回np.array of L_tokens/R_tokens/counts 给python
cpdef count_u16pair_batch(
    object tokens_offsets
):
    # reset 进程单例内存池 / 基于单例内存池的计数器
    reset_process()

    cdef np.ndarray[np.uint16_t, ndim=1, mode="c"] tokens_flat = tokens_offsets[0]
    cdef np.ndarray[np.int64_t, ndim=1, mode="c"] offsets = tokens_offsets[1]

    cdef int64_t _LENGTH = tokens_flat.shape[0] # token_flat's total length
    if _LENGTH != offsets[-1]:
        raise ValueError(f"tokens_flat length {_LENGTH} mismatch with last offset {offsets[-1]}")
    
    # 制作 L_tokens: token pair 左边的 tokens 和 R_tokens: token pair 右边的 tokens 
    mask = pynp.full(shape=(_LENGTH,), fill_value=True)
    chunk_ends_ = (offsets-1)[1:]
    chunk_starts_ = offsets[:-1]

    # ends_ == starts_ 的，说明chunk长度为1, 不需要统计paircounts. filter out
    _where_equal_ = chunk_ends_ == chunk_starts_
    mask[ chunk_ends_[_where_equal_] ] = False

    mask_cp = mask.copy()

    # 去掉所有 chunk 末尾的 token, 就是所有 L_tokens
    mask[chunk_ends_] = False
    cdef np.ndarray[np.uint16_t, ndim=1, mode="c"] L_tokens = tokens_flat[mask] # 可以为空
    
    # 去掉所有 chunk 开头的 token, 就是所有 R_tokens
    mask_cp[chunk_starts_] = False
    cdef np.ndarray[np.uint16_t, ndim=1, mode="c"] R_tokens = tokens_flat[mask_cp] # 可以为空

    # 检查 L_tokens 和 R_tokens 长度
    cdef size_t len = L_tokens.shape[0]
    if len != R_tokens.shape[0]:
        raise ValueError(f"Left & Right tokens length mismatch")
    
    if len == 0:
        return (pynp.array([], dtype=pynp.uint16),
                pynp.array([], dtype=pynp.uint16),
                pynp.array([], dtype=pynp.uint64))

    cdef const uint16_t[:] L_tokens_view = L_tokens
    cdef const uint16_t* L_tokens_ptr = &L_tokens_view[0]

    cdef const uint16_t[:] R_tokens_view = R_tokens
    cdef const uint16_t* R_tokens_ptr = &R_tokens_view[0]
    
    # 在进程内部 调用 c_local_count_u16pair_batch, 对 u16-token batch data 计算 pair-counts
    cdef u16token_pair_counts_ptrs result = c_local_count_u16pair_batch(
        L_tokens_ptr,
        R_tokens_ptr,
        len
    )

    cdef size_t size = result.size

    # 转换result里的C指针到numpy array, 构建output tokens array & counts array
    cdef uint16_t* raw_L_ptr = result.L_tokens_ptr # 接受C指针
    cdef uintptr_t L_addr = <uintptr_t><void*> raw_L_ptr # C pointer->void*转换, 然后是平台安全的指针->地址整数转换
    py_L_tokens_ptr = ctypes.cast(L_addr, ctypes.POINTER(ctypes.c_uint16)) # 地址整数 -> python地址
    output_L_tokens_np = pynp.ctypeslib.as_array(py_L_tokens_ptr, shape=(size,))

    cdef uint16_t* raw_R_ptr = result.R_tokens_ptr # 接受C指针
    cdef uintptr_t R_addr = <uintptr_t><void*> raw_R_ptr # C pointer ->void*转换, 然后是平台安全的指针->地址整数转换
    py_R_tokens_ptr = ctypes.cast(R_addr, ctypes.POINTER(ctypes.c_uint16))
    output_R_tokens_np = pynp.ctypeslib.as_array(py_R_tokens_ptr, shape=(size,))

    cdef uint64_t* raw_counts_ptr = result.counts_ptr # 接受C指针
    cdef uintptr_t counts_addr = <uintptr_t><void*> raw_counts_ptr # C pointer->void*转换, 然后是平台安全的指针->地址整数转换
    py_counts_ptr = ctypes.cast(counts_addr, ctypes.POINTER(ctypes.c_uint64))
    output_counts_np = pynp.ctypeslib.as_array(py_counts_ptr, shape=(size,))

    # 打包, tuple-pack 3 output np arrays with batch order
    return (output_L_tokens_np, output_R_tokens_np, output_counts_np)





# with GIL 版本, 给 工作进程 使用 以绕开 GIL
# 返回np.array of merged_tokens_flat/offsets给python
cpdef merge_u16pair_batch(
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
        raise ValueError(f"tokens_flat length {_LENGTH} mismatch with last offsets {offsets[num_chunks]}")
    
    # const uint16_t[::1]保证 memoryview是只读+内存连续的
    # 因为tokens_flat来自 np.array(..., dtype=..., copy=False) 共享了只读数据
    cdef const uint16_t[:] tokens_flat_view = tokens_flat
    cdef const int64_t[:] offsets_view = offsets

    # 零拷贝获取数据地址: get input ptr from memoryview input(zero-copy)
    cdef const uint16_t* tokens_flat_ptr = &tokens_flat_view[0]
    cdef const int64_t* offsets_ptr = &offsets_view[0]

    # 在进程内部 调用 c_local_merge_u16pair_batch
    cdef bool if_filter_len1 = True
    cdef merged_u16token_offset_ptrs result = c_local_merge_u16pair_batch(
        tokens_flat_ptr,
        offsets_ptr,
        num_chunks,
        pair_L,
        pair_R,
        new_token,
        if_filter_len1
    )

    cdef size_t merged_num_chunks = result.merged_num_chunks

    if merged_num_chunks == 0:
        return (pynp.array([], dtype=pynp.uint16), pynp.array([0], dtype=pynp.int64))

    cdef int64_t* raw_merged_offsets_ptr = result.merged_offsets_ptr # 接受C指针
    cdef uintptr_t merged_offsets_addr = <uintptr_t><void*> raw_merged_offsets_ptr # C pointer ->void* 转换, 然后是平台安全的指针->地址整数转换
    py_merged_offsets_ptr = ctypes.cast(merged_offsets_addr, ctypes.POINTER(ctypes.c_int64))
    merged_offsets = pynp.ctypeslib.as_array(py_merged_offsets_ptr, shape=(merged_num_chunks+1,))

    cdef int64_t _MERGED_LENGTH = merged_offsets[merged_num_chunks] # merged_offsets 长度是 merged_num_chunks+1, 这里取最后一个
    cdef uint16_t* raw_tokens_ptr = result.merged_tokens_flat_ptr # 接受C指针
    cdef uintptr_t tokens_addr = <uintptr_t><void*> raw_tokens_ptr # C pointer ->void* 转换, 然后是平台安全的指针->地址整数转换
    py_tokens_ptr = ctypes.cast(tokens_addr, ctypes.POINTER(ctypes.c_uint16))
    merged_tokens_flat = pynp.ctypeslib.as_array(py_tokens_ptr, shape=(_MERGED_LENGTH,))

    # 打包, pack merged info with batch order
    return (merged_tokens_flat, merged_offsets)