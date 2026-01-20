# distutils: language = c++
# cython: language_level=3, boundscheck=True, wraparound=True

import numpy as np
import sys
cimport numpy as cnp
from libc.stdint cimport uint32_t, int64_t, uint64_t, uintptr_t
import ctypes

cnp.import_array()

cdef extern from *:
    ctypedef bint bool

cdef extern from "mt_pair_count_merge.h":

    # 初始化线程环境: tls内存池 / 基于tls内存池的可复用计数器
    void init_thread(size_t block_size, size_t alignment, size_t capacity)

    # 重置线程环境：重置tls内存池 / 清空可复用计数器
    void reset_thread()

    # 销毁线程的tls内存池 / 基于该tls内存池的可复用计数器，准备退出程序
    void release_thread()

    # 声明 C++ 中的 u32token_pair_counts_ptrs 结构体
    struct u32token_pair_counts_ptrs:
        uint32_t* L_tokens_ptr
        uint32_t* R_tokens_ptr
        uint64_t* counts_ptr
        size_t size;

     # 声明 C++ 中的 c_tls_count_u32pair_batch 函数 nogil
    u32token_pair_counts_ptrs c_tls_count_u32pair_batch(
        const uint32_t* L_tokens,
        const uint32_t* R_tokens,
        const size_t len
    ) nogil

    # 声明 C++ 中的 merged_u32token_offset_ptrs 结构体 for merge_pair_func
    struct merged_u32token_offset_ptrs:
        uint32_t* merged_tokens_flat_ptr
        int64_t* merged_offsets_ptr
        size_t merged_num_chunks
        int64_t merged_num_tokens

    # 声明 C++ 中的 c_merge_u32pair_batch 函数 nogil
    merged_u32token_offset_ptrs c_tls_merge_u32pair_batch(
        const uint32_t* tokens_flat,
        const int64_t* offsets,
        const size_t num_chunks,
        const uint32_t pair_L,
        const uint32_t pair_R,
        const uint32_t new_token,
        const bool if_filter_len1
    ) nogil





# 创建内存池/计数器接口给python. block_size size_t 从python侧传入, alignment设为64
cpdef initialize_thread(size_t block_size):
    init_thread(block_size, 64, 1024)




cpdef count_u32pair_batch(
    object tokens_offsets
):
    # reset 进程tls内存池 / 基于tls内存池的计数器
    reset_thread()

    cdef cnp.ndarray[cnp.uint32_t, ndim=1, mode="c"] tokens_flat = tokens_offsets[0]
    cdef cnp.ndarray[cnp.int64_t, ndim=1, mode="c"] offsets = tokens_offsets[1]

    cdef int64_t _LENGTH = tokens_flat.shape[0] # token_flat's total length
    if _LENGTH != offsets[-1]:
        sys.exit(1)
    
    # 制作 L_tokens: token pair 左边的 tokens 和 R_tokens: token pair 右边的 tokens 
    mask = np.full(shape=(_LENGTH,), fill_value=True)
    chunk_ends_ = (offsets-1)[1:]
    chunk_starts_ = offsets[:-1]

    # ends_ == starts_ 的，说明chunk长度为1, 不需要统计paircounts. filter out
    _where_equal_ = chunk_ends_ == chunk_starts_
    mask[ chunk_ends_[_where_equal_] ] = False

    mask_cp = mask.copy()

    # 去掉所有 chunk 末尾的 token, 就是所有 L_tokens
    mask[chunk_ends_] = False
    cdef cnp.ndarray[cnp.uint32_t, ndim=1, mode="c"] L_tokens = tokens_flat[mask] # 可以为空
    
    # 去掉所有 chunk 开头的 token, 就是所有 R_tokens
    mask_cp[chunk_starts_] = False
    cdef cnp.ndarray[cnp.uint32_t, ndim=1, mode="c"] R_tokens = tokens_flat[mask_cp] # 可以为空

    # 检查 L_tokens 和 R_tokens 长度.
    cdef size_t len = L_tokens.shape[0]
    if len != R_tokens.shape[0]:
        sys.exit(1)
    
    if len == 0:
        return (np.array([], dtype=np.uint32),
                np.array([], dtype=np.uint32),
                np.array([], dtype=np.uint64))

    cdef const uint32_t[:] L_tokens_view = L_tokens
    cdef const uint32_t[:] R_tokens_view = R_tokens

    cdef const uint32_t* L_tokens_ptr = &L_tokens_view[0]
    cdef const uint32_t* R_tokens_ptr = &R_tokens_view[0]

    cdef u32token_pair_counts_ptrs result
    cdef size_t size

    cdef uint32_t *raw_L_ptr, *raw_R_ptr
    cdef uintptr_t L_addr, R_addr

    cdef uint64_t* raw_counts_ptr
    cdef uintptr_t counts_addr

    # ----- NOGIL 区域: 核心计算 -----
    with nogil:
        # 调用 c_tls_count_u32pair_batch, 对 u32-token batch data 计算 pair-counts
        result = c_tls_count_u32pair_batch(
            L_tokens_ptr,
            R_tokens_ptr,
            len
        )

        size = result.size

        # 转换result里的C指针到numpy array, 构建output tokens array & counts array
        raw_L_ptr = result.L_tokens_ptr # 接受C指针
        L_addr = <uintptr_t><void*> raw_L_ptr # C pointer->void*转换, 然后是平台安全的指针->地址整数转换

        raw_R_ptr = result.R_tokens_ptr # 接受C指针
        R_addr = <uintptr_t><void*> raw_R_ptr # C pointer ->void*转换, 然后是平台安全的指针->地址整数转换

        raw_counts_ptr = result.counts_ptr # 接受C指针
        counts_addr = <uintptr_t><void*> raw_counts_ptr # C pointer->void*转换, 然后是平台安全的指针->地址整数转换

    # ----- NOGIL 区域结束 -----

    py_L_tokens_ptr = ctypes.cast(L_addr, ctypes.POINTER(ctypes.c_uint32)) # 地址整数 -> python地址
    py_R_tokens_ptr = ctypes.cast(R_addr, ctypes.POINTER(ctypes.c_uint32))
    py_counts_ptr = ctypes.cast(counts_addr, ctypes.POINTER(ctypes.c_uint64))

    output_L_tokens_np = np.ctypeslib.as_array(py_L_tokens_ptr, shape=(size,))
    output_R_tokens_np = np.ctypeslib.as_array(py_R_tokens_ptr, shape=(size,))
    output_counts_np = np.ctypeslib.as_array(py_counts_ptr, shape=(size,))

    # 打包, tuple-pack 3 output np arrays with batch order
    return (output_L_tokens_np, output_R_tokens_np, output_counts_np)






cpdef merge_u32pair_batch(
    object tokens_offsets,
    cnp.uint32_t pair_L,
    cnp.uint32_t pair_R,
    cnp.uint32_t new_token,
):
    # reset 进程单例内存池 / 基于单例内存池的计数器
    reset_thread()

    # 得到 tokens flattened
    cdef cnp.ndarray[cnp.uint32_t, ndim=1, mode="c"] tokens_flat = tokens_offsets[0]
    # 得到 offsets
    cdef cnp.ndarray[cnp.int64_t, ndim=1, mode="c"] offsets = tokens_offsets[1]

    cdef size_t num_chunks = offsets.shape[0] - 1
    if num_chunks <= 0:
        return (np.array([], dtype=np.uint32), np.array([0], dtype=np.int64))
    
    cdef int64_t _LENGTH = tokens_flat.shape[0] # token_flat's total length
    if _LENGTH != offsets[num_chunks]:
        sys.exit(1)
    
    # const uint32_t[::1]保证 memoryview是只读+内存连续的
    # 因为tokens_flat来自 np.array(..., dtype=..., copy=False) 共享了只读数据
    cdef const uint32_t[:] tokens_flat_view = tokens_flat
    cdef const int64_t[:] offsets_view = offsets

    # 零拷贝获取数据地址: get input ptr from memoryview input(zero-copy)
    cdef const uint32_t* tokens_flat_ptr = &tokens_flat_view[0]
    cdef const int64_t* offsets_ptr = &offsets_view[0]

    cdef bool if_filter_len1 = True
    cdef merged_u32token_offset_ptrs result

    cdef size_t merged_num_chunks
    cdef int64_t _MERGED_LENGTH

    cdef int64_t* raw_merged_offsets_ptr
    cdef uintptr_t merged_offsets_addr

    cdef uint32_t* raw_tokens_ptr
    cdef uintptr_t tokens_addr

    # ----- NOGIL 区域: 核心计算 -----
    with nogil:
        result = c_tls_merge_u32pair_batch(
            tokens_flat_ptr,
            offsets_ptr,
            num_chunks,
            pair_L,
            pair_R,
            new_token,
            if_filter_len1
        )

        merged_num_chunks = result.merged_num_chunks
        _MERGED_LENGTH = result.merged_num_tokens

        raw_merged_offsets_ptr = result.merged_offsets_ptr # 接受C指针
        merged_offsets_addr = <uintptr_t><void*> raw_merged_offsets_ptr # C pointer ->void* 转换, 然后是平台安全的指针->地址整数转换

        raw_tokens_ptr = result.merged_tokens_flat_ptr # 接受C指针
        tokens_addr = <uintptr_t><void*> raw_tokens_ptr # C pointer ->void* 转换, 然后是平台安全的指针->地址整数转换

    # ----- NOGIL 区域结束 -----
    
    py_merged_offsets_ptr = ctypes.cast(merged_offsets_addr, ctypes.POINTER(ctypes.c_int64))
    py_tokens_ptr = ctypes.cast(tokens_addr, ctypes.POINTER(ctypes.c_uint32))

    if merged_num_chunks == 0:
        return (np.array([], dtype=np.uint32), np.array([0], dtype=np.int64))
    
    merged_offsets = np.ctypeslib.as_array(py_merged_offsets_ptr, shape=(merged_num_chunks+1,))
    merged_tokens_flat = np.ctypeslib.as_array(py_tokens_ptr, shape=(_MERGED_LENGTH,))

    # 打包, pack merged info with batch order
    return (merged_tokens_flat, merged_offsets)