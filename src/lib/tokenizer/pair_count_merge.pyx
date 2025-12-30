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

cdef extern from "pair_count_merge.h":

    # 初始化进程环境: 单例内存池 / 基于单例内存池的可复用计数器
    void init_process(size_t block_size, size_t alignment, size_t capacity)

    # 重置进程环境：重置单例内存池 / 清空可复用计数器
    void reset_process()

    # 销毁进程的单例内存池 / 基于该单例内存池的可复用计数器，准备退出程序
    void release_process()

    # 声明 C++ 中的 L_R_token_counts_ptrs 结构体
    struct L_R_token_counts_ptrs:
        uint16_t* L_tokens_ptr
        uint16_t* R_tokens_ptr
        uint64_t* counts_ptr
        size_t size;

     # 声明 C++ 中的 c_count_pair_batch 函数
    L_R_token_counts_ptrs c_count_pair_batch(
        const uint16_t* L_tokens,
        const uint16_t* R_tokens,
        const int64_t len
    )

    # 声明 C++ 中的 token_filter_len_ptrs 结构体
    struct token_filter_len_ptrs:
        uint16_t* output_tokens_flat_ptr
        bool* output_filter_ptr
        int64_t* output_tokens_lens_ptr

    # 声明 C++ 中的 c_merge_pair_batch 函数
    token_filter_len_ptrs c_merge_pair_batch(
        const uint16_t* tokens_flat,
        const int64_t* offsets,
        const size_t num_chunks,
        const uint16_t pair_L,
        const uint16_t pair_R,
        const uint16_t new_token
    )







# 创建内存池/计数器接口给python. block_size size_t 从python侧传入, alignment设为64
# 经过测算，output needed size 的chunk_lens/fitler/tokens 所占空间分别大概是 8倍/10倍/20倍 batch_size bytes
# 为了保证tokens在同一个block而不是large alloc, block_size 设定为 40倍 batch_size 比较好
# 计数器capacity: 对于 32000 次merge, 最终pair的种类不超过 32256*32256 = 10亿左右. 
# 初始计数器设在 16384*16384*2 = 2^29 次 = 536870912 就好。这样如果初始分配的capacity不够，一次rehash就差不多就足够了
# 计数器和count_pair_batch使用单线程
cpdef initialize(size_t block_size):
    init_process(block_size, 64, 536870912)





# 只是给python提供了reset进程的接口，但实际上count_pair_batch和merge_pair_batch每一次执行前都reset了
cpdef reset():
    reset_process()






cpdef close():
    release_process()





# 返回np.array of L_tokens/R_tokens/counts 给python
cpdef count_pair_batch(
    object tokens_offsets_border
):
    # reset 进程单例内存池 / 基于单例内存池的计数器
    reset_process()

    cdef np.ndarray[np.uint16_t, ndim=1, mode="c"] tokens_flat = tokens_offsets_border[0][0]
    cdef np.ndarray[np.int64_t, ndim=1, mode="c"] offsets = tokens_offsets_border[0][1]

    cdef int64_t _LENGTH = tokens_flat.shape[0] # token_flat's total length
    if _LENGTH != offsets[-1]:
        sys.exit(1)
    
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

    # 检查 L_tokens 和 R_tokens 长度.
    cdef size_t len = L_tokens.shape[0]
    if len != R_tokens.shape[0]:
        sys.exit(1)
    
    if len == 0:
        return (pynp.array([], dtype=pynp.uint16),
                pynp.array([], dtype=pynp.uint16),
                pynp.array([], dtype=pynp.uint64)), tokens_offsets_border[1]

    cdef const uint16_t[:] L_tokens_view = L_tokens
    cdef const uint16_t* L_tokens_ptr = &L_tokens_view[0]

    cdef const uint16_t[:] R_tokens_view = R_tokens
    cdef const uint16_t* R_tokens_ptr = &R_tokens_view[0]
    
    # 调用 c_count_pair_batch
    cdef L_R_token_counts_ptrs result = c_count_pair_batch(
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
    return (output_L_tokens_np, output_R_tokens_np, output_counts_np), tokens_offsets_border[1]



# 返回np.array of merged_tokens_flat/offsets给python
cpdef merge_pair_batch(
    object tokens_offsets_border,
    np.uint16_t pair_L,
    np.uint16_t pair_R,
    np.uint16_t new_token,
):
    # reset 进程单例内存池 / 基于单例内存池的计数器
    reset_process()

    # 得到 tokens flattened
    cdef np.ndarray[np.uint16_t, ndim=1, mode="c"] tokens_flat = tokens_offsets_border[0][0]
    # 得到 offsets
    cdef np.ndarray[np.int64_t, ndim=1, mode="c"] offsets = tokens_offsets_border[0][1]

    cdef size_t num_chunks = offsets.shape[0] - 1
    if num_chunks <= 0:
        return (pynp.array([], dtype=pynp.uint16), pynp.array([0], dtype=pynp.int64)), tokens_offsets_border[1]
    
    cdef int64_t _LENGTH = tokens_flat.shape[0] # token_flat's total length
    if _LENGTH != offsets[num_chunks]:
        sys.exit(1)
    
    # const uint16_t[::1]保证 memoryview是只读+内存连续的
    # 因为tokens_flat来自 np.array(..., dtype=..., copy=False) 共享了只读数据
    cdef const uint16_t[:] tokens_flat_view = tokens_flat
    cdef const int64_t[:] offsets_view = offsets

    # get input ptr from memoryview input(zero-copy)
    cdef const uint16_t* tokens_flat_ptr = &tokens_flat_view[0]
    cdef const int64_t* offsets_ptr = &offsets_view[0]

    # deploy cpp function
    cdef token_filter_len_ptrs result = c_merge_pair_batch(
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
    return (valid_merged_tokens_flat, valid_merged_offsets), tokens_offsets_border[1]
