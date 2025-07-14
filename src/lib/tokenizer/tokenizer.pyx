# distutils: language = c++
# cython: language_level=3, boundscheck=True, wraparound=True

import numpy as np
cimport numpy as np

np.import_array()

cdef extern from "tokenizer.h":
    void c_merge_pair_batch(
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



def merge_pair_batch_parallel(
    memoryview tokens_flat, # memoryview of int32
    memoryview offsets, # memoryview of int64
    pair_L, # int32
    pair_R, # int32
    new_token, # int32
    **kwargs
):
    '''
    Cython version `merge_pair_batch_parallel`
    '''
    cdef int num_chunks = offsets.shape[0] - 1
    if num_chunks <= 0:
        return np.array([], dtype=np.int32), np.array([0], dtype=np.int64)
    
    cdef long _length = tokens_flat.shape[0] # token_flat's total length

    cdef int[:] tokens_flat_view = <int[:_length]> tokens_flat
    cdef long[:] offsets_view = <long[:num_chunks+1]> offsets

    # get input ptr from memoryview input(zero-copy)
    cdef int* tokens_flat_ptr = &tokens_flat_view[0]
    cdef long* offsets_ptr = &offsets_view[0]

    # get output ptr from allocate memory
    cdef int* output_tokens_flat = <int*>calloc(_length, sizeof(int))
    cdef bool* output_filter = <bool*>calloc(_length, sizeof(bool))

    # output_tokens_lens Initialized as tokens lens from offsets
    cdef long* output_tokens_lens = <long*>calloc(num_chunks, sizeof(long))
    for i in range(num_chunks):
        output_tokens_lens[i] = offsets_view[i+1] - offsets_view[i]
    
    # deploy cpp function
    c_merge_pair_batch(
        tokens_flat_ptr,
        offsets_ptr,
        num_chunks,
        pair_L,
        pair_R,
        new_token,
        output_tokens_flat,
        output_filter,
        output_tokens_lens
    )

    # build output tokens array from memory output_tokens_flat/output_filter
    # build output tokens offsets from memory output_tokens_lens




