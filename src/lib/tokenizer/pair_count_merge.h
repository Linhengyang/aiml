// pair_count_merge.H
// statement for pair_count_merge_api.cpp

#ifndef TOKENIZER_H
#define TOKENIZER_H
#include "memory_pool.h"  // 引入 memory_pool.h 以便访问 MemoryPool 类
#include <cstddef>
#include <cstdint>




extern "C" {

// 结构体，用于封装count_pair_batch函数返回的多个data指针
struct L_R_token_counts_ptrs {
    uint16_t* L_tokens_ptr;
    uint16_t* R_tokens_ptr;
    uint64_t* counts_ptr;
};


L_R_token_counts_ptrs c_count_pair_batch(
    const uint16_t* L_tokens,
    const uint16_t* R_tokens,
    const int num_threads
);


void count_pair_core_threadsafe(
    const uint16_t* L_tokens,
    const uint16_t* R_tokens,
    const int num_threads
);


// 结构体，用于封装merge_pair_batch函数返回的多个data指针
struct token_filter_len_ptrs {
    uint16_t* output_tokens_flat_ptr;
    bool* output_filter_ptr;
    int64_t* output_tokens_lens_ptr;
};


token_filter_len_ptrs c_merge_pair_batch(
    const uint16_t* tokens_flat,
    const int64_t* offsets,
    const size_t num_chunks,
    const uint16_t pair_L,
    const uint16_t pair_R,
    const uint16_t new_token
);


void merge_pair_core_parallel(
    const uint16_t* tokens_flat,
    const int64_t* offsets,
    const size_t num_chunks,
    const uint16_t pair_L,
    const uint16_t pair_R,
    const uint16_t new_token,
    uint16_t* output_tokens_flat, // all -1 init. in-place change in this function
    bool* output_filter, // all false init. in-place change in this function
    int64_t* output_tokens_lens // input tokens lens init. in-place change in this function
);


// 创建内存池
void init_memory_pool(size_t block_size, size_t alignment);


// 缩小内存池
void shrink_memory_pool();


// 重置内存池
void reset_memory_pool();


// 销毁内存池
void release_memory_pool();


}



#endif