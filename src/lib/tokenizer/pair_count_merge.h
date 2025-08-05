// pair_count_merge.H
// statement for pair_count_merge_api.cpp

#ifndef TOKENIZER_H
#define TOKENIZER_H

#include <cstddef>
#include <cstdint>
#include "mempool_counter.h" // 引入 mempool_counter.h 以便访问 counter_mt/counter_st 类



extern "C" {

using counter_st = counter<std::pair<uint16_t, uint16_t>, false>;
using counter_mt = counter<std::pair<uint16_t, uint16_t>, true>;

// 结构体，用于封装count_pair_batch函数返回的多个data指针, 和(L,R) pair-freq 总数
struct L_R_token_counts_ptrs {
    uint16_t* L_tokens_ptr;
    uint16_t* R_tokens_ptr;
    uint64_t* counts_ptr;
    size_t size;
};


L_R_token_counts_ptrs c_count_pair_batch(
    const uint16_t* L_tokens,
    const uint16_t* R_tokens,
    const int64_t len,
    const int num_threads
);


void count_pair_core_multi_thread(
    counter_mt& counter,
    const uint16_t* L_tokens,
    const uint16_t* R_tokens,
    const int64_t len,
    const int num_threads
);


void count_pair_core_single_thread(
    counter_st& counter,
    const uint16_t* L_tokens,
    const uint16_t* R_tokens,
    const int64_t len
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