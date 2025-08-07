// pair_count_merge.h
// statement for pair_count_merge_api.cpp

#ifndef TOKENIZER_H
#define TOKENIZER_H

#include <cstddef>
#include <cstdint>
#include "mempool_counter.h" // 引入 mempool_counter.h 以便访问 counter 模板类
#include "memory_pool_global.h"
#include <vector>
#include "mempool_hash_table_mt.h"
#include "mempool_hash_table_st.h"

using counter_key_type = std::pair<uint16_t, uint16_t>;

// 这里 counter_hasher 是一个函数类
struct hasher_type {
    size_t operator()(const counter_key_type& pair) const {
        return (static_cast<size_t>(pair.first) << 16) | pair.second;
    }
};

using counter_st = counter<counter_key_type, false, global_mempool, hasher_type>;
using counter_mt = counter<counter_key_type, true, global_mempool, hasher_type>;


// 声明全局变量
extern hasher_type pair_hasher; // 全局使用的哈希器
extern counter_st* global_counter_st;
extern counter_mt* global_counter_mt;


extern "C" {


// 创建全局内存池
void init_global_mempool(size_t block_size, size_t alignment);


// 创建全局counter
void init_global_counter(size_t capacity, int num_threads);


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


// 重置counter准备下一个epoch复用. 要放在 内存池重置前面使用
void reset_global_counter();


// 重置内存池准备merge pair复用
void reset_globabl_mempool();


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


// 缩小内存池
void shrink_global_mempool();


// 销毁计数器
void delete_global_counter();


// 销毁内存池
void release_global_mempool();


} // end of extern C


#endif