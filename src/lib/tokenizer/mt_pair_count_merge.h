// multi-thread version of pair count/merge related statements
#pragma once
#include <cstddef>
#include <cstdint>
#include "memory_pool.h"
#include "mempooled_counter.h"








// 定义 counter_key_type
using counter_key_type = uint32_t;

// 定义哈希 counter_key 的哈希器. 这里 hasher 是一个函数类, 通过实例化得到哈希器 hasher myHasher;
struct hasher {
    size_t operator()(const counter_key_type& key) const {
        return static_cast<size_t>(key); // 无符号整数自身就是很好的哈希值, 无需复杂变换
    }
};

using counter_st = counter<counter_key_type, false, mempool, hasher>;


extern "C" {


void init_tls_pool(size_t block_size, size_t alignment);


mempool& get_tls_pool();


// 线程初始化（只做一次即可，允许重复调用作“已初始化”检查）
void init_thread(size_t block_size, size_t alignment, size_t capacity);


// 重置线程局部的 内存池 / 基于该内存池的可复用计数器，使得它们处于可复用状态
void reset_thread();


// 销毁线程局部的 内存池 / 基于该内存池的可复用计数器，准备退出程序.
void release_thread();



// 结构体，用于封装 c_count_u16pair_batch 函数返回的多个data指针, 和(L,R) pair-freq 总数
// 这里的 token 是 uint16_t 类型, 表示范围 0-65535  --> 不适用于超过此规模的 大号词表
struct u16token_pair_counts_ptrs {
    uint16_t* L_tokens_ptr;
    uint16_t* R_tokens_ptr;
    uint64_t* counts_ptr;
    size_t size;
};


// 给单一线程的 thread-local-storage count uint16_t token-pair batch data 的 core: 采用 counter for count
u16token_pair_counts_ptrs tls_dict_count_u16pair_core(
    uint32_t* keys,
    const size_t len,
    mempool& pool,
    counter_st* counter
);


// 给单一线程的 thread-local-storage count uint16_t token-pair batch data 的 core: 采用 sort for count
u16token_pair_counts_ptrs tls_sort_count_u16pair_core(
    uint32_t* keys,
    const size_t len,
    mempool& pool
);


// 给单一线程的 thread-local-storage count uint16_t token-pair batch data 的函数
u16token_pair_counts_ptrs c_tls_count_u16pair_batch(
    const uint16_t* L_tokens,
    const uint16_t* R_tokens,
    const size_t len
);


// 结构体，用于封装 merge_u16pair_core 函数返回的 merged tokens_flat/offsets 指针
// 这里的 token 是 uint16_t 类型, 表示范围 0-65535  --> 不适用于超过此规模的 大号词表
struct merged_u16token_offset_ptrs {
    uint16_t* merged_tokens_flat_ptr;
    int64_t* merged_offsets_ptr;
    size_t merged_num_chunks;
    int64_t merged_num_tokens;
};


// 给单一线程的 thread-local-storage merge uint16_t token-pair batch data 的 core
std::pair<uint16_t*, int64_t*> tls_merge_u16pair_core(
    const uint16_t* tokens_flat,
    const int64_t* offsets,
    const size_t num_chunks,
    const uint16_t pair_L,
    const uint16_t pair_R,
    const uint16_t new_token,
    const bool if_filter_len1,
    mempool& pool
);


// 给单一线程的 thread-local-storage merge merge uint16_t token-pair batch data 的函数
merged_u16token_offset_ptrs c_tls_merge_u16pair_batch(
    const uint16_t* tokens_flat,
    const int64_t* offsets,
    const size_t num_chunks,
    const uint16_t pair_L,
    const uint16_t pair_R,
    const uint16_t new_token,
    const bool if_filter_len1
);


} // end of extern C