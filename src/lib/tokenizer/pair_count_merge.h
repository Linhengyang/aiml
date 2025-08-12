// pair_count_merge.h
// statement for pair_count_merge_api.cpp


#pragma once
#include <cstddef>
#include <cstdint>
#include "mempool_counter.h"
#include "memory_pool_singleton.h"
#include "mempool_hash_table_mt.h"
#include "mempool_hash_table_st.h"

// 定义 counter_key_type
using counter_key_type = std::pair<uint16_t, uint16_t>;

// 定义从一般输入 到 counter_key_type 的 构造. 这里可以直接使用 counter_key_type
struct key_maker {
    counter_key_type operator()(const uint16_t& L, const uint16_t& R) const {
        return counter_key_type(L, R);
    }
};

// 定义哈希 counter_key 的哈希器. 这里 counter_hasher 是一个函数类, 通过实例化得到哈希器 hasher hasher;
struct hasher {
    size_t operator()(const counter_key_type& pair) const {
        return (static_cast<size_t>(pair.first << 16) | pair.second);
    }
};





// 用一个pair数据结构来代表两个uint16_t数据, 太奢侈了. 实际上 first<<16|second 与 (first,second)是双射:
// uint32_t combo = pair.first << 16 | pair.second; // 这个赋值计算合法
// 在后续 size_t index = combo % _capacity; 也是合法的, 会自动作类型提升
// 逆反射：uint16_t a = combo >> 16; uint16_t b = combo & 0xFFFF; // 这个赋值计算合法



// // 定义 counter_key_type
// using counter_key_type = uint32_t;

// // 定义从一般输入 到 counter_key_type 的 构造器
// struct key_maker {
//     counter_key_type operator()(const uint16_t& L, const uint16_t& R) const {
//         return L << 16 | R;
//     }
// };

// // 定义哈希 counter_key 的哈希器. 这里 hasher 是一个函数类, 通过实例化得到哈希器 hasher myHasher;
// struct hasher {
//     uint32_t operator()(const counter_key_type& key) const {
//         return key;
//     }
// };





using counter_st = counter<counter_key_type, false, unsafe_singleton_mempool, hasher>;
using counter_mt = counter<counter_key_type, true, singleton_mempool, hasher>;


// 全局对象在 .SO 被python导入后就存在主进程，python解释器没结束, 全局对象就一直存在且复用
// 所以全局指针 delete 清空之后必须 置空set to nullptr, 不然就成了悬垂指针.
// // 声明全局变量
// extern hasher pair_hasher; // 全局使用的哈希器
// extern counter_st* global_counter_st;
// extern counter_mt* global_counter_mt;

// 全局对象在多进程里不推荐使用：
// 在linux下，多进程的启动方式是folk，子进程通过copy-on-write来继承父进程地址空间的一切，包括全局对象。之后进程之间各用各的副本，互相隔离
// 在windows/macOS下，子进程的启动方式是spawn，子进程会重新import模块，.so重新加载，全局对象会在各子进程里各自初始化，互相隔离
// 但是folk有坑：一是和spawn存在不同的行为（folk可以在主进程初始化好对象后，靠copy传给子进程；spawn做不到）
// 二是folk对锁/线程有陷阱：folk后的std::mutex等可能处于不可预期状态，进而导致子进程卡死

// 多进程的推荐办法：在子进程启动后，进程内初始化一切，特别是带锁/线程的结构；进程退出前统一释放


extern "C" {


// 进程内初始化（只做一次即可，允许重复调用作“已初始化”检查）
void init_process(size_t block_size, size_t alignment, size_t capacity, int num_threads);


// 重置进程的单例内存池 / 基于该单例内存池的可复用计数器，使得它们处于可复用状态
void reset_process();


// 销毁进程的单例内存池 / 基于该单例内存池的可复用计数器，准备退出程序
void release_process();


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
    const uint16_t new_token,
    const int num_threads
);


} // end of extern C
