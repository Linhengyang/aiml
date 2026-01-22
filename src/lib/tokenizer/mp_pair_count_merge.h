// multi-process: mp_pair_count_merge.h
// statement for mp_pair_count_merge_api.cpp


#pragma once
#include <cstddef>
#include <cstdint>
#include "memory_pool_singleton.h"
#include "memory_pool.h"
// #include "mempooled_concurrent_hashtable.h"
// #include "mempooled_hashtable.h"
#include "mempooled_counter.h"


/*
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
*/





// 实际上 first<<16|second 与 (first,second)是双射:
// uint32_t combo = pair.first << 16 | pair.second; // 这个赋值计算合法
// 在后续 size_t index = combo % _capacity; 也是合法的, 会自动作类型提升
// 逆反射：uint16_t a = combo >> 16; uint16_t b = combo & 0xFFFF; // 这个赋值计算合法

// 不用pair of uint16_t，用 uint32_t，是因为 uint32_t 作为 key，可以满足排序计数。性能方面差别不大。 



// 定义 counter_key_type
using counter_key_type = uint32_t;

// // 定义从一般输入 到 counter_key_type 的 构造器
// struct key_maker {
//     counter_key_type operator()(const uint16_t& L, const uint16_t& R) const {
//         return L << 16 | R;
//     }
// };

// 定义哈希 counter_key 的哈希器. 这里 hasher 是一个函数类, 通过实例化得到哈希器 hasher myHasher;
struct hasher {
    size_t operator()(const counter_key_type& key) const {
        return static_cast<size_t>(key); // 无符号整数自身就是很好的哈希值, 无需复杂变换
    }
};

using counter_st = counter<counter_key_type, false, singleton_mempool, hasher>;

// // 实测 多线程并发写同一个全局哈希表，速度非常慢。真正的多线程写法是分数据段+线程独立资源+合并统计。
// using counter_mt = counter<counter_key_type, true, threadsafe_singleton_mempool, hasher>;


/*
全局对象在 .SO 被python导入后就存在主进程，python解释器没结束, 全局对象就一直存在且复用
只要一个指针在被delete之后，可能会被再次使用，就应该在delete之后置空 set to nullptr.
所以全局指针 delete 清空之后必须 置空set to nullptr, 不然就成了未被销毁的悬垂指针. 局部指针倒没关系(delete之后马上要被销毁了所以无所谓)
// 声明全局变量
extern hasher pair_hasher; // 全局使用的哈希器
extern counter_st* global_counter_st;
extern counter_mt* global_counter_mt;



全局对象在多进程里不推荐使用：
在linux下，多进程的启动方式是folk，子进程通过copy-on-write来继承父进程地址空间的一切，包括全局对象。之后进程之间各用各的副本，互相隔离
在windows/macOS下，子进程的启动方式是spawn，子进程会重新import模块，.so重新加载，全局对象会在各子进程里各自初始化，互相隔离
但是folk有坑：一是和spawn存在不同的行为（folk可以在主进程初始化好对象后，靠copy传给子进程；spawn做不到）
二是folk对锁/线程有陷阱：folk后的std::mutex等可能处于不可预期状态，进而导致子进程卡死

多进程的推荐办法：在子进程启动后，进程内初始化一切，特别是带锁/线程的结构；进程退出前统一释放
*/




// 计数器遍历计数的性能低于 排序计数. 舍弃 计数器遍历计数, 采用排序计数


// C++ 编译器会对函数名进行修饰（如 _Z3fooi），而 C 编译器不会（保持 foo）
// 使用 extern "C" 告诉 C++ 编译器：“按 C 的方式处理这些符号”，确保链接成功
extern "C" {


// 进程内初始化（只做一次即可，允许重复调用作“已初始化”检查）
void init_process(size_t block_size, size_t alignment, size_t capacity);


// 重置进程的单例内存池 / 基于该单例内存池的可复用计数器，使得它们处于可复用状态
void reset_process();


// 销毁进程的单例内存池 / 基于该单例内存池的可复用计数器，准备退出程序
void release_process();



// 结构体，用于封装 c_count_u16pair_batch 函数返回的多个data指针, 和(L,R) pair-freq 总数
// 这里的 token 是 uint16_t 类型, 表示范围 0-65535  --> 不适用于超过此规模的 大号词表
struct u16token_pair_counts_ptrs {
    uint16_t* L_tokens_ptr;
    uint16_t* R_tokens_ptr;
    uint64_t* counts_ptr;
    size_t size;
};


// 给单一进程用的 count uint16_t token-pair batch data 的 core: 采用 sort for count
u16token_pair_counts_ptrs local_sort_count_u16pair_core(
    uint32_t* keys,
    const size_t len,
    singleton_mempool& pool
);


// 给单一进程用的 count uint16_t token-pair batch data 的 core: 采用 counter for count
u16token_pair_counts_ptrs local_dict_count_u16pair_core(
    uint32_t* keys,
    const size_t len,
    singleton_mempool& pool,
    counter_st* counter
);


// 给单一进程用的 count uint16_t token-pair batch data 的函数
u16token_pair_counts_ptrs c_local_count_u16pair_batch(
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
};


// 给单一进程用的 merge uint16_t token-pair batch data 的 core
std::pair<uint16_t*, int64_t*> local_merge_u16pair_core(
    const uint16_t* tokens_flat,
    const int64_t* offsets,
    const size_t num_chunks,
    const uint16_t pair_L,
    const uint16_t pair_R,
    const uint16_t new_token,
    const bool if_filter_len1,
    singleton_mempool& pool
);


// 给单一进程用的 merge uint16_t token-pair batch data 的函数
merged_u16token_offset_ptrs c_local_merge_u16pair_batch(
    const uint16_t* tokens_flat,
    const int64_t* offsets,
    const size_t num_chunks,
    const uint16_t pair_L,
    const uint16_t pair_R,
    const uint16_t new_token,
    const bool if_filter_len1
);







// 废弃的 local_merge_u16pair 相关代码
#if 0
struct merged_u16token_filter_len_ptrs {
    uint16_t* output_tokens_flat_ptr;
    bool* output_filter_ptr;
    int64_t* output_tokens_lens_ptr;
};


// 给单一进程用的 merge uint16_t token-pair batch data 的 core
merged_u16token_filter_len_ptrs _deprecated_local_merge_u16pair_core(
    const uint16_t* tokens_flat,
    const int64_t* offsets,
    const size_t num_chunks,
    const uint16_t pair_L,
    const uint16_t pair_R,
    const uint16_t new_token,
    singleton_mempool& pool
);


// 给单一进程用的 merge uint16_t token-pair batch data 的函数
merged_u16token_filter_len_ptrs _deprecated_c_local_merge_u16pair_batch(
    const uint16_t* tokens_flat,
    const int64_t* offsets,
    const size_t num_chunks,
    const uint16_t pair_L,
    const uint16_t pair_R,
    const uint16_t new_token
);
#endif

} // end of extern C
