#include <stdexcept>
#include <cstddef>
#include <iostream>
#include <cstdint>
#include <atomic>
#include "pair_count_merge.h"
#include "memory_pool_singleton.h"
#include "memory_pool.h"
#include "mempool_counter.h"
#include "mempool_hash_table_mt.h"
#include "mempool_hash_table_st.h"


// namespace 定义作用域, 在里面声明的变量函数类, 不会污染全局作用域, 要用显式调用 name::func
// 匿名的命名空间, 意思是里面定义的链接仅限本.cpp文件使用, 不会暴露给其他编译单元. Cython可以正常调用 extern C 内部的 c_count_pair_batch
// 进程内的静态对象也一并在这里定义, 比如 进程的单例内存池 / 基于单例内存池的可复用计数器 / 原子变量 g_inited 以判断是否需要初始化前二者
// 保证进程内有各自所需的静态对象，更不容易被误用
namespace {

    counter_st* g_counter_st = nullptr;
    counter_mt* g_counter_mt = nullptr;
    
    std::atomic<bool> g_inited{false};

    /* 默认构造哈希器 pair_hasher */
    hasher pair_hasher;

}



extern "C" {


// 创建进程的单例内存池 / 基于该单例内存池的可复用计数器
void init_process(size_t block_size, size_t alignment, size_t capacity) {
    // 只在子进程内调用: 允许多次调用，但只有第一次真正执行初始化，根据原子变量 g_inited 执行
    bool expected = false;
    if (g_inited.compare_exchange_strong(expected, true)) {
        // 初始化 单例内存池（进程内）/ 基于单例内存池的 可复用计数器
        unsafe_singleton_mempool::get(block_size, alignment);
        g_counter_st = new counter_st(pair_hasher, capacity, unsafe_singleton_mempool::get());

        singleton_mempool::get(block_size, alignment);
        g_counter_mt = new counter_mt(pair_hasher, capacity, singleton_mempool::get());

        const size_t BYTES_IN_GB = 1024ULL * 1024ULL * 1024ULL;
        std::cout << "global memory pool with " << block_size/BYTES_IN_GB << "GB initialized" << std::endl;
    }

}


// 重置进程的单例内存池 / 基于该单例内存池的可复用计数器，使得它们处于可复用状态
void reset_process() {

    if (g_counter_st) g_counter_st->clear();
    if (g_counter_mt) g_counter_mt->clear();
    
    if (unsafe_singleton_mempool::exist()) {
        unsafe_singleton_mempool::get().shrink();
        unsafe_singleton_mempool::get().reset();
    }
    // 线程安全版本的略过
    if (singleton_mempool::exist()) {
        singleton_mempool::get().shrink();
        singleton_mempool::get().reset();
    }
}


// 销毁进程的单例内存池 / 基于该单例内存池的可复用计数器，使得它们处于可复用状态
void release_process() {
    // 先销毁 计数器. delete 后必须要置空指针，以防止UAF / 二次delete
    if (g_counter_st) { delete g_counter_st; g_counter_st = nullptr; }
    if (unsafe_singleton_mempool::exist()) { unsafe_singleton_mempool::destroy(); }

    if (g_counter_mt) { delete g_counter_mt; g_counter_mt = nullptr; }
    if (singleton_mempool::exist()) { singleton_mempool::destroy(); } 

    std::cout << "global memory pool released" << std::endl;

    // 复位 g_inited 允许重新初始化
    g_inited.store(false, std::memory_order_relaxed);
}



L_R_token_counts_ptrs c_count_pair_batch(
    const uint16_t* L_tokens,
    const uint16_t* R_tokens,
    const int64_t len
) {
    try
    {
        auto& pool = unsafe_singleton_mempool::get();
        // keys 数组储存 len 个 L(uint16)R(uint16) 组成的 uint32
        uint32_t* keys = static_cast<uint32_t*>(pool.allocate(len*sizeof(uint32_t)));
        for (int64_t i = 0; i < len; ++i) {
            keys[i] == uint32_t(L_tokens[i] << 16 | R_tokens[i]);
        }

        count_pair_core(
            g_counter_st,
            keys,
            len
        );
        size_t size = g_counter_st->size();
        uint16_t* L_uniq = static_cast<uint16_t*>(pool.allocate(size*sizeof(uint16_t)));
        uint16_t* R_uniq = static_cast<uint16_t*>(pool.allocate(size*sizeof(uint16_t)));
        uint64_t* counts = static_cast<uint64_t*>(pool.allocate(size*sizeof(uint64_t)));
        
        size_t i = 0;
        for (auto it = g_counter_st->cbegin(); it != g_counter_st->cend(); ++it) {
            auto [pair, freq] = *it;
            L_uniq[i] = pair >> 16;
            R_uniq[i] = pair & 0xFFFF;
            counts[i] = freq;
            ++i;
        }

        return L_R_token_counts_ptrs{L_uniq, R_uniq, counts, size};
    }
    catch(const std::exception& e)
    {
        throw std::runtime_error("Error in c_count_pair_batch");
    }
}



token_filter_len_ptrs c_merge_pair_batch(
    const uint16_t* tokens_flat,
    const int64_t* offsets,
    const size_t num_chunks, // num_chunks = len(offsets) - 1
    const uint16_t pair_L,
    const uint16_t pair_R,
    const uint16_t new_token
) {
    try
    {
        auto& pool = unsafe_singleton_mempool::get();

        // num_chunks = len(offsets) - 1 = len(output_tokens_lens)
        // need size = sizeof(long) * num_chunks
        int64_t* output_tokens_lens = static_cast<int64_t*>(pool.allocate(num_chunks*sizeof(int64_t)));

        // 初始化数组
        for (size_t i = 0; i < num_chunks; ++i) {
            output_tokens_lens[i] = offsets[i+1] - offsets[i];
        }

        // offsets 的最后一个值是 tokens_flat 的长度，也是 output_tokens_flat/output_filter 的长度
        int64_t _LENGTH = offsets[num_chunks];

        // _LENGTH 长度
        // need size = sizeof(bool) * _LENGTH
        bool* output_filter = static_cast<bool*>(pool.allocate(_LENGTH*sizeof(bool)));
        for (int64_t i = 0; i < _LENGTH; ++i) {
            output_filter[i] = false; // 全部初始化为 false
        }

        // _LENGTH 长度
        // need size = sizeof(int) * _LENGTH
        uint16_t* output_tokens_flat = static_cast<uint16_t*>(pool.allocate(_LENGTH*sizeof(uint16_t)));
        for (int64_t i = 0; i < _LENGTH; ++i) {
            output_tokens_flat[i] = -1; // 全部初始化为 -1
        }

        merge_pair_core(
            tokens_flat,
            offsets,
            num_chunks,
            pair_L,
            pair_R,
            new_token,
            output_tokens_flat,
            output_filter,
            output_tokens_lens
        );
        
        token_filter_len_ptrs result = token_filter_len_ptrs{output_tokens_flat, output_filter, output_tokens_lens};

        return result;
    }
    catch(const std::exception& e)
    {
        throw std::runtime_error("Error in c_merge_pair_batch");
    }
}


} // end extern "C"