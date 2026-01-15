#include <stdexcept>
#include <cstddef>
#include <iostream>
#include <cstdint>
#include <atomic>
#include "mp_pair_count_merge.h"
#include "memory_pool_singleton.h"
#include "memory_pool.h"
// #include "mempooled_counter.h"
#include "mempooled_concurrent_hashtable.h"
#include "mempooled_hashtable.h"

// 概念解析:
// 静态存储期：程序启动时创建，结束时销毁，存储在一个非栈、非普通堆的地方。python调用时, import .so文件时即被创建
// 外部链接性：可以通过 extern 关键字被其他 .cpp 文件引用
// 内部链接性：其他 .cpp 文件无法通过 extern 访问

// 全局变量：在所有函数外部定义的变量，不属于任何类或命名空间, 作用域为整个程序  <--- 静态存储期 + 外部链接性
// int global_var = 42;
// 静态全局变量：在函数外部，用 static 修饰的变量. 实现“文件私有”的全局状态，作用域为当前文件, 避免命名冲突 <--- 静态存储期 + 内部链接性
// static int static_global = 100;
// 静态局部变量：在函数内部，用 static 修饰的变量. 作用域仅为函数内部. 只初始化一次(即函数调用时) <--- 静态存储期，无任何链接性, 出了函数就不可访问.
// void func() {
//     static int count = 0;  // 静态局部变量
//     ++count;
//     std::cout << count << "\n";
// }

// namespace 定义作用域, 在里面声明的变量函数类, 不会污染全局作用域, 要用显式调用 name::func
// 匿名的命名空间, 意思是里面定义的链接仅限本.cpp文件使用, 实现“文件私有”的效果，避免命名冲突，不会暴露给其他编译单元.
// 且里面的定义等价于 静态static，它具有内部链接 ---> 所以语义上就是“静态全局变量”，但比 static 更现代。
// 所以相当于静态对象也一并在这里被定义并初始化, 比如 进程的单例内存池/基于单例内存池的可复用计数器/原子变量 g_inited 以判断是否需要初始化前二者
// 保证进程内有各自所需的静态对象，更不容易被误用
namespace {

    counter_st* local_counter = nullptr;
    // counter_mt* g_counter_mt = nullptr;
    
    std::atomic<bool> g_inited{false};

    /* 默认构造哈希器 pair_hasher */
    hasher pair_hasher;

}


// Cython可以正常调用 extern C 内部的 c_count_pair_batch: 接口层必备 extern "C"
extern "C" {


// 创建进程的单例内存池 / 基于该单例内存池的可复用计数器
void init_process(size_t block_size, size_t alignment, size_t capacity) {
    // 只在子进程内调用: 允许多次调用，但只有第一次真正执行初始化，根据原子变量 g_inited 执行
    bool expected = false;
    if (g_inited.compare_exchange_strong(expected, true)) {
        /* 初始化 单例内存池（进程内）/ 基于单例内存池的 可复用计数器 */
        
        singleton_mempool::get(block_size, alignment);
        local_counter = new counter_st(pair_hasher, capacity, singleton_mempool::get());

        // threadsafe_singleton_mempool::get(block_size, alignment);
        // g_counter_mt = new counter_mt(pair_hasher, capacity, threadsafe_singleton_mempool::get());

        const size_t BYTES_IN_GB = 1024ULL * 1024ULL * 1024ULL;
        std::cout << "singleton memory pool with " << block_size/BYTES_IN_GB << "GB initialized" << std::endl;
    }

}


// 重置进程的单例内存池 / 基于该单例内存池的可复用计数器，使得它们处于可复用状态
void reset_process() {

    if (local_counter) local_counter->clear();
    // if (g_counter_mt) g_counter_mt->clear();
    
    if (singleton_mempool::exist()) {
        singleton_mempool::get().shrink();
        singleton_mempool::get().reset();
    }
    // 线程安全版本的略过
    // if (threadsafe_singleton_mempool::exist()) {
    //     threadsafe_singleton_mempool::get().shrink();
    //     threadsafe_singleton_mempool::get().reset();
    // }
}


// 销毁进程的单例内存池 / 基于该单例内存池的可复用计数器，准备退出程序
void release_process() {
    // 先销毁 计数器. 由于 计数器是静态的,存在复用的可能性, 故 delete 后必须要置空指针，以防止UAF / 二次delete
    if (local_counter) { delete local_counter; local_counter = nullptr; }
    if (singleton_mempool::exist()) { singleton_mempool::destroy(); }

    // if (g_counter_mt) { delete g_counter_mt; g_counter_mt = nullptr; }
    // if (threadsafe_singleton_mempool::exist()) { threadsafe_singleton_mempool::destroy(); }

    std::cout << "singleton memory pool released" << std::endl;

    // 复位 g_inited 允许重新初始化
    g_inited.store(false, std::memory_order_relaxed);
}



u16token_pair_counts_ptrs c_local_count_u16pair_batch(
    const uint16_t* L_tokens,
    const uint16_t* R_tokens,
    const size_t len
) {
    try
    {
        if (len <= 0) {
            return u16token_pair_counts_ptrs{nullptr, nullptr, nullptr, 0};
        }

        auto& pool = singleton_mempool::get();
        // keys 数组储存 len 个 L(uint16)R(uint16) 组成的 uint32
        uint32_t* keys = static_cast<uint32_t*>(pool.allocate(len*sizeof(uint32_t)));

        for (size_t i = 0; i < len; ++i) {
            const uint32_t l = static_cast<uint32_t>(L_tokens[i]) & 0xFFFFu;
            const uint32_t r = static_cast<uint32_t>(R_tokens[i]) & 0xFFFFu;
            keys[i] = ( l<<16 ) | r;
        }
        
        // 根据是否有 counter 来决定使用哪一个计数函数
        if (local_counter) {
            u16token_pair_counts_ptrs result = local_dict_count_u16pair_core(
                keys,
                len,
                pool,
                local_counter
            );
            return result;
        }
        else {
            u16token_pair_counts_ptrs result = local_sort_count_u16pair_core(
                keys,
                len,
                pool
            );
            return result;
        }
    }
    catch(const std::exception& e)
    {
        throw std::runtime_error("Error in c_local_count_u16pair_batch");
    }
}


merged_u16token_filter_len_ptrs c_local_merge_u16pair_batch(
    const uint16_t* tokens_flat,
    const int64_t* offsets,
    const size_t num_chunks, // num_chunks = len(offsets) - 1
    const uint16_t pair_L,
    const uint16_t pair_R,
    const uint16_t new_token
) {
    try
    {
        auto& pool = singleton_mempool::get();

        merged_u16token_filter_len_ptrs result = local_merge_u16pair_core(
            tokens_flat,
            offsets,
            num_chunks,
            pair_L,
            pair_R,
            new_token,
            pool
        );

        return result;
    }
    catch(const std::exception& e)
    {
        throw std::runtime_error("Error in c_local_merge_u16pair_batch");
    }
}




merged_u16token_offset_ptrs c_local_merge_u16pair_batch_v2(
    const uint16_t* tokens_flat,
    const int64_t* offsets,
    const size_t num_chunks, // num_chunks = len(offsets) - 1
    const uint16_t pair_L,
    const uint16_t pair_R,
    const uint16_t new_token,
    const bool if_filter_len1
) {
    try
    {
        auto& pool = singleton_mempool::get();

        merged_u16token_offset_ptrs result = local_merge_u16pair_core_v2(
            tokens_flat,
            offsets,
            num_chunks,
            pair_L,
            pair_R,
            new_token,
            pool
        );

        if(if_filter_len1) {
            /*过滤 merge 后 length = 1 的chunk*/
        }

        return result;
    }
    catch(const std::exception& e)
    {
        throw std::runtime_error("Error in c_local_merge_u16pair_batch");
    }
}


} // end extern "C"