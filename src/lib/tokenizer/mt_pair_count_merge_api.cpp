// multi-thread version of pair count/merge related functions
#include <stdexcept>
#include <cstddef>
#include <iostream>
#include <cstdint>
#include <atomic>
#include "mt_pair_count_merge.h"
#include "optional"

namespace {
    thread_local std::optional<mempool> tls_pool;

    thread_local counter_st* counter_tls = nullptr;

    static hasher pair_hasher;

}



extern "C" {


void init_tls_pool(size_t block_size, size_t alignment) {
    /* 初始化 thread-local-storage 内存池 */
    if (!tls_pool.has_value()) {
        tls_pool.emplace(block_size, alignment);
    }
}

mempool& get_tls_pool() {
    /* 若 thread-local-storage 内存池不存在, 报错 */
    if (!tls_pool.has_value()) {
        throw std::runtime_error("thread-local memory pool not initialzied.");
    }
    return *tls_pool;
}


// 创建线程的 thread-local-storage 内存池 / 基于该内存池的可复用计数器
void init_thread(size_t block_size, size_t alignment, size_t capacity) {

    /* 初始化 thread-local-storage 内存池 */
    init_tls_pool(block_size, alignment);

    /* 基于该内存池的 可复用计数器 */
    if (!counter_tls) {
        counter_tls = new counter_st(pair_hasher, capacity, get_tls_pool());

        const size_t BYTES_IN_GB = 1024ULL * 1024ULL * 1024ULL;
        std::cout << "thread-local memory pool with " << block_size/BYTES_IN_GB << "GB initialized" << std::endl;
    }

}

// 重置线程的 thread-local-storage 内存池 / 基于该内存池的可复用计数器，使得它们处于可复用状态
void reset_thread() {

    if (counter_tls) counter_tls->clear();
    get_tls_pool().shrink();
    get_tls_pool().reset();
}

// python的线程池不提供类似进程池的 atexit / finalize 等注册机制.
// 理论上 OS 在结束线程时, 会自动调用 thread-local-storage 的对象的析构, 并释放线程的资源. 故 counter_st 和 mempool 理论上会被自动析构释放
// 但实际上, 若线程非正常退出(比如被 os._exit / kill -9 / python crash 等), 所以 mempool 的析构应该设计可容忍 crash
void release_thread() {
    if (counter_tls) { delete counter_tls; counter_tls = nullptr; }
    
    //  OS级线程批处理运行, 不需要这个: OS会自动收回分配的资源. 不过这里还是写
    get_tls_pool().release();
}



u32token_pair_counts_ptrs c_tls_count_u32pair_batch(
    const uint32_t* L_tokens,
    const uint32_t* R_tokens,
    const size_t len
) {
    try
    {
        if (len <= 0) {
            return u32token_pair_counts_ptrs{nullptr, nullptr, nullptr, 0};
        }

        auto& pool = get_tls_pool();
        // keys 数组储存 len 个 L(uint32)R(uint32) 组成的 uint64
        uint64_t* keys = static_cast<uint64_t*>(pool.allocate(len*sizeof(uint64_t)));

        for (size_t i = 0; i < len; ++i) {
            const uint64_t l = static_cast<uint64_t>(L_tokens[i]);
            const uint64_t r = static_cast<uint64_t>(R_tokens[i]);
            keys[i] = ( l << 32 ) | r;
        }
        
        // 根据是否有 counter 来决定使用哪一个计数函数
        if (counter_tls) {
            u32token_pair_counts_ptrs result = tls_dict_count_u32pair_core(
                keys,
                len,
                pool,
                counter_tls
            );
            return result;
        }
        else {
            u32token_pair_counts_ptrs result = tls_sort_count_u32pair_core(
                keys,
                len,
                pool
            );
            return result;
        }
    }
    catch(const std::exception& e)
    {
        throw std::runtime_error("Error in c_tls_count_u32pair_batch");
    }
}



merged_u32token_offset_ptrs c_tls_merge_u32pair_batch(
    const uint32_t* tokens_flat,
    const int64_t* offsets,
    const size_t num_chunks, // num_chunks = len(offsets) - 1
    const uint32_t pair_L,
    const uint32_t pair_R,
    const uint32_t new_token,
    const bool if_filter_len1
) {
    try
    {
        auto& pool = get_tls_pool();

        auto [merged_tokens_flat_ptr, merged_offsets_ptr] = tls_merge_u32pair_core(
            tokens_flat,
            offsets,
            num_chunks,
            pair_L,
            pair_R,
            new_token,
            if_filter_len1,
            pool
        );

        if(if_filter_len1) {
            /*过滤 merge 后 length = 1 的chunk*/
            int64_t* merged_filtered_offsets = static_cast<int64_t*>(pool.allocate((num_chunks+1)*sizeof(int64_t)));

            size_t j = 0;
            for(size_t i = 0; i < num_chunks; i++) {
                if(merged_offsets_ptr[i] != merged_offsets_ptr[i+1]) {
                    merged_filtered_offsets[j] = merged_offsets_ptr[i];
                    ++j;
                }
            }
            merged_filtered_offsets[j] = merged_offsets_ptr[num_chunks];

            return merged_u32token_offset_ptrs{merged_tokens_flat_ptr, merged_filtered_offsets, j, merged_filtered_offsets[j]};
        }

        return merged_u32token_offset_ptrs{merged_tokens_flat_ptr, merged_offsets_ptr, num_chunks, merged_offsets_ptr[num_chunks]};
    }
    catch(const std::exception& e)
    {
        throw std::runtime_error("Error in c_tls_merge_u32pair_batch");
    }
}


} // end extern "C"

