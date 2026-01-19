// multi-thread version of pair count/merge related functions
#include <stdexcept>
#include <cstddef>
#include <iostream>
#include <cstdint>
#include <atomic>
#include "mt_pair_count_merge.h"


thread_local counter_st* counter_tls = nullptr;

static hasher pair_hasher;


extern "C" {

// 创建线程的 thread-local-storage 内存池 / 基于该内存池的可复用计数器
void init_thread(size_t block_size, size_t alignment, size_t capacity) {

    if (counter_tls == nullptr) {
        /* 初始化 thread-local-storage 内存池 / 基于该内存池的 可复用计数器 */

        thread_local mempool pool(block_size, alignment);

        counter_tls = new counter_st(pair_hasher, capacity, pool);

        const size_t BYTES_IN_GB = 1024ULL * 1024ULL * 1024ULL;
        std::cout << "thread-local memory pool with " << block_size/BYTES_IN_GB << "GB initialized" << std::endl;
    }

}

// 重置线程的 thread-local-storage 内存池 / 基于该内存池的可复用计数器，使得它们处于可复用状态
void reset_thread() {

    if (counter_tls) counter_tls->clear();
    
    thread_local pool 
}



} // end extern "C"

