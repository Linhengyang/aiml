#include <stdexcept>
#include <cstddef>
#include <iostream>
#include <cstdint>
#include "pair_count_merge.h"
#include "memory_pool_global.h"


// namespace 定义作用域, 在里面声明的变量函数类, 不会污染全局作用域, 要用显式调用 name::func
// 匿名的命名空间, 意思是里面定义的链接仅限本.cpp文件使用, 不会暴露给其他编译单元. Cython可以正常调用 extern C 内部的 c_count_pair_batch
namespace {
    template<typename CounterT>
    L_R_token_counts_ptrs extract_result_from_counter(CounterT& counter) {
        size_t size = counter.size();
        auto& pool = memory_pool::get_mempool();

        uint16_t* L = static_cast<uint16_t*>(pool.allocate(size*sizeof(uint16_t)));
        uint16_t* R = static_cast<uint16_t*>(pool.allocate(size*sizeof(uint16_t)));
        uint64_t* counts = static_cast<uint64_t*>(pool.allocate(size*sizeof(uint64_t)));

        size_t i = 0;
        for(auto it = counter.cbegin(); it != counter.cend(); ++it) {
            auto [pair, freq] = *it;
            L[i] = pair.first;
            R[i] = pair.second;
            counts[i] = freq;
            ++i;
        }

        return L_R_token_counts_ptrs{L, R, counts, size};
    }
}


extern "C" {

L_R_token_counts_ptrs c_count_pair_batch(
    const uint16_t* L_tokens,
    const uint16_t* R_tokens,
    const int64_t len,
    const int num_threads
) {
    try
    {
        // 不同的分支下, counter 是不同的类型, 所以没办法把 extract_result 部分统一到外部使用
        if (num_threads == 1) {
            counter_st counter = counter_st(1024, memory_pool::get_mempool());
            count_pair_core_single_thread(counter, L_tokens, R_tokens, len);

            return extract_result_from_counter(counter);
        }
        else {
            counter_mt counter = counter_mt(1024, memory_pool::get_mempool());
            count_pair_core_multi_thread(counter, L_tokens, R_tokens, len, num_threads);

            return extract_result_from_counter(counter);
        }

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
        // num_chunks = len(offsets) - 1 = len(output_tokens_lens)
        // need size = sizeof(long) * num_chunks
        int64_t* output_tokens_lens = static_cast<int64_t*>(global_mempool::get().allocate(num_chunks*sizeof(int64_t)));

        // 初始化数组
        for (size_t i = 0; i < num_chunks; ++i) {
            output_tokens_lens[i] = offsets[i+1] - offsets[i];
        }

        // offsets 的最后一个值是 tokens_flat 的长度，也是 output_tokens_flat/output_filter 的长度
        int64_t _LENGTH = offsets[num_chunks];

        // _LENGTH 长度
        // need size = sizeof(bool) * _LENGTH
        bool* output_filter = static_cast<bool*>(global_mempool::get().allocate(_LENGTH*sizeof(bool)));
        for (int64_t i = 0; i < _LENGTH; ++i) {
            output_filter[i] = false; // 全部初始化为 false
        }

        // _LENGTH 长度
        // need size = sizeof(int) * _LENGTH
        uint16_t* output_tokens_flat = static_cast<uint16_t*>(global_mempool::get().allocate(_LENGTH*sizeof(uint16_t)));
        for (int64_t i = 0; i < _LENGTH; ++i) {
            output_tokens_flat[i] = -1; // 全部初始化为 -1
        }

        merge_pair_core_parallel(
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



// 创建内存池（全局单例）
void init_memory_pool(size_t block_size, size_t alignment) {
    global_mempool::get(block_size, alignment);
    const size_t BYTES_IN_GB = 1024ULL * 1024ULL * 1024ULL;
    std::cout << "global memory pool with " << block_size/BYTES_IN_GB << "GB initialized" << std::endl;
}




// 缩小内存池(如果存在)
void shrink_memory_pool() {
    if (global_mempool::exist()) {
        global_mempool::get().shrink();
    }
}



// 重置内存池
void reset_memory_pool() {
    global_mempool::get().reset();
}



// 销毁内存池
void release_memory_pool() {
    global_mempool::destroy();
    std::cout << "global memory pool released" << std::endl;
}

} // end extern "C"