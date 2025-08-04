#include <stdexcept>
#include <cstddef>
#include <iostream>
#include <cstdint>
#include "merge_pair.h"
#include "memory_pool.h"


extern "C" {

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
        int64_t* output_tokens_lens = static_cast<int64_t*>(memory_pool::get_mempool().allocate(num_chunks*sizeof(int64_t)));

        // 初始化数组
        for (size_t i = 0; i < num_chunks; ++i) {
            output_tokens_lens[i] = offsets[i+1] - offsets[i];
        }

        // offsets 的最后一个值是 tokens_flat 的长度，也是 output_tokens_flat/output_filter 的长度
        int64_t _LENGTH = offsets[num_chunks];

        // _LENGTH 长度
        // need size = sizeof(bool) * _LENGTH
        bool* output_filter = static_cast<bool*>(memory_pool::get_mempool().allocate(_LENGTH*sizeof(bool)));
        for (int64_t i = 0; i < _LENGTH; ++i) {
            output_filter[i] = false; // 全部初始化为 false
        }

        // _LENGTH 长度
        // need size = sizeof(int) * _LENGTH
        uint16_t* output_tokens_flat = static_cast<uint16_t*>(memory_pool::get_mempool().allocate(_LENGTH*sizeof(uint16_t)));
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
        throw std::runtime_error("Error in merge_pair_core_parallel");
    }
}



// 创建内存池（全局单例）
void init_memory_pool(size_t block_size, size_t alignment) {
    memory_pool::get_mempool(block_size, alignment);
    const size_t BYTES_IN_GB = 1024ULL * 1024ULL * 1024ULL;
    std::cout << "global memory pool with " << block_size/BYTES_IN_GB << "GB initialized" << std::endl;
}




// 缩小内存池(如果存在)
void shrink_memory_pool() {
    if (memory_pool::mempool_exist()) {
        memory_pool::get_mempool().shrink();
    }
}



// 重置内存池
void reset_memory_pool() {
    memory_pool::get_mempool().reset();
}



// 销毁内存池
void release_memory_pool() {
    memory_pool::mempool_destroy();
    std::cout << "global memory pool released" << std::endl;
}

} // end extern "C"