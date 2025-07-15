#include <stdexcept>
#include <cstddef>
#include <iostream>
#include "tokenizer.h"
#include "memory_pool.h"


extern "C" {

return_bundle c_merge_pair_batch(
    const int* tokens_flat,
    const long* offsets,
    size_t num_chunks, // num_chunks = len(offsets) - 1
    int pair_L,
    int pair_R,
    int new_token
) {
    try
    {
        size_t size = 1<<20;

        // num_chunks = len(offsets) - 1 = len(output_tokens_lens)
        long* output_tokens_lens = static_cast<long*>(memory_pool::get_mempool().allocate(size));

        // 初始化数组
        for (size_t i = 0; i < num_chunks; ++i) {
            output_tokens_lens[i] = offsets[i+1] - offsets[i];
        }

        // offsets 的最后一个值是 tokens_flat 的长度，也是 output_tokens_flat/output_filter 的长度
        long _LENGTH = offsets[num_chunks];

        // _LENGTH 长度
        bool* output_filter = static_cast<bool*>(memory_pool::get_mempool().allocate(size));
        for (size_t i = 0; i < _LENGTH; ++i) {
            output_filter[i] = false; // 全部初始化为 false
        }

        // _LENGTH 长度
        int* output_tokens_flat = static_cast<int*>(memory_pool::get_mempool().allocate(size));
        for (size_t i = 0; i < _LENGTH; ++i) {
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
        
        return_bundle result = return_bundle{output_tokens_flat, output_filter, output_tokens_lens};

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
    std::cout << "global memory pool initialized" << std::endl;
}




// 缩小内存池
void shrink_memory_pool() {
    memory_pool::get_mempool().shrink();
}



// 重置内存池
void reset_memory_pool() {
    memory_pool::get_mempool().reset();
}



// 销毁内存池
void release_memory_pool() {
    memory_pool::get_mempool().release();
}

} // end extern "C"