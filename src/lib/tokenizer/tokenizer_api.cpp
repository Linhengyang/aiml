#include <stdexcept>
#include <cstddef>
#include "tokenizer.h"
#include "../share/memory_pool.h"


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
        // num_chunks = len(offsets) - 1 = len(output_tokens_lens)
        long* output_tokens_lens = nullptr; //TODO, allocate from memory_pool::get_mempool().allocate()

        // 初始化数组
        for (size_t i = 0; i < num_chunks; ++i) {
            output_tokens_lens[i] = offsets[i+1] - offsets[i];
        }

        // offsets 的最后一个值是 tokens_flat 的长度，也是 output_tokens_flat/output_filter 的长度
        long _LENGTH = offsets[num_chunks];

        // _LENGTH 长度
        bool* output_filter = nullptr; //TODO, allocate from memory_pool::get_mempool().allocate()
        for (size_t i = 0; i < _LENGTH; ++i) {
            output_filter[i] = false; // 全部初始化为 false
        }

        // _LENGTH 长度
        int* output_tokens_flat = nullptr; //TODO, allocate from memory_pool::get_mempool().allocate()
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



// 重置内存池
void reset_memory_pool() {
    memory_pool::get_mempool().reset();
}



// 销毁内存池
void release_memory_pool() {
    memory_pool::get_mempool().release();
}

} // end extern "C"