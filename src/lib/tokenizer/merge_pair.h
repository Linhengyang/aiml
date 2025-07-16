// tokenizer.H
// statement for tokenizer.cpp

#ifndef TOKENIZER_H
#define TOKENIZER_H
#include "memory_pool.h"  // 引入 memory_pool.h 以便访问 MemoryPool 类
#include <cstddef>




extern "C" {

// 结构体，用于封装多个函数返回指针
struct return_bundle {
    int* output_tokens_flat_ptr;
    bool* output_filter_ptr;
    long* output_tokens_lens_ptr;
};


return_bundle c_merge_pair_batch(
    const int* tokens_flat,
    const long* offsets,
    const size_t num_chunks,
    const int pair_L,
    const int pair_R,
    const int new_token
);



void merge_pair_core_parallel(
    const int* tokens_flat,
    const long* offsets,
    const size_t num_chunks,
    const int pair_L,
    const int pair_R,
    const int new_token,
    int* output_tokens_flat, // all -1 init. in-place change in this function
    bool* output_filter, // all false init. in-place change in this function
    long* output_tokens_lens // input tokens lens init. in-place change in this function
);


// 创建内存池
void init_memory_pool(size_t block_size, size_t alignment);


// 缩小内存池
void shrink_memory_pool();


// 重置内存池
void reset_memory_pool();


// 销毁内存池
void release_memory_pool();


}



#endif