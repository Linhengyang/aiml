// tokenizer.cpp
// core functions for utils/text/tokenizer
#include <vector>
#include <omp.h>
#include <cstddef>
#include <cstdint>

extern "C" {

void count_pair_core(
    const uint16_t* L_tokens,
    const uint16_t* R_tokens,
    // concurrent_hash_table* table, 这里需要一个支持并发读写的 hash table
    const int num_threads
) {
    // 大致逻辑
    // for L, R in zip(L_tokens, R_tokens):
    //      key = hash(L, R)
    //      table[key] ++ 
    
}

}