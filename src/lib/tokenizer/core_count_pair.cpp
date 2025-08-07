// tokenizer.cpp
// core functions for utils/text/tokenizer
#include <vector>
#include <omp.h>
#include <cstddef>
#include <cstdint>
#include "pair_count_merge.h"
#include "mempool_counter.h"

extern "C" {

void count_pair_core_multi_thread(
    counter_mt& counter,
    const uint16_t* L_tokens,
    const uint16_t* R_tokens,
    const int64_t len,
    // concurrent_hash_table* table, 这里需要一个支持并发读写的 hash table
    const int num_threads
) {
    // 大致逻辑
    // for L, R in zip(L_tokens, R_tokens):
    //      key = hash(L, R)
    //      table[key] ++ 
    
}




void count_pair_core_single_thread(
    counter_st& counter,
    const uint16_t* L_tokens,
    const uint16_t* R_tokens,
    const int64_t len
) {
    // 大致逻辑
    // for L, R in zip(L_tokens, R_tokens):
    //      key = hash(L, R)
    //      table[key] ++ 
    
}

} // end of extern C