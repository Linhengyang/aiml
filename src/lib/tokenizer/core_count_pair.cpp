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
    // 单线程遍历 L_tokens/R_tokens, 以统计频次
    for(int64_t j = 0; j < len; ++j) {
        counter( counter_key_type(L_tokens[j], R_tokens[j]) );
    }
}

} // end of extern C