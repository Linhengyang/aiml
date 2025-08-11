// tokenizer.cpp
// core functions for utils/text/tokenizer
#include <vector>
#include <omp.h>
#include <cstddef>
#include <cstdint>
#include "pair_count_merge.h"
#include "mempool_counter.h"


namespace {
    // 从 一般输入 到 counter_key_type 的构造器
    key_maker get_key;
}

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
        // current key
        counter_key_type cur_key = get_key(L_tokens[j], R_tokens[j]);
        // count current key
        counter( cur_key );
    }
    // 非常慢. (L,R) -> pair(L,R) -hash-> 定位 hash table bucket ->
    //      若冲突 --> 链表遍历 -->
    //      若新key --> 分配node -->
    //      随机内存访问 + 分支, cpu缓存命中率极差, 内存池是堆内存+调度开销 --> O(n)的算法复杂度, 实际效果很慢


}

} // end of extern C