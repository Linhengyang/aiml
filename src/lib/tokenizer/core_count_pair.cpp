// // // tokenizer.cpp
// // // core functions for utils/text/tokenizer


// #include <vector>
// #include <omp.h>
// #include <cstddef>
// #include <cstdint>
// #include "pair_count_merge.h"
// #include "mempool_counter.h"




// extern "C" {

// void count_pair_core(
//     counter_st* counter,
//     uint32_t* keys,
//     const int64_t len
// ) {
//     // (L,R) -> pair(L,R) -hash-> 定位 hash table bucket ->
//     //      若冲突 --> 链表遍历 -->
//     //      若新key --> 分配node -->
//     //      随机内存访问 + 分支, cpu缓存命中率极差, 内存池是堆内存+调度开销 --> O(n)的算法复杂度, 实际效果很慢


// }

// } // end of extern C