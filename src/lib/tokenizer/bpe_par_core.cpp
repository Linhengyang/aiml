// bpe_par_core.h

#pragma once
#include "bpe_core.h"

// 匿名的命名空间, 等价于声明 静态存储 & 本文件私有
namespace {
    // thread_local 声明线程隔离变量

}





extern "C" {

std::vector<std::pair<std::pair<uint32_t, uint32_t>, uint64_t>> c_par_bpe(
    const int num_merges,
    const size_t num_words,
    const uint32_t* tokens_ptr,
    const int64_t* offsets_ptr,
    const uint64_t* freqs_ptr
) {
    
}



std::vector<std::pair<uint64_t, uint64_t>> par_bpe_loop_core(
    max_octanory_heap& max_heap,
    std::vector<Word>& unique_words,
    const std::vector<uint64_t>& freqs,
    std::unordered_map<uint64_t, uint64_t>& pair_counts,
    hashmap& where_to_update,
    const int num_merges
) {
    
}


} // end extern "C"