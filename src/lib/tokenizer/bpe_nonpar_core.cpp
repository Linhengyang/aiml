// bpe_nonpar_core.h

#pragma once
#include "bpe_core.h"

// 匿名的命名空间, 等价于声明 静态存储 & 本文件私有
namespace {

    static hasher pair_hasher;

}





extern "C" {

std::vector<std::pair<std::pair<uint32_t, uint32_t>, uint64_t>> c_nonpar_bpe(
    const int num_merges,
    const size_t num_words,
    const uint32_t* tokens_ptr,
    const int64_t* offsets_ptr,
    const uint64_t* freqs_ptr
) {

    // 1. 从 unique_words 的 buffer 地址(tokens_ptr / offsets_ptr), 循环 num_words 构建这么多个 word 为vector
    std::vector<Word> unique_words;
    unique_words.reserve(num_words);

    for (size_t i = 0; i < num_words; ++i) {
        unique_words.emplace_back(tokens_ptr + offsets_ptr[i], tokens_ptr + offsets_ptr[i+1]);
    }

    // 直接使用范围拷贝构造得到 freqs
    std::vector<uint64_t> freqs(freqs_ptr, freqs_ptr + num_words);
    
    // 2. 构建 pair_counts(hashmap{u64: u64}) 在 系统堆内存 + where_to_update(hashmap{u64: unordered_set}) 在 内存池
    std::unordered_map<uint64_t, uint64_t, hasher> pair_counts;
    mempool pool(17179869184LL, 64); // 创建一个 16GB 的内存池
    hashmap where_to_update(pair_hasher, 65536, &pool);

    for (size_t i = 0; i< num_words; ++i) {
        // pair_iterator on every word
        for(auto pair_it = unique_words[i].begin_pair(); pair_it != unique_words[i].end_pair(); ++pair_it) {
            uint64_t token_pair = *pair_it;
            pair_counts[token_pair] += freqs[i]; // std::unordered_map<uint64_t, uint64_t, hasher> 类型的 [] 操作符会自动给新key插入默认值0

        }
    }

    mempool pool(17179869184LL, 64); // 创建一个 16GB 的内存池
   
}


std::vector<std::pair<uint64_t, uint64_t>> nonpar_bpe_loop_core(
    max_octanory_heap& max_heap,
    std::vector<Word>& unique_words,
    const std::vector<uint64_t>& freqs,
    std::unordered_map<uint64_t, uint64_t>& pair_counts,
    const int num_merges
) {

}


} // end extern "C"