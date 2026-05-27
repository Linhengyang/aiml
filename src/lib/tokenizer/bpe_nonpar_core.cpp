// bpe_nonpar_core.h

#pragma once
#include "bpe_core.h"
#include <tuple>

// 匿名的命名空间, 等价于声明 静态存储 & 本文件私有
namespace {

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
    // 范围地址(拷贝)构造
    for (size_t i = 0; i < num_words; ++i) {
        unique_words.emplace_back(tokens_ptr + offsets_ptr[i], tokens_ptr + offsets_ptr[i+1]);
    }

    // 直接使用范围拷贝构造得到 freqs
    std::vector<uint64_t> freqs(freqs_ptr, freqs_ptr + num_words);
    
    // 2. 构建 pair_counts(hashmap{u64: u64}) 在 系统堆内存 + where_to_update(hashmap{u64: unordered_set}) 在 内存池
    std::unordered_map<uint64_t, uint64_t, hasher> pair_counts;

    // 初始状态总共只有 256 种token, 故token_pair种类一共就 65536; 后续token_pair种类上限为(256+num_merges)^2
    mempool pool(8589934592LL, 64); // 初始化一个 8GB 的内存池
    hashmap where_to_update(65536, &pool); // 初始 where_to_update 的 capacity 取 65536 即可

    for (size_t i = 0; i< num_words; ++i) {
        // pair_iterator on every word
        for(auto pair_it = unique_words[i].begin_pair(); pair_it != unique_words[i].end_pair(); ++pair_it) {
            uint64_t token_pair = *pair_it;
            pair_counts[token_pair] += freqs[i]; // std::unordered_map<uint64_t, uint64_t, hasher> 类型的 [] 操作符会自动给新key插入默认值0
            where_to_update.atomic_upsert(token_pair, [i](position_set& positions) { positions.insert(i); }, position_set{});
        }
    }

    // 3. 移动语义遍历 where_to_update: pair & move(positions) + pair_counts[pair] --merge_node构造--> heapify--> max_octanory_heap. 置空 where_to_update
    /*TODO*/

    // 4. 调用 nonpar_bpe_loop_core
    std::vector<std::pair<uint64_t, uint64_t>> merges = nonpar_bpe_loop_core(
        max_heap,
        unique_words,
        freqs,
        pair_counts,
        where_to_update,
        num_merges
    );

    // 5. 转换并返回 merges(vector of ((u32, u32), u64))

}


std::vector<std::pair<uint64_t, uint64_t>> nonpar_bpe_loop_core(
    max_octanory_heap& max_heap,
    std::vector<Word>& unique_words,
    const std::vector<uint64_t>& freqs,
    std::unordered_map<uint64_t, uint64_t, hasher>& pair_counts,
    hashmap& where_to_update,
    const int num_merges
) {
    // 初始化一个要返回的收集容器 vector of pair<uint64 as tokens-pair, uint64 as p_counts> merges
    std::vector<std::pair<uint64_t, uint64_t>> merges;
    merges.reserve(num_merges);

    // 初始化一个大容量的、收集各pos的local_changes的容器 vector of tuple<uint64 as tokens-pair, int as change_signal, size_t as position> changes
    // 作用是 循环 num_merges 次以下操作: 对所有pos的word执行merge, 合并所有local_changes(添加pos)到容器changes, 然后扫描它以更新pair_counts&where_to_update
    // 最后 clear() + shrink_to_fit() 即清空但保留足够空间, 以下一次循环复用
    std::vector<std::tuple<uint64_t, int, size_t>> changes;
    changes.reserve(unique_words.size());

    size_t merge_cnts = 0; // 需要用 merge_cnts 来计算 new_token
    while (true) {
        if (merge_cnts >= num_merges) {
            break;
        }

        if (max_heap.empty()) {
            // TODO: 这里要保证merges能被返回
            throw std::runtime_error("Pop failed: Heap is empty!");
        }

        merge_node top = max_heap.pop(); // pop堆顶
        if (top.p_cnts != pair_counts.at(top.token_pair)) {
            top.p_cnts = pair_counts.at(top.token_pair);
            max_heap.push(std::move(top)); // push堆底(传入移动右值, 函数内部始终用移动 保证移动语义)
            continue;
        }

        ++merge_cnts;
    }
}


} // end extern "C"