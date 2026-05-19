// bpe_loop_core.h
#pragma once
#include <cstddef>
#include <vector>
#include <cstdint>
#include "heap.h"
#include <unordered_set>
#include <unordered_map>
#include "mempooled_hashtable.h"
#include "mempooled_concurrent_hashtable.h"
#include "memory_pool.h"


// 拼合两个 u32 token 成为一个 u64 的方法
uint64_t combine_2tokens(const uint32_t left_token, const uint32_t right_token) {
    return (static_cast<uint64_t>(left_token) << 32) | right_token;
}


// 分拆一个 u64 成为两个 u32 token 的方法
std::pair<uint32_t, uint32_t> split_2tokens(const uint64_t combined_tokens) {
    return {static_cast<uint32_t>(combined_tokens >> 32), static_cast<uint32_t>(combined_tokens & 0xFFFFFFFFULL)};
}


class Word {

private:
    std::vector<uint32_t> token_ids; // 表示单个 word 的底层数据 u32 tokens

public:

    // 构造函数之 范围构造. 在 BPE 中只需要这种拷贝构造函数就够了 
    explicit Word(const uint32_t* begin, const uint32_t* end):
        token_ids(begin, end) // 这里采用 vector 的一种范围构造方法 range constructor: 输入首尾两个(符合类型的)迭代器(指针也是迭代器), 拷贝范围内的数据到新内存, 构建实例
    {}

    // 析构函数之 自然析构
    ~Word() {}

    // 当本 word 实例执行一次 merge(token-pair, new_token) 操作后, 新增/消失 的 token-pair with its signal
    std::vector<std::pair<uint64_t, int>> merge(const uint64_t to_merge_token_pair, const uint32_t merged_token) {
        std::vector<std::pair<uint64_t, int>> changes = std::vector<std::pair<uint64_t, int>>();
        auto [left_token, right_token] = split_2tokens(to_merge_token_pair);
        int i = 0;
        while (true)
        {
            // 遍历 i 到 last token 的位置时(或根本没有token), 没有 pair 了, 退出循环
            if (i > token_ids.size() - 1) {
                break;
            }
            // 判断是否匹配到 to_merge_token_pair. 只有当匹配时 才会生成 changes
            if (token_ids[i] == left_token && token_ids[i+1] == right_token) {
                // 如果 token_ids[i] == left_token 的左边还有 token, 那么因为这次merge, 发生了一次 -1 的 token-pair 消失, 一次 +1 的 token-pair 增加
                if (i > 0) {
                    uint32_t prev_token = token_ids[i - 1];
                    changes.push_back( {combine_2tokens(prev_token, left_token), -1} );
                    changes.push_back( {combine_2tokens(prev_token, merged_token), 1} );
                }
                // 如果 token_ids[i+1] == right_token 的右边还有 token, 那么因为这次merge, 发生了一次 -1 的 token-pair 消失, 一次 +1 的 token-pair 增加
                if (i+1 < token_ids.size()-1) {
                    uint32_t folw_token = token_ids[i+2];
                    changes.push_back( {combine_2tokens(right_token, folw_token), -1} );
                    changes.push_back( {combine_2tokens(merged_token, folw_token), 1} );
                }
                // 对 word.token_ids 作 in-place 修改: merge [i] & [i+1] -> merged_token
                token_ids[i] = merged_token;
                //vec.begin() + i 获取指向索引 i 的迭代器；erase() 删除该位置元素，后续元素自动前移；vec.size() 会自动减 1
                token_ids.erase(token_ids.begin() + i+1);
            }
            ++i;
        }
        return changes;
    };

    /*
    * 只读 token-pair 迭代器
    * 
    * 用法: 单一线程下 for(auto win2_it = word.begin_pair(); it != word.end_pair(); ++it) {uint64_t token_pair = *win2_it; //code//}
    */
    class pair_iterator {
    private:
        const Word* _word;
        size_t _index;
    public:
        pair_iterator(const Word* word, size_t index) // 这种写法代表 word是一个指向 const Word的指针: 不能通过word来改变其指向的数据, 只能通过word来读取数据.
            :_word(word),
            _index(index)
        {}

        // *it 迭代器对象解引用 --> 只读返回
        uint64_t operator*() const {
            return combine_2tokens(_word->token_ids[_index], _word->token_ids[_index+1]); // 返回 u64 作为两个 u32 tokens 组成的 pair
        }

        // ++it 迭代器对象 前置自增 --> 改变自身状态后, 返回自身引用
        pair_iterator& operator++() {
            ++_index;
            return *this;
        }

        // it++ 迭代器对象 后置自增 --> 用前置自增改变自身状态, 返回迭代器自增前原值副本
        pair_iterator operator++(int) {
            pair_iterator tmp = *this; // 自增前 拷贝赋值
            ++(*this);
            return tmp;
        }

        // 给出两个迭代器状态是否相等的判决方法: 相同的 word实例指针 以及 相同的遍历index位置
        bool operator==(const pair_iterator& other) const {
            return _index == other._index && _word == other._word;
        }

        // 给出两个迭代器状态是否不相等的判决方法, 必须是 operator == 操作的反面
        bool operator!=(const pair_iterator& other) const {
            return !(*this == other); // this是本对象指针, *this就是返回本对象
        }
    };  // end of pair-iterator definition

    // begin_pair 方法返回的迭代器应该处于 begin 的状态, 即指向 first it: _index指向首token
    pair_iterator begin_pair() {
        return pair_iterator(this, 0);
    }

    // end 方法返回的迭代器应该处于 end 的临界状态, 即刚结束迭代的 状态: _index指向末token
    // 注意两个特殊情况: 当token_ids.size=1时, 天然满足1-1=0就是末token; 但是当token_ids.size=0时, 0-1=-1是一个失效的位置index,并不是刚结束迭代的状态
    pair_iterator end_pair() {
        if (token_ids.empty()) {
            // 若 word 为空, 直接返回 begin 状态作为结束状态
            return pair_iterator(this, 0);
        }
        return pair_iterator(this, token_ids.size()-1);
    }
};


// 定义 不可重复的集合容器(set)
using position_set = std::unordered_set<size_t>;



// 定义 优先队列的节点(node)
struct merge_node {
    uint64_t token_pair;
    uint64_t p_cnts;
    position_set positions;
};


// 定义 严格弱序优先级函数(compare)
struct node_comparator {
    bool operator()(const merge_node& a, const merge_node& b) {
        return a.p_cnts > b.p_cnts;
    }
};


// 定义 优先队列(8-ary max_heap)
using max_octanory_heap = octanary_heap<merge_node, node_comparator>;


// 定义 哈希表的 哈希器. 这里 hasher 是一个函数类, 通过实例化得到哈希器 hasher myHasher;
struct hasher {
    size_t operator()(const uint64_t& key) const {
        return static_cast<size_t>(key);
    }
};


// 定义 基于内存池的哈希表(pool_hashtable & threadsafe_pool_hashtable)
// where_to_update 用在两个地方: bpe 里用于 优先队列的初始化(随后置空), 以及 bpe_loop_core 里用于接收新产生的 token_pair-positions KV对.
// 前者用完后即置空, 可并行(如果需要并行则需要 where_to_update 线程安全); 后者在循环中不断「插入-移动语义-空」，但没有并行的需求
// 所以似乎在 bpe_loop_core 内部，loop开始前重新 初始化一个无需线程安全的 where_to_update(hashmap{u64: unordered_set}) 即可, 无需把循环之前的 where_to_update 传进去
using hashmap = pooled_hashtable<uint64_t, position_set, mempool, hasher>;
using threadsafe_hashmap = pooled_concurrent_hashtable<uint64_t, position_set, threadsafe_mempool, hasher>;
 

extern "C" {

// unique_words & freqs --window2_token遍历--> pair_counts(hashmap{u64: u64}), where_to_update(hashmap{u64: unordered_set})
// 移动语义遍历where_to_update: pair & move(positions) + pair_counts --> C++ Merge构造 --heapify--> max_heap(8-ary heap of Merge), 销毁where_to_update
// 调用 nonpar_bpe_loop_core: merges(vector of (u64, u64)) = nonpar_bpe_loop_core(max_heap, unique_words, freqs(const), pair_counts, num_merges)
// 转换 merges(vector of (u64, u64)) --> merges(vector of ((u32, u32), u64)), 并返回
std::vector<std::pair<std::pair<uint32_t, uint32_t>, uint64_t>> c_nonpar_bpe(
    const int num_merges,
    const size_t num_words,
    const uint32_t* tokens_ptr,
    const int64_t* offsets_ptr,
    const uint64_t* freqs_ptr
);


std::vector<std::pair<uint64_t, uint64_t>> nonpar_bpe_loop_core(
    max_octanory_heap& max_heap,
    std::vector<Word>& unique_words,
    const std::vector<uint64_t>& freqs,
    std::unordered_map<uint64_t, uint64_t>& pair_counts,
    const int num_merges
);


std::vector<std::pair<std::pair<uint32_t, uint32_t>, uint64_t>> c_par_bpe(
    const int num_merges,
    const size_t num_words,
    const uint32_t* tokens_ptr,
    const int64_t* offsets_ptr,
    const uint64_t* freqs_ptr
);


std::vector<std::pair<uint64_t, uint64_t>> par_bpe_loop_core(
    max_octanory_heap& max_heap,
    std::vector<Word>& unique_words,
    const std::vector<uint64_t>& freqs,
    std::unordered_map<uint64_t, uint64_t>& pair_counts,
    const int num_merges
);


} // end of extern C