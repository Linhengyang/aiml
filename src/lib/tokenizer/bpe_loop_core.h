// bpe_loop_core.h
#pragma once
#include <cstddef>
#include <vector>
#include <cstdint>


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




extern "C" {

std::vector<std::pair<std::pair<uint32_t, uint32_t>, uint64_t>> c_non_par_bpe(
    const int num_merges,
    const size_t num_words,
    const uint32_t* tokens_ptr,
    const int64_t* offsets_ptr,
    const uint64_t* freqs_ptr
);

std::vector<std::pair<uint64_t, uint64_t>> bpe_non_par_loop_core(
    
);

} // end of extern C