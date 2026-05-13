// bpe_loop_core.h
#pragma once
#include <cstddef>
#include <vector>
#include <cstdint>




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
    std::vector<std::pair<uint64_t, int>> merge(const uint64_t to_merge_token_pair, const uint32_t new_token) {
        //TODO
    };

    /*
    * 只读 token-pair 迭代器
    * 
    * 用法: 单一线程下 for(auto win2_it = word.begin_pair(); it != word.end_pair(); ++it) {uint64_t token_pair = *win2_it; //code//}
    */
    class pair_iterator {
    
    private:
        const Word& word; // 迭代器

    public:


    
    }



}




extern "C" {



std::vector<std::pair<uint64_t, uint64_t>> bpe_core_loop(
    int num_merges,

);


} // end of extern C