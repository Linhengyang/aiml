// core function for count pair using non-singleton memory pool]
// core_merge_pair.cpp
// core functions for count pair in tokenizer
#include <cstddef>
#include <cstdint>
#include <functional>
#include <mt_pair_count_merge.h>
#include <mempooled_counter.h>

extern "C" {

u32token_pair_counts_ptrs tls_dict_count_u32pair_core(
    uint64_t* keys,
    const size_t len,
    mempool& pool,
    counter_st* counter
) {
    for(size_t i = 0; i < len; ++i) {
        counter->increment(keys[i]);
    }

    size_t size = counter->size();

    uint32_t* L_uniq = static_cast<uint32_t*>(pool.allocate(size*sizeof(uint32_t)));
    uint32_t* R_uniq = static_cast<uint32_t*>(pool.allocate(size*sizeof(uint32_t)));
    uint64_t* counts = static_cast<uint64_t*>(pool.allocate(size*sizeof(uint64_t)));

    size_t index = 0;
    for(auto it = counter->cbegin(); it != counter->cend(); ++it, ++index) {
        auto [k, v] = *it;
        L_uniq[index] = static_cast<uint32_t>(k >> 32);
        R_uniq[index] = static_cast<uint32_t>(k & 0xFFFFFFFF);
        counts[index] = v;
    }
    
    return u32token_pair_counts_ptrs{L_uniq, R_uniq, counts, size};
}



u32token_pair_counts_ptrs tls_sort_count_u32pair_core(
    uint64_t* keys,
    const size_t len,
    mempool& pool
) {
    uint32_t* L_uniq = static_cast<uint32_t*>(pool.allocate(len*sizeof(uint32_t)));
    uint32_t* R_uniq = static_cast<uint32_t*>(pool.allocate(len*sizeof(uint32_t)));
    uint64_t* counts = static_cast<uint64_t*>(pool.allocate(len*sizeof(uint64_t)));

    // 排序, 可并行
    std::sort(keys, keys+len);
    
    // 线性遍历计数
    uint64_t prev = keys[0]; uint64_t cnt = 1; size_t size = 0;
    for (size_t i = 1; i < len; ++i) {
        if (keys[i] == prev) {
            ++cnt; }
        else {
            // flush(prev, cnt)
            L_uniq[size] = static_cast<uint32_t>(prev >> 32);
            R_uniq[size] = static_cast<uint32_t>(prev & 0xFFFFFFFF);
            counts[size] = cnt; ++size;

            prev = keys[i]; cnt = 1; }
    }
    L_uniq[size] = static_cast<uint32_t>(prev >> 32);
    R_uniq[size] = static_cast<uint32_t>(prev & 0xFFFFFFFF);
    counts[size] = cnt; ++size;
    
    return u32token_pair_counts_ptrs{L_uniq, R_uniq, counts, size};
}


} // end of extern C