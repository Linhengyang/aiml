// core function for count pair using non-singleton memory pool]
// core_merge_pair.cpp
// core functions for count pair in tokenizer
#include <cstddef>
#include <cstdint>
#include <functional>
#include <mt_pair_count_merge.h>
#include <mempooled_counter.h>

extern "C" {

u16token_pair_counts_ptrs tls_dict_count_u16pair_core(
    uint32_t* keys,
    const size_t len,
    mempool& pool,
    counter_tls* counter
) {
    for(size_t i = 0; i < len; ++i) {
        counter->increment(keys[i]);
    }

    size_t size = counter->size();

    uint16_t* L_uniq = static_cast<uint16_t*>(pool.allocate(size*sizeof(uint16_t)));
    uint16_t* R_uniq = static_cast<uint16_t*>(pool.allocate(size*sizeof(uint16_t)));
    uint64_t* counts = static_cast<uint64_t*>(pool.allocate(size*sizeof(uint64_t)));

    size_t index = 0;
    for(auto it = counter->cbegin(); it != counter->cend(); ++it, ++index) {
        auto [k, v] = *it;
        L_uniq[index] = static_cast<uint16_t>(k >> 16);
        R_uniq[index] = static_cast<uint16_t>(k & 0xFFFF);
        counts[index] = v;
    }
    
    return u16token_pair_counts_ptrs{L_uniq, R_uniq, counts, size};
}

} // end of extern C