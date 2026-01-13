// core_merge_pair.cpp
// core functions for count pair in tokenizer
#include <cstddef>
#include <cstdint>
#include <functional>
#include <pair_count_merge.h>

extern "C" {

size_t local_count_u16pair_core(
    const uint32_t* keys,
    const size_t len,
    uint16_t* L_uniq, // in-place change in this function
    uint16_t* R_uniq, // in-place change in this function
    uint64_t* counts // in-place change in this function
) {
    // 排序, 可并行
    std::sort(keys, keys+len);
    
    // 线性遍历计数
    uint32_t prev = keys[0]; uint64_t cnt = 1; size_t size = 0;
    for (size_t i = 1; i < len; ++i) {
        if (keys[i] == prev) {
            ++cnt; }
        else {
            // flush(prev, cnt)
            L_uniq[size] = static_cast<uint16_t>(prev >> 16);
            R_uniq[size] = static_cast<uint16_t>(prev & 0xFFFF);
            counts[size] = cnt; ++size;

            prev = keys[i]; cnt = 1; }
    }
    L_uniq[size] = static_cast<uint16_t>(prev >> 16);
    R_uniq[size] = static_cast<uint16_t>(prev & 0xFFFF);
    counts[size] = cnt; ++size;

    return size;
}

} // end of extern C