// core_merge_pair.cpp
// core functions for merge pair in tokenizer
#include <omp.h>
#include <cstddef>
#include <cstdint>
#include <mp_pair_count_merge.h>

extern "C" {


merged_u16token_offset_ptrs local_merge_u16pair_core(
    const uint16_t* tokens_flat,
    const int64_t* offsets,
    const size_t num_chunks,
    const uint16_t pair_L,
    const uint16_t pair_R,
    const uint16_t new_token,
    const bool if_filter_len1,
    singleton_mempool& pool
) {
    // offsets 的最后一个值是 tokens_flat 的长度，也是 output_tokens_flat 的长度
    int64_t _LENGTH = offsets[num_chunks];

    uint16_t* merged_tokens_flat = static_cast<uint16_t*>(pool.allocate(_LENGTH*sizeof(uint16_t)));

    // merged_offsets 和 offsets 一样, 都是 各chunk首token的 index，末尾append一个最终长度
    // 所用 总共有 num_chunks+1 个值, 第一个值是 0, 最后一个值是 merged_tokens 的总数
    int64_t* merged_offsets = static_cast<int64_t*>(pool.allocate((num_chunks+1)*sizeof(int64_t)));
    merged_offsets[0] = 0;

    int64_t num_merges = 0;
    int64_t num_filtered = 0;

    // 遍历所有 chunk: chunk_1 --> chunk_k --> chunk_num_chunks
    for(size_t k = 1; k <= num_chunks; ++k) {
        // 遍历 chunk_k: j 作为 token_flat 的 index, 从 offsets[k-1] --> offsets[k]-1, 即遍历了 chunk_k
        for(size_t j = offsets[k-1]; j < offsets[k];) {
            // 匹配到了 pair
            if(j < offsets[k]-1 && tokens_flat[j] == pair_L && tokens_flat[j+1] == pair_R) {
                merged_tokens_flat[j-num_merges-num_filtered] = new_token;
                j += 2;
                num_merges += 1;
            }
            // 没有匹配到 pair
            else {
                merged_tokens_flat[j-num_merges-num_filtered] = tokens_flat[j];
                j += 1;
            }
        }
        // chunk_k 遍历结束: 确定 chunk_k 的边界
        merged_offsets[k] = offsets[k] - num_merges - num_filtered;

        // 过滤 len=1 chunk
        if(if_filter_len1 && (merged_offsets[k] - merged_offsets[k-1] == 1)) {
            merged_offsets[k] -= 1;
            num_filtered += 1;
        }
    }
    // 所有 chunk 遍历结束
    return merged_u16token_offset_ptrs{merged_tokens_flat, merged_offsets, num_chunks};
}







#if 0
merged_u16token_filter_len_ptrs _deprecated_local_merge_u16pair_core(
    const uint16_t* tokens_flat,
    const int64_t* offsets,
    const size_t num_chunks,
    const uint16_t pair_L,
    const uint16_t pair_R,
    const uint16_t new_token,
    singleton_mempool& pool
) {

    int64_t* output_tokens_lens = static_cast<int64_t*>(pool.allocate(num_chunks*sizeof(int64_t)));
    // 初始化数组
    for (size_t i = 0; i < num_chunks; ++i) {
        output_tokens_lens[i] = offsets[i+1] - offsets[i];
    }

    // offsets 的最后一个值是 tokens_flat 的长度，也是 output_tokens_flat/output_filter 的长度
    int64_t _LENGTH = offsets[num_chunks];

    // _LENGTH 长度
    // need size = sizeof(bool) * _LENGTH
    bool* output_filter = static_cast<bool*>(pool.allocate(_LENGTH*sizeof(bool)));
    for (int64_t i = 0; i < _LENGTH; ++i) {
        output_filter[i] = false; // 全部初始化为 false
    }

    // _LENGTH 长度
    uint16_t* output_tokens_flat = static_cast<uint16_t*>(pool.allocate(_LENGTH*sizeof(uint16_t)));
    for (int64_t i = 0; i < _LENGTH; ++i) {
        output_tokens_flat[i] = -1; // 全部初始化为 -1
    }

    #pragma omp parallel for
    for(size_t i = 0; i < num_chunks; i++) {
        int64_t start = offsets[i];
        int64_t end = offsets[i+1];
        int64_t len_tokens = end - start;
        for(int64_t j = 0; j < len_tokens;) {
            if(j < len_tokens-1 && tokens_flat[start+j] == pair_L && tokens_flat[start+j+1] == pair_R) {
                output_tokens_lens[i] -= 1;
                output_tokens_flat[start+j] = new_token;
                output_filter[start+j] = true;
                j += 2;
            }else {
                output_tokens_flat[start+j] = tokens_flat[start+j];
                output_filter[start+j] = true;
                j += 1;
            }
        }
    }

    return merged_u16token_filter_len_ptrs{output_tokens_flat, output_filter, output_tokens_lens};
}
#endif


} // end of extern C