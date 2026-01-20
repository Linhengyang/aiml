// core function for merge pair using non-singleton memory pool
// core_merge_pair.cpp
// core functions for merge pair in tokenizer
#include <omp.h>
#include <cstddef>
#include <cstdint>
#include <mt_pair_count_merge.h>
#include <utility>

extern "C" {

std::pair<uint32_t*, int64_t*> tls_merge_u32pair_core(
    const uint32_t* tokens_flat,
    const int64_t* offsets,
    const size_t num_chunks,
    const uint32_t pair_L,
    const uint32_t pair_R,
    const uint32_t new_token,
    const bool if_filter_len1,
    mempool& pool
) {
    // offsets 的最后一个值是 tokens_flat 的长度，也是 output_tokens_flat 的长度
    int64_t _LENGTH = offsets[num_chunks];

    uint32_t* merged_tokens_flat = static_cast<uint32_t*>(pool.allocate(_LENGTH*sizeof(uint32_t)));

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
    return {merged_tokens_flat, merged_offsets};
}

} // end of extern C