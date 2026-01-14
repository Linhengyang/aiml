// core_merge_pair.cpp
// core functions for merge pair in tokenizer
#include <omp.h>
#include <cstddef>
#include <cstdint>
#include <mp_pair_count_merge.h>

extern "C" {

void local_merge_u16pair_core(
    const uint16_t* tokens_flat,
    const int64_t* offsets,
    const size_t num_chunks,
    const uint16_t pair_L,
    const uint16_t pair_R,
    const uint16_t new_token,
    uint16_t* output_tokens_flat, // all -1 init. in-place change in this function
    bool* output_filter, // all false init. in-place change in this function
    int64_t* output_tokens_lens // input tokens lens init. in-place change in this function
) {
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
}

} // end of extern C