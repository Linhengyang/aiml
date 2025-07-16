// tokenizer.cpp
// core functions for utils/text/tokenizer
#include <vector>
#include <omp.h>
#include <cstddef>


extern "C" {

void merge_pair_core_parallel(
    const int* tokens_flat,
    const long* offsets,
    const size_t num_chunks,
    const int pair_L,
    const int pair_R,
    const int new_token,
    int* output_tokens_flat, // all -1 init. in-place change in this function
    bool* output_filter, // all false init. in-place change in this function
    long* output_tokens_lens // input tokens lens init. in-place change in this function
) {
    #pragma omp parallel for
    for(size_t i = 0; i < num_chunks; i++) {
        int start = offsets[i];
        int end = offsets[i+1];
        int len_tokens = end - start;
        for(int j = 0; j < len_tokens;) {
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

}