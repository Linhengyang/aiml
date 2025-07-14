// tokenizer.cpp
// core functions for utils/text/tokenizer
#include <vector>
#include <cstring>
#include <omp.h>

extern "C" {

void merge_pair_core_parallel(
    const int* tokens_flat,
    const long* offsets,
    int num_chunks,
    int pair_L,
    int pair_R,
    int new_token,
    int* output_tokens_flat,
    bool* output_filter,
    long* output_tokens_lens
) {
    #pragma omp parallel for
    for(int i = 0; i < num_chunks; i++) {
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