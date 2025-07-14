// tokenizer.H
// statement for tokenizer.cpp

#ifndef TOKENIZER_H
#define TOKENIZER_H

#include <cstddef>


extern "C" {

void c_merge_pair_batch(
    const int* tokens_flat,
    const long* offsets,
    int num_chunks,
    int pair_L,
    int pair_R,
    int new_token,
    int* output_tokens_flat,
    bool* output_filter,
    int* output_tokens_lens
);

void merge_pair_core_parallel(
    const int* tokens_flat,
    const long* offsets,
    int num_chunks,
    int pair_L,
    int pair_R,
    int new_token,
    int* output_tokens_flat,
    bool* output_filter,
    int* output_tokens_lens
);

}



#endif