#include <stdexcept>
#include "tokenizer.h"

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
) {
    try
    {
        merge_pair_core_parallel(
            tokens_flat,
            offsets,
            num_chunks,
            pair_L,
            pair_R,
            new_token,
            output_tokens_flat,
            output_filter,
            output_tokens_lens
        );
    }
    catch(const std::exception& e)
    {
        throw std::runtime_error("Error in merge_pair_core_parallel");
    }
}

}