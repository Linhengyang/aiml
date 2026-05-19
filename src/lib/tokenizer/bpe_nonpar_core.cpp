// bpe_loop_core.h

#pragma once
#include "bpe_core.h"

// 匿名的命名空间, 等价于声明 静态存储 & 本文件私有
namespace {
    // thread_local 声明线程隔离变量

}





extern "C" {

std::vector<std::pair<std::pair<uint32_t, uint32_t>, uint64_t>> c_nonpar_bpe(
    const int num_merges,
    const size_t num_words,
    const uint32_t* tokens_ptr,
    const int64_t* offsets_ptr,
    const uint64_t* freqs_ptr
) {
    
}

} // end extern "C"