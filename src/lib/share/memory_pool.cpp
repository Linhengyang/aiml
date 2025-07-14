// memory_pool.cpp

#include "memory_pool.h"
#include <stdexcept>



memory_pool::memory_pool(size_t block_size): _block_size(block_size){
    if (block_size == 0) {
        throw std::invalid_argument("block_size must be > 0");
    }
}



memory_pool::~memory_pool() {
    release();
}



void memory_pool::release() {
    for(auto block: _blocks) {
        // 释放 _blocks vector 中每一个 内存block
        std::free(block); // 用 std::malloc 分配的, 要用 std::free 来释放
    }
    _blocks.clear(); // 清空 _blocks 容器里的所有元素
    _current_block = nullptr;
    _offset = 0;
}


