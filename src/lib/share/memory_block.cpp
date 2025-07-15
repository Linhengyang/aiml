// memory_block.cpp

#include "memory_block.h"
#include <stdexcept>
#include <cstdint>
#include <cstring>



// 分配并返回大小为 size 的内存首地址, 且这个首地址是 alignment 对齐 
void* block::aligned_malloc(size_t size, size_t alignment) {
#if defined(_WIN32) || defined(_WIN64)
    return _aligned_malloc(size, alignment)
#else
    void* ptr = nullptr;

    // Allocate memory of `size` bytes with an alignment of `alignment`
    int err = posix_memalign(&ptr, alignment, size);

    if (err != 0) {
        return nullptr;
    }
    return ptr;
#endif
}



// block 构造函数
block::block(size_t capacity, size_t alignment): _aligned_data(nullptr), _capacity(0), _offset(0), _alignment(alignment) {
    
    

}


