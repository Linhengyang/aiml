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




// 销毁内存
void block::aligned_free(void* ptr) {
#if defined(_WIN32) || defined(_WIN64)
    _aligned_free(ptr);
#else
    std::free(ptr);
#endif
}




// block 构造函数:
// 申请大小至少为 capacity 且首地址 alignment 对齐的内存块。初始化offset和capacity，记录用户可使用的空间地址_aligned_data
block::block(size_t capacity, size_t alignment): _aligned_block_start(nullptr), _capacity(0), _offset(0), _alignment(alignment) {
    
    // 确保对齐要求为2的幂
    if ((alignment & (alignment - 1)) != 0 || alignment == 0) {
        throw std::invalid_argument("alignment must be a power of 2.");
    }

    // 寻找首地址对齐的、大小为capacity+alignment的内存块。+alignment是为了保证能获取对齐首地址
    void* raw_mem = aligned_malloc(capacity+alignment, alignment);

    if (!raw_mem) {
        throw std::bad_alloc();
    }

    // aligned_malloc 返回的一定是 alignment 对齐的. 无须额外调整
    // raw_mem 对于 block（size为capacity）来说，是浪费了一点 alignment 空间。无伤大雅
    _aligned_block_start = static_cast<char*>(raw_mem);

    _capacity = capacity;

    _offset = 0;

}




// block 析构函数
// 由于这是一个线性内存池，其真正从系统"申请内存"只在构造时aligned_malloc得到block_start地址，故只要释放这个地址即可
block::~block() {
    aligned_free(_aligned_block_start);
}





// 本内存block的标记为可从头复用状态. 标记本block尚未用过
void block::reset() {
    _offset = 0;
}





// 在本内存块中寻找大小为size的内存空间且首地址对齐。若不成功，返回nullptr
// _aligned_block_start + _offset 即为下一次可分配的起点
// 经过alignment后如果距离 capacity 还足够size，那么返回 alignment 后的起点 _aligned_position
// _offset = _aligned_position - _aligned_block_start + size
// 若不足够，返回nullptr表示此次分配失败
void* block::allocate(size_t size) {

    // 首先找到下一次分配的对齐地址
    uintptr_t next_address = reinterpret_cast<uintptr_t>(_aligned_block_start + _offset); //下一次可分配的起点地址值
    uintptr_t aligned_next_address = (next_address + _alignment - 1) & ~(_alignment - 1); // 上跳对齐
    char* _aligned_next_start = reinterpret_cast<char*>(aligned_next_address); //重新转换成地址

    if (_aligned_next_start + size > _aligned_block_start + _capacity) {
        // start = 0, capacity = 5
        // next_start = 2, size = 3 --> 2+3 = 5 == start + capacity
        // 这个时候不越界, 因为 2 3 4 满足分配要求.
        // 故只有严格大于才是越界
        return nullptr;
    }

    _offset = _aligned_next_start - _aligned_block_start + size; //char*指针-指针，得到的元素数就是字节数

    return _aligned_next_start;
}