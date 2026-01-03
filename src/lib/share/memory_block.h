// memory_block.h

#ifndef MEMORY_BLOCK_H
#define MEMORY_BLOCK_H


#include <stdexcept>
#include <cstdint>
#include <cstring>
#include <cstddef>


// 内部内存对齐的内存块
// block是独立的类，负责自身的创建和销毁
class block {

private:
    
    // 每次在block中分配内存时，寻址到的内存起始位置. 保证alignment对齐
    char* _aligned_block_start; //char*是指向单字节的指针, 故可以拿来指代最小偏移粒度为单字节的内存block

    size_t _capacity; //单个内存block的字节容量

    size_t _offset; //标记当前内存block已使用的位置偏移量.
    // _aligned_block_start + _offset 即为下一次可分配的起点
    // 经过alignment后如果距离 capacity 还足够，那么返回 alignment 后的起点 _aligned_position
    // _offset = _aligned_position - _aligned_block_start + size

    size_t _alignment; //对齐字节数量. 8/16/64 等


public:

    // 构造函数. 默认对齐64字节. 构造 block 的时候, 内存申请实际发生
    explicit block(size_t capacity, size_t alignment=64);

    // 析构函数. 析构 block 时, 内存释放实际发生
    ~block();

    // 成员函数
    // getter: const 成员函数
    inline size_t get_offset() const { return _offset; }
    inline size_t get_capacity() const { return _capacity; }
    inline size_t get_remaining() const { return _capacity - _offset; }

    // allocate & reset. allocate 仅是分配地址, 没有内存申请; reset 仅是地址复位, 没有内存释放
    void* allocate(size_t size); // 在本内存块中寻找大小为size的内存空间且首地址对齐。若不成功，返回nullptr

    void reset(); // 本内存block的标记为可从头复用状态


private:

    static void* aligned_malloc(size_t size, size_t alignment);

    static void aligned_free(void* ptr);

};



#endif