// memory_block.h

#ifndef MEMORY_BLOCK_H
#define MEMORY_BLOCK_H


#include <stdexcept>
#include <cstdint>
#include <cstring>


// 内部内存对齐的内存块
// block是独立的类，负责自身的创建和销毁
class block {

private:

    char* _aligned_data; //char*是指向单字节的指针, 故可以拿来指代最小偏移粒度为单字节的内存block

    const size_t _capacity; //单个内存block的字节容量

    size_t _offset; //当前内存block已使用的位置偏移量

    size_t _alignment; //对齐字节数量. 8/16/64 等

public:

    // 构造函数. 默认对齐64字节
    explicit block(size_t capacity, size_t alignment = 64);

    // 析构函数
    ~block();

    // 成员函数
    // getter: const 成员函数
    inline size_t get_offset() const { return _offset; }
    inline size_t get_capacity() const { return _capacity; }
    inline size_t get_remaining() const { return _capacity - _offset; }

    // allocate & reset
    void* allocate(size_t size);
    void reset();


private:

    static void* aligned_malloc(size_t size, size_t alignment);

    static void aligned_free(void* ptr);

};



#endif