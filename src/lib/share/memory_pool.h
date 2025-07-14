// memory_pool.h


#ifndef MEMORY_POOL_H
#define MEMORY_POOL_H

#include <vector>
#include <cstddef>
#include <cstdlib>
#include <mutex>

class memory_pool
{

private:
    const size_t _block_size; //内存池中, 单个内存block的字节量

    std::vector<char*> _blocks; //char*是指向单字节的指针, 故可以拿来指代最小偏移粒度为单字节的内存block

    std::vector<void*> _large_alloc; //大于 _block_size 的内存申请, 单独申请. 在这里记录申请结果

    char* _current_block = nullptr; //内存池的当前正在使用的内存block指针

    size_t _offset = 0; //当前内存block已使用的位置偏移量

    std::mutex _mtx; //互斥锁, 保证多线程安全

public:
    explicit memory_pool(size_t _block_size = 4096); //默认给单个内存block申请4096字节的内存
    // explicit的意思是禁止对 memory_pool 类对象作隐式转换

    ~memory_pool(); //会调用 release() 释放内存池

    void* allocate(size_t size); //分配指定大小的内存, 如果当前块不足以容纳, 则申请新块
    // void*是 不定类型的指针，经常用于 malloc 和 new
    // allocate 拿到地址之后, 返回 void*, 后续用户根据实际需要, 将该地址转换为正确类型, 如:
    // void* ptr;
    // int*pInt = statc_cast<int*>(ptr);

    void dealloc_large(void* ptr, size_t size); //如果 ptr是大对象(大于_block_size)的，会被释放;否则不会被释放

    void reset(); //清空内存池, 复用下一次

    void release(); //释放内存池, 全部释放(block 和 large alloc都释放)

};


#endif