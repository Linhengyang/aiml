// memory_pool.h

#ifndef MEMORY_POOL_H
#define MEMORY_POOL_H

#include <vector>
#include <cstddef>
#include <cstdlib>
#include <mutex>








class memory_pool {

private:

    // 构造和析构函数写在private里，保证只能由 get_memory 方法获取内存池单例
    explicit memory_pool(size_t capacity = 4096); //默认给单个内存block申请 4kb 的内存
    // explicit的意思是禁止对 memory_pool 类对象作隐式转换

    ~memory_pool(); //会调用 release() 释放内存池


    const size_t _block_size; //内存池中, 单个内存block的字节量

    std::vector<block*> _blocks;

    std::vector<void*> _large_alloc; //大于 _block_size 的内存申请, 单独申请. 在这里记录申请结果

    block* _current_block = nullptr; //内存池的当前正在使用的内存block指针

    std::mutex _mtx; //互斥锁, 保证多线程安全

    // 禁止拷贝构造和赋值操作
    memory_pool(const memory_pool&) = delete;
    memory_pool& operator=(const memory_pool&) = delete;


public:

    // 获取 memory_pool 单例实例
    static memory_pool& get_mempool() {
        static memory_pool instance;  // 静态局部变量，确保全局只有一个实例
        return instance;
    }

    void* allocate(size_t size); //分配指定大小的内存, 如果当前块不足以容纳, 则申请新块
    // void*是 不定类型的指针，经常用于 malloc 和 new
    // allocate 拿到地址之后, 返回 void*, 后续用户根据实际需要, 将该地址转换为正确类型, 如:
    // void* ptr;
    // int*pInt = statc_cast<int*>(ptr);

    void dealloc_large(void* ptr, size_t size); //如果 ptr是大对象(大于_block_size)的，会被释放;否则不会被释放

    void reset(); //复用内存池

    void release(); //释放内存池, 全部释放(block 和 large alloc都释放)

};


#endif