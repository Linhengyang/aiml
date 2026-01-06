// memory_pool.h


// memory_pool 只负责一次次由 allocate 方法返回分配好的符合条件的 内存地址，并不在乎其上会被定义成什么数据结构.
// 具体构造是在拿到该地址后，由具体非平凡结构的构造函数placement new，或平凡构造.
// 其上的数据结构也只负责自身的析构，而不应涉及 memory pool 的复位reset / 释放 release 等操作.

#ifndef MEMORY_POOL_H
#define MEMORY_POOL_H

#include <vector>
#include <cstddef>
#include <cstdlib>
#include <mutex>
#include <memory>
#include "memory_block.h"




class mempool {

private:

    const size_t _block_size; //内存池中, 单个内存block的字节量

    const size_t _alignment; //内存池的对齐参数

    std::vector<block*> _blocks; // 内存块 指针数组

    std::vector<void*> _large_allocs; //大于 _block_size 的内存申请, 单独申请. 在这里记录申请结果

    block* _current_block = nullptr; //内存池的当前正在使用的内存block起始位置指针

    // 互斥锁以保障线程安全. 非单例模式下只需满足自身成员变量安全访问即可, 故使用类成员的锁而非静态
    // mutable关键字修饰非const非static非引用成员变量, 表达即使在 const 成员函数里, 被mutable修饰的成员变量仍然可以改变
    
    // 虽然当前没有 const 成员函数，仍然建议保留 mutable 修饰 _mtx，因为：
    // 它为未来扩展提供灵活性
    // 它符合 C++ 并发编程的最佳实践
    // 它清晰表达了“这个成员仅用于同步，不影响逻辑状态”的意图
    mutable std::mutex _mtx;

    // 尽管 mempool 不是全局单例模式, 但仍然禁止拷贝构造和赋值操作.
    mempool(const mempool&) = delete;
    mempool& operator=(const mempool&) = delete;


public:

    // 构造函数. 构造 内存池 实例
    explicit mempool(size_t block_size = 1048576, size_t alignment = 64); //默认给单个内存block申请 1MB 的内存, 64字节对齐

    // 析构函数. 调用 release 释放 内存池 实例
    ~mempool();

    // 非静态方法，调用方法:  实例.方法名

    // 分配指定大小的内存, 非静态, 依赖成员变量 _blocks 等
    void* allocate(size_t size); // 如果当前块不足以容纳, 则申请新块

    void dealloc_large(void* ptr) ; //如果 ptr 记录在 _large_allocs 中，会被释放; 否则不会被释放

    void shrink(size_t max_num = 1) ; //缩小block数量, 释放部分尚未使用的block. 必须要在 reset之前用. 因为reset会重置所有block.used=false

    // 复用内存池的公共接口
    void reset(); // 全部复用，所有已经申请好的内存block都reset成从头可用

    // 带锁释放内存池的公共接口
    void release(); // 全部释放(block 和 large alloc都释放), 带锁以线程安全

}; // end of mempool





// un_threadsafe_mempool 线程不安全、不带锁的 内存池：单线程使用




class un_threadsafe_mempool {

private:

    const size_t _block_size; //内存池中, 单个内存block的字节量

    const size_t _alignment; //内存池的对齐参数

    std::vector<block*> _blocks; // 内存块 指针数组

    std::vector<void*> _large_allocs; //大于 _block_size 的内存申请, 单独申请. 在这里记录申请结果

    block* _current_block = nullptr; //内存池的当前正在使用的内存block起始位置指针

    // 尽管 un_threadsafe_mempool 不是全局单例模式, 但仍然禁止拷贝构造和赋值操作.
    un_threadsafe_mempool(const un_threadsafe_mempool&) = delete;
    un_threadsafe_mempool& operator=(const un_threadsafe_mempool&) = delete;


public:

    // 构造函数. 构造 内存池 实例
    explicit un_threadsafe_mempool(size_t block_size = 1048576, size_t alignment = 64); //默认给单个内存block申请 1MB 的内存, 64字节对齐

    // 析构函数. 调用 release 释放 内存池 实例
    ~un_threadsafe_mempool();

    // 非静态方法，调用方法:  实例.方法名

    // 分配指定大小的内存, 非静态, 依赖成员变量 _blocks 等
    void* allocate(size_t size); // 如果当前块不足以容纳, 则申请新块

    void dealloc_large(void* ptr) ; //如果 ptr 记录在 _large_allocs 中，会被释放; 否则不会被释放

    void shrink(size_t max_num = 1) ; //缩小block数量, 释放部分尚未使用的block. 必须要在 reset之前用. 因为reset会重置所有block.used=false

    // 复用内存池的公共接口
    void reset(); // 全部复用，所有已经申请好的内存block都reset成从头可用

    // 带锁释放内存池的公共接口
    void release(); // 全部释放(block 和 large alloc都释放), 带锁以线程安全

}; // end of un_threadsafe_mempool



#endif