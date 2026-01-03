// memory_pool.h


// TODO: memory_pool 的 全局单例模式要去掉. 为了支持 hash table 的复用, hash table 所使用的 memory pool 应该是和它自身单一绑定的
// 这样，一个hash table clear 自身时, reset 自身的 memory pool, 才使得这个hash table 可以复用. 不然 reset 会干扰其他所有分配在 memory pool上的对象

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


#endif