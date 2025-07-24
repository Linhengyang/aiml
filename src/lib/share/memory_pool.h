// memory_pool.h

#ifndef MEMORY_POOL_H
#define MEMORY_POOL_H

#include <vector>
#include <cstddef>
#include <cstdlib>
#include <mutex>
#include <memory>
#include <memory_block.h>


// void*是 不定类型的指针，经常用于 malloc 和 new
// allocate 拿到地址之后, 返回 void*, 后续用户根据实际需要, 将该地址转换为正确类型, 如:
// void* ptr;
// int* pInt = statc_cast<int*>(ptr);

class memory_pool {

private:

    // 构造和析构函数写在private里，保证只能由 get_mempool 方法创建内存池实例
    explicit memory_pool(size_t block_size = 4096, size_t alignment = 8); //默认给单个内存block申请 4kb 的内存, 8字节对齐
    // explicit的意思是禁止对 memory_pool 类对象作隐式转换

    ~memory_pool(); //会调用 release() 释放内存池

    const size_t _block_size; //内存池中, 单个内存block的字节量

    const size_t _alignment; //内存池的对齐参数

    std::vector<block*> _blocks;

    std::vector<void*> _large_allocs; //大于 _block_size 的内存申请, 单独申请. 在这里记录申请结果

    block* _current_block = nullptr; //内存池的当前正在使用的内存block起始位置指针


    void release_no_lock(); //不带锁释放内存池, 全部释放(block 和 large alloc都释放), 私有供析构和智能指针reset使用


    // 静态成员是保证该类只能有一个实现，在类层面定义. 静态成员变量必须要在类外定义, 类内只是申明

    // 自定义删除器结构体
    struct Deleter {
        void operator()(memory_pool* ptr) const {
            delete ptr;  // 这里可以访问 private ~memory_pool()
        }
    };

    // 声明删除器为友元，让它能访问私有析构
    friend struct Deleter;

    // 静态成员变量 单例指针
    static std::unique_ptr<memory_pool, Deleter> _instance;



    // 静态成员变量 互斥锁
    static std::mutex _mtx;
    // 互斥锁, 保证多线程安全。比如在allocate方法里加入互斥锁，那么会发生如下：
    // 线程A在函数域内调用对象mempool的allocate方法以获得内存，这个时候lock_guard类就在线程A的作用域里生成了，传入互斥量mtx_，生成对象 lock 
    // 这个时候如果另一个线程试图调用mempool的allocate方法，它就拿不到互斥量mtx_，所以就只能堵塞，这样就能保证线程A allocate拿到的内存只有线程A能用。
    // 当线程A函数在return后，对象 lock 就离开了线程A的作用域，自动销毁，那么互斥量mtx_就被释放了

    // 禁止拷贝构造和赋值操作
    memory_pool(const memory_pool&) = delete;
    memory_pool& operator=(const memory_pool&) = delete;


public:

    // 带参数调用: 创建 memory_pool 单例
    static memory_pool& get_mempool(size_t block_size, size_t alignment) {
        std::lock_guard<std::mutex> lock(_mtx);
        if (!_instance) {
            _instance.reset(new memory_pool(block_size, alignment));
            // _instance = std::unique_ptr<memory_pool, Deleter>(new memory_pool(block_size, alignment));
        }
        return *_instance;
    }

    // 不带参数调用: 获得已经实现的单例. 若尚未创建，报错
    static memory_pool& get_mempool() {
        std::lock_guard<std::mutex> lock(_mtx);
        if(!_instance) {
            throw std::runtime_error("memory_pool not created");
        }
        return *_instance;
    }

    // 探知单例是否已经创建
    static bool mempool_exist() {
        std::lock_guard<std::mutex> lock(_mtx);
        return _instance != nullptr;
    }

    // 销毁单例.
    static void mempool_destroy() {
        std::lock_guard<std::mutex> lock(_mtx); // 加锁
        _instance.reset(); // 智能指针 reset 会以友好身份调用 析构函数. 所以资源会被自动release
    }

    void* allocate(size_t size); //分配指定大小的内存, 如果当前块不足以容纳, 则申请新块

    void dealloc_large(void* ptr); //如果 ptr 记录在 _large_allocs 中，会被释放; 否则不会被释放

    void shrink(size_t max_num = 1); //缩小block数量, 释放部分尚未使用的block. 必须要在 reset之前用. 因为reset会重置所有block.used=false

    void reset(); //复用内存池（全部复用，所有已经申请好的内存block都reset成从头可用）

    void release(); //带锁释放内存池, 全部释放(block 和 large alloc都释放), 带锁以线程安全

};



#endif