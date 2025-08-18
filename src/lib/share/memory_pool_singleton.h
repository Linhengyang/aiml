// singleton_mempool.h

#ifndef MEMORY_POOL_SINGLETON_H
#define MEMORY_POOL_SINGLETON_H

#include <vector>
#include <cstddef>
#include <cstdlib>
#include <mutex>
#include <memory>
#include "memory_block.h"



// void*是 不定类型的指针，经常用于 malloc 和 new
// allocate 拿到地址之后, 返回 void*, 后续用户根据实际需要, 将该地址转换为正确类型, 如:
// void* ptr;
// int* pInt = statc_cast<int*>(ptr);

class singleton_mempool {

private:

    // 构造和析构函数写在private里，保证只能由 get_mempool 方法创建内存池实例
    explicit singleton_mempool(size_t block_size = 1048576, size_t alignment = 64); //默认给单个内存block申请1MB内存,64字节对齐
    // explicit的意思是禁止对 singleton_mempool 类对象作隐式转换

    // 析构函数被private, 导致如下后果:
    // 1. 禁止在栈上创建对象，对象离开作用域时，编译器需要调用析构函数，而它不能访问private
    // 2. 禁止delete表达式来调用析构
    // 3. 无法成为其他类的成员变量，因为包含该成员的类在被销毁时，需要调用成员对象的析构函数
    // 4. 无法作为函数的返回值(按值返回)，因为返回过程中临时对象需要被析构
    // 5. 禁止std::vector/std::list等标准容器存储, 因为容器需要访问元素的析构函数
    // 单例模式下的对象，适合把析构函数private：这样可以由 唯一指定的接口来释放销毁对象，以此手动控制单例的生命周期
    ~singleton_mempool();

    const size_t _block_size; //内存池中, 单个内存block的字节量

    const size_t _alignment; //内存池的对齐参数

    std::vector<block*> _blocks;

    std::vector<void*> _large_allocs; //大于 _block_size 的内存申请, 单独申请. 在这里记录申请结果

    block* _current_block = nullptr; //内存池的当前正在使用的内存block起始位置指针

    void release_no_lock(); //不带锁释放内存池, 全部释放(block 和 large alloc都释放), 私有供析构和智能指针reset使用

    // 单例模式: 静态成员是保证该类只能有一个实现，在类层面定义. 静态成员变量必须要在类外定义, 类内只是申明

    // 自定义删除器结构体
    struct Deleter {
        void operator()(singleton_mempool* ptr) const {
            delete ptr;  // 这里可以访问 private 的析构. 删除器内部不加锁, 因为调用删除器前 destroy 会加锁
        }
    };

    // 声明删除器为友元，让它能访问私有析构
    friend struct Deleter;

    // 声明静态成员变量 单例指针
    static std::unique_ptr<singleton_mempool, Deleter> _instance;

    // 声明静态成员变量 互斥锁
    static std::mutex _mtx;
    // 互斥锁, 保证多线程安全。比如在allocate方法里加入互斥锁，那么会发生如下：
    // 线程A在函数域内调用对象 singleton_mempool 的allocate方法以获得内存，这个时候lock_guard类就在线程A的作用域里生成了，
    // 传入互斥量mtx_，生成对象 lock. 这个时候如果另一个线程试图调用 singleton_mempool 的allocate方法，它就拿不到互斥量mtx_，
    // 所以就只能堵塞，这样就能保证线程A allocate拿到的内存只有线程A能用。
    // 当线程A函数在return后，对象 lock 就离开了线程A的作用域，自动销毁，那么互斥量mtx_就被释放了

    // 禁止拷贝构造和赋值操作
    singleton_mempool(const singleton_mempool&) = delete;
    singleton_mempool& operator=(const singleton_mempool&) = delete;


public:
    // 类的静态方法不依赖实例（故也不能使用非静态成员/方法），适合用来定义单例：用单例类就可以调用，单例类约等于单例实例
    // 调用方法:  类名::静态方法名
    // 带参数调用: 创建 singleton_mempool 单例
    static singleton_mempool& get(size_t block_size, size_t alignment) {
        std::lock_guard<std::mutex> lock(_mtx);
        if (!_instance) {
            _instance.reset(new singleton_mempool(block_size, alignment));
            // _instance = std::unique_ptr<singleton_mempool, Deleter>(new singleton_mempool(block_size, alignment));
        }
        return *_instance;
    }

    // 不带参数调用: 获得已经实现的单例（加锁以保证线程安全）. 若尚未创建，报错
    static singleton_mempool& get() {
        std::lock_guard<std::mutex> lock(_mtx);
        if(!_instance) {
            throw std::runtime_error("singleton_mempool not created");
        }
        return *_instance;
    }

    // 探知单例是否已经创建
    static bool exist() {
        std::lock_guard<std::mutex> lock(_mtx);
        return _instance != nullptr;
    }

    // 带锁释放内存池的公共接口，效果和release相同，但是是静态方法，可以并推荐用 类名::方法 来使用
    static void destroy() {
        std::lock_guard<std::mutex> lock(_mtx); // 加锁
        _instance.reset(); // 智能指针 reset 会调用 删除器, 删除器作为友好类, 可以访问访问 private 的析构
    }

    // 非静态方法，调用方法:  实例.方法名, 单例模式下这样调用：类名::get().方法名

    // 分配指定大小的内存, 非静态, 依赖成员变量 _blocks 等
    void* allocate(size_t size); // 如果当前块不足以容纳, 则申请新块

    void dealloc_large(void* ptr); //如果 ptr 记录在 _large_allocs 中，会被释放; 否则不会被释放

    void shrink(size_t max_num = 1); //缩小block数量, 释放部分尚未使用的block. 必须要在 reset之前用. 因为reset会重置所有block.used=false

    // 复用内存池的公共接口
    void reset(); // 全部复用，所有已经申请好的内存block都reset成从头可用

    // 带锁释放内存池的公共接口，效果和 destroy 相同，但是是非静态方法
    void release() ; // 全部释放(block 和 large alloc都释放), 带锁以线程安全

}; // end of singleton_mempool




// unsafe_singleton_mempool 线程不安全、不带锁的 单例模式内存池：单线程使用




class unsafe_singleton_mempool {

private:

    explicit unsafe_singleton_mempool(size_t block_size = 1048576, size_t alignment = 64);

    ~unsafe_singleton_mempool();

    const size_t _block_size;

    const size_t _alignment;

    std::vector<block*> _blocks;

    std::vector<void*> _large_allocs;

    block* _current_block = nullptr;

    // 单例模式: 静态成员是保证该类只能有一个实现，在类层面定义. 静态成员变量必须要在类外定义, 类内只是申明

    struct Deleter {
        void operator()(unsafe_singleton_mempool* ptr) const {
            delete ptr;
        }
    };

    friend struct Deleter;

    static std::unique_ptr<unsafe_singleton_mempool, Deleter> _instance;

    unsafe_singleton_mempool(const unsafe_singleton_mempool&) = delete;
    unsafe_singleton_mempool& operator=(const unsafe_singleton_mempool&) = delete;


public:

    static unsafe_singleton_mempool& get(size_t block_size, size_t alignment) {
        if (!_instance) {
            _instance.reset(new unsafe_singleton_mempool(block_size, alignment));
        }
        return *_instance;
    }

    static unsafe_singleton_mempool& get() {
        if(!_instance) {
            throw std::runtime_error("unsafe_singleton_mempool not created");
        }
        return *_instance;
    }

    static bool exist() {
        return _instance != nullptr;
    }

    static void destroy() {
        _instance.reset();
    }

    void* allocate(size_t size);

    void dealloc_large(void* ptr);

    void shrink(size_t max_num = 1);

    void reset();

    void release() ;

}; // end of unsafe_singleton_mempool




#endif