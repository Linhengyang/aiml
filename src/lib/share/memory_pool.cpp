// memory_pool.cpp

#include "memory_pool.h"
#include <stdexcept>
#include <algorithm>




memory_pool::memory_pool(size_t block_size): _block_size(block_size){
    if (block_size == 0) {
        throw std::invalid_argument("block_size must be > 0");
    }
}



memory_pool::~memory_pool() {
    release();
}



void memory_pool::release() {
    for(auto block: _blocks) {
        // 释放 _blocks vector 中每一个 内存block
        std::free(block); // 用 std::malloc 分配的, 要用 std::free 来释放
    }
    _blocks.clear(); // 清空 _blocks 容器里的所有元素

    for(auto ptr: _large_alloc) {
        // 释放 _large_alloc vector 中每一个 ptr
        std::free(ptr);
    }
    _large_alloc.clear();

    _current_block = nullptr;
    _offset = 0;
}



// 申请分配 size 字节数量的内存.
// 如果 size 是小于等于内存block，那么分配相应大小的可复用的内存. 统一用release释放所有block, 用reset复用block
// 如果size是超过内存block的大容量，那么分配相应大小不可复用的内存. 可以用 dealloc_large(ptr, size)单独释放, 也可以随着release统一释放
void* memory_pool::allocate(size_t size) {
    // 线程A在函数域内调用对象mempool的allocate方法以获得内存，这个时候lock_guard类就在线程A的作用域里生成了，并传入互斥量mtx_，生成对象 lock 。
    // 这个时候如果另一个线程试图调用mempool的allocate方法，它就拿不到互斥量mtx_，所以就只能堵塞，这样就能保证线程A allocate拿到的内存只有线程A能用。
    // 当线程A函数在return后，对象 lock 就离开了线程A的作用域，自动销毁，那么互斥量mtx_就被释放了
    std::lock_guard<std::mutex> lock(_mtx);

    size = (size + 7) & (~7); // 把申请字节数向上对齐到最近的8的倍数. 现代cpu的特性. 线程的特性

    if(size > _block_size) { // 大于 _block_size 的内存申请: 单独申请, 并记录在 _larg_alloc 中
        void* ptr = std::malloc(size);
        if(!ptr) {
            throw std::bad_alloc();
        }
        _large_alloc.push_back(ptr); // 记录该次 large allocate. release里一起释放

        return ptr;
    }


    if(!_current_block || _offset + size > _block_size) {
        // 若当前内存块block指针为空, 或者当前内存块block偏移量+申请量 > 内存块block总量
        // 说明当前内存块block对于线程不够用了，要申请新的内存块block
        _current_block = static_cast<char*>(std::malloc(_block_size));
        if(!_current_block) {
            throw std::bad_alloc();
        }
        _blocks.push_back(_current_block); // 记录新的内存block
        _offset = 0; // 新的内存block偏移量为0
    }

    void* ptr = _current_block + _offset;
    _offset += size;
    
    return ptr;
}



//如果 ptr是大对象(大于_block_size)的，会被释放;否则不会被释放
void memory_pool::dealloc_large(void* ptr, size_t size) {
    // 单独释放大内存, 加互斥锁
    std::lock_guard<std::mutex> lock(_mtx);

    size = (size + 7) & (~7);

    if(size > _block_size) {
        std::free(ptr); // 大对象单独释放

        // ptr已经释放了，要从 _large_alloc vector 中删除, 避免在release中再次释放
        auto it = std::find(_large_alloc.begin(), _large_alloc.end(), ptr);
        if(it != _large_alloc.end()) {
            _large_alloc.erase(it);
        }
    }
}



void memory_pool::reset() {
    std::lock_guard<std::mutex> lock(_mtx);
    _offset = 0; // 重置当前内存block的偏移量为0, 意思是该block要重新复用
}