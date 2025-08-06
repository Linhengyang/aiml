// global_mempool.cpp

#include "memory_pool_global.h"
#include "memory_block.h"
#include <stdexcept>
#include <cstdint>
#include <cstring>
#include <cstddef>
#include <algorithm>



// TODO: memory_pool 的 全局单例模式要去掉. 为了支持 hash table 的复用, hash table 所使用的 memory pool 应该是和它自身单一绑定的
// 这样，一个hash table clear 自身时, reset 自身的 memory pool, 才使得这个hash table 可以复用. 不然 reset 会干扰其他所有分配在 memory pool上的对象

// 定义静态成员变量
// 用智能指针来管理单例
std::unique_ptr<global_mempool, global_mempool::Deleter> global_mempool::_instance = nullptr;
std::mutex global_mempool::_mtx;




// 构造函数. 放到 private 里导致它不对外使用.
global_mempool::global_mempool(size_t block_size, size_t alignment):
    _block_size(block_size),
    _blocks(std::vector<block*>()),
    _large_allocs(std::vector<void*>()),
    _alignment(alignment)
{

    // 确保对齐要求为2的幂
    if ((alignment & (alignment - 1)) != 0 || alignment == 0) {
        throw std::invalid_argument("alignment must be a power of 2.");
    }
    // block_size 不可以是 0
    if (block_size == 0) {
        throw std::invalid_argument("block_size must be > 0");
    }

}


// 析构函数. 放到 private 里导致外部不可使用（所以不加锁）. 必须通过公共接口destroy/release调用. 这些方法会加锁
global_mempool::~global_mempool() {
    // 析构时调用不加锁的 release_no_lock 方法. 因为析构已经被私有, 而释放内存池的公共接口 mempool_destroy 会带锁
    // 所以为了避免死锁, 这里无需
    release_no_lock();
}



void global_mempool::release_no_lock() {
    // 不加锁的 release 内存池, 给加锁的release等外部函数使用, 在外部加锁使用

    for(auto block: _blocks) {
        // 释放 _blocks vector 中每一个内存block. 调用 block 的析构函数
        delete block;
    }

    _blocks.clear(); // 清空 _blocks 容器里的所有元素

    for(auto ptr: _large_allocs) {
        // 释放 _large_alloc vector 中每一个 ptr
        std::free(ptr);
    }

    _large_allocs.clear();

    _current_block = nullptr;

}




void global_mempool::release() {

    // public方法 release 涉及到修改共享变量, 加锁只能由一个线程执行.
    std::lock_guard<std::mutex> lock(_mtx);

    global_mempool::release_no_lock();

}



// 申请分配 size 字节数量的内存.
void* global_mempool::allocate(size_t size) {

    std::lock_guard<std::mutex> lock(_mtx);

    if(size == 0) {return nullptr;}

    size = (size + _alignment) & ~(_alignment-1); // size 上跳对齐

    // 大于 _block_size 的内存申请
    if(size > _block_size) { // 单独申请, 并记录在 _larg_allocs 中
        void* ptr = std::malloc(size);
        if(!ptr) {
            throw std::bad_alloc();
        }
        _large_allocs.push_back(ptr); // 记录该次 large allocate. release里一起释放

        return ptr;
    }

    // 小于等于 _block_size 的内存申请
    if (!_current_block || _current_block->get_remaining() < size) { // 若当前 block 不够分配size
        // 遍历 _blocs，寻找剩余空间足够的 block
        bool found = false;

        for (auto block : _blocks) {
            if (block->get_remaining() >= size) {
                _current_block = block;
                found = true;
                break;
            }
        }

        // 遍历之后还是没有找到足够分配 size 的block, 就新创建一个 block
        if (!found) {

            block* new_block = new block(_block_size, _alignment);

            if(!new_block) {
                throw std::bad_alloc();
            }

            _blocks.push_back(new_block); // 记录新的内存block
            _current_block = new_block;
        }

    }

    return _current_block->allocate(size);
}





//如果 ptr是大对象(大于_block_size)的，会被释放;否则不会被释放
//如果ptr在 _large_allocs 里，那么释放它，然后从 _large_allocs 中删除它。否则不作任何操作
void global_mempool::dealloc_large(void* ptr) {
    // 单独释放大内存, 加互斥锁
    std::lock_guard<std::mutex> lock(_mtx);

    // 在 _large_allocs 中寻找 ptr
    auto it = std::find(_large_allocs.begin(), _large_allocs.end(), ptr);
    if(it != _large_allocs.end()) {
        std::free(ptr); // 找到了 ptr. 大对象单独释放
        _large_allocs.erase(it); // 从 _large_allocs中删除它
    }
}






// reset 内存池: reset 所有内存 block, release 所有 large_allocation
void global_mempool::reset() {

    std::lock_guard<std::mutex> lock(_mtx);

    for(auto block: _blocks) {
        block->reset();
    }

    if (!_blocks.empty()) {
        _current_block = _blocks[0];
    }
    else {
        _current_block = nullptr;
    }

    // 释放 _large_allocs 中每一个 ptr, 清空 _large_allocs
    for(auto ptr: _large_allocs) {
        std::free(ptr);
    }

    _large_allocs.clear();

}








// shrink 内存池: 把 _offset == 0 的block销毁掉（_current_block除外,保证极端情况下不会销毁所有block）
// 为了防止大起大落（第一波用了很多内存块，第二波用了很少，第三波又用了很多），每次shrink最多max_num个
// 必须要在 reset之前用. 因为reset会重置所有block._offset=0
// 下一步紧接着就是 reset
void global_mempool::shrink(size_t max_num) {
    std::lock_guard<std::mutex> lock(_mtx);

    // 如果当前block指向空, 说明还没有创建任何一个block, 那么无可shrink，直接退出
    if(!_current_block) {
        return;
    }

    std::vector<block*> _released_block = {};

    for (auto block: _blocks) {
        // 遍历 block. 如果 block 不是 _current_block, 且 usage = 0, 销毁
        if (block != _current_block && block->get_offset() == 0) {
            // 把这个block放进已删除列表
            _released_block.push_back(block);
            // 释放这个block
            delete block;
            // 数量达到最大值了，就退出shrink
            if(_released_block.size() == max_num) {
                break;
            }
        }
    }

    // 把已删除列表中的block从_blocks中删除
    for (auto block: _released_block) {
        auto it = std::find(_blocks.begin(), _blocks.end(), block);
        if(it != _blocks.end()) {
            _blocks.erase(it);
        }
    }
}