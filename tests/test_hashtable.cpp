#include "mempool_hash_table_mt.h"
#include "mempool_hash_table_st.h"
#include "memory_pool_singleton.h"
#include <iostream>
#include <cassert>

// 定义哈希 counter_key 的哈希器. 这里 hasher 是一个函数类, 通过实例化得到哈希器 hasher myHasher;
struct hasher {
    uint32_t operator()(const uint32_t& key) const {
        return key;
    }
};


int main() {

    // 创建 内存池单例
    size_t block_size = 40LL * 172470436LL;
    singleton_mempool& pool = singleton_mempool::get(block_size, 64);

    // 创建 哈希器
    hasher my_hasher;

    // 创建 线程安全的 哈希表
    hash_table_mt_chain<uint32_t, uint64_t, singleton_mempool, hasher> hashtable(my_hasher, 1, &pool);

    // 插入 node
    hashtable.insert(1, 4);

    // get node
    uint64_t v = 0;
    std::cout << "get key = 0 " << hashtable.get(2, v) << " where val = " << v << std::endl;

    // 插入 node, 触发rehash
    hashtable.insert(2, 5);

    // 输出size
    std::cout << hashtable.size() << std::endl;

    // clear 哈希表
    hashtable.clear();

    // 输出size
    std::cout << hashtable.size() << std::endl;

    // 手动销毁 哈希表
    hashtable.destroy();

    // 手动销毁内存池
    pool.destroy();

    return 0;
}