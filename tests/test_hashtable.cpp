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

// constexpr size_t next_pow2(size_t x) {
//     if (x <= 1) return 1;
//     --x; x |= x >> 1; x |= x >> 2; x |= x >> 4; x |= x >> 8; x |= x >> 16;
// #if SIZE_MAX > 0xFFFFFFFFu
//     x |= x >> 32;
// #endif
//     return x + 1;
// };

int main() {

    // 创建 内存池单例
    size_t block_size = 40LL * 172470436LL;
    singleton_mempool& pool = singleton_mempool::get(block_size, 64);

    // 创建 哈希器
    hasher my_hasher;

    // 创建 线程安全的 哈希表
    hash_table_mt_chain<uint32_t, uint64_t, singleton_mempool, hasher> hashtable(my_hasher, 5e8, &pool);

    // 手动销毁 哈希表
    hashtable.destroy();

    // 手动销毁内存池
    pool.destroy();

    return 0;
}