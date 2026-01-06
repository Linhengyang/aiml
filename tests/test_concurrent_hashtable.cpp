#include "mempool_hash_table_mt.h"
#include "mempool_hash_table_st.h"
#include "memory_pool_singleton.h"
#include <iostream>
#include <cassert>
#include <thread>

using namespace std;

// 定义哈希 counter_key 的哈希器. 这里 hasher 是一个函数类, 通过实例化得到哈希器 hasher myHasher;
struct hasher {
    uint32_t operator()(const uint32_t& key) const {
        return key;
    }
};


void test_concurrent_hash_map() {

    // 创建 内存池单例
    size_t block_size = 40LL * 172470436LL;
    unsafe_singleton_mempool& pool = unsafe_singleton_mempool::get(block_size, 64);

    // 创建 哈希器
    hasher my_hasher;

    hash_table_mt_chain<uint32_t, int, unsafe_singleton_mempool, hasher> map(my_hasher, 128, &pool);
    const int num_threads = 2;
    const int ops_per_thread = 16;

    // 启动写线程：每个线程插入自己的 key 范围
    vector<thread> writers;
    for (int t = 0; t < num_threads; ++t) {
        writers.emplace_back([&, t]() {
            for (int i = 0; i < ops_per_thread; ++i) {
                uint32_t key = t * ops_per_thread + i;
                map.insert(key, key * 2);
            }
        });
    }

    // 启动读线程：随机读取已写入的 key（这里简化为等待写完再读）
    for (auto& w : writers) {
        w.join();
    }

    // 验证所有写入都成功
    assert(map.size() == num_threads * ops_per_thread);

    for (int t = 0; t < num_threads; ++t) {
        for (int i = 0; i < ops_per_thread; ++i) {
            int key = t * ops_per_thread + i;
            int val;
            bool found = map.get(key, val);
            assert(found);
            assert(val == key * 2);
        }
    }

    // // 测试删除
    // vector<thread> deleters;
    // for (int t = 0; t < num_threads / 2; ++t) {
    //     deleters.emplace_back([&, t]() {
    //         for (int i = 0; i < ops_per_thread; ++i) {
    //             int key = t * ops_per_thread + i;
    //             map.remove(key);
    //         }
    //     });
    // }

    // for (auto& d : deleters) {
    //     d.join();
    // }

    // // 检查剩余元素数量
    // assert(map.size() == (num_threads / 2) * ops_per_thread);

    // clear 哈希表
    map.clear();

    // 输出size
    cout << map.size() << endl;

    // 手动销毁 哈希表
    map.destroy();

    // 手动销毁内存池
    pool.destroy();

    cout << "✅ ConcurrentHashMap test passed!" << endl;
}

int main() {
    test_concurrent_hash_map();
    return 0;
}