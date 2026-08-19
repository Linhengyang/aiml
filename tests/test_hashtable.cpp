#include "mempooled_hashtable.h"
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

    // 创建 内存池单例: 单线程版本即可
    size_t block_size = 40LL * 172470436LL;
    singleton_mempool& pool = singleton_mempool::get(block_size, 64);

    // 创建 哈希器
    hasher my_hasher;

    // 创建 哈希表: 单线程版本
    pooled_hashtable<int, int, singleton_mempool, hasher> hashtable(my_hasher, 1, &pool);

    // 插入 node
    hashtable.insert(1, 4);

    // get node
    int v = 0;
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


    // 测试 迭代器
    using MyMap = pooled_hashtable<std::string, std::string, singleton_mempool>;

    std::cout << "--- Test 1: Full Drain ---\n";
    {
        MyMap map(16, &pool);
        map.insert("key1", "value1");
        map.insert("key2", "value2");
        map.insert("key3", "value3");

        for (auto&& [k, v] : map.drain()) {
            std::cout << "Drained: " << k << " -> " << v << "\n";
        }
        // 析构时 _fully_drained = true，不会调用 clear()
    }

    std::cout << "\n--- Test 2: Partial Drain (Break Early) ---\n";
    {
        MyMap map(16,  &pool);
        map.insert("A", "1");
        map.insert("B", "2");
        map.insert("C", "3");
        map.insert("D", "4");

        int count = 0;
        for (auto&& [k, v] : map.drain()) {
            std::cout << "Drained: " << k << "\n";
            count++;
            if (count == 2) {
                std::cout << "Breaking early!\n";
                break; // 中途退出，测试是否会 Double Destruct
            }
        }
        // 析构时 _fully_drained = false，会调用 map.clear()
        // 如果 operator* 中手动析构了，这里 clear() 会引发崩溃。
        // 修正后的代码将析构移到 operator++，此处安全通过！
    }

    std::cout << "\nAll tests passed safely!\n";


    // 手动销毁内存池
    pool.destroy();

    return 0;
}