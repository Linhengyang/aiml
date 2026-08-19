#include "mempooled_concurrent_hashtable.h"
#include "memory_pool_singleton.h"
#include <iostream>
#include <cassert>
#include <thread>
#include <string>
#include <cassert>
#include <chrono>

using namespace std;

// 定义哈希 counter_key 的哈希器. 这里 hasher 是一个函数类, 通过实例化得到哈希器 hasher myHasher;
struct hasher {
    int operator()(const int& key) const {
        return key;
    }
};


void test1_concurrent_hash_map() {

    // 创建 内存池单例: 并发哈希表必须要使用 线程安全的内存池
    size_t block_size = 40LL * 172470436LL;
    threadsafe_singleton_mempool& pool = threadsafe_singleton_mempool::get(block_size, 64);

    // 创建 哈希器
    hasher my_hasher;

    pooled_concurrent_hashtable<int, int, threadsafe_singleton_mempool, hasher> map(my_hasher, 4096, &pool);
    const int num_threads = 8;
    const int ops_per_thread = 10000;

    // 启动写线程：每个线程插入自己的 key 范围
    vector<thread> writers;
    for (int t = 0; t < num_threads; ++t) {
        writers.emplace_back([&, t]() {
            for (int i = 0; i < ops_per_thread; ++i) {
                int key = t * ops_per_thread + i;
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

    // 测试删除-插入
    vector<thread> deleters;
    for (int t = 0; t < num_threads / 2; ++t) {
        deleters.emplace_back([&, t]() {
            for (int i = 0; i < ops_per_thread; ++i) {
                int key = t * ops_per_thread + i;
                int val;
                bool got = map.pop(key, val);
                assert(got);
                assert(val == key * 2);
                map.insert(key, key * 3);
            }
        });
    }

    for (auto& d : deleters) {
        d.join();
    }

    // 检查剩余元素数量
    assert(map.size() == (num_threads) * ops_per_thread);

    // clear 哈希表
    map.clear();

    // 输出size
    cout << map.size() << endl;

    // 重新启动写线程：reset pool 之后, 复用 hashtable
    pool.reset();

    // 再次启动写线程：每个线程插入自己的 key 范围
    vector<thread> writer2;
    for (int t = 0; t < num_threads; ++t) {
        writer2.emplace_back([&, t]() {
            for (int i = 0; i < ops_per_thread; ++i) {
                int key = t * ops_per_thread + i;
                map.insert(key, key * 2);
            }
        });
    }

    // 启动读线程：随机读取已写入的 key（这里简化为等待写完再读）
    for (auto& w : writer2) {
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

    // 这次是 销毁 哈希表
    map.destroy();

    // 销毁内存池
    pool.destroy();

    cout << "✅ ConcurrentHashMap test passed!" << endl;
}



void test2_concurrent_hash_map() {
    // 创建 内存池单例: 并发哈希表必须要使用 线程安全的内存池
    size_t block_size = 40LL * 172470436LL;
    threadsafe_singleton_mempool& pool = threadsafe_singleton_mempool::get(block_size, 64);

    // 创建 哈希器
    // hasher my_hasher;

    using MapType = pooled_concurrent_hashtable<int, std::string, threadsafe_singleton_mempool>;
    MapType map(1024, &pool, 64);

    std::cout << "=== Test 1: Basic Insert & Get ===" << std::endl;
    for (int i = 0; i < 1000; ++i) {
        map.insert(i, "value_" + std::to_string(i));
    }
    assert(map.size() == 1000);

    std::string val;
    assert(map.get(500, val) && val == "value_500");
    std::cout << "Basic test passed. Size: " << map.size() << std::endl;

    std::cout << "\n=== Test 2: Concurrent Insert & Pop ===" << std::endl;
    map.clear();
    pool.reset();

    const int NUM_THREADS = 8;
    const int OPS_PER_THREAD = 50000;
    std::vector<std::thread> threads;

    auto start = std::chrono::high_resolution_clock::now();

    for (int t = 0; t < NUM_THREADS; ++t) {
        threads.emplace_back([&, t]() {
            for (int i = 0; i < OPS_PER_THREAD; ++i) {
                int key = t * OPS_PER_THREAD + i;
                map.insert(key, "thread_" + std::to_string(t));
            }
            for (int i = 0; i < OPS_PER_THREAD; ++i) {
                int key = t * OPS_PER_THREAD + i;
                std::string out;
                map.pop(key, out);
            }
        });
    }

    for (auto& th : threads) th.join();
    
    auto end = std::chrono::high_resolution_clock::now();
    std::cout << "Concurrent Insert/Pop finished. Final Size: " << map.size() 
              << " (Expected 0). Time: " 
              << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << " ms" << std::endl;
    assert(map.size() == 0);

    std::cout << "\n=== Test 3: Concurrent Clear (Generation Stress Test) ===" << std::endl;
    std::atomic<bool> stop_flag{false};

    // 线程 A: 疯狂 insert 和 pop
    std::thread worker([&]() {
        int i = 0;
        while (!stop_flag.load(std::memory_order_relaxed)) {
            map.insert(i, "worker_val");
            std::string out;
            map.pop(i, out);
            i++;
        }
    });

    // 线程 B: 疯狂 clear 和 reset arena (模拟极端并发破坏)
    std::thread clearer([&]() {
        for (int i = 0; i < 100; ++i) {
            map.clear();
            pool.reset(); // 模拟外部 reset arena
            std::this_thread::sleep_for(std::chrono::microseconds(100));
        }
        stop_flag.store(true, std::memory_order_relaxed);
    });

    worker.join();
    clearer.join();

    std::cout << "Stress test with clear/reset passed without crashing!" << std::endl;

    map.destroy();
    std::cout << "\nAll tests passed successfully!" << std::endl;
}

int main() {
    test2_concurrent_hash_map();
    return 0;
}