#include "memory_pool_singleton.h"
#include <iostream>
#include <cassert>

int main() {
    // 开始测试前，确保未初始化
    assert(!singleton_mempool::exist());
    std::cout << "✔ 未初始化状态正确\n";

    // 创建单例
    singleton_mempool& pool = singleton_mempool::get(1024, 16);
    assert(singleton_mempool::exist());
    std::cout << "✔ 单例创建成功\n";

    // 再次获取，确保是同一个实例
    singleton_mempool& pool2 = singleton_mempool::get();
    assert(&pool == &pool2);
    std::cout << "✔ 获取同一个单例实例成功\n";

    // 检查已初始化状态
    assert(singleton_mempool::exist());
    std::cout << "✔ 单例已初始化状态检查成功\n";

    // 分配内存
    void* ptr = pool.allocate(64);
    assert(ptr != nullptr);
    std::cout << "✔ 分配内存成功\n";

    // 对齐检查（可选，简单方式）
    assert(reinterpret_cast<uintptr_t>(ptr) % 16 == 0);
    std::cout << "✔ 分配内存对齐正确\n";

    // 释放内存
    pool.release();
    std::cout << "✔ 释放内存成功\n";

    // 销毁单例
    singleton_mempool::destroy();
    assert(!singleton_mempool::exist());
    std::cout << "✔ 单例销毁成功\n";

    std::cout << "\n✅ 所有 memory_pool 单例测试通过\n";
    return 0;
}