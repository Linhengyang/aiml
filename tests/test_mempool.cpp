#include "memory_pool.h"
#include <iostream>
#include <cassert>

int main() {
    // 开始测试前，确保未初始化
    assert(!memory_pool::mempool_exist());
    std::cout << "✔ 未初始化状态正确\n";

    // 创建单例
    memory_pool& pool = memory_pool::get_mempool(1024, 16);
    assert(memory_pool::mempool_exist());
    std::cout << "✔ 单例创建成功\n";

    // 再次获取，确保是同一个实例
    memory_pool& pool2 = memory_pool::get_mempool();
    assert(&pool == &pool2);
    std::cout << "✔ 获取同一个单例实例成功\n";

    // 检查已初始化状态
    assert(memory_pool::mempool_exist());
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
    memory_pool::mempool_destroy();
    assert(!memory_pool::mempool_exist());
    std::cout << "✔ 单例销毁成功\n";

    std::cout << "\n✅ 所有 memory_pool 单例测试通过\n";
    return 0;
}