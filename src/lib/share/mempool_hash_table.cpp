// mempool_hash_table.cpp

#include <iostream>
#include <vector>
#include <functional>
#include <cstddef>
#include <memory_pool.h>
#include <mempool_hash_table.h>
#include <type_traits>




// 哈希表的构造函数. 传入哈希表的capacity, 和内存池
// 初始化 _table vector为长度为 capacity 的 nullptr
template <typename TYPE_K, typename TYPE_V>
hash_table_chain<TYPE_K, TYPE_V>::hash_table_chain(size_t capacity, memory_pool& pool)
    : _capacity(capacity), _pool(pool), _table(capacity, nullptr) {}





template <typename TYPE_K, typename TYPE_V>
hash_table_chain<TYPE_K, TYPE_V>::~hash_table_chain() {
    clear(); // 自动析构所有节点(如果需要)
}





template <typename TYPE_K, typename TYPE_V>
void hash_table_chain<TYPE_K, TYPE_V>::clear() {
    // 对于每个 bucket, 作为哈希冲突的 node 的链表头, 循环以显式析构所有node(如果需要)
    for (auto& bucket : _table) {
        HashTableNode* curr = bucket;
        while (curr) {
            HashTableNode* next = curr->next;
            destroy_node(curr);
            curr = next;
        }
        // bucket 自身置空. 此时该bucket无法从 哈希表对象访问. 但内存并未释放, 等待内存池统一释放
        bucket = nullptr;
    }
}





template <typename TYPE_K, typename TYPE_V>
bool hash_table_chain<TYPE_K, TYPE_V>::get(const TYPE_K& key, TYPE_V& value) {
    // 计算 bucket index
    size_t index = hash(key) % _capacity;
    // 得到 bucket, 即哈希冲突的链表头
    HashTableNode* current = _table[index];

    while (current) {
        if (current->key == key) {
            value = current->value; // 改变 value 地址的值
            return true;
        }
        current = current->next;
    }
    return false; // 没找到
}







template <typename TYPE_K, typename TYPE_V>
bool hash_table_chain<TYPE_K, TYPE_V>::insert(const TYPE_K& key, const TYPE_V& value) {
    // 计算 bucket index
    size_t index = hash(key) % _capacity;


    // 在 内存池 上分配新内存给新节点, raw_mem 内存
    void* raw_mem = pool.allocate(sizeof(HashTableNode));
    if (!raw_mem) {
        return false; // 如果内存分配失败
    }
    // placement new 构造
    HashTableNode* new_node = new(raw_mem) HashTableNode{key, value, _table[index]};

    // 使用最新的链表头作为bucket
    _table[index] = new_node;
    return true;
}