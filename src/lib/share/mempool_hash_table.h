// mempool_hash_table.h

#ifndef MEMPOOL_HASH_TABLE_H
#define MEMPOOL_HASH_TABLE_H

#include <iostream>
#include <vector>
#include <functional>
#include <cstddef>
#include "memory_pool.h"
#include <type_traits>


template <typename TYPE_K, typename TYPE_V>
class hash_table_chain {

private:

    // 哈希表的 node. node 会根据 hash(key) 进入 bucket, 在 bucket 中形成一个链表
    struct HashTableNode {
        TYPE_K key; // 键
        TYPE_V value; // 值
        HashTableNode* next; // 下一个节点, 用于解决哈希冲突
    };

    // 如果 node 存在非平凡析构对象, 那么对 HashTableNode 显式调用析构
    void destroy_node(HashTableNode* node) {
        if constexpr (!std::is_trivially_destructible<HashTableNode>::value) {
            node->~HashTableNode();
        }
    }

    size_t _capacity; // 哈希表的容量, bucket数量

    // 数组 of buckets, 每个 bucket 是链表头, 每个链表是哈希冲突的 nodes
    std::vector<HashTableNode*> _table;

    // 引用传入的内存池
    memory_pool& _pool; // void* allocate(size_t size); void release() 接口

    // 使用标准库的 hash 函数, 对 TYPE_K 类型的输入 key, 作hash算法, 返回值
    size_t hash(const TYPE_K& key) const {
        return std::hash<TYPE_K>()(key);
    }

public:

    // 哈希表的构造函数. 传入哈希表的capacity, 和内存池
    hash_table_chain(size_t capacity, memory_pool& pool): _capacity(capacity), _pool(pool) {
        _table.resize(_capacity, nullptr); // 长度为 _capacity 的 HashTableNode* vector, 全部初始化为nullptr
    }

    // 析构函数, 会调用 clear 方法来释放所有 HashTableNode 中需要显式析构的部分, 但不负责内存释放
    ~hash_table_chain() {
        clear(); // 自动析构所有节点(如果需要)
    }

    // 哈希表关键方法之 get(key&, value&) --> change value, return true if success
    bool get(const TYPE_K& key, TYPE_V& value) {
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
    };

    // 哈希表关键方法之 insert(key&, value&) --> change table, return true if success
    // 使用 placement new
    bool insert(const TYPE_K& key, const TYPE_V& value) {
        // 计算 bucket index
        size_t index = hash(key) % _capacity;

        // 在 内存池 上分配新内存给新节点, raw_mem 内存
        void* raw_mem = _pool.allocate(sizeof(HashTableNode));
        if (!raw_mem) {
            return false; // 如果内存分配失败
        }
        // placement new 构造
        HashTableNode* new_node = new(raw_mem) HashTableNode{key, value, _table[index]};

        // 使用最新的链表头作为bucket
        _table[index] = new_node;
        return true;
    };

    // 哈希表是构建在传入的 内存池 上的数据结构, 它不应该负责 内存池 的销毁
    // 内存池本身是只可以重用/整体销毁，不可精确销毁单次allocate的内存
    // 故哈希表的"清空"应该是数据不再可访问的意思, 但其分配的内存不会在这里被销毁.
    // 同时, 哈希表node中需要显式调用析构的，在这里一并显式析构
    void clear() {
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
    };

};




#endif