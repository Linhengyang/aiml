// mempool_hash_table_st.h

#ifndef MEMPOOL_HASH_TABLE_SINGLE_THREAD_H
#define MEMPOOL_HASH_TABLE_SINGLE_THREAD_H


#include <vector>
#include <functional>
#include <cstddef>
#include <type_traits>
#include "memory_pool.h"


template <typename TYPE_K, typename TYPE_V>
class hash_table_st_chain {

private:

    // 哈希表的 node. node 会根据 hash(key) 进入 bucket, 在 bucket 中形成一个链表
    struct HashTableNode {
        TYPE_K key; // 键
        TYPE_V value; // 值
        HashTableNode* next; // 下一个节点, 用于解决哈希冲突

        // 提供placement new 构造支持
        HashTableNode(const TYPE_K& k, const TYPE_V& v, HashTableNode* ptr): key(k), value(v), next(ptr) {}
    };

    // 如果 node 存在非平凡析构对象, 那么对 HashTableNode 显式调用析构
    void destroy_node(HashTableNode* node) {
        if constexpr (!std::is_trivially_destructible<HashTableNode>::value) {
            node->~HashTableNode();
        }
    }

    size_t _capacity; // 哈希表的容量, bucket数量

    const float _max_load_factor = 0.80f; // 默认最大负载因子. 当 node 数量/_capacity 超过时, 触发扩容

    size_t _size = 0; // node数量

    // 数组 of buckets, 每个 bucket 是链表头, 每个链表是哈希冲突的 nodes
    std::vector<HashTableNode*> _table;

    // 引用传入的内存池
    memory_pool& _pool; // void* allocate(size_t size); void release() 接口

    // 使用标准库的 hash 函数, 对 TYPE_K 类型的输入 key, 作hash算法, 返回值
    size_t hash(const TYPE_K& key) const {
        return std::hash<TYPE_K>()(key);
    }

    /*
    * 扩容 rehash
    * @param new_capacity
    * 
    * 行为:对每一个node重新计算bucket, 然后将其重新挂载到新的bucket链表的头部
    */
    void rehash(size_t new_capacity) {
        // 初始化一个新的 table
        std::vector<HashTableNode*> _new_table(new_capacity, nullptr);

        for (size_t i = 0; i < _capacity; i++) {
            HashTableNode* current = _table[i]; // 从该bucekt的链表头开始
            while (current) { // 当前node非空
                HashTableNode* next = current->next; // 先取出next node
                size_t new_index = hash(current->key) % new_capacity; // 计算得出新bucket
                current->next = _new_table[new_index]; // 当前node挂载到新bucket链表头
                _new_table[new_index] = current; // 更新确认新bucket的链表头
                current = next; // 遍历下一个node
            }
            // 旧_table会被舍弃
            _table[i] = nullptr;
        }
        // 所有bucket所有node重新挂载完毕后, 切换 _table/_capacity
        _table = std::move(_new_table);
        _capacity = new_capacity;
    }


public:

    // 哈希表的构造函数. 传入哈希表的capacity, 和内存池
    explicit hash_table_st_chain(size_t capacity, memory_pool& pool): _capacity(capacity), _pool(pool) {
        _table.resize(_capacity, nullptr); // 长度为 _capacity 的 HashTableNode* vector, 全部初始化为nullptr
    }

    // 析构函数, 会调用 clear 方法来释放所有 HashTableNode 中需要显式析构的部分, 但不负责内存释放
    ~hash_table_st_chain() {
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
    }

    /*
    * 插入或更新键值对
    * @param key
    * @param value
    * @return 如果插入或更新成功, 返回true; 如果内存分配失败返回false
    * 
    * 行为: 若 key 已经存在, 则更新对应的 value; 否则新建节点插入. 插入后检查是否需要扩容
    */
    bool insert(const TYPE_K& key, const TYPE_V& value) {
        // 计算 bucket index
        size_t index = hash(key) % _capacity;

        // 首先查找 key 是否已经存在. 若 key 存在, 修改原 value 到 新value
        HashTableNode* current = _table[index];
        while (current) {
            if (current->key == key) {
                current->value = value; // 修改 node 的value
                return true; // 完成 insert, return true 退出
            }
            current = current->next;
        }
        // 如果执行到这里, 说明要么 currrent 是 nullptr, 要么 _table[index] 链表里没有 key
        // 那么就要执行新建节点, 并将新节点放到 _table[index] 这个bucket的头部

        // 在 内存池 上分配新内存给新节点, raw_mem 内存
        void* raw_mem = _pool.allocate(sizeof(HashTableNode));
        if (!raw_mem) {
            return false; // 如果内存分配失败
        }
        // placement new 构造
        HashTableNode* new_node = new(raw_mem) HashTableNode{key, value, _table[index]};

        // 更新确认该 bucket 的链表头
        _table[index] = new_node;
        
        // node数量自加1. 原子线程安全
        _size++;

        // 单线程, 只需检查是否满足负载因子，触发扩容
        if (_size >= _capacity*_max_load_factor) {
            rehash( _capacity*2 ); // 扩容为两倍
        }

        return true;
    }

    // 哈希表是构建在传入的 内存池 上的数据结构, 它不应该负责 内存池 的销毁
    // 内存池本身是只可以重用/整体销毁，不可精确销毁单次allocate的内存
    // 故哈希表的"清空"应该是数据不再可访问的意思, 但其分配的内存不会在这里被销毁.
    // 同时, 哈希表node中需要显式调用析构的，在这里一并显式析构
    void clear() {

        // 对于每个 bucket, 作为哈希冲突的 node 的链表头, 循环以显式析构所有node(如果需要)
        for (size_t i = 0; i < _capacity; i++) {

            HashTableNode* curr = _table[i];
            while (curr) {
                HashTableNode* next = curr->next;
                destroy_node(curr);
                curr = next;
            }
            // bucket 自身置空. 此时该bucket无法从 哈希表对象访问. 但内存并未释放, 等待内存池统一释放
            _table[i] = nullptr;
        }
        // node数量 置0
        _size = 0;
    }


    // 输出当前哈希表 k-v 数量
    size_t size() const {
        return _size; // 原子读取
    }

};




#endif