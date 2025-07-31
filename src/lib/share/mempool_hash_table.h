// mempool_hash_table.h

#ifndef MEMPOOL_HASH_TABLE_H
#define MEMPOOL_HASH_TABLE_H


#include <vector>
#include <functional>
#include <cstddef>
#include <type_traits>
#include <shared_mutex>
#include <atomic>
#include "memory_pool.h"



// 不要让多个桶锁落入同一个 cache line. cpu总是会加载一整个cache line, 多个线程的桶锁若落入同一个cache line, 会引发竞争性能下降
// 对齐桶锁到 cache line size 边界, 并填充一些使得 padded mutex 至少能占满一整个 cache line
constexpr size_t CACHE_LINE_SIZE = 64;

template <typename TYPE_LOCK>
struct padded_mutex {
    
    // alignas, C++11引入的关键字, 指定变量的内存对齐方式
    alignas(CACHE_LINE_SIZE) TYPE_LOCK lock; // alignas强制TYPE_LOCK类变量 lock 按64字节内存对齐

    // padding数组, 使得当sizeof(TYPE_LOCK)小于 CACHE_LINE_SIZE 时, lock占据+padding部分正好占满一个完整的cache line.
    // 当 sizeof(TYPE_LOCK)大于 CACHE_LINE_SIZE 时, 前面alignas 对齐就够了. 此时pad至少1以满足部分编译器的要求
    char padding[CACHE_LINE_SIZE - sizeof(TYPE_LOCK) > 0 ? CACHE_LINE_SIZE - sizeof(TYPE_LOCK) : 1];
};




template <typename TYPE_K, typename TYPE_V>
class hash_table_chain {

private:

    // 哈希表的 node. node 会根据 hash(key) 进入 bucket, 在 bucket 中形成一个链表
    struct HashTableNode {
        TYPE_K key; // 键
        TYPE_V value; // 值
        HashTableNode* next; // 下一个节点, 用于解决哈希冲突

        // 提供placement new 构造支持
        HashTableNode(const TYPE_K& k, const TYPE_V&v, HashTableNode* ptr): key(k), value(v), next(ptr) {}
    };

    // 如果 node 存在非平凡析构对象, 那么对 HashTableNode 显式调用析构
    void destroy_node(HashTableNode* node) {
        if constexpr (!std::is_trivially_destructible<HashTableNode>::value) {
            node->~HashTableNode();
        }
    }

    size_t _capacity; // 哈希表的容量, bucket数量

    const float _max_load_factor = 0.80f; // 默认最大负载因子. 当 node 数量/_capacity 超过时, 触发扩容

    std::atomic<size_t> _size{0}; // node数量, 原子保证自加符++线程安全

    // 数组 of buckets, 每个 bucket 是链表头, 每个链表是哈希冲突的 nodes
    std::vector<HashTableNode*> _table;

    // 引用传入的内存池
    memory_pool& _pool; // void* allocate(size_t size); void release() 接口

    // 使用标准库的 hash 函数, 对 TYPE_K 类型的输入 key, 作hash算法, 返回值
    size_t hash(const TYPE_K& key) const {
        return std::hash<TYPE_K>()(key);
    }

    // 锁整张表的锁. rehash/clear等对整张表进行操作时, 独占该锁, 使得其他任何线程不能对table进行任何操作
    std::shared_mutex _table_mutex;

    // 锁单个bucket的锁. insert操作时, 独占该锁, 使得其他任何线程不能对bucket进行任何操作
    std::vector<padded_mutex<std::shared_mutex>> _bucket_mutexs; //实际是pad 之后的桶锁

    // 原子变量 _rehashing, 当多个线程并发执行insert并都通过了rehash条件检查后,应该只允许第一个线程执行rehash
    // 此时需要一个原子变量, 执行原子级的翻转操作: 当原子为false时的线程进入rehash, 翻转它为true, 其他线程只会得到true
    std::atomic<bool> _rehashing{false};

    /*
    * 扩容 rehash
    * @param new_capacity
    * 
    * 行为:对每一个node重新计算bucket, 然后将其重新挂载到新的bucket链表的头部
    */
    void rehash(size_t new_capacity) {
        // rehash 的调用在 insert 里，调用前会加 独占表锁, 故这里不再加独占表锁避免死锁

        // 初始化一个新的 table
        std::vector<HashTableNode*> _new_table(new_capacity, nullptr);

        // 初始化 新的 bucket mutexs 桶锁序列
        std::vector<padded_mutex<std::shared_mutex>> _new_bucket_mutexs(new_capacity);

        // 目前 size 是只增的, 且rehash 一定是扩容. 但为了逻辑闭环, 以及支持未来可能有缩容的rehash, 应该重新统计 node 总个数
        size_t actual_node_count = 0; // 独占表锁下, 此变量线程安全

        for (size_t i = 0; i < _capacity; i++) {

            // 搬迁当前 bucket 时, 也独占该bucket桶锁. 作用到本i次 for-loop 结束
            std::unique_lock<std::shared_mutex> _lock_bucket_for_rehash_(_bucket_mutexs[i].lock);

            HashTableNode* current = _table[i]; // 从该bucekt的链表头开始
            while (current) { // 当前node非空
                HashTableNode* next = current->next; // 先取出next node
                size_t new_index = hash(current->key) % new_capacity; // 计算得出新bucket
                {
                    // 挂载 current node 到新table的新bucket. 也独占新bucket桶锁. 作用到_new_table[new_index]修改完毕
                    std::unique_lock<std::shared_mutex> _lock_newbucket_for_rehash_(_new_bucket_mutexs[new_index].lock);
                    current->next = _new_table[new_index]; // 当前node挂载到新bucket链表头
                    _new_table[new_index] = current; // 更新确认新bucket的链表头
                }
                current = next; // 遍历下一个node
                actual_node_count++; // 每搬迁一个node就计数
            }
            // 旧_table会被舍弃
            _table[i] = nullptr;
        }
        // 所有bucket所有node重新挂载完毕后, 切换 _table/_bucket_mutexs/_capacity/_size
        _table = std::move(_new_table);
        _bucket_mutexs = std::move(_new_bucket_mutexs);
        _capacity = new_capacity;
        _size.store(actual_node_count, std::memory_order_relaxed); // 独占表锁下, 只要变更值就可以了, 不需要考虑其他线程是否全局一致.
    }


public:

    // 哈希表的构造函数. 传入哈希表的capacity, 和内存池
    explicit hash_table_chain(size_t capacity, memory_pool& pool): _capacity(capacity), _pool(pool) {
        _table.resize(_capacity, nullptr); // 长度为 _capacity 的 HashTableNode* vector, 全部初始化为nullptr
        _bucket_mutexs.resize(_capacity); // 初始化桶锁序列
    }

    // 析构函数, 会调用 clear 方法来释放所有 HashTableNode 中需要显式析构的部分, 但不负责内存释放
    ~hash_table_chain() {
        clear(); // 自动析构所有节点(如果需要)
    }

    /*
    * 根据键找值
    * @param key
    * @param value, 取值地址
    * @return 如果取值成功返回true; 如果取值失败返回false
    * 
    * 行为: 取 key 对应的 value 存到 input arg value. 若失败返回false, 若成功返回true
    * 读写要分离: 读的时候不允许写（单桶），允许并发读（单桶）
    */
    // 哈希表关键方法之 get(key&, value&) --> change value, return true if success
    bool get(const TYPE_K& key, TYPE_V& value) {
        // 读取 key-value 时, 不允许对整表有 rehash/clear 操作.
        // 但是是允许对多个不同bucket作并发读取的, 所以共享占用表锁. 作用到get结束
        std::shared_lock<std::shared_mutex> _lock_from_rehash_clear_(_table_mutex);

        // 计算 bucket index
        size_t index = hash(key) % _capacity;

        // 读取 key-value 时, 允许单个bucket上并发读. 不允许insert操作. 所以共享占用桶锁. 作用到get结束
        std::shared_lock<std::shared_mutex> _lock_from_insert_(_bucket_mutexs[index].lock);

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
    * 读写要分离: 写的时候不允许读（单桶），不允许并发写（单桶），允许多桶并发写
    */
    bool insert(const TYPE_K& key, const TYPE_V& value) {
        // 写入 key-value 时, 不允许对整表有 rehash/clear 操作.
        // 但是是允许对多个不同bucket作并发写入的, 所以共享占用. 作用到rehash判断前
        std::shared_lock<std::shared_mutex> _lock_from_rehash_clear_(_table_mutex);

        // 计算 bucket index
        size_t index = hash(key) % _capacity;
        {
            // 写入 key-value 时, 不允许其他线程对相应bucket有读写操作. 所以独占桶锁. 作用到本桶更新完毕
            std::unique_lock<std::shared_mutex> _lock_from_insert_read_(_bucket_mutexs[index].lock);
            
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
            // std::memory_order_relaxed 就可以保证原子安全. 但未来若需要在某些线程里仅靠_size来判断是否有数据写入, 这个模式不安全.
            // 这个模式下, 其他线程不一定能看到 自增后的 _size. 可以用 _size.fetch_add(1) 默认模式, 最严格, 保证全局一致.
            _size.fetch_add(1);

        }

        // 可能会有 rehash 操作, 需要释放共享的表锁
        _lock_from_rehash_clear_.unlock();
        
        // 扩容检查. 因为在临近扩容时,由于多并发写入, 会有多个进程近乎同时判断出需要rehash. 但只有一个能执行rehash
        bool expected = false;
        // 临近状态下, 多个线程都满足第一个条件, 但是第二个条件: 原子变量 _rehashing == expected(false) 只能原子级满足
        // compare_exchange_strong 保证了一旦原子变量 _rehashing 满足 == false, 马上将转化为true并返回true.
        // 如此其他线程在这里只会得到一个为 true 的_rehashing, 从而无法进入内部.
        if (_size >= _capacity*_max_load_factor && _rehashing.compare_exchange_strong(expected, true)) {

            // 独占 _table_mutex 表锁, rehash 时其他任何线程不能对table作任何操作. 作用到rehash结束
            std::unique_lock<std::shared_mutex> _lock_table_for_rehash_(_table_mutex);

            // 二次检查. _size只增, 似乎没必要二次检查. 但实际上为未来增加remove(k,v)作准备, 增强健壮性
            if (_size >= _capacity*_max_load_factor) {
                rehash( _capacity*2 ); // 扩容为两倍
            }
            _rehashing.store(false);
        }

        return true;
    }

    // 哈希表是构建在传入的 内存池 上的数据结构, 它不应该负责 内存池 的销毁
    // 内存池本身是只可以重用/整体销毁，不可精确销毁单次allocate的内存
    // 故哈希表的"清空"应该是数据不再可访问的意思, 但其分配的内存不会在这里被销毁.
    // 同时, 哈希表node中需要显式调用析构的，在这里一并显式析构
    void clear() {

        // 清空 hash table时，独占 表锁
        std::unique_lock<std::shared_mutex> _lock_table_for_clear_(_table_mutex);

        // 对于每个 bucket, 作为哈希冲突的 node 的链表头, 循环以显式析构所有node(如果需要)
        for (size_t i = 0; i < _capacity; i++) {

            // 析构本bucket上的nodes, 以及要置空本bucket时, 独占 桶锁. 作用到本i次for-loop结束
            std::unique_lock<std::shared_mutex> _lock_bucket_for_clear_(_bucket_mutexs[i].lock);

            HashTableNode* curr = _table[i];
            while (curr) {
                HashTableNode* next = curr->next;
                destroy_node(curr);
                curr = next;
            }
            // bucket 自身置空. 此时该bucket无法从 哈希表对象访问. 但内存并未释放, 等待内存池统一释放
            _table[i] = nullptr;
        }
        // node数量 置0, 原子线程安全
        _size.store(0, std::memory_order_relaxed);
    }

    
    // 输出当前哈希表 k-v 数量
    size_t size() const {
        return _size.load(); // 原子读取
    }



    /*
    * 迭代器
    * 线程安全的迭代器, 到底是指什么? 
    * 首先, 单线程下, 迭代哈希表时也不应该insert/remove/change key操作, 因为这些都可能导致rehash, 会导致 iterator 失效
    * 所以在单线程下, 迭代哈希表时最多 只允许change value, 不允许其他任何操作. 单线程下可以不用 只读迭代器. 允许迭代器change value
    * 
    * 那么在多线程下, 首先迭代器在运行时，肯定也要禁止任何线程作 change value 之外的操作. 问题是, 是否允许change value(即使它线程安全)?
    * 答案是: 否. 在缺乏同步机制的前提下, 当某个线程在执行迭代遍历时, 若其他线程在 thread-safe change value, 会导致两个可能的严重后果
    * 1. 撕裂读取：迭代器在读取 key-value 时，可能读取到value的一部分后，另一部分被另一个线程改变了，导致读到了一个”混合value”
    * 2. 后续逻辑破坏：迭代读到了一个value, 但实际上这个value在随后就被改了然而迭代器线程并不知情, 可能会导致后续逻辑错误
    * 所以归纳一下：
    *   1. 若非const迭代器, 只能单线程迭代, 且加锁不允许其他线程作只读之外的任何操作.
    *      这样迭代器允许change value, 但若迭代器change value, 其他线程不能作任何操作(读写都不可以)
    *   2. 若要并发迭代，必须都是const迭代. 且加锁不允许其他线程作只读之外的任何操作.
    * 归并一下同类项，迭代器应该这样设计:
    *   1. 非const迭代器, 应该在迭代时上独占表锁, 其他任何线程不能对表有任何操作(读写都不行)
    *   2. const迭代器, 允许并发迭代, 应该共享表锁(禁止了需要独占表锁的rehash/clear), 共享桶锁(禁止了需要独占桶锁的insert/remove)
    */
};




#endif