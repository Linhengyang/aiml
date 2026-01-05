// mempool_hash_table_mt.h

#ifndef MEMPOOL_HASH_TABLE_MULTI_THREAD_H
#define MEMPOOL_HASH_TABLE_MULTI_THREAD_H


#include <vector>
#include <functional>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <new>
#include <type_traits>
#include <shared_mutex>
#include <mutex>
#include <atomic>
#include <cstring>


constexpr size_t next_pow2(size_t x) {
    if (x <= 1) return 1;
    --x; x |= x >> 1; x |= x >> 2; x |= x >> 4; x |= x >> 8; x |= x >> 16;
#if SIZE_MAX > 0xFFFFFFFFu
    x |= x >> 32;
#endif
    return x + 1;
};


// 不要让多个桶锁落入同一个 cache line. cpu总是会加载一整个cache line, 多个线程的桶锁若落入同一个cache line, 会引发竞争性能下降
// 对齐桶锁到 cache line size 边界, 并填充一些使得 padded mutex 至少能占满一整个 cache line
constexpr size_t CACHE_LINE_SIZE = 64;


struct padded_mutex {
    
    // alignas, C++11引入的关键字, 指定变量的内存对齐方式
    alignas(CACHE_LINE_SIZE) std::shared_mutex lock; // alignas强制TYPE_LOCK类变量 lock 按64字节内存对齐

    // padding数组, 使得当sizeof(TYPE_LOCK)小于 CACHE_LINE_SIZE 时, lock占据+padding部分正好占满一个完整的cache line.
    // 当 sizeof(TYPE_LOCK)大于 CACHE_LINE_SIZE 时, 前面alignas 对齐就够了. 此时pad至少1以满足部分编译器的要求
    char padding[CACHE_LINE_SIZE - sizeof(std::shared_mutex) > 0 ? CACHE_LINE_SIZE - sizeof(std::shared_mutex) : 1];

    padded_mutex() = default; // padded_mutex 要用在桶锁vector中，而vector初始化需要元素有默认构造

    padded_mutex(const padded_mutex&) = delete; // 禁止拷贝
    padded_mutex& operator=(const padded_mutex&) = delete; // 禁止赋值
    padded_mutex(padded_mutex&&) = delete; // 禁止移动. shared_mutex 不可移动
    padded_mutex& operator=(padded_mutex&&) = delete;
};



// 哈希表的本质就是管理 _capacity 个 链表，其中链表的node是 hashtablenode
// 低效做法：初始化时，就初始化好 _capacity 个空链表, 插入node时是一个一个插入
// clear表时 遍历_capacity个链表，并对每个链表逐node析构，置空所有链表头; destroy表就是clear的基础上再释放链表数组使得链表和node都不再可访问
// 高效做法：初始化时，就初始化一个 _occupied_bucket_indices 空数组，用以存储 0至_capacity 的index以指明哪些链表被插入node非空
// node除了插入bucket链表，还插入一个gc链，clear时直接沿着这条gc链析构就行，然后根据_occupied置空相应链表头

template <typename TYPE_K, typename TYPE_V, typename TYPE_MEMPOOL, typename HASH_FUNC = std::hash<TYPE_K>>
class hash_table_mt_chain {

private:

    struct HashTableNode {
        TYPE_K key;
        TYPE_V value;
        HashTableNode* next;
        HashTableNode* gc_next = nullptr;
        HashTableNode(const TYPE_K& k, const TYPE_V& v, HashTableNode* ptr): key(k), value(v), next(ptr) {}
    };

    void destroy_node(HashTableNode* node) {
        if constexpr (!std::is_trivially_destructible<HashTableNode>::value) {
            node->~HashTableNode();
        }
    }

    size_t _capacity;

    const float _max_load_factor = 0.80f;

    std::atomic<size_t> _size{0};

    HashTableNode** _table = nullptr;

    // 分配容量为 n 的节点指针数组 到数组头 _table
    void alloc_table_ptrs(size_t n) {
        if (n == 0) {
            _table = nullptr;
            return;
        }
        _table = static_cast<HashTableNode**>(std::calloc(n, sizeof(HashTableNode*)));
        if (!_table) throw std::bad_alloc();
    }

    // 释放节点指针数组，相当于 vector.clear(). 但节点内存并没有释放，由mempool管理
    void free_table_ptrs() noexcept {
        std::free(_table);
        _table = nullptr;
    }

    // gc 链表: 链起所有node
    std::atomic<HashTableNode*> _all_nodes_head{nullptr};

    // 记录非空bucket index. 桶置空操作时只需遍历这些桶即可. index类型要与 _capacity 类型对齐, 因为它是 hash成员函数的输出 取_capacity余
    // 要么给 _occuped_indices 另外加一个 锁, 要么去掉. TODO: 去掉 _occupied_indices
    std::vector<size_t> _occupied_indices;

    TYPE_MEMPOOL* _pool;

    HASH_FUNC _hasher;

    // 使用哈希器, 对 TYPE_K 类型的输入 key, 作hash算法, 返回值类型必须是 size_t 与 _capacity 对齐
    size_t hash(const TYPE_K& key) const {
        return _hasher(key);
    }

    std::shared_mutex _table_mutex; // 读写锁: 读锁可并发, 写锁必排他

    std::vector<padded_mutex> _stripes;
    size_t _stripe_mask;  // 

    inline std::shared_mutex& bucket_lock(size_t bucket_index) noexcept {
        return _stripes[bucket_index & _stripe_mask].lock;
    }

    std::atomic<bool> _rehashing{false};

    void rehash(size_t new_capacity) {
        // rehash 的调用在 insert 里，调用前会加 独占表锁, 故这里不再加独占表锁避免死锁

        HashTableNode** _new_table = static_cast<HashTableNode**>(std::calloc(new_capacity, sizeof(HashTableNode*)));
        if (!_new_table) throw std::bad_alloc();
        
        // 新的非空桶列表
        std::vector<size_t> _new_occupied_indices;
        _new_occupied_indices.reserve(_occupied_indices.size());

        // 新的gc链表
        HashTableNode* _new_all_nodes_head = nullptr;

        size_t actual_node_count = 0;

        // 遍历非空桶
        for (size_t old_index: _occupied_indices) {

            HashTableNode* curr = _table[old_index];
            while (curr) {
                HashTableNode* next = curr->next;
                size_t new_index = hash(curr->key) % new_capacity;
                const bool was_empty = (_new_table[new_index] == nullptr);
                // 头插到新桶
                curr->next = _new_table[new_index];
                _new_table[new_index] = curr;
                // 若空桶->非空, 记录
                if (was_empty) _new_occupied_indices.push_back(new_index);
                // 头插到新的gc链表
                curr->gc_next = _new_all_nodes_head;
                _new_all_nodes_head = curr;
                // 更新计数
                ++actual_node_count;

                curr = next;
            }
        }

        // 切换
        std::free(_table);
        _table = _new_table;
        _capacity = new_capacity;
        _size.store(actual_node_count, std::memory_order_relaxed);
        _occupied_indices.swap(_new_occupied_indices);
        _all_nodes_head.store(_new_all_nodes_head, std::memory_order_relaxed);
    }


public:

    explicit hash_table_mt_chain(const HASH_FUNC& hasher, size_t capacity, TYPE_MEMPOOL* pool, size_t stripe_hint = 4096):
        _hasher(hasher),
        _capacity(capacity),
        _pool(pool),
        _stripe_mask(next_pow2(stripe_hint)-1),
        // vector类具备构造函数重载: vector(size_type n, const T& value = T()), 当T(这里是shared_mutex)可默认构造且noexcept时, vector执行T的默认构造函数n次
        // .reserve 方法只是预留容量. 这里是实打实构造了 n 个独立实例 ---> 也就是说, 这里构造了 next_pow2(stripe_hint) 个 独立的互斥锁
        _stripes(next_pow2(stripe_hint))
    {
        alloc_table_ptrs(_capacity);
    }

    explicit hash_table_mt_chain(size_t capacity, TYPE_MEMPOOL* pool, size_t stripe_hint = 4096):
        _hasher(),
        _capacity(capacity),
        _pool(pool),
        _stripe_mask(next_pow2(stripe_hint)-1),
        // vector类具备构造函数重载: vector(size_type n, const T& value = T()), 当T(这里是shared_mutex)可默认构造且noexcept时, vector执行T的默认构造函数n次
        // .reserve 方法只是预留容量. 这里是实打实构造了 n 个独立实例 ---> 也就是说, 这里构造了 next_pow2(stripe_hint) 个 独立的互斥锁
        _stripes(next_pow2(stripe_hint))
    {
        alloc_table_ptrs(_capacity);
    }

    ~hash_table_mt_chain() {
        destroy();
    }

    bool get(const TYPE_K& key, TYPE_V& value) {
        // 并发读: 此操作(get)不独占表锁
        std::shared_lock<std::shared_mutex> _lock_from_rehash_clear_(_table_mutex);

        if (_capacity == 0 || !_table) return false;

        size_t index = hash(key) % _capacity;

        // 并发读: 此操作(get)不独占桶锁(条带锁)
        std::shared_lock<std::shared_mutex> _lock_from_insert_(bucket_lock(index));

        for (HashTableNode* cur = _table[index]; cur; cur = cur->next) {
            if (cur->key == key) {
                value = cur->value;
                return true;
            }
        }
        return false;
    }

    bool insert(const TYPE_K& key, const TYPE_V& value) {
        // 并发读: 此操作(insert)不独占表锁
        std::shared_lock<std::shared_mutex> _lock_table_from_rehash_clear_(_table_mutex);
        
        if (_capacity == 0 || !_table) return false;

        size_t index = hash(key) % _capacity;
        {
            // 独占写: 此操作(花括号内部) 独占桶锁(条带锁)，即只有此操作发生
            std::unique_lock<std::shared_mutex> _lock_bucket_for_insert_(bucket_lock(index));

            for (HashTableNode* cur = _table[index]; cur; cur = cur->next) {
                if (cur->key == key) {
                    cur->value = value;
                    return true;
                }
            }
            const bool was_empty = (_table[index] == nullptr);
            void* raw_mem = _pool->allocate(sizeof(HashTableNode));
            if (!raw_mem) return false;

            HashTableNode* new_node = new(raw_mem) HashTableNode{key, value, _table[index]};
            _table[index] = new_node;
            // 新的 node 要线程安全地插入gc链: 独占的桶锁(条带锁)仅锁住了当前桶(条带), 但是gc链是全局的, 可能有其他桶(条带)在写入, 故这里要线程安全
            HashTableNode* old = _all_nodes_head.load(std::memory_order_relaxed);
            do {
                new_node->gc_next = old; // 头插 gc 链
            } while (!_all_nodes_head.compare_exchange_weak(old, new_node, std::memory_order_release, std::memory_order_relaxed));

            // 如果 空 -> 非空, 那么记录桶号 ERROR: 表级读锁, 桶/条带级写锁, 无法保护 _occupied_indices 的 push_back 操作的线程安全
            if (was_empty) _occupied_indices.push_back(index);

            // node数量自加1. 原子线程安全
            // std::memory_order_relaxed 就可以保证原子安全. 但未来若需要在某些线程里仅靠_size来判断是否有数据写入, 这个模式不安全.
            // 这个模式下, 其他线程不一定能看到 自增后的 _size. 可以用 _size.fetch_add(1) 默认模式, 最严格, 保证全局一致.
            _size.fetch_add(1);

        }

        _lock_table_from_rehash_clear_.unlock(); // 解开并发读的表锁, 是因为 rehash 操作需要 独占表锁
        
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

    template <typename Func>
    bool atomic_upsert(const TYPE_K& key, Func&& updater, const TYPE_V& default_val) {
        // 并发读: 此操作(update by insert)不独占表锁
        std::shared_lock<std::shared_mutex> _lock_table_from_rehash_clear_(_table_mutex);

        if (_capacity == 0 || !_table) return false;

        size_t index = hash(key) % _capacity;
        {
            // 独占写: 此操作(花括号内部) 独占桶锁(条带锁)，即只有此操作发生
            std::unique_lock<std::shared_mutex> _lock_bucket_for_insert_(bucket_lock(index));

            for (HashTableNode* cur = _table[index]; cur; cur = cur->next) {
                if (cur->key == key) {
                    std::forward<Func>(updater)(cur->value);
                    return true;
                }
            }
            const bool was_empty = (_table[index] == nullptr);
            void* raw_mem = _pool->allocate(sizeof(HashTableNode));
            if (!raw_mem) return false;

            HashTableNode* new_node = new(raw_mem) HashTableNode{key, default_val, _table[index]};
            _table[index] = new_node;

            // 新的 node 要线程安全地插入gc链: 独占的桶锁(条带锁)仅锁住了当前桶(条带), 但是gc链是全局的, 可能有其他桶(条带)在写入, 故这里要线程安全
            HashTableNode* old = _all_nodes_head.load(std::memory_order_relaxed);
            do {
                new_node->gc_next = old; // 头插 gc 链
            } while (!_all_nodes_head.compare_exchange_weak(old, new_node, std::memory_order_release, std::memory_order_relaxed));

            // 如果 空 -> 非空, 那么记录桶号 ERROR: 表级读锁, 桶/条带级写锁, 无法保护 _occupied_indices 的 push_back 操作的线程安全
            if (was_empty) _occupied_indices.push_back(index);

            _size.fetch_add(1);
        }
        
        _lock_table_from_rehash_clear_.unlock(); // 解开并发读的表锁, 是因为 rehash 操作需要 独占表锁

        bool expected = false;

        if (_size >= _capacity*_max_load_factor && _rehashing.compare_exchange_strong(expected, true)) {
            // 独占 _table_mutex 表锁, rehash 时其他任何线程不能对table作任何操作. 作用到rehash结束
            std::unique_lock<std::shared_mutex> _lock_table_for_rehash_(_table_mutex);

            if (_size >= _capacity*_max_load_factor) {
                rehash( _capacity*2 );
            }
            _rehashing.store(false);
        }

        return true;
    }

    void clear() {
        // 清空全表时, 清空过程要全程 独占表锁
        std::unique_lock<std::shared_mutex> _lock_table_(_table_mutex);

        if (_capacity == 0 || !_table) {
            _size.store(0, std::memory_order_relaxed);
            _occupied_indices.clear();
            _all_nodes_head.store(nullptr, std::memory_order_relaxed);
            return;
        }

        if constexpr(!std::is_trivially_destructible<HashTableNode>::value) {
            HashTableNode* curr = _all_nodes_head.exchange(nullptr, std::memory_order_acquire);
            while (curr) {
                HashTableNode* next = curr->gc_next;
                destroy_node(curr);
                curr = next;
            }
        } else {
            _all_nodes_head.store(nullptr, std::memory_order_relaxed);
        }

        // clear 和 destroy 的区别就在于: clear 保留了桶结构, _table 链表头数组不释放(但全部置空). 各链表不再可访问.
        // 经过内存池的reset操作后, 本哈希表可以重新复用.
        for (size_t index: _occupied_indices) {
            _table[index] = nullptr;
        }

        _occupied_indices.clear();
        _size.store(0, std::memory_order_relaxed);

    }

    void destroy() {
        // 析构全表时, 析构过程要全程 独占表锁
        std::unique_lock<std::shared_mutex> _lock_table_(_table_mutex);

        if (_capacity == 0 || !_table) {
            _size.store(0, std::memory_order_relaxed);
            _occupied_indices.clear();
            _all_nodes_head.store(nullptr, std::memory_order_relaxed);
            return;
        }

        if constexpr(!std::is_trivially_destructible<HashTableNode>::value) {
            HashTableNode* curr = _all_nodes_head.exchange(nullptr, std::memory_order_acquire);
            while (curr) {
                HashTableNode* next = curr->gc_next;
                destroy_node(curr);
                curr = next;
            }
        } else {
            _all_nodes_head.store(nullptr, std::memory_order_relaxed);
        }
        
        // destroy 和 clear 的区别就在于: destroy 摧毁了桶结构, _table 链表头数组释放, 各链表不再可访问
        // 本哈希表不再可复用.
        free_table_ptrs();
        _capacity = 0;

        _occupied_indices.clear();
        _size.store(0, std::memory_order_relaxed);
    }

    size_t size() const {
        return _size.load();
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
    *   1. 非const迭代器, 应该在迭代时上独占表锁, 其他任何线程不能对表有任何操作(读写都不行). 迭代器自身可change value,
    *   2. const迭代器, 允许并发迭代, 应该共享表锁(禁止了需要独占表锁的rehash/clear), 共享桶锁(禁止了需要独占桶锁的insert/remove)
    *      迭代器是只读的. 哈希表不可被任何change, 即线程A迭代bucket_i时, 不该允许线程B在bucket_i作insert和remove
    *      这里似乎可以允许线程B在bucket_j作insert和remove, 因为线程A在迭代bucket_i时, 对其他桶似乎可以不作要求. 只不过这样的话，
    *      多线程并发迭代的结果可能会不一样. 如果要求保证并发迭代的结果一致, 那么线程A在迭代bucket_i时, 应该对全部bucket都共享锁.
    *      可是这种需求有更好的实现方式: 先单线程迭代一遍哈希表并dump成副本, 然后多线程使用该副本. 所以这里不对全部桶上共享锁.
    * 
    *      并发迭代有不同的设计模式: 1. 多个线程并发无误遍历一遍哈希表（总共一遍），2. 多个线程各自并发无误遍历一遍哈希表（总共多遍）
    *      前者多个线程并发遍历一遍哈希表,（迭代器的_node指针是线程local的, 不能多线程共享. 遍历过程中_node指针很多跳转, 共享需要极其精细的
    *      同步机制, 那就不现实.）即使是为了加速迭代也应该使用分片（sharding）多线程迭代的方式（每个线程负责一部分bucket）. 
    *      那么这样的迭代器和全迭代肯定是不同设计的（需要输入bucket id以发送给不同线程，以实现sharding并行扫描），是高性能哈希表TBB/folly::F14的做法,
    *      并不是常规iterator的职责范围. 在这里首先实现的是“多个线程各自并发无误遍历一遍哈希表（总共多遍）”的const只读迭代器。
    */

    /*
    * const只读迭代器
    * 
    * 用法: 单一线程下 for(auto it = hash_table.cbegin(); it != hash_table.cend(); ++it) {auto [k, v] = *it; //code//}
    */
    class const_iterator {

    public:

        const_iterator(const hash_table_mt_chain* hash_table, size_t bucket_index, HashTableNode* node)
            :_hash_table(hash_table),
            _bucket_index(bucket_index),
            _node(node)
        {
            _null_node_advance_to_next_valid_bucket();
        }

        // *it 迭代器对象解引用 --> 只读返回
        std::pair<const TYPE_K&, const TYPE_V&> operator*() const {
            return {_node->key, _node->value}; // 返回 pair(key, value)临时对象
        }

        // 迭代器对象前置++
        const_iterator& operator++() {
            if (_node) {
                _node = _node->next;
            }

            if (!_node) {
                _bucket_index++;
                _null_node_advance_to_next_valid_bucket();
            }
            return *this;
        }

        // 迭代器对象后置++
        const_iterator operator++(int) {
            const_iterator tmp = *this;
            ++(*this);
            return tmp;
        }

        // 迭代器对象 == 运算
        bool operator==(const const_iterator& other) const {
            return _node == other._node && _hash_table == other._hash_table;
        }

        // 迭代器对象 != 运算
        bool operator!=(const const_iterator& other) const {
            return !(*this == other);
        }

    private:

        const hash_table_mt_chain* _hash_table;

        size_t _bucket_index;

        HashTableNode* _node;

        // 迭代器内部不加锁逻辑。锁在迭代器外部调用
        void _null_node_advance_to_next_valid_bucket() {
            while (!_node && _bucket_index < _hash_table->_capacity) {

                _node = (_hash_table->_table)[_bucket_index];

                if (_node) break;

                _bucket_index++;
            }
        }

    }; // end of const_iterator definition

    const_iterator cbegin() const {
        return const_iterator(this, 0, nullptr); // 会自动定位到第一个有效节点
    }

    const_iterator cend() const {
        return const_iterator(this, _capacity, nullptr); // 尾后迭代器: 返回的迭代器应该处于 end 的临界状态, 即刚结束迭代的 状态
    }

    /*
    * 非const迭代器
    * 
    * 用法: for(auto it = hash_table.begin(); it != hash_table.end(); ++it) {auto& [k, v] = *it; //code//}
    */
    class iterator {

    public:

        // 迭代器的构造函数
        iterator(hash_table_mt_chain* hash_table, size_t bucket_index, HashTableNode* node)
            :_hash_table(hash_table),
            _bucket_index(bucket_index),
            _node(node)
        {
            _null_node_advance_to_next_valid_bucket();
        }

        // *it 迭代器对象解引用 --> v可变返回
        std::pair<const TYPE_K&, TYPE_V&> operator*() const {
            return {_node->key, _node->value}; // 返回 pair(key, value)临时对象
        }

        // 迭代器对象前置++
        iterator& operator++() {
            if (_node) {
                _node = _node->next;
            }

            if (!_node) {
                _bucket_index++;
                _null_node_advance_to_next_valid_bucket(); // 
            }
            return *this;
        }

        // 迭代器对象后置++
        iterator operator++(int) {
            iterator tmp = *this;
            ++(*this);
            return tmp;
        }

        // 迭代器对象 == 运算
        bool operator==(const iterator& other) const {
            return _node == other._node && _hash_table == other._hash_table;
        }

        // 迭代器对象 != 运算
        bool operator!=(const iterator& other) const {
            return !(*this == other);
        }

    private:

        hash_table_mt_chain* _hash_table;

        size_t _bucket_index;

        HashTableNode* _node;

        void _null_node_advance_to_next_valid_bucket() {
            while (!_node && _bucket_index < _hash_table->_capacity) {
                _node = (_hash_table->_table)[_bucket_index];
                if (_node) break;
                _bucket_index++;
            }
        }

    }; // end of iterator definition
    
    iterator begin() const {
        return iterator(this, 0, nullptr);
    }

    iterator end() const {
        return iterator(this, _capacity, nullptr);
    }

}; // end of hash_table_mt_chain definition




#endif