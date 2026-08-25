// mempooled_concurrent_hashtable.h
// 哈希表的 node* 指针数组适合放在内存池外, 而 nodes 适合放在内存池里
// 这是因为 哈希表涉及到扩容(rehash), 而扩容后旧数组若在内存池里, 则无法回收复用(内存池reset之前). 放在系统内存则可以由系统立即回收复用.

// 内存池上的哈希表由两部分组成: nodes 和 buckets(链表头node指针数组). 其中 nodes 在insert时逐一分配在内存池上
// 而 buckets 由创建方式分配内存, 即:
// 方法1: HashTable* map = new HashTable(capacity, &mempool); 此时 buckets 分配在 堆内存 上, 由new/delete手动管理哈希表的生命周期
// 方法2: HashTable map(capacity, &mempool); 此时 buckets 分配在 栈内存 上, 由函数调用自动管理哈希表的生命周期
// 这样的好处是 rehash 后原buckets相关空间可以即时被系统回收.

// 推荐方法1, 且将哈希表指针存储在 静态区. 这样可以全程手动控制哈希表的生命周期, 且资源做到最大程度的可复用和即时回收.

#ifndef MEMPOOLED_CONCURRENT_HASHTABLE_H
#define MEMPOOLED_CONCURRENT_HASHTABLE_H


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
#include <new>
#include <stdexcept>
#include <array>

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



// 读写锁
// 写锁是核心概念：写锁提供互斥 --> 写锁是排他的，不光“排写锁”，还“排读锁”。在写锁 .lock 作用之后, 其他任何需要该锁的 操作都会被阻塞
// 写锁除了提供互斥, 还提供线程之间的数据同步: 对同一个 mutex, 线程A的 unlock synchronize-with 线程B的lock, 所以B能看到A的所有修改, 即
// 线程A在 持有锁期间对共享数据的所有 写操作, 在它 unlock 之后, 线程B lock同一把锁时, 一定拿到线程A在持锁期间的所有修改 --> 跨线程的内存同步
// ----> synchronize-with 的语义. 注意只在 临界区内部有同步关系, 所以要避免锁外数据写入操作








template <typename TYPE_K, typename TYPE_V, typename TYPE_MEMPOOL, typename HASH_FUNC = std::hash<TYPE_K>>
class pooled_concurrent_hashtable {

private:

    struct HashTableNode {
        TYPE_K key;
        TYPE_V value;
        HashTableNode* next;
        HashTableNode* free_next = nullptr;
        // 禁止赋值(拷贝or移动)
        HashTableNode(const HashTableNode& other) = delete;
        HashTableNode& operator=(const HashTableNode& other) = delete;
        HashTableNode(HashTableNode&& other) = delete;
        HashTableNode& operator=(HashTableNode&& other) = delete;
        

        // 业务(普通)构造函数
        // 拷贝 key & value 资源 构造node
        HashTableNode(const TYPE_K& k, const TYPE_V& v, HashTableNode* ptr): key(k), value(v), next(ptr) {}
        // 移动 key & value 资源 构造node
        HashTableNode(TYPE_K&& k, TYPE_V&& v, HashTableNode* ptr): key(std::move(k)), value(std::move(v)), next(ptr) {}
        // 哈希表node 在 atomic_upsert 方法里有一个比较特殊的情形需要重载: key实参右值以移动/临时, 而default_value作为重复使用的对象, 必须const&
        HashTableNode(TYPE_K&& k, const TYPE_V& v, HashTableNode* ptr): key(std::move(k)), value(v), next(ptr) {}
    };
    
    void destroy_node(HashTableNode* node) {
        if constexpr (!std::is_trivially_destructible<HashTableNode>::value) {
            node->~HashTableNode();
        }
    }


    size_t _capacity;
    const float _max_load_factor = 0.75f;
    std::atomic<size_t> _size{0};
    HashTableNode** _table = nullptr;
    
    void alloc_table_ptrs(size_t n) {
        if (n == 0) {
            _table = nullptr;
            return;
        }
        _table = static_cast<HashTableNode**>(std::calloc(n, sizeof(HashTableNode*)));
        if (!_table) throw std::bad_alloc();
    }

    void free_table_ptrs() noexcept {
        std::free(_table);
        _table = nullptr;
    }

    // 空闲 free 链表: 链起 析构后的 poped nodes. 采用 TLC 设计: tls free_list + lock on global free_list
    // 对于 insert node, 线程优先从tls free_list中无锁取地址, 如果取不到, 从 global free_list refill 地址再取, 如果还是失败, 从 arena 分配
    // 对于 pop node, 线程优先把地址回收到 tls free_list, 如果塞满了, flush 到 global free_list

    static constexpr size_t TLS_FREE_LIST_MAX = 128;
    static constexpr size_t FREE_LIST_BATCH   = 32;

    // 全局 free_list(带锁)
    HashTableNode* _global_free_head = nullptr;
    std::mutex _global_free_mutex;

    // 跟着哈希表实例走的全局原子 代际信号量. 只在 clear/destroy 操作中自增, 用处是在线程之间同步是否发生 clear/destroy 操作: 线程在从 tls free_list 取存node前, 都要检查代际generation是否一致
    std::atomic<uint64_t> _generation{0};

    // TLS free_list(侵入式链表):存储了tls free_list的链表头, tls链表的size, 以及tls链表的代际(每clear/destroy一次, 代际+1)
    struct TLSFreeList {
        HashTableNode* head  = nullptr;
        size_t count = 0;
        uint64_t generation = 0;
    };

    // 废弃方案: 直接将 tls free_list 从 thread_local关键字定义. 原因: thread_local 只能用于：命名空间作用域变量、类的 static 成员、函数内 static 局部变量
    // 即 thread_local关键字的变量, 跟着线程走而不是对象, 所以必须要static(进入静态存储期)这样一来, 所有 同类型哈希表的不同实例, 在同一个线程处理时, 将共用同一个 tls free_list
    // 其实仔细分析一下, 由于此哈希表是基于 arena mempool 的, 也就是说 tls free_list 都是arena上的地址 --> 只要不同实例在同一 arena 上, 似乎不同实例之间共同复用一个 tls free_list 也无所谓
    
    // static thread_local TLSFreeList* tls_free_list;

    // 新方案: 维护一个 tls free lists注册表: tls registry, 它自身是 static thread_local 的, 也就是说同类型哈希表不同实例, 在同一线程下共享这个registry. thread_local天生线程安全
    // 但是, 注册表内部维护了所有该同类型哈希表 不同实例的指针 <-> tls free_list 的对应关系. 从而每个实例在要使用 tls free_list 时, 先根据自身指针this从注册表中找到自己的tls free_list再使用
    struct TLSRegistry {
        static constexpr size_t MAX_INSTANCES = 4; // 一个线程最多同时操作4个同类哈希表实例
        struct Entry { // 哈希表实例指针(不允许在这里通过指针改变哈希表) <-> tls free_list 的对应关系
            const pooled_concurrent_hashtable* owner = nullptr;
            TLSFreeList free_list;
        };
        std::array<Entry, MAX_INSTANCES> entries{};
        size_t count = 0;

        TLSFreeList* get_or_create_tls_free_list(pooled_concurrent_hashtable* owner) {
            // 线性查找，N很小，速度极快
            for (size_t i = 0; i < count; ++i) {
                if (entries[i].owner == owner) return &entries[i].free_list;
            }
            // 查找完毕没找到, 但仍然有空槽位, 为此哈希表实例 创建一个 tls free_list
            if (count < MAX_INSTANCES) {
                entries[count].owner = owner;
                entries[count].free_list = {nullptr, 0, 0};
                return &entries[count++].free_list;
            }
            return nullptr; // Fallback: 超过限制，降级为不使用 TLS
            // 此时 get node from tls/refill tls / flush tls 这三个操作都放弃; push node to tls改成直接push to global free_list
        }
    };

    inline static thread_local TLSRegistry tls_registry;

    // 根据 本哈希表实例的指针, 本线程可以根据此函数, 找到 本线程local 的 tls free_list
    inline TLSFreeList* get_tls_free_list() {
        return tls_registry.get_or_create_tls_free_list(this);
    }

    // TLS free_list 的操作: 纯单线程, 零锁零原子

    // 从 TLS free_list 中得到 node: tls_free_list 贡献复用地址, 用于 insert/atomic_upsert
    inline HashTableNode* get_node_from_tls() {
        // 尝试找到 本实例 本线程local 的 tls free_list
        TLSFreeList* tls_free_list = get_tls_free_list();
        // 如果失败, get node失败, 返回nullptr
        if (!tls_free_list) return nullptr;

        // 检查 本地 tls free_list代际是否匹配 _generation
        uint64_t global_gen = _generation.load(std::memory_order_acquire);
        if (tls_free_list->generation != global_gen) {
            // 代际不匹配，说明表在 insert/upsert 释放锁后, 被 clear/destroy
            // 必须丢弃 TLS 中缓存的所有旧节点(因为arena可能要/已 reset, 不能复用了, 随着 arena reset即可), 然后同步 generation
            tls_free_list->head = nullptr;
            tls_free_list->count = 0;
            tls_free_list->generation = global_gen;
            return nullptr;
        }
        // 代际匹配
        if (tls_free_list->count == 0) return nullptr; // 当count为0时, 返回nullptr
        HashTableNode* node = tls_free_list->head; // tls free_list 的链表头
        tls_free_list->head = node->free_next; // 更新 tls_free_list 的链表头和count
        --tls_free_list->count;
        return node;
    }

    // TLS free_list 回收 node: tls_free_list 回收可复用地址, 用于 pop
    inline void push_node_to_tls(HashTableNode* node) {
        // 尝试找到 本实例 本线程local 的 tls free_list
        TLSFreeList* tls_free_list = get_tls_free_list();
        // 如果失败, 直接 push node 到 global free_list
        if (!tls_free_list) {
            // Fallback: 直接放入 global
            std::lock_guard<std::mutex> lock(_global_free_mutex);
            node->free_next = _global_free_head;
            _global_free_head = node;
            return;
        }
        // 检查 本地 tls free_list代际是否等于 _generation
        uint64_t global_gen = _generation.load(std::memory_order_acquire);
        if (tls_free_list->generation != global_gen) {
            // 代际不匹配, 说明在 pop 释放锁后，有人执行了 clear/destroy
            // 此时 node 指向的内存可能已被 arena reset，不能放入 free_list
            // 直接丢弃该 node，并丢弃 TLS 中缓存的所有旧节点, 然后同步 generation
            tls_free_list->head = nullptr;
            tls_free_list->count = 0;
            tls_free_list->generation = global_gen;
            return; // 丢弃 node
        }

        node->free_next = tls_free_list->head;
        tls_free_list->head = node;
        ++tls_free_list->count;
    }

    // global free_list 与 TLS free_list 之间的交互操作: 需要给 global free_list 上锁

    // TLS free_list空了, 从 global free_list 批量refill: 锁住 global free_list, 从其 push 最多 FREE_LIST_BATCH 个node 到 tls free_list
    void refill_tls_from_global() {
        TLSFreeList* tls_free_list = get_tls_free_list();
        if (!tls_free_list) return;

        std::lock_guard<std::mutex> lock(_global_free_mutex);
        uint64_t global_gen = _generation.load(std::memory_order_acquire);
        if (tls_free_list->generation != global_gen) {
            // 代际不匹配, 说明有人执行了 clear/destroy. 这二操作会清空 global free_list
            // 放弃 refill tls free_list, 且清空并同步它
            tls_free_list->head = nullptr;
            tls_free_list->count = 0;
            tls_free_list->generation = global_gen;
            return; 
        }

        for (size_t i = 0; i < FREE_LIST_BATCH && _global_free_head; ++i) {
            HashTableNode* node = _global_free_head;
            _global_free_head = node->free_next;
            node->free_next = tls_free_list->head;
            tls_free_list->head = node;
            ++tls_free_list->count;
        }
    }

    // TLS free_list满了, 向 global free_list 批量flush: 锁住 global free_list, 其从 tls get 最多 FREE_LIST_BATCH 个node
    void flush_tls_to_global() {
        TLSFreeList* tls_free_list = get_tls_free_list();
        if (!tls_free_list) return;

        std::lock_guard<std::mutex> lock(_global_free_mutex);
        uint64_t global_gen = _generation.load(std::memory_order_acquire);
        if (tls_free_list->generation != global_gen) {
            // 代际不匹配, 说明有人执行了 clear/destroy. 这二操作会清空 global free_list
            // 放弃 flush to global free_list. 清空 tls free_list 并同步它
            tls_free_list->head = nullptr;
            tls_free_list->count = 0;
            tls_free_list->generation = global_gen;
            return; 
        }

        for (size_t i = 0; i < FREE_LIST_BATCH && tls_free_list->count > 0; ++i) {
            HashTableNode* node = tls_free_list->head;
            tls_free_list->head = node->free_next;
            --tls_free_list->count;
            node->free_next = _global_free_head;
            _global_free_head = node;
        }
    }

    TYPE_MEMPOOL* _pool;

    HASH_FUNC _hasher;

    size_t hash(const TYPE_K& key) const {
        return _hasher(key);
    }

    mutable std::shared_mutex _table_mutex; 
    mutable std::vector<padded_mutex> _stripes;
    size_t _stripe_mask;
    inline std::shared_mutex& bucket_lock(size_t bucket_index) noexcept {
        return _stripes[bucket_index & _stripe_mask].lock;
    }

    std::atomic<size_t> _resize_threshold{0};

    void rehash(size_t new_capacity) {
        HashTableNode** _new_table = static_cast<HashTableNode**>(std::calloc(new_capacity, sizeof(HashTableNode*)));
        if (!_new_table) throw std::bad_alloc();
        size_t actual_node_count = 0;

        for (size_t old_index = 0; old_index < _capacity; ++old_index) {

            HashTableNode* curr = _table[old_index];
            while (curr) {
                HashTableNode* next = curr->next;
                size_t new_index = hash(curr->key) % new_capacity;
                curr->next = _new_table[new_index];
                _new_table[new_index] = curr;
                ++actual_node_count;
                curr = next;
            }
        }

        std::free(_table);
        _table = _new_table;
        _capacity = new_capacity;
        _size.store(actual_node_count, std::memory_order_relaxed);
        _resize_threshold.store(static_cast<size_t>(new_capacity * _max_load_factor), std::memory_order_relaxed); // 更新 下一次 rehash 的 size 阈值
        // global free_list 和 tls free_list 都不需要变动
    }


public:

    explicit pooled_concurrent_hashtable(const HASH_FUNC& hasher, size_t capacity, TYPE_MEMPOOL* pool, size_t stripe_hint = 4096):
        _hasher(hasher),
        _capacity(capacity),
        _pool(pool),
        _stripe_mask(next_pow2(stripe_hint)-1),
        _stripes(next_pow2(stripe_hint))
    {
        _resize_threshold.store(static_cast<size_t>(capacity * _max_load_factor), std::memory_order_relaxed);
        alloc_table_ptrs(_capacity);
    }

    explicit pooled_concurrent_hashtable(size_t capacity, TYPE_MEMPOOL* pool, size_t stripe_hint = 4096):
        _hasher(),
        _capacity(capacity),
        _pool(pool),
        _stripe_mask(next_pow2(stripe_hint)-1),
        _stripes(next_pow2(stripe_hint))
    {
        _resize_threshold.store(static_cast<size_t>(capacity * _max_load_factor), std::memory_order_relaxed);
        alloc_table_ptrs(_capacity);
    }

    ~pooled_concurrent_hashtable() {
        destroy();
    }


    bool get(const TYPE_K& key, TYPE_V& value) {
        std::shared_lock<std::shared_mutex> _lock_from_rehash_clear_(_table_mutex);
        if (_capacity == 0 || !_table) return false;
        size_t index = hash(key) % _capacity;

        std::shared_lock<std::shared_mutex> _lock_from_insert_(bucket_lock(index));

        for (HashTableNode* cur = _table[index]; cur; cur = cur->next) {
            if (cur->key == key) {
                value = cur->value;
                return true;
            }
        }
        return false;
    }

    template <typename K, typename V>
    bool insert(K&& key, V&& value) {
        std::shared_lock<std::shared_mutex> _lock_table_from_rehash_clear_(_table_mutex);
        if (_capacity == 0 || !_table) return false;
        size_t index = hash(key) % _capacity;
        {
            std::unique_lock<std::shared_mutex> _lock_bucket_for_insert_(bucket_lock(index));
            for (HashTableNode* cur = _table[index]; cur; cur = cur->next) {
                if (cur->key == key) {
                    cur->value = std::forward<V>(value);
                    return true;
                }
            }
            HashTableNode* new_node = nullptr;
            
            // 优先复用 空闲链表 里的地址
            // 单线程版本
            /*
            new_node = _free_nodes_head; // 获取第一个空闲地址
            _free_nodes_head = new_node->free_next; // 更新空闲列表
            */
            // 并发安全 TLC 版本:
            // 首先尝试从 tls free_list 中找地址(最快, 无锁)
            if (HashTableNode* node = get_node_from_tls()) {
                new_node = node;
            }
            // 如果从 tls free_list 拿到的是nullptr, 那么的先从 global free_list 批量补充 tls free_list(如有), 再从 tls free_list 取node(不一定有)
            else {
                refill_tls_from_global();
                new_node = get_node_from_tls(); // 不一定有: 如果 global free_list也空了, 那么这里 new_node = nullptr
            }

            if (new_node) {
                new(&new_node->key) TYPE_K(std::forward<K>(key));
                new(&new_node->value) TYPE_V(std::forward<V>(value));
                new_node->next = _table[index];
                new_node->free_next = nullptr;
            }
            else {
                void* raw_mem = _pool->allocate(sizeof(HashTableNode));
                if (!raw_mem) return false;

                new_node = new(raw_mem) HashTableNode{std::forward<K>(key), std::forward<V>(value), _table[index]};
            }
            
            _table[index] = new_node;
            _size.fetch_add(1, std::memory_order_relaxed);
        }

        _lock_table_from_rehash_clear_.unlock();
        if (_size.load(std::memory_order_relaxed) >= _resize_threshold.load(std::memory_order_relaxed))
        {
            std::unique_lock<std::shared_mutex> _lock_table_for_rehash_(_table_mutex);
            if (_size.load(std::memory_order_relaxed) >= _resize_threshold.load(std::memory_order_relaxed)) {
                rehash( _capacity*2 );
            }
        }

        return true;
    }

    template <typename K, typename FUNC>
    bool atomic_upsert(K&& key, FUNC&& updater, const TYPE_V& default_val) {
        std::shared_lock<std::shared_mutex> _lock_table_from_rehash_clear_(_table_mutex);
        if (_capacity == 0 || !_table) return false;
        size_t index = hash(key) % _capacity;
        {
            std::unique_lock<std::shared_mutex> _lock_bucket_for_insert_(bucket_lock(index));
            for (HashTableNode* cur = _table[index]; cur; cur = cur->next) {
                if (cur->key == key) {
                    std::forward<FUNC>(updater)(cur->value);
                    return true;
                }
            }
            HashTableNode* new_node = nullptr;
            
            // 优先复用 空闲列表 里的地址
            // 单线程版本
            /*
            new_node = _free_nodes_head; // 获取第一个空闲地址
            _free_nodes_head = new_node->free_next; // 更新空闲列表
            */
            // 并发安全 TLC 版本:
            // 首先尝试从 tls free_list 中找地址(最快, 无锁)
            if (HashTableNode* node = get_node_from_tls()) {
                new_node = node;
            }
            // 如果从 tls free_list 拿到的是nullptr, 那么的先从 global free_list 批量补充 tls free_list(如有), 再从 tls free_list 取node(不一定有)
            else {
                refill_tls_from_global();
                new_node = get_node_from_tls(); // 不一定有: 如果 global free_list也空了, 那么这里 new_node = nullptr
            }

            if (new_node) {
                new(&new_node->key) TYPE_K(std::forward<K>(key));
                new(&new_node->value) TYPE_V(default_val);
                new_node->next = _table[index];
                new_node->free_next = nullptr;
            }
            else {
                void* raw_mem = _pool->allocate(sizeof(HashTableNode));
                if (!raw_mem) return false;

                new_node = new(raw_mem) HashTableNode{std::forward<K>(key), default_val, _table[index]};
            }
            std::forward<FUNC>(updater)(new_node->value);
            _table[index] = new_node;
            _size.fetch_add(1, std::memory_order_relaxed);
        }

        _lock_table_from_rehash_clear_.unlock();
        if (_size.load(std::memory_order_relaxed) >= _resize_threshold.load(std::memory_order_relaxed))
        {
            std::unique_lock<std::shared_mutex> _lock_table_for_rehash_(_table_mutex);
            if (_size.load(std::memory_order_relaxed) >= _resize_threshold.load(std::memory_order_relaxed)) {
                rehash( _capacity*2 );
            }
        }

        return true;
    }

    bool pop(const TYPE_K& key, TYPE_V& value) {
        std::shared_lock<std::shared_mutex> _lock_table_from_rehash_clear_(_table_mutex);
        if (_capacity == 0 || !_table) return false;
        size_t index = hash(key) % _capacity;

        // 预设一个 node_to_recycle: 如果它在后续的过程中被更新到确实存在, 那么执行挂载到 free_list 的动作可以在 桶锁/条带锁 之外: 因为它是线程local的
        HashTableNode* node_to_recycle = nullptr;
        {
            std::unique_lock<std::shared_mutex> _lock_bucket_for_pop_(bucket_lock(index));
            HashTableNode* head = _table[index];
            HashTableNode* parent = nullptr;

            while (head) {
                if (head->key == key) { // 已定位到 待摘除node: head
                    // 1. 先拷贝 value
                    value = head->value;

                    // 2. 从 bucket链表中摘除 head
                    if (!parent) {
                        _table[index] = head->next;
                    }
                    else {
                        parent->next = head->next;
                    }

                    // 3. 先析构 key 和 value
                    if constexpr(!std::is_trivially_destructible<TYPE_K>::value) head->key.~TYPE_K();
                    if constexpr(!std::is_trivially_destructible<TYPE_V>::value) head->value.~TYPE_V();

                    // 4. 清理指针(防御性编程)
                    head->next = nullptr;
                    head->free_next = nullptr;
                    _size.fetch_sub(1, std::memory_order_relaxed);

                    // 应该把待摘除node 即 head 挂载到 空闲列表.
                    // 如果是单线程版本, 在这里就可以执行这个挂载操作了(如下). 执行完就可以return true跳出循环.
                    /*
                    head->free_next = _free_nodes_head; // 更新 head
                    _free_nodes_head = head; // _free_nodes_head 改成 head
                    */
                    // TLC版本在这里确定好 node_to_recycle, 然后在桶锁之外执行挂载 free_list 操作
                    node_to_recycle = head;
                    break;
                }
                parent = head;
                head = head->next;
            }
        }

        // 5. TLC版本里, 挂载 free_list 操作是 thread-local + locked global 串行的, 所以不需要 bucket lock
        if (node_to_recycle) {
            push_node_to_tls(node_to_recycle);
            TLSFreeList* tls_free_list = get_tls_free_list();
            if (tls_free_list && tls_free_list->count > TLS_FREE_LIST_MAX) {
                flush_tls_to_global();
            }
            return true;
        }
        
        return false;
    }

    void clear() {
        std::unique_lock<std::shared_mutex> _lock_table_(_table_mutex);
        for (size_t index = 0; index < _capacity; ++index) {
            HashTableNode* head = _table[index];
            if constexpr(!std::is_trivially_destructible<HashTableNode>::value) {
                while (head) {
                    HashTableNode* next = head->next;
                    destroy_node(head);
                    head = next;
                }
            }
            _table[index] = nullptr;
        }

        // 全表clear时置空 global free_list, 等待 reset 内存池全表复用而不是复用空闲链表上的地址
        {
            std::lock_guard<std::mutex> lock(_global_free_mutex);
            _global_free_head = nullptr; 
        }

        // 增加 generation，使所有线程的 TLS 缓存 懒更新失效(线程要用的时候检查generation发现失效)
        _generation.fetch_add(1, std::memory_order_release);
        _size.store(0, std::memory_order_relaxed);

    }

    void destroy() {
        std::unique_lock<std::shared_mutex> _lock_table_(_table_mutex);
        for (size_t index = 0; index < _capacity; ++index) {
            HashTableNode* head = _table[index];
            if constexpr(!std::is_trivially_destructible<HashTableNode>::value) {
                while (head) {
                    HashTableNode* next = head->next;
                    destroy_node(head);
                    head = next;
                }
            }
        }
        
        free_table_ptrs();
        {
            std::lock_guard<std::mutex> lock(_global_free_mutex);
            _global_free_head = nullptr;
        }
        _capacity = 0;
        _resize_threshold.store(0, std::memory_order_relaxed);
        _generation.fetch_add(1, std::memory_order_release);
        _size.store(0, std::memory_order_relaxed);
    }

    size_t size() const {
        return _size.load();
    }

#if 0
    // _capacity 的修改 必须在 表级写锁下, 这保证了它的线程安全性, 以及“同一把表级写锁在线程之间的内存同步性”，故不需要引入原子类型
    size_t _capacity;

    const float _max_load_factor = 0.75f;

    // _size的修改 存在并发写入的可能(不同条带/桶), 故 _size 类型需要引入原子类型
    std::atomic<size_t> _size{0};

    // _table 的修改 必须在 表级写锁下, 这保证了它的线程安全性, 以及“同一把表级写锁在线程之间的内存同步性”，故不需要引入原子类型
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

    // gc链表 --> 已废弃
    // std::atomic<HashTableNode*> _all_nodes_head{nullptr};

    // 空闲 free 链表: 链起所有 析构后的 poped nodes. 其修改 存在并发写入的可能(不同条带/桶), 故需要引入原子类型
    std::atomic<HashTableNode*> _free_nodes_head{nullptr};

    // 非空桶记录  --> 已废弃
    // 记录非空bucket index. 桶置空操作时只需遍历这些桶即可. index类型要与 _capacity 类型对齐, 因为它是 hash成员函数的输出 取_capacity余
    // 要么给 _occuped_indices 另外加一个 锁, 要么去掉. 选择 去掉 _occupied_indices
    // std::vector<size_t> _occupied_indices;

    TYPE_MEMPOOL* _pool;

    HASH_FUNC _hasher;

    // 使用哈希器, 对 TYPE_K 类型的输入 key, 作hash算法, 返回值类型必须是 size_t 与 _capacity 对齐
    size_t hash(const TYPE_K& key) const {
        return _hasher(key);
    }

    // mutable 关键字: 被修饰的成员变量, 即使在const成员函数中也可以修改. 此豁免多用于 锁: const函数不改变用户视角的数据, 却对_mutex变量有加锁操作
    // 本质是告诉编译器: 虽然加锁操作在物理上修改了mutex的状态, 但逻辑上该成员函数并未改变数据内容, 所以允许在const方法中加锁

    mutable std::shared_mutex _table_mutex; // 读写锁: 读锁可并发, 写锁必排他

    mutable std::vector<padded_mutex> _stripes;
    size_t _stripe_mask;  // 从 桶编号 bucket_index 映射到 条带编号 stripe

    inline std::shared_mutex& bucket_lock(size_t bucket_index) noexcept {
        return _stripes[bucket_index & _stripe_mask].lock;
    }

    // std::atomic<bool> _rehashing{false};
    // 原方案用一个 _rehashing 原子bool类型, 来表达 "当前该哈希表是否正在经历rehash".
    // 在 insert&upsert 操作之后, 要检查是否要 rehash, 如果要rehash, 那么必须要上 表级写锁. 这种争抢表级写锁的耗时很高, 严重影响并发, 所以要有一个预检查
    // _rehashing 的问题在于: 其与 表级写锁lock 并非完全一致, 即存在可能_rehashing为True时, 线程恰好被OS调度了, 表级写锁未能lock, 从而其他线程错过 rehash
    // 工业级实践中, Java/Rust 的 concurrentHashMap 都引入一个扩容阈值 resize threshold 的原子变量, 来作为 rehash 的预检查
    std::atomic<size_t> _resize_threshold{0}; // 新增, 替代 _rehashing 用于 rehash 的预检查

    void rehash(size_t new_capacity) {
        // rehash 的调用在 insert 里，调用前会加 独占表锁, 故这里不再加独占表锁避免死锁

        // 初始化一个新的 table
        HashTableNode** _new_table = static_cast<HashTableNode**>(std::calloc(new_capacity, sizeof(HashTableNode*)));
        if (!_new_table) throw std::bad_alloc();

        // 重新计算 _size, 为缩容式 rehash 留下余地. 不过缩容式rehash必要性不大: 避免频繁rehash
        size_t actual_node_count = 0;

        // 遍历 _table 所有元素. _table 是通过 std::calloc 分配的指针地址, 无法用 range-for 的方式来遍历. 必须通过 index
        // index 从 0 到 _capacity, _capacity 是线程安全的: 因为 _capacity 的修改必然在表级写锁下, 必然排他, 故表级读锁下的_capacity也必然安全
        for (size_t old_index = 0; old_index < _capacity; ++old_index) {

            HashTableNode* curr = _table[old_index];
            while (curr) {
                HashTableNode* next = curr->next; // 先取出next node
                size_t new_index = hash(curr->key) % new_capacity; // 计算得出新bucket

                // 头插到新桶
                curr->next = _new_table[new_index]; // 当前node挂载到新bucket链表头
                _new_table[new_index] = curr; // 更新确认新bucket的链表头

                // _free_nodes_head 不需要变动

                // 更新计数
                ++actual_node_count;

                curr = next;
            }
        }

        // 切换: 表级写锁下
        std::free(_table);
        _table = _new_table;
        _capacity = new_capacity;
        _size.store(actual_node_count, std::memory_order_relaxed);
        _resize_threshold.store(static_cast<size_t>(new_capacity * _max_load_factor), std::memory_order_relaxed); // 更新 下一次 rehash 的 size 阈值
        // _free_nodes_head 不需要变动
    }


public:

    explicit pooled_concurrent_hashtable(const HASH_FUNC& hasher, size_t capacity, TYPE_MEMPOOL* pool, size_t stripe_hint = 4096):
        _hasher(hasher),
        _capacity(capacity),
        _pool(pool),
        _stripe_mask(next_pow2(stripe_hint)-1),
        // vector类具备构造函数重载: vector(size_type n, const T& value = T())
        // 当T(这里是shared_mutex)可默认构造且noexcept时, vector执行T的默认构造函数n次
        // .reserve 方法只是预留容量. 这里是实打实构造了 n 个独立实例 ---> 也就是说, 这里构造了 next_pow2(stripe_hint) 个 独立的互斥锁
        _stripes(next_pow2(stripe_hint))
    {
        _resize_threshold.store(static_cast<size_t>(capacity * _max_load_factor), std::memory_order_relaxed);
        alloc_table_ptrs(_capacity);
    }

    explicit pooled_concurrent_hashtable(size_t capacity, TYPE_MEMPOOL* pool, size_t stripe_hint = 4096):
        _hasher(),
        _capacity(capacity),
        _pool(pool),
        _stripe_mask(next_pow2(stripe_hint)-1),
        // vector类具备构造函数重载: vector(size_type n, const T& value = T()), 当T(这里是shared_mutex)可默认构造且noexcept时, vector执行T的默认构造函数n次
        // .reserve 方法只是预留容量. 这里是实打实构造了 n 个独立实例 ---> 也就是说, 这里构造了 next_pow2(stripe_hint) 个 独立的互斥锁
        _stripes(next_pow2(stripe_hint))
    {
        _resize_threshold.store(static_cast<size_t>(capacity * _max_load_factor), std::memory_order_relaxed);
        alloc_table_ptrs(_capacity);
    }

    ~pooled_concurrent_hashtable() {
        destroy();
    }

    /*
    * 根据键获取值
    * @param key: 不可变引用即可. 查询不应该改变源. 即使源是右值(移动或临时资源等), const& 也能有效接收
    * @param value: 可变引用, 存储查询到的值
    * @return 如果查询成功, 返回true; 如果查询失败返回false
    * 
    * 行为: 若 key 存在, 则获取对应的 value 到可变引用, 返回 true; 否则返回 false
    */
    bool get(const TYPE_K& key, TYPE_V& value) {
        // 表读锁: 此操作(get) 要排除 rehash & clear 等需要独占(写锁)表锁的行为
        std::shared_lock<std::shared_mutex> _lock_from_rehash_clear_(_table_mutex);

        if (_capacity == 0 || !_table) return false;

        size_t index = hash(key) % _capacity;

        // 并发读: 此操作(get)不独占桶锁(条带锁)
        std::shared_lock<std::shared_mutex> _lock_from_insert_(bucket_lock(index));

        for (HashTableNode* cur = _table[index]; cur; cur = cur->next) {
            if (cur->key == key) {
                // 获取值回调. 触发 node.value 的拷贝赋值: 生命周期分离, 这里不返回引用, 保证哈希表的资源生命周期不影响外部变量对象
                value = cur->value;
                return true;
            }
        }
        return false;
    }

    /*
    * 语义等同于 std::unordered_ma.insert_or_assign: 有则更新，无则插入
    * @param key: 可能是左值/常左值, 此时 key 类型为 TYPE_K&/const TYPE_K&, 源对象不会被掏空; 也可能是右值(临时/移动), 此时 key 类型为 TYPE_K&&, 源对象会被掏空
    * @param value: 同 key
    * key-pair 组成的 HashTableNode 在创建时, 如果 key / value 是右值引用, 那么可以调用 节点的移动构造 来节省拷贝成本
    * ---> 模板函数
    * @return 如果插入或更新成功, 返回true; 如果内存分配失败返回false
    * 
    * 行为: 若 key 已经存在, 则更新对应的 value; 否则新建节点key 插入默认值作为value, 然后再更新value. 插入后检查是否需要扩容
    */
    template <typename K, typename V>
    bool insert(K&& key, V&& value) {
        // 并发锁: 此操作(insert)不独占表锁
        std::shared_lock<std::shared_mutex> _lock_table_from_rehash_clear_(_table_mutex);
        
        if (_capacity == 0 || !_table) return false;

        size_t index = hash(key) % _capacity;
        {
            // 独占写: 此操作(花括号内部) 独占桶锁(条带锁)，即只有此操作发生
            std::unique_lock<std::shared_mutex> _lock_bucket_for_insert_(bucket_lock(index));

            for (HashTableNode* cur = _table[index]; cur; cur = cur->next) {
                if (cur->key == key) {
                    // 如果用value(具名变量作为左值), 会触发TYPE_V的拷贝赋值. 但语义上若value在参数签名处为 右值引用时, 调用本意应该是移动构造
                    cur->value = std::forward<V>(value); // 用std::forward完美转发, 保持 value 的右值语义(如果最开始是右值), 得以触发TYPE_V的移动赋值(如果有)
                    return true;
                }
            }

            // 如果执行到这里, 说明要么 _table[index] 是 nullptr, 要么 _table[index] 链表里没有 key

            // 那么就要执行新建节点, 并将新节点放到 _table[index] 这个bucket的头部
            HashTableNode* new_node = nullptr;
            
            // 首先复用 _free_nodes_head 里的地址. _free_nodes_head 是全局变量, 要考虑并发安全
            // 单线程版本
            /*
            new_node = _free_nodes_head; // 获取第一个空闲地址
            _free_nodes_head = std::launder(new_node)->free_next; // 更新空闲列表
            */
            // 并发安全 CAS 版本:
            while (_free_nodes_head.load(std::memory_order_relaxed) != nullptr) { 
                // 尝试从 free list 中复用: 获取 _free_nodes_head(relaxed表示无同步成本)
                HashTableNode* curr_head = _free_nodes_head.load(std::memory_order_relaxed);

                if (curr_head != nullptr && _free_nodes_head.compare_exchange_weak(
                    curr_head,
                    // 废弃方案
                    // std::launder(curr_head)->free_next,
                    // 新方案
                    curr_head->free_next,
                    std::memory_order_acquire, std::memory_order_relaxed)
                ) {
                    // 语义: curr_head 是 全局变量 _free_nodes_head 尝试读到的旧值. 对比这个全局变量和旧值
                    // 如果一致, 那么执行全局变量更新为 free_next(do 中算出来的新值); 如果不一致, 循环尝试再取一次全局变量, 直到重试成功

                    // std::memory_order_acquire: 消费者(读操作)内存序, 代表CPU保证 --> 共享变量在执行该内存序操作后的所有指令, 必须不能重排到该内存序前面
                    // 代表一种消费者逻辑: 必须先acquire东西再消费. 这里 insert 函数对 全局变量_free_nodes_head而言就是消费者: 它消费_free_nodes_head上的空闲地址

                    // CAS 成功, 获取到的空闲列表头地址 --> new_node, 退出循环
                    new_node = curr_head;
                    break;
                }
                // CAS 失败, 重试: 重新去尝试获取空闲列表头. 如果此时空闲列表头已为空, 说明没有空闲地址了, 退出循环, new_node保持为nullptr
            }

            if (new_node) {
                // 废弃方案:
                // 在 new_node指向的地址上(已析构), placement new 构造, 并用头插法在构造时直接把该index代表的bucket插入new_node->next
                // new(new_node) HashTableNode{std::forward<K>(key), std::forward<V>(value), _table[index]};
                // 新方案:
                new_node->next = _table[index];
                new_node->free_next = nullptr;
                new(&new_node->key) TYPE_K(std::forward<K>(key));
                new(&new_node->value) TYPE_V(std::forward<V>(value));

                // 完美转发以保持key和value的 左/右 值引用性质, 才能触发对应的 HashTableNode 构造函数(左(常)值引用-->拷贝, 右值引用-->移动)
                // 如果传入的是右值引用，那么源对象会被掏空. 这样调用的本意就是转移资源，所以不介意源被掏空.
            }
            else {
                // 空闲列表为空, 申请新内存
                void* raw_mem = _pool->allocate(sizeof(HashTableNode));
                if (!raw_mem) return false;

                new_node = new(raw_mem) HashTableNode{std::forward<K>(key), std::forward<V>(value), _table[index]};
            }
            
            _table[index] = new_node;

            // // 新的 node 要线程安全地插入gc链: 独占的桶锁(条带锁)仅锁住了当前桶(条带), 但是gc链是全局的, 可能有其他桶(条带)在写入, 故这里要线程安全 --> 废弃
            // HashTableNode* old = _all_nodes_head.load(std::memory_order_relaxed);
            // do {
            //     new_node->gc_next = old; // 头插 gc 链
            // } while (!_all_nodes_head.compare_exchange_weak(old, new_node, std::memory_order_release, std::memory_order_relaxed));

            // node数量自加1. 原子线程安全
            // std::memory_order_relaxed 就可以保证原子安全. 但未来若需要在某些线程里仅靠_size来判断是否有数据写入, 这个模式不安全.
            // 这个模式下, 其他线程不一定能看到 自增后的 _size. 可以用 _size.fetch_add(1) 默认模式, 最严格, 保证全局一致.
            _size.fetch_add(1, std::memory_order_relaxed);

        }

        _lock_table_from_rehash_clear_.unlock(); // 解开并发读的表锁, 是因为 rehash 操作需要 独占表锁.
        // 写锁unlock->写锁lock  写锁unlock->读锁lock 之间存在 synchronizes-with 数据同步
        // 但这里是 读锁unlock, 它与后续 读锁lock / 写锁lock 之间不存在数据同步
        // 但这不影响安全: 因为 这里 读锁lock期间 _table和_capacity都不会有改动, 而 _size 虽然有变动, 但是它有自己的同步机制std::atomic

        // 释放表级读锁之后, 可能会由其他线程加锁(本表所有方法都是锁内操作), 那么本线程会阻塞到表级写锁之前.
        // 如果其他线程加的是表级读锁, 那么其他线程必定是在执行 get / insert(before rehash). 这些执行中 _capacity 都不会变
        // 1. 如果其他线程执行的是 get, 那么 _capacity/_size 不会被改变, 本线程继续时, 各成员变量都没有竞态风险
        // 2. 如果其他线程执行的是 insert(before rehash), 那么_capacity不会被改变, 但_size会增加, 本线程继续时, 应该以新的_size去判断是否要 再一次rehash

        // 如果其他线程加的就是表级写锁, 那么其他线程必定是在执行 rehash / clear / destroy 三者之一.
        // 3. 如果其他线程执行的是 rehash, 那么 _capacity 会被改变(安全增大), 本线程继续时, 应该以新的 _capacity 去判断是否要 再一次rehash
        // 4. 如果其他线程执行的是 clear/destroy, 那么_size会被清零, 后面不应该执行任何rehash, 本线程继续时, rehash应该被跳过, 也就是以新的_size去判断

        // _size 是原子变量, 能同步自身, 所以1.2.4.都不会有问题; 独占写锁的 unlock-lock 之间具备内存同步语义, 故3.也不是问题, 即
        // 本线程在 unique_lock 表锁时, 会同步 表级写锁 作用域内所有修改, 即 _size 和 _capacity 都得到同步.
        // ----> 多次 rehash 前后, 数据必然得到同步, 因为每一次 rehash 都是 表级写锁 操作, 必然同步内存.

        // 扩容检查. 因为在临近扩容时, 由于多并发写入, 会有多个进程近乎同时判断出需要rehash. 希望减少rehash次数.
        // 但另一方面, 为了保证总是在锁内读取 _size和_capacity, 检查是否要rehash时必须要加 表级写锁 --> 每一次insert都要表级写锁lock, 成本太高
        // ----> 二次检查以减少 触发 表级写锁. 原方案是 _rehashing --update--> _resize_threshold

        // UPDATE: 这里 存在锁外读取 _capacity，以及 _rehashing 与 _table_mutex 可能冲突 --> 去除 _rehashing, 引入一个原子变量 _capacity_threshold
        
        // // 临近状态下, 多个线程都满足第一个条件, 但是第二个条件: 原子变量 _rehashing == expected(false) 只能原子级满足
        // // compare_exchange_strong 保证了一旦原子变量 _rehashing 满足 == false, 马上将转化为true并返回true.
        // // 如此其他线程在这里只会得到一个为 true 的_rehashing, 从而无法进入内部.
        // bool expected = false;
        // if (_size >= _capacity*_max_load_factor && _rehashing.compare_exchange_strong(expected, true))
        if (_size.load(std::memory_order_relaxed) >= _resize_threshold.load(std::memory_order_relaxed))
        {
            // 独占 _table_mutex 表锁, rehash 时其他任何线程不能对table作任何操作. 作用到rehash结束
            std::unique_lock<std::shared_mutex> _lock_table_for_rehash_(_table_mutex);

            // 二次检查. 写锁作用域内的 共享状态才会被 synchronize-with, 所以这里会同步 写锁作用域内的 其他线程的操作
            // if (_size >= _capacity*_max_load_factor)
            if (_size.load(std::memory_order_relaxed) >= _resize_threshold.load(std::memory_order_relaxed))
            { // _size 和 _capacity 会同步
                rehash( _capacity*2 );
            }
            // _rehashing.store(false);
        }

        return true;
    }

    /*
    * @param key: 可能是左值/常左值, 此时 key 类型为 TYPE_K&/const TYPE_K&, 源对象不会被掏空; 也可能是右值(临时/移动), 此时 key 类型为 TYPE_K&&, 源对象会被掏空
    * @param updater: 
        updater 应该是一个函数指针, 比如 函数指针 std::function<void(TYPE_V&)> 或
        函数指针的左值引用 std::function<void(TYPE_V&)>& 或
        函数指针的const&引用 const std::function<void(TYPE_V&)>& 这样可以const引用右值(lambda函数)
        这里采用最灵活的模板写法, &&万能引用，然后在内部用 std::forward<Func>(updater) 替代 updater 来实现完美转发
    * @param default_val: 不同于 key, key的源对象不在乎会不会掏空 --> 计数或插入了就行. 而 default_value 完全很可能是重复使用的, 所以不应该被掏空 --> 用const&
    * key-pair 组成的 HashTableNode 在创建时, 如果 key / value 是右值引用, 那么可以调用 节点的移动构造 来节省拷贝成本
    * ---> 模板函数
    * @return 如果插入或更新成功, 返回true; 如果内存分配失败返回false
    * 
    * 行为: 若 key 已经存在, 则更新对应的 value; 否则新建节点插入. 插入后检查是否需要扩容
    */
    template <typename K, typename FUNC>
    bool atomic_upsert(K&& key, FUNC&& updater, const TYPE_V& default_val) {
        // 并发锁: 此操作(update by insert)不独占表锁
        std::shared_lock<std::shared_mutex> _lock_table_from_rehash_clear_(_table_mutex);

        if (_capacity == 0 || !_table) return false;

        size_t index = hash(key) % _capacity;
        {
            // 独占写: 此操作(花括号内部) 独占桶锁(条带锁)，即只有此操作发生
            std::unique_lock<std::shared_mutex> _lock_bucket_for_insert_(bucket_lock(index));

            // 在该bucket中遍历寻找, 以尝试执行 update 逻辑
            for (HashTableNode* cur = _table[index]; cur; cur = cur->next) {
                if (cur->key == key) {
                    std::forward<FUNC>(updater)(cur->value);
                    return true;
                }
            }

            // 如果执行到这里, 说明要么 _table[index] 是 nullptr, 要么 _table[index] 链表里没有 key, 无法执行 update 逻辑

            // 那么就要执行新建节点, 并将新节点放到 _table[index] 这个bucket的头部
            HashTableNode* new_node = nullptr;
            
            // 首先复用 _free_nodes_head 里的地址. _free_nodes_head 是全局变量, 要考虑并发安全
            // 单线程版本
            /*
            new_node = _free_nodes_head; // 获取第一个空闲地址
            _free_nodes_head = std::launder(new_node)->free_next; // 更新空闲列表
            */
            // 并发安全 CAS 版本:
            while (_free_nodes_head.load(std::memory_order_relaxed) != nullptr) { 
                // 尝试从 free list 中复用: 获取 _free_nodes_head (relaxed表示无同步成本) 的 tls副本
                HashTableNode* curr_head = _free_nodes_head.load(std::memory_order_relaxed);

                if (curr_head != nullptr && _free_nodes_head.compare_exchange_weak(
                    curr_head,
                    // 废弃方案:
                    // std::launder(curr_head)->free_next,
                    // 新方案:
                    curr_head->free_next,
                    std::memory_order_acquire, std::memory_order_relaxed)
                ) {
                    // 语义: curr_head 是 全局变量 _free_nodes_head 尝试读到的旧值. 对比这个全局变量和旧值
                    // 如果一致, 那么执行全局变量更新为 free_next(do 中算出来的新值); 如果不一致, 循环尝试再取一次全局变量, 直到重试成功

                    // std::memory_order_acquire: 消费者(读操作)内存序, 代表CPU保证 --> 共享变量在执行该内存序操作后的所有指令, 必须不能重排到该内存序前面
                    // 代表一种消费者逻辑: 必须先acquire东西再消费. 这里 insert 函数对 全局变量_free_nodes_head而言就是消费者: 它消费_free_nodes_head上的空闲地址

                    // CAS 成功, 获取到的空闲列表头地址 --> new_node, 退出循环
                    new_node = curr_head;
                    break;
                }
                // CAS 失败, 重试: 重新去尝试获取空闲列表头. 如果此时空闲列表头已为空, 说明没有空闲地址了, 退出循环, new_node保持为nullptr
            }

            if (new_node) {
                // 废弃方案:
                // 在 new_node指向的地址上(已析构), placement new 构造, 并用头插法在构造时直接把该index代表的bucket插入new_node->next
                // new(new_node) HashTableNode{std::forward<K>(key), default_val, _table[index]};
                // 新方案:
                new_node->next = _table[index];
                new_node->free_next = nullptr;
                new(&new_node->key) TYPE_K(std::forward<K>(key));
                new(&new_node->value) TYPE_V(default_val);
            }
            else {
                // 空闲列表为空, 申请新内存
                void* raw_mem = _pool->allocate(sizeof(HashTableNode));
                if (!raw_mem) return false;

                new_node = new(raw_mem) HashTableNode{std::forward<K>(key), default_val, _table[index]};
            }

            // 更新插入后的默认值value
            std::forward<FUNC>(updater)(new_node->value);

            _table[index] = new_node;

            // // 新的 node 要线程安全地插入gc链: 独占的桶锁(条带锁)仅锁住了当前桶(条带), 但是gc链是全局的, 可能有其他桶(条带)在写入, 故这里要线程安全 --> 废弃
            // HashTableNode* old = _all_nodes_head.load(std::memory_order_relaxed);
            // do {
            //     new_node->gc_next = old; // 头插 gc 链
            // } while (!_all_nodes_head.compare_exchange_weak(old, new_node, std::memory_order_release, std::memory_order_relaxed));

            _size.fetch_add(1, std::memory_order_relaxed);
        }
        
        _lock_table_from_rehash_clear_.unlock();

        if (_size.load(std::memory_order_relaxed) >= _resize_threshold.load(std::memory_order_relaxed))
        {
            std::unique_lock<std::shared_mutex> _lock_table_for_rehash_(_table_mutex);

            if (_size.load(std::memory_order_relaxed) >= _resize_threshold.load(std::memory_order_relaxed))
            {
                rehash( _capacity*2 );
            }
        }

        return true;
    }

    /*
    * 从哈希表获取值, 并移除键值对
    * @param key: 不可变引用. pop不会涉及 new hashtable node的构造，故在key-value资源移动/拷贝之间的优化空间非常少.
    * @param value: 可变引用, 存取查询到的值
    * @return 如果查询到, 则将值拷贝进入value, 从哈希表移除键值对, 返回true; 如果未查询到则返回false
    * 
    * 行为: 若 key 存在, 则获取对应的 value 到可变引用, 返回 true; 否则返回 false
    */
    bool pop(const TYPE_K& key, TYPE_V& value) {
        // 并发锁: 此操作(pop)不独占表锁
        std::shared_lock<std::shared_mutex> _lock_table_from_rehash_clear_(_table_mutex);

        if (_capacity == 0 || !_table) return false;

        // 计算 bucket index
        size_t index = hash(key) % _capacity;
        {
            // 独占写: 此操作(花括号内部) 独占桶锁(条带锁)，即只有此操作发生
            std::unique_lock<std::shared_mutex> _lock_bucket_for_pop_(bucket_lock(index));

            // 遍历查询 key

            // 若 key-hash 不存在, 直接返回 false 结束
            HashTableNode* head = _table[index];
            if (!head) return false;

            // 若 key-hash 存在, 遍历该链表以查询 key
            HashTableNode* parent = nullptr; // 为了"删除"节点, 需要跟随保留父节点指针

            while (head) {
                if (head->key == key) {
                    // 已定位到待摘除的node
                    // 获取 value. 触发 node.value 的拷贝赋值: 生命周期分离, 这里不返回引用, 保证哈希表的资源生命周期不影响外部变量对象
                    value = head->value;

                    // 摘除 node
                    if (!parent) { // parent为空, 说明其未曾更新, 说明头节点head就是待删除节点
                        _table[index] = head->next; // 摘除head
                    }
                    else { // 如果 parent 不为空, 说明待删节点head不是头节点
                        parent->next = head->next; // parent 一定不是空指针: next重挂, 从而 head 从链表中脱离
                    }

                    // 防御性编程 置空 head 的 next以防止非法访问
                    head->next = nullptr;

                    // node数量自减1. 原子线程安全
                    _size.fetch_sub(1, std::memory_order_relaxed);

                    // 把 head 挂到 free_list 上, 然后析构 head. 这样该地址可被 insert/upsert 等插入方法复用
                    // _free_nodes_head 是全局变量, 需要 CAS(compare and swap) 操作以保证 head 挂入时安全

                    // 单线程版本
                    /*
                    head->free_next = _free_nodes_head; // 更新 head
                    _free_nodes_head = head; // _free_nodes_head 改成 head
                    */
                    // _free_nodes_head作为被输入到多个线程的指针, 它是共享变量. 它的取值&更新, 在线程之间存在竞争问题; 而head已经在桶(条带)锁之下, 不会有竞争问题

                    // 并发安全 CAS 版本: 时间顺序详解如下
                    // do 部分 <-- 竞争的线程AB各自读到了 _free_nodes_head 并执行了 head_A 更新 和 head_B 更新
                    // while 部分 <-- 线程A更快, 首先执行 compare_exchage: _free_nodes_head 对比 tls_head. 此时一致
                    //                compare_exchange给线程A执行 _free_nodes_head 改成 head_A, 返回 True, 从而线程A退出循环
                    // while 部分 <-- 线程B执行 compare_exchange: _free_nodes_head 对比 tls_head, 此时不一致(前者已经被线程A修改成head_A)
                    //                compare_exchange 直接返回 False, 线程B重新进入do 部分 <-- 线程B读到了更新后的 _free_nodes_head, 再一次执行 head_B 更新
                    // while 部分 <-- 线程B执行 compare_exchange: _free_nodes_head 对比 tls_head. 此时终于一致
                    //                compare_exchange给线程B执行 _free_nodes_head 改成 head_B, 返回 True, 从而线程B退出循环
                    HashTableNode* tls_head;
                    do {
                        tls_head = _free_nodes_head.load(std::memory_order_relaxed);
                        head->free_next = tls_head;
                    } while (!_free_nodes_head.compare_exchange_weak(tls_head, head, std::memory_order_release, std::memory_order_relaxed));
                    // 语义: tls_head 是 全局变量 _free_nodes_head 在 do 中读到的旧值. 对比这个全局变量和旧值
                    // 如果一致, 那么执行全局变量更新为 head(do 中算出来的新值); 如果不一致, 循环do 用 全局变量的新值再来一次, 直到重试成功

                    // atomic_var.compare_exchange_weak(expect_val, new_val, success_memory_order, failure_memory_order) 语义:
                    // 对比 atomic_var 和 expect_val
                    //      如果相同, 则执行更新: atomic_var <- new_val, 内存序 success_memory_order, 返回 True;
                    //      如果不相同, 内存序 success_memory_order, 返回 False

                    // std::memory_order_release: 生产者(写操作)内存序, 代表CPU保证 --> 共享变量在执行该内存序操作前的所有指令, 必须不能重排到该内存序后面
                    // 代表一种生产者逻辑: 东西全部生产完毕了才能release. 这里 pop 函数对 全局变量_free_nodes_head而言就是生产者: 它生产空闲地址发布到_free_nodes_head上

                    // 废弃方案: 析构整个被摘除的 node
                    // destroy_node(head);

                    // 新方案: 只析构数据成员变量 key 和 value, head指向的 HashTableNode 作为空壳仍然valid等待复用
                    if constexpr(!std::is_trivially_destructible<TYPE_K>::value) head->key.~TYPE_K();
                    if constexpr(!std::is_trivially_destructible<TYPE_V>::value) head->value.~TYPE_V();

                    return true;
                }
                parent = head;
                head = head->next;
            }

        }
        
        // 如果执行到这里, 说明从 head 遍历到 nullptr 都没能查找到 key. 那么这个是不应该的: key-hash在此index
        // throw std::runtime_error("Error in concurrent hashtable pop");
        return false;
    }

    void clear() {
        // 清空全表时, 清空过程要全程 独占表锁
        std::unique_lock<std::shared_mutex> _lock_table_(_table_mutex);

        // 遍历所有(非空)buckets, 首先对每个链表头, 沿着链表头析构所有node, 然后将该链表头置空
        for (size_t index = 0; index < _capacity; ++index) {
            HashTableNode* head = _table[index];
            // 若 node 需要非平凡析构. constexpr 关键字的意思是在编译期求值: 即编译期即可知道括号内是true还是false
            if constexpr(!std::is_trivially_destructible<HashTableNode>::value) {
                // 遍历所有buckets, 沿着链表头析构所有node
                while (head) {
                    HashTableNode* next = head->next;
                    destroy_node(head);
                    head = next;
                }
            } // 若 node 不需要非平凡析构：就跳过析构环节

            _table[index] = nullptr; // _table指针数组(buckets)保持结构.
        }

        _free_nodes_head.store(nullptr, std::memory_order_relaxed); // 全表clear时置空 空闲链表, 等待 reset 内存池全表复用. 不会复用空闲链表上的空壳node地址.
        // 不要沿着空闲链表去析构那些空壳node. 它们内部的非平凡析构成员(如果是)key和value已经被析构了, 只剩下平凡析构的两个指针. 再次析构会造成double free
        _size.store(0, std::memory_order_relaxed);

    }

    // clear 不破坏表结构, 即 bucket 数组仍然存在. destroy 在 clear 基础上, 释放 bucket 数组 _table, _capacity置0 即完全破坏表结构
    // destroy 之后 哈希表不可复用. 但是所使用过的内存未释放, 等待mempool在外部统一释放
    void destroy() {
        // 析构全表时, 析构过程要全程 独占表锁
        std::unique_lock<std::shared_mutex> _lock_table_(_table_mutex);

        // 遍历所有(非空)buckets, 首先对每个链表头, 沿着链表头析构所有node, 然后将该链表头置空
        // for (size_t index: _occupied_indices)
        for (size_t index = 0; index < _capacity; ++index) {
            HashTableNode* head = _table[index];
            // 若 node 需要非平凡析构. constexpr 关键字的意思是在编译期求值: 即编译期即可知道括号内是true还是false
            if constexpr(!std::is_trivially_destructible<HashTableNode>::value) {
                // 遍历所有buckets, 沿着链表头析构所有node
                while (head) {
                    HashTableNode* next = head->next;
                    destroy_node(head);
                    head = next;
                }
            } // 若 node 不需要非平凡析构：就跳过析构环节
        }
        
        // destroy 和 clear 的区别就在于: destroy 摧毁了桶结构, _table 链表头数组释放, 各链表不再可访问, _capacity/_resize_threshold置零
        // 本哈希表不再可复用. 但内存尚未释放, 等待内存池操作
        free_table_ptrs();

        _size.store(0, std::memory_order_relaxed);
        _capacity = 0;
        _resize_threshold.store(0, std::memory_order_relaxed);
        _free_nodes_head.store(nullptr, std::memory_order_relaxed);
        // 不要沿着空闲链表去析构那些空壳node. 它们内部的非平凡析构成员(如果是)key和value已经被析构了, 只剩下平凡析构的两个指针. 再次析构会造成double free
    }

    size_t size() const {
        return _size.load();
    }
#endif


    // 迭代相关
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
    *   1. 非const迭代器, 应该在迭代时上独占表锁, 其他任何线程不能对表有任何操作(读写都不行). 迭代器可change value
    *      : 不依靠数据结构解决业务层的问题
    *      --> 如果需要强一致性的全局遍历(可修改value), 应该是业务层对整个容器加锁 + unsafe遍历(*返回MutableProxy)
    *      --> 如果需要弱一致性(即程序运行时不出问题但不保证前后一致,允许漏看多看)的全局遍历(可修改value), key只读快照遍历 + insert/atomic_upsert调用

    *   2. const迭代器, 允许并发迭代, 应该共享表锁(禁止了需要独占表锁的rehash/clear), 共享桶锁(禁止了需要独占桶锁的insert/atomic_upsert/pop)
    *      迭代器是只读的. 哈希表不可被任何change, 即线程A迭代bucket_i时, 不该允许线程B在bucket_i作insert和remove
    *      这里似乎可以允许线程B在bucket_j作insert和remove, 因为线程A在迭代bucket_i时, 对其他桶似乎可以不作要求. 只不过这样的话，
    *      多线程并发迭代的结果可能会不一样. 如果要求保证并发迭代的结果一致, 那么线程A在迭代bucket_i时, 应该对全部bucket都共享锁.
    *      可是这种需求有更好的实现方式: 先单线程迭代一遍哈希表并dump成副本, 然后多线程使用该副本. 所以这里不对全部桶上共享锁.
    *      并发迭代有不同的设计模式: 1. 多个线程并发无误遍历一遍哈希表（总共一遍），2. 多个线程各自并发无误遍历一遍哈希表（总共多遍）
    *      前者多个线程并发遍历一遍哈希表,（迭代器的_node指针是线程local的, 不能多线程共享. 遍历过程中_node指针很多跳转, 共享需要极其精细的
    *      同步机制, 那就不现实.）即使是为了加速迭代也应该使用分片（sharding）多线程迭代的方式（每个线程负责一部分bucket）. 
    *      那么这样的迭代器和全迭代肯定是不同设计的（需要输入bucket id以发送给不同线程，以实现sharding并行扫描），是高性能哈希表TBB/folly::F14的做法,
    *      并不是常规iterator的职责范围. 在这里首先实现的是“多个线程各自并发无误遍历一遍哈希表（总共多遍）”的const只读迭代器。
    *      : 完全阻塞了hashtable的表级操作(rehash/clear)
    *      : ++it的时候存在共享桶锁交接, 这个间隙里如果有线程独占桶锁并执行了桶的改变(insert/upsert/pop), 会造成遍历前后不一致
    *      : 返回引用的悬垂问题: 返回了const T& 后迭代器内部锁就释放了, 此时若其他线程删除了node, 就会出现use-after-free问题
    *      --> 如果允许阻塞写 <==> 强一致性的 只读遍历, 那么复用 独占表锁 + unsafe遍历(*返回ConstProxy)
    *      --> 如果不允许阻塞写 <==> 弱一致性的 只读遍历, 那么 key只读快照遍历 + get调用
    */


    // 迭代相关的正确设计模式: 只要是迭代(const / value-mutable / drain), 都要阻塞写——即独占写锁给全局表锁. 这样完全放弃了并发, 好处是得到了完全的强一致性迭代

    // 至于需要并发的场景, 那么只能提供弱一致性(某个状态下的可运行状态). 不提供 并发+强一致性遍历 的原因, 是其极难处理且严重影响性能.
    // 应该在业务侧避免这种需求, 不在基建侧提供这种能力. 基建侧只提供 key只读快照, 供 弱一致性(不阻塞写，故而不保证前后一致,允许漏看多看)的迭代遍历
    /*
    * 配合 get(只读) / insert(改变value) / atomic_upsert(改变value) 并发调用, 完成相应迭代目的
    * 用法:
    * auto keys_snapshot = hashtable.get_readonly_kesy(); // keys_snapshot 是 vector<TYPE_K>. 对其的元素遍历+get(元素)/insert(元素)/upsert(元素)可以并发
    * for (const auto& key: keys_snapshot) {
    *     V value;
    *     hashtable.get(key, value); // 只读遍历
    *     hashtable.insert(key, value); // 改value遍历
    *     hashtable.atomic_upsert(key, som_func, some_default_value); // 改value遍历-回调
    * }
    */
    std::vector<TYPE_K> get_readonly_keys() const {
        std::vector<TYPE_K> keys_snapshot;
        { // 上 表读锁: 要排除 rehash & clear 等需要独占(写锁)表锁的行为
            std::shared_lock<std::shared_mutex> _lock_table_from_rehash_clear_(_table_mutex);
            keys_snapshot.reserve( size() ); // 预设大小
            // 遍历所有 bucket
            for (size_t i = 0; i < _capacity; ++i) {
                // 上 桶(条带)读锁: 要排除 insert/atomic_upsert/pop 等需要独占(写锁)桶锁的行为
                std::shared_lock<std::shared_mutex> _lock_from_insert_(bucket_lock(i));
                for (HashTableNode* node = _table[i]; node; node = node->next) {
                    keys_snapshot.push_back(node->key);
                }
            } // 该单次循环结束时 释放对应的共享桶(条带)锁
        } // 释放共享表锁
        return keys_snapshot;
    }



    struct ConstProxy {
        const TYPE_K& key;
        const TYPE_V& value;
    };

    /*
    * 不加锁、线程不安全的 只读迭代器
    */
    class unsafe_const_iterator {
        // pooled_concurrent_hashtable 为 friend, 因为要允许它访问私有的构造方法. 构造方法私有是为了防止暴露误用
        // ---> 嵌套类自动是母类的 friend, 而母类访问嵌套类的 private 需要 申明母类是friend
        friend class pooled_concurrent_hashtable;
    public:
        ConstProxy operator*() const {
            return ConstProxy{_node->key, _node->value};
        }
        unsafe_const_iterator& operator++() {
            if (_node) {
                _node = _node->next;
            }
            if (!_node) {
                _bucket_index++;
                _null_node_advance_to_next_valid_bucket();
            }
            return *this;
        }
        unsafe_const_iterator operator++(int) {
            unsafe_const_iterator tmp = *this;
            ++(*this);
            return tmp;
        }
        bool operator==(const unsafe_const_iterator& other) const {
            return _node == other._node && _hash_table == other._hash_table;
        }
        bool operator!=(const unsafe_const_iterator& other) const {
            return !(*this == other);
        }
    private:
        explicit unsafe_const_iterator(const pooled_concurrent_hashtable* hash_table, size_t bucket_index, HashTableNode* node)
            :_hash_table(hash_table),
            _bucket_index(bucket_index),
            _node(node)
        {
            _null_node_advance_to_next_valid_bucket();
        }
        const pooled_concurrent_hashtable* _hash_table;
        size_t _bucket_index;
        HashTableNode* _node;
        void _null_node_advance_to_next_valid_bucket() {
            while (!_node && _bucket_index < _hash_table->_capacity) {
                _node = (_hash_table->_table)[_bucket_index];
                if (_node) break;
                _bucket_index++;
            }
        }
    };

    // 暴露 unsafe_const_iterator 迭代器接口. 仅供 write_lock_const_view 内部或明确知道风险的外部使用
    unsafe_const_iterator unsafe_const_begin() const { return unsafe_const_iterator(this, 0, nullptr); }
    unsafe_const_iterator unsafe_const_end() const { return unsafe_const_iterator(this, _capacity, nullptr); }

    /*
    * 用 RAII视图(view) 提供安全的 给全表上 写锁的 接口. 目的是把 全表上锁 的操作交给业务层, 从而可以在业务层实现强一致性(阻塞写入表)的迭代遍历
    * 此 view 返回的是 const迭代
    * 用法(强一致性场景/阻塞表级写入): for循环持续期内, 表都上了写锁
    * for (auto&& [k, v] : hashtable.const_iter_map_locked_view()) {
    *       ..code using k(const K&类型), v(const V&类型)...
    *   }
    */
    class write_lock_const_view {
        // pooled_concurrent_hashtable 为友元, 因为要允许它访问私有的构造方法. 构造方法私有是为了防止暴露误用
        friend class pooled_concurrent_hashtable;
    private:
        const pooled_concurrent_hashtable& _map;
        std::unique_lock<std::shared_mutex> _map_write_lock;
        explicit write_lock_const_view(const pooled_concurrent_hashtable& hashtable): // 不希望数据源hashtable改动数据, 所以const引用之; 但是其内部锁已经被mutable修饰, 故而还是可以传递给 unique_lock 供改变锁状态, 达到"数据只读, 锁可写"的目的
            _map(hashtable),
            _map_write_lock(hashtable._table_mutex) // 成员对象必须在初始化列表中初始化
        {
            // 在此 write_lock_const_view 被构造出来(临时对象)后, 其有效存续期间, _table_mutex 传入 独占写锁_map_write_lock, 从而全表上写锁 阻塞写
            // 在for循环中构造它, for循环结束后自然析构, 从而释放 写锁
        }
    public:
        // 禁用拷贝, 防止锁被意外释放或多次释放
        write_lock_const_view(const write_lock_const_view&) = delete; // 禁用拷贝构造
        write_lock_const_view& operator=(const write_lock_const_view&) = delete; // 禁用拷贝赋值
        // 没有禁止的理由, 就得允许移动. 因为可能有编译器优化依靠移动. 这里需要显式确认
        write_lock_const_view(write_lock_const_view&&) = default; // 显式确认 default 移动构造
        write_lock_const_view& operator=(write_lock_const_view&&) = default; // 显式确认 default 移动赋值

        // 在 view 中封装 hashtable 的 unsafe_const_begin & unsafe_const_end 方法(无论是否私密, 作为嵌套类的view类 自动是 hashtable的friend, 可以访问)
        // 包装成 begin 和 end 提供给 for循环. 在 for循环中 ++it会自动调用 返回类型(即 unsafe_const_iterator) 的++操作符
        unsafe_const_iterator begin() { return _map.unsafe_const_begin(); }
        unsafe_const_iterator end() { return _map.unsafe_const_end(); }
    };

    // 提供获取view的接口
    write_lock_const_view const_iter_map_locked_view() const {
        return write_lock_const_view(*this);
    }





    struct MutableProxy {
        const TYPE_K& key;
        TYPE_V& value;
    };

    /*
    * 不加锁、线程不安全的 可变迭代器
    */
    class unsafe_iterator {
        // pooled_concurrent_hashtable 为 friend, 因为要允许它访问私有的构造方法. 构造方法私有是为了防止暴露误用
        // ---> 嵌套类自动是母类的 friend, 而母类访问嵌套类的 private 需要 申明母类是friend
        friend class pooled_concurrent_hashtable;
    public:
        MutableProxy operator*() const {
            return MutableProxy{_node->key, _node->value};
        }
        unsafe_iterator& operator++() {
            if (_node) {
                _node = _node->next;
            }
            if (!_node) {
                _bucket_index++;
                _null_node_advance_to_next_valid_bucket();
            }
            return *this;
        }
        unsafe_iterator operator++(int) {
            unsafe_iterator tmp = *this;
            ++(*this);
            return tmp;
        }
        bool operator==(const unsafe_iterator& other) const {
            return _node == other._node && _hash_table == other._hash_table;
        }
        bool operator!=(const unsafe_iterator& other) const {
            return !(*this == other);
        }
    private:
        explicit unsafe_iterator(pooled_concurrent_hashtable* hash_table, size_t bucket_index, HashTableNode* node)
            :_hash_table(hash_table),
            _bucket_index(bucket_index),
            _node(node)
        {
            _null_node_advance_to_next_valid_bucket();
        }
        pooled_concurrent_hashtable* _hash_table; // 迭代器所迭代的容器, 在这里是哈希表. 从这里得到bucket/node等内部结构
        size_t _bucket_index; // 遍历哈希表的所有桶, 0 -> _capacity-1
        HashTableNode* _node; // 遍历所有桶的所有node
        void _null_node_advance_to_next_valid_bucket() {
            while (!_node && _bucket_index < _hash_table->_capacity) {
                _node = (_hash_table->_table)[_bucket_index];
                if (_node) break;
                _bucket_index++;
            }
        }
    };

    // 暴露 unsafe_iterator 迭代器接口. 仅供 write_lock_view 内部或明确知道风险的外部使用
    unsafe_iterator unsafe_begin() { return unsafe_iterator(this, 0, nullptr); }
    unsafe_iterator unsafe_end() { return unsafe_iterator(this, _capacity, nullptr); }

    /*
    * 用 RAII视图(view) 提供安全的 给全表上 写锁的 接口. 目的是把 全表上锁 的操作交给业务层, 从而可以在业务层实现强一致性(阻塞写入表)的迭代遍历
    * 此 view 返回的是 可变迭代
    * 用法(强一致性场景/阻塞表级写入): for循环持续期内, 表都上了写锁
    * for (auto&& [k, v] : hashtable.iter_map_locked_view()) {
    *       v(V&类型) = some code using k(const K&类型)
    *   }
    */
    class write_lock_view {
        // pooled_concurrent_hashtable 为友元, 因为要允许它访问私有的构造方法. 构造方法私有是为了防止暴露误用
        friend class pooled_concurrent_hashtable;
    private:
        pooled_concurrent_hashtable& _map;
        std::unique_lock<std::shared_mutex> _map_write_lock;
        explicit write_lock_view(pooled_concurrent_hashtable& hashtable):
            _map(hashtable),
            _map_write_lock(hashtable._table_mutex)
        {
            // 在此 write_lock_view 被构造出来(临时对象)后, 其有效存续期间, _table_mutex 传入 独占写锁_map_write_lock, 从而全表上写锁 阻塞写
            // 在for循环中构造它, for循环结束后自然析构, 从而释放 写锁
        }
    public:
        // 禁用拷贝, 防止锁被意外释放或多次释放
        write_lock_view(const write_lock_view&) = delete; // 禁用拷贝构造
        write_lock_view& operator=(const write_lock_view&) = delete; // 禁用拷贝赋值
        write_lock_view(write_lock_view&&) = default; // 显式确认 default 移动构造
        write_lock_view& operator=(write_lock_view&&) = default; // 显式确认 default 移动赋值

        // 在 view 中封装 hashtable 的 unsafe_begin & unsafe_end 方法(无论是否私密, 作为嵌套类的view类 自动是 hashtable的friend, 可以访问)
        // 包装成 begin 和 end 提供给 for循环. 在 for循环中 ++it会自动调用 返回类型(即unsafe_itgerator) 的++操作符
        unsafe_iterator begin() { return _map.unsafe_begin(); }
        unsafe_iterator end() { return _map.unsafe_end(); }
    };

    // 提供获取view的接口
    write_lock_view iter_map_locked_view() {
        return write_lock_view(*this);
    }



    struct DrainProxy {
        // 代理对象, 用于零拷贝转移. 这里必须是值类型, 因为代理类型作为 operator* 的返回类型, 需要被触发 移动构造 成临时值, 才能将 kv 资源窃取出来, 从而达到drain语义
        TYPE_K key;
        TYPE_V value;

        // 禁止深拷贝: 这个 drain遍历返回的结果, 强制只能移动使用. 实际上尽量使用 C++17的结构化绑定 auto&& [k,v]
        DrainProxy(const DrainProxy&) = delete;
        DrainProxy& operator=(const DrainProxy&) = delete;

        // 允许移动: 显式
        DrainProxy(DrainProxy&&) = default;
        DrainProxy& operator=(DrainProxy&&) = default;

    };

    /*
    * drain语义迭代器: 破坏式遍历、移动转移资源、遍历后原容器为空
    */
    class unsafe_drain_iterator {
        // drain_iterator的构造方法为private为防止误用. 只能在 write_lock_drain_range 内部调用
        friend class write_lock_drain_range;
    private:
        // 显式构造
        explicit unsafe_drain_iterator(pooled_concurrent_hashtable* hash_table, size_t bucket_index, HashTableNode* node) 
            :_hash_table(hash_table),
            _bucket_index(bucket_index),
            _node(node)
        {
            _null_node_advance_to_next_valid_bucket();
        }
        pooled_concurrent_hashtable* _hash_table;
        size_t _bucket_index;
        HashTableNode* _node;
        void _null_node_advance_to_next_valid_bucket() {
            while (!_node && _bucket_index < _hash_table->_capacity) {
                _node = (_hash_table->_table)[_bucket_index];
                if (_node) break;
                _bucket_index++;
            }
        }
    public:
        // 不同于其他 迭代器, 因为 drain是破坏性的, 相当于rehash, 故禁用拷贝, 防止多个迭代器竞争移动同一张表
        unsafe_drain_iterator(const unsafe_drain_iterator&) = delete;
        unsafe_drain_iterator& operator=(const unsafe_drain_iterator&) = delete;
        // 允许移动，原迭代器失效
        unsafe_drain_iterator(unsafe_drain_iterator&&) = default;
        unsafe_drain_iterator& operator=(unsafe_drain_iterator&&) = default;
        DrainProxy operator*() {
            return DrainProxy{std::move(_node->key), std::move(_node->value)};
        }
        unsafe_drain_iterator& operator++() {
            if (_node) {
                HashTableNode* curr = _node;
                HashTableNode* next_node = _node->next;
                if constexpr(!std::is_trivially_destructible<TYPE_K>::value) curr->key.~TYPE_K();
                if constexpr(!std::is_trivially_destructible<TYPE_V>::value) curr->value.~TYPE_V();
                _hash_table->_table[_bucket_index] = next_node;
                _hash_table->_size.fetch_sub(1, std::memory_order_relaxed);;
                // 可以设计成 moved-from 节点在析构后加入 free_list. 不过其实没有必要, 因为drain之后全表应该处于clear状态
                _node = next_node;
            }
            if (!_node) {
                _bucket_index++;
                _null_node_advance_to_next_valid_bucket();
            }
            return *this;
        }
        // it++ 迭代器对象自增后, 返回自增前的自身拷贝. 由于 drain_iterator 禁止了拷贝构造, 且 input_iterator 也不需要返回值的后置++ 

        bool operator==(const unsafe_drain_iterator& other) const {
            return _node == other._node && _hash_table == other._hash_table;
        }
        bool operator!=(const unsafe_drain_iterator& other) const {
            return !(*this == other);
        }
    };

    
    // 不暴露 unsafe_drain_iterator 的 任何构造接口, 只允许在 drain_map_locked_view() 接口中构造 drain_range 使用

    // drain range
    class write_lock_drain_range {
        // pooled_concurrent_hashtable 为友元, 因为要允许它访问私有的构造方法. 构造方法私有是为了防止暴露误用
        friend class pooled_concurrent_hashtable;
    private:
        pooled_concurrent_hashtable* _map;
        bool _fully_drained = false;
        std::unique_lock<std::shared_mutex> _map_write_lock;
        explicit write_lock_drain_range(pooled_concurrent_hashtable* hashtable):
            _map(hashtable),
            _map_write_lock(hashtable->_table_mutex)
        {
            // 在此 write_lock_drain_range 被构造出来(临时对象)后, 其有效存续期间, _table_mutex 传入 独占写锁_map_write_lock, 从而全表上写锁 阻塞写
            // 在for循环中构造它, for循环结束后自然析构, 从而释放 写锁
        }
    
    public:
        // 禁用拷贝, 防止锁被意外释放或多次释放
        write_lock_drain_range(const write_lock_drain_range&) = delete; // 禁用拷贝构造
        write_lock_drain_range& operator=(const write_lock_drain_range&) = delete; // 禁用拷贝赋值
         // 显式确认移动
        write_lock_drain_range(write_lock_drain_range&&) = default;
        write_lock_drain_range& operator=(write_lock_drain_range&&) = default;

        // drain write_lock_drain_range 的析构: 在退出(无论是正常还是非正常)for循环时, write_lock_drain_range 被析构, 此时要清空已经被drain破坏掉的哈希表为 空表状态
        ~write_lock_drain_range() {
            if (!_map) return;
            if (_fully_drained) {
                // 当明确已经全部 drain, 走快速 置空置零命令. 此时还在 表级写锁_map_write_lock的作用之下, 所以下面操作安全
                std::fill(_map->_table, _map->_table + _map->_capacity, nullptr);
                _map->_generation.fetch_add(1, std::memory_order_release);
                _map->_size.store(0, std::memory_order_relaxed);
                _map->_global_free_head = nullptr; 
            }
            else {
                // 当中途break, 兜底清理剩余node. 如果直接调用 _map的clear方法, 有死锁风险. 把clear的内部逻辑去掉上锁在这里重写一份
                // _map->clear();
                for (size_t index = 0; index < _map->_capacity; ++index) {
                    HashTableNode* head = _map->_table[index];
                    if constexpr(!std::is_trivially_destructible<HashTableNode>::value) {
                        while (head) {
                            HashTableNode* next = head->next;
                            destroy_node(head);
                            head = next;
                        }
                    }
                    _map->_table[index] = nullptr;
                }
                {
                    std::lock_guard<std::mutex> lock(_map->_global_free_mutex);
                    _map->_global_free_head = nullptr; 
                }
                _map->_generation.fetch_add(1, std::memory_order_release);
                _map->_size.store(0, std::memory_order_relaxed);
            }
        }

        // 作为 unsafe_drain_iterator的 friend, write_lock_drain_range 封装 unsafe_drain_iterator 的 首迭代器 和 尾后迭代器为 begin & end 成员方法
        unsafe_drain_iterator begin() {
            return unsafe_drain_iterator(_map, 0, nullptr);
        }
        unsafe_drain_iterator end() {
            _fully_drained = true;
            return unsafe_drain_iterator(_map, _map->_capacity, nullptr);
        }
    };

    write_lock_drain_range drain_map_locked_view() {
        return write_lock_drain_range(this);
    }

}; // end of pooled_concurrent_hashtable definition


#endif