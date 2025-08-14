// mempool_hash_table_mt.h

#ifndef MEMPOOL_HASH_TABLE_MULTI_THREAD_H
#define MEMPOOL_HASH_TABLE_MULTI_THREAD_H


#include <vector>
#include <functional>
#include <cstddef>
#include <type_traits>
#include <shared_mutex>
#include <mutex>
#include <atomic>


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
    padded_mutex(padded_mutex&&) = default;
    padded_mutex& operator=(padded_mutex&&) = default;
};




template <typename TYPE_K, typename TYPE_V, typename TYPE_MEMPOOL, typename HASH_FUNC = std::hash<TYPE_K>>
class hash_table_mt_chain {

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

    std::atomic<size_t> _size{0}; // node数量, 原子保证自加符++线程安全

    // 数组 of buckets, 每个 bucket 是链表头, 每个链表是哈希冲突的 nodes
    // std::vector<HashTableNode*> _table;
    // 节点指针vector太慢了. 若初始化大容量（比如亿级）表时，vector.resize()会非常耗时. 采用原始指针数组
    HashTableNode** _table = nullptr; // 指向 节点指针 的指针, 以代表 节点指针数组（头）

    // 分配容量为 n 的节点指针数组 到数组头 _table
    void alloc_table_ptrs(size_t n) {
        if (n == 0) {
            _table = nullptr;
            return;
        }
        // calloc: 分配空间为 n 个 节点指针 的空间, 并零初始化, 避免 .resize的逐元素置空
        _table = static_cast<HashTableNode**>(std::calloc(n, sizeof(HashTableNode*)));
        if (!_table) throw std::bad_alloc();
    }

    // 释放节点指针数组，相当于 vector.clear(). 但节点内存并没有释放，由mempool管理
    void free_table_ptrs() noexcept {
        std::free(_table);
        _table = nullptr;
    }

    // 指针传入内存池（模板方式传入。用纯虚类接口的方式传入过不了编译器）
    TYPE_MEMPOOL* _pool; // void* allocate(size_t size)

    // 成员遍历哈希器, 定义了 operator() 即可供函数式调用 _hasher(key)
    HASH_FUNC _hasher;

    // 使用标准库的 hash 函数, 对 TYPE_K 类型的输入 key, 作hash算法, 返回值
    size_t hash(const TYPE_K& key) const {
        return _hasher(key);
    }

    // 锁整张表的锁. rehash/clear等对整张表进行操作时, 独占该锁, 使得其他任何线程不能对table进行任何操作
    mutable std::shared_mutex _table_mutex;
    // 用 mutable 修饰: 只读迭代器传入 const hash_table 时, hash_table 传入 _table_mutex 可以上锁

    // 锁单个bucket的锁. insert操作时, 独占该锁, 使得其他任何线程不能对bucket进行任何操作
    // mutable std::vector<padded_mutex> _bucket_mutexs;
    // C++标准库认为 mutex 的加锁是 写操作, 即使是共享锁 shared_lock 也是改变状态. 所以没办法用 const 修饰 mutex. 所以一般来说
    // std::vector<padded_mutex<std::shared_mutex>> _bucket_mutexs 即可.
    // 但是在 const_iterator 中, 确实不希望只读迭代器修改锁状态, 希望用const修饰mutex. 那么就需要在声明mutex时加 mutable修饰
    // 指明在后续该 mutex

    // 当 capacity 非常大时，给每个 bucket 一个桶锁 的桶锁序列是不可接受的. 构建桶锁非常耗时耗资源. 用固定大小的条带锁解耦桶锁数量和capacity
    // 条带锁 stripe 会增大锁竞争的概率（多个bucket同时等待一个锁的概率存在）
    mutable std::vector<padded_mutex> _stripes; // 定长的桶锁序列 _stripes, 所有bucket会映射到其中一个锁
    size_t _stripe_mask;  // 

    // 上锁函数: 根据给定的 桶 index, 定位到唯一的 条带锁 以上锁
    inline std::shared_mutex& bucket_lock(size_t bucket_index) noexcept {
        return _stripes[bucket_index & _stripe_mask].lock;
    }

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
        // 独占表锁之后, 多线程里就只有 一个 rehash 可以运行了, 其他都会被阻塞. 桶锁是多余的

        // 初始化一个新的 table
        HashTableNode** _new_table = nullptr;
        {
            // 可能抛异常
            _new_table = static_cast<HashTableNode**>(std::calloc(new_capacity, sizeof(HashTableNode*)));
            if (!_new_table) throw std::bad_alloc();
        }

        // 初始化 新的 条带锁序列（粗粒度桶锁），长度不变仍然是 next_pow2(stripe_hint) = _stripe_mask + 1
        std::vector<padded_mutex> _new_stripes(_stripe_mask+1);

        // 目前 size 是只增的, 且rehash 一定是扩容. 但为了逻辑闭环, 以及支持未来可能有缩容的rehash, 应该重新统计 node 总个数
        size_t actual_node_count = 0; // 独占表锁下, 此变量线程安全

        for (size_t i = 0; i < _capacity; i++) {

            // 搬迁当前 bucket 时, 也独占该bucket桶锁. 作用到本i次 for-loop 结束. 在加表锁的前提下，桶锁是多余的
            // std::unique_lock<std::shared_mutex> _lock_bucket_for_rehash_(_bucket_mutexs[i].lock);
            std::unique_lock<std::shared_mutex> _lock_(bucket_lock(i)); // 在加表锁的前提下，桶锁是多余的. 未来测试去除

            HashTableNode* current = _table[i]; // 从该bucekt的链表头开始
            while (current) { // 当前node非空
                HashTableNode* next = current->next; // 先取出next node
                size_t new_index = hash(current->key) % new_capacity; // 计算得出新bucket
                // 挂载 current node 到新table的新bucket.
                {
                    // 新的条带锁 对应 锁：上写锁. 由于 表锁的作用，rehash应该只有一个线程可以执行, 故新表 _new_table 是线程安全的
                    // 这里可能新桶锁也是不需要的. 未来测试去除
                    std::unique_lock<std::shared_mutex> _new_lock_(_new_stripes[new_index & _stripe_mask]);
                    current->next = _new_table[new_index]; // 当前node挂载到新bucket链表头
                    _new_table[new_index] = current; // 更新确认新bucket的链表头
                }
                current = next; // 遍历下一个node
                ++actual_node_count; // 每搬迁一个node就计数
            }
            // 旧_table会被舍弃
            _table[i] = nullptr;
        }

        // 所有bucket所有node重新挂载完毕后, 旧指针数组释放置空后，切换 _table/_bucket_mutexs/_capacity/_size
        free_table_ptrs();
        _table = _new_table;

        _stripes = std::move(_new_stripes);
        _capacity = new_capacity;
        _size.store(actual_node_count, std::memory_order_relaxed); // 独占表锁下, 只要变更值就可以了, 不需要考虑其他线程是否全局一致.
    }


public:

    // 重载的哈希表的构造函数. 传入哈希表的哈希器, capacity, 和内存池.
    explicit hash_table_mt_chain(const HASH_FUNC& hasher, size_t capacity, TYPE_MEMPOOL* pool, size_t stripe_hint = 4096):
        _hasher(hasher), // 这里哈希器采用参数传入的实现了 operator()支持函数式调用hasher(key)的结构体
        _capacity(capacity),
        _pool(pool),
        // 桶锁 vector 的 resize 会导致编译不通过. 不要用 _bucket_mutexs.resize(_capacity) 初始化
        _stripe_mask(next_pow2(stripe_hint)-1),
        _stripes(_stripe_mask+1) // 延迟初始化 桶锁 vector, 构建少量(少于_capacity)
    {
        // _table.resize(_capacity, nullptr); // 长度为 _capacity 的 HashTableNode* vector, 全部初始化为nullptr
        alloc_table_ptrs(_capacity); // calloc 零初始化

    }

    // 重载的哈希表的构造函数. 传入哈希表的capacity, 和内存池.
    explicit hash_table_mt_chain(size_t capacity, TYPE_MEMPOOL* pool, size_t stripe_hint = 4096):
        _hasher(), // 这里哈希器采用模板的默认构造 std::hash<TYPE_K>
        _capacity(capacity),
        _pool(pool),
        _stripe_mask(next_pow2(stripe_hint)-1),
        _stripes(_stripe_mask+1) // 延迟初始化 桶锁 vector, 构建少量(少于_capacity)
    {
        // _table.resize(_capacity, nullptr); // 长度为 _capacity 的 HashTableNode* vector, 全部初始化为nullptr
        alloc_table_ptrs(_capacity); // calloc 零初始化
    }

    // 析构函数, 会调用 destroy 方法来释放所有 HashTableNode 中需要显式析构的部分, clear buckets 数组 _table, _size 和 _capacity 置0
    // 但不负责内存释放. 由内存池在外部统一释放
    ~hash_table_mt_chain() {
        destroy();
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

        if (_capacity == 0 || !_table) return false;

        // 计算 bucket index
        size_t index = hash(key) % _capacity;

        // 读取 key-value 时, 允许单个bucket上并发读. 不允许insert操作. 所以共享占用桶锁. 作用到get结束
        std::shared_lock<std::shared_mutex> _lock_from_insert_(bucket_lock(index));

        for (HashTableNode* cur = _table[index]; cur; cur = cur->next) {
            if (cur->key == key) {
                value = cur->value; // 改变 value 地址的值
                return true;
            }
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

        if (_capacity == 0 || !_table) return false;

        // 计算 bucket index
        size_t index = hash(key) % _capacity;
        {
            // 写入 key-value 时, 不允许其他线程对相应bucket有读写操作. 所以独占桶锁. 作用到本桶更新完毕
            std::unique_lock<std::shared_mutex> _lock_from_insert_read_(bucket_lock(index));
            
            // 首先查找 key 是否已经存在. 若 key 存在, 修改原 value 到 新value
            for (HashTableNode* cur = _table[index]; cur; cur = cur->next) {
                if (cur->key == key) {
                    cur->value = value;
                    return true;
                }
            }
            // 如果执行到这里, 说明要么 currrent 是 nullptr, 要么 _table[index] 链表里没有 key
            // 那么就要执行新建节点, 并将新节点放到 _table[index] 这个bucket的头部

            // 在 内存池 上分配新内存给新节点, raw_mem 内存
            void* raw_mem = _pool->allocate(sizeof(HashTableNode));
            if (!raw_mem) return false; // 如果内存分配失败

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

    /*
    * @param key: const TYPE_K&
    * @param updater: void lambda_func(TYPE_V* input_arg)
    * @param initial: const TYPE_V&
    * @return 如果插入或更新成功, 返回true; 如果内存分配失败返回false
    * 
    * 行为:按key查找, 若找到value, 以updater原子更新updater(value); 若未找到, 以default先初始化value, 再原子更新updater(value)
    * get+insert复合操作: 贡献表锁以拒绝表结构变化(禁止rehash/clear/destroy), 独占桶锁以拒绝单桶的竞态读(get)、竞态写(insert/upsert)
    * 允许不同桶并发读写(其他桶的读/写不受影响)
    */
    template <typename Func>
    bool atomic_upsert(const TYPE_K& key, Func&& updater, const TYPE_V& default_val) {

        // 加表锁, 防止表结构变动
        std::shared_lock<std::shared_mutex> _lock_from_rehash_clear_(_table_mutex);

        if (_capacity == 0 || !_table) return false;

        // 基本照搬 insert 逻辑, 除了修改节点value/插入新节点时, 分别使用updater/default_val来更新/插入
        size_t index = hash(key) % _capacity;

        {
            std::unique_lock<std::shared_mutex> _lock_from_insert_read_(bucket_lock(index));

            for (HashTableNode* cur = _table[index]; cur; cur = cur->next) {
                if (cur->key == key) {
                    std::forward<Func>(updater)(cur->value); // 用forward 完美转发 updater
                    return true;
                }
            }

            void* raw_mem = _pool->allocate(sizeof(HashTableNode));
            if (!raw_mem) return false;

            // 插入新节点 (key, default_val)
            HashTableNode* new_node = new(raw_mem) HashTableNode{key, default_val, _table[index]};

            _table[index] = new_node;

            _size.fetch_add(1);
        }

        _lock_from_rehash_clear_.unlock();

        bool expected = false;

        if (_size >= _capacity*_max_load_factor && _rehashing.compare_exchange_strong(expected, true)) {

            std::unique_lock<std::shared_mutex> _lock_table_for_rehash_(_table_mutex);

            if (_size >= _capacity*_max_load_factor) {
                rehash( _capacity*2 );
            }
            _rehashing.store(false);
        }

        return true;
    }

    // 哈希表是构建在传入的 内存池 上的数据结构, 它不应该负责 内存池 的销毁
    // 内存池本身是只可以 整体复用/整体销毁，不可精确销毁单次allocate的内存
    // 哈希表的"清空"：原数据全部析构, 不再可访问, 但其分配的内存不会在这里被销毁. 保持 bucket 结构
    // 由于保持了 bucket 结构 和 内存池, 故 reset 内存池之后, 本哈希表即可重新复用(insert/upsert node)
    void clear() {

        // 清空 hash table时，独占 表锁
        std::unique_lock<std::shared_mutex> _lock_table_for_clear_(_table_mutex);

        // 对于每个 bucket, 作为哈希冲突的 node 的链表头, 循环以显式析构所有node(如果需要)
        for (size_t i = 0; i < _capacity; i++) {

            // 析构本bucket上的nodes, 以及要置空本bucket时, 独占 桶锁. 作用到本i次for-loop结束
            std::unique_lock<std::shared_mutex> _lock_bucket_for_clear_(bucket_lock(i));

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


    // clear 不破坏表结构, 即 bucket 数组仍然存在. destroy 在 clear 基础上, 释放 bucket 数组 _table
    // destroy 之后 哈希表不可复用. 但是所使用过的内存未释放, 等待mempool在外部统一释放
    void destroy() {

        // 独占表锁：destroy 会对表结构发生改动, 故需要独占表锁以排除其余任何操作
        std::unique_lock<std::shared_mutex> _lock_table_for_clear_(_table_mutex);

        for (size_t i = 0; i < _capacity; i++) {

            std::unique_lock<std::shared_mutex> _lock_bucket_for_clear_(bucket_lock(i));

            HashTableNode* curr = _table[i];
            while (curr) {
                HashTableNode* next = curr->next;
                destroy_node(curr);
                curr = next;
            }

            _table[i] = nullptr;
        }

        _size.store(0, std::memory_order_relaxed);

        free_table_ptrs(); // 释放 节点指针数组, 置空 _table

        _capacity = 0; // _capacity 置零
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

        // 迭代器的构造函数
        const_iterator(const hash_table_mt_chain* hash_table, size_t bucket_index, HashTableNode* node)
            :_hash_table(hash_table),
            _bucket_index(bucket_index),
            _node(node)
        {
            _null_node_advance_to_next_valid_bucket(); // 在内部, 目标桶加桶锁
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
                _bucket_index++; // 尝试跳转到next桶. _bucket_index 是线程local, 所以自增是安全的.
                _null_node_advance_to_next_valid_bucket(); // 在内部, 目标桶加桶锁
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

        // 延迟构造
        // std::shared_lock<std::shared_mutex> _table_lock(_table_mutex);
        // 这种是 定义一个 _table_lock变量, 类型为shared_lock<shared_mutex>, 并传入 _table_mutex 即刻构造-->完成加锁
        // std::shared_lock<std::shared_mutex> _table_lock; 只是声明了一个 类型为shared_lock<shared_mutex>的_table_lock变量
        // 会在const_iterator构造函数初始化列表里传入 _table_mutex 调用shared_lock的构造函数-->届时才加锁, 所以本质是延迟构造加锁
        // std::shared_lock<std::shared_mutex> _table_lock;
        
        // 因为迭代器内部, _bucket_index 是在变化的, 故所有桶锁要一并传进来, 然后在迭代器内部根据_bucket_index加锁
        // 哈希表的 桶锁 vector 地址. 不希望迭代器改变它. 但是 加const 修饰会导致编译不通过
        // padded_mutex* _bucket_mutexs;

        // 迭代器内部不加锁逻辑。锁在迭代器外部调用

        void _null_node_advance_to_next_valid_bucket() {
            // null node 跳转到下一个 valid bucket head node. _bucket_index 和 _node 都是线程局部的, 所以它们都安全
            // _node 在跳转的时候, 应该有目标桶锁, 以限制该桶有insert/remove操作
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
            return !(*this == other); // this是本对象指针, *this就是返回本对象
        }

    private:

        hash_table_mt_chain* _hash_table;

        size_t _bucket_index;

        HashTableNode* _node;

        // 延迟构造
        // 会在iterator构造函数初始化列表里传入 _table_mutex 调用unique_lock的构造函数-->届时才加锁, 所以本质是延迟构造加锁
        // std::unique_lock<std::shared_mutex> _table_lock;

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