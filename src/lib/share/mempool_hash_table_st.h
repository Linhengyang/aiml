// mempool_hash_table_st.h

#ifndef MEMPOOL_HASH_TABLE_SINGLE_THREAD_H
#define MEMPOOL_HASH_TABLE_SINGLE_THREAD_H


#include <functional>
#include <cstddef>
#include <type_traits>
#include <cstring>


template <typename TYPE_K, typename TYPE_V, typename TYPE_MEMPOOL, typename HASH_FUNC = std::hash<TYPE_K>>
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

    // 数组 of buckets, 每个 bucket 是链表的头, 每个链表是哈希冲突的 nodes, 由第一个node代表
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

    // 使用哈希器, 对 TYPE_K 类型的输入 key, 作hash算法, 返回值
    size_t hash(const TYPE_K& key) const {
        return _hasher(key);
    }

    /*
    * 扩容 rehash
    * @param new_capacity
    * 
    * 行为:对每一个node重新计算bucket, 然后将其重新挂载到新的bucket链表的头部
    */
    void rehash(size_t new_capacity) {
        // 初始化一个新的 table
        HashTableNode** _new_table = nullptr;
        {
            // 可能抛异常
            _new_table = static_cast<HashTableNode**>(std::calloc(new_capacity, sizeof(HashTableNode*)));
            if (!_new_table) throw std::bad_alloc();
        }

        // 重新计算 _size, 为缩容式 rehash 留下余地
        size_t actual_node_count = 0;

        for (size_t i = 0; i < _capacity; i++) {
            HashTableNode* current = _table[i]; // 从该bucekt的链表头开始
            while (current) { // 当前node非空
                HashTableNode* next = current->next; // 先取出next node
                size_t new_index = hash(current->key) % new_capacity; // 计算得出新bucket
                current->next = _new_table[new_index]; // 当前node挂载到新bucket链表头
                _new_table[new_index] = current; // 更新确认新bucket的链表头
                current = next; // 遍历下一个node
                ++actual_node_count;
            }
        }
        // 所有bucket所有node重新挂载完毕后, 旧指针数组释放置空后，切换 _table/_capacity
        free_table_ptrs();
        _table = _new_table;

        _capacity = new_capacity;
        _size = actual_node_count; // 逻辑闭环：万一未来支持缩容
    }


public:

    // 重载的哈希表的构造函数. 传入哈希表的哈希器, capacity, 和内存池.
    explicit hash_table_st_chain(const HASH_FUNC& hasher, size_t capacity, TYPE_MEMPOOL* pool):
        _hasher(hasher), // 这里哈希器采用参数传入的实现了 operator()支持函数式调用hasher(key)的结构体
        _capacity(capacity),
        _pool(pool)
    {
        // _table.resize(_capacity, nullptr); // 长度为 _capacity 的 HashTableNode* vector, 全部初始化为nullptr
        alloc_table_ptrs(_capacity); // calloc 零初始化
    }

    // 重载的哈希表的构造函数. 传入哈希表的capacity, 和内存池.
    explicit hash_table_st_chain(size_t capacity, TYPE_MEMPOOL* pool):
        _hasher(), // 这里哈希器采用模板的默认构造 std::hash<TYPE_K>
        _capacity(capacity),
        _pool(pool)
    {
        // _table.resize(_capacity, nullptr); // 长度为 _capacity 的 HashTableNode* vector, 全部初始化为nullptr
        alloc_table_ptrs(_capacity); // calloc 零初始化
    }

    // 析构函数, 会调用 destroy 方法来释放所有 HashTableNode 中需要显式析构的部分, 释放 buckets数组 _table并置空, _size 和 _capacity 置0
    // 但不负责内存释放. 由内存池在外部统一释放
    ~hash_table_st_chain() {
        destroy();
    }

    // 哈希表关键方法之 get(key&, value&) --> change value, return true if success
    bool get(const TYPE_K& key, TYPE_V& value) {
        if (_capacity == 0 || !_table) return false;

        // 计算 bucket index
        size_t index = hash(key) % _capacity;

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
    */
    bool insert(const TYPE_K& key, const TYPE_V& value) {
        if (_capacity == 0 || !_table) return false;

        // 计算 bucket index
        size_t index = hash(key) % _capacity;

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
        ++_size;

        // 单线程, 只需检查是否满足负载因子，触发扩容
        if (_size >= _capacity*_max_load_factor) {
            rehash( _capacity*2 ); // 扩容为两倍
        }

        return true;
    }

    // updater 应该是一个函数指针, 比如 函数指针 std::function<void(TYPE_V&)> 或
    // 函数指针的左值引用 std::function<void(TYPE_V&)>& 或
    // 函数指针的const &引用 const std::function<void(TYPE_V&)>& 这样可以const引用右值(lambda函数)
    // 这里采用最灵活的模板写法, 用 && 保证右值引用，然后在内部用 std::forward<Func>(updater) 替代 updater 来实现完美转发
    template <typename Func>
    bool atomic_upsert(const TYPE_K& key, Func&& updater, const TYPE_V& default_val) {
        if (_capacity == 0 || !_table) return false;

        // 基本照搬 insert 逻辑, 除了修改节点value/插入新节点时, 分别使用updater/default_val来更新/插入
        size_t index = hash(key) % _capacity;

        for (HashTableNode* cur = _table[index]; cur; cur = cur->next) {
            if (cur->key == key) {
                std::forward<Func>(updater)(cur->value); // 用forward 完美转发 updater
                return true;
            }
        }
        
        void* raw_mem = _pool->allocate(sizeof(HashTableNode));
        if (!raw_mem) {
            return false;
        }

        HashTableNode* new_node = new(raw_mem) HashTableNode{key, default_val, _table[index]};

        _table[index] = new_node;

        _size++;

        if (_size >= _capacity*_max_load_factor) {
            rehash( _capacity*2 );
        }

        return true;
    }


    // 哈希表是构建在传入的 内存池 上的数据结构, 它不应该负责 内存池 的销毁
    // 内存池本身是只可以 整体复用/整体销毁，不可精确销毁单次allocate的内存
    // 哈希表的"清空"：原数据全部析构, 不再可访问, 但其分配的内存不会在这里被销毁. 保持 bucket 结构
    // 由于保持了 bucket 结构 和 内存池, 故 reset 内存池之后, 本哈希表即可重新复用(insert/upsert node)
    void clear() {
        if (_capacity == 0 || !_table) {
            _size = 0;
            return;
        }
        if constexpr(!std::is_trivially_destructible<HashTableNode>::value) {
            // 对于每个 bucket, 作为哈希冲突的 node 的链表头, 循环以显式析构所有node(如果需要)
            for (size_t i = 0; i < _capacity; i++) {

                HashTableNode* curr = _table[i];
                if (!curr) continue; // 空桶直接跳过
                while (curr) {
                    HashTableNode* next = curr->next;
                    destroy_node(curr);
                    curr = next;
                }
            }
        }
        // _table 指针数组全部置空
        std::memset(_table, 0, _capacity * sizeof(HashTableNode*));
        // node数量 置0
        _size = 0;

    }

    // clear 不破坏表结构, 即 bucket 数组仍然存在. destroy 在 clear 基础上, 释放 bucket 数组 _table
    // destroy 之后 哈希表不可复用. 但是所使用过的内存未释放, 等待mempool在外部统一释放
    void destroy() {

        for (size_t i = 0; i < _capacity; i++) {

            HashTableNode* curr = _table[i];
            while (curr) {
                HashTableNode* next = curr->next;
                destroy_node(curr);
                curr = next;
            }
        }
        // node数量 置0
        _size = 0;

        free_table_ptrs(); // 释放 节点指针数组, 置空 _table

        _capacity = 0; // _capacity 置零
    }


    // 输出当前哈希表 k-v 数量
    size_t size() const noexcept {
        return _size; // 原子读取
    }

    /*
    * 只读迭代器
    * 
    * 用法: 单一线程下 for(auto it = hash_table.cbegin(); it != hash_table.cend(); ++it) {auto [k, v] = *it; //code//}
    */
    class const_iterator {

    public:

        const_iterator(const hash_table_st_chain* hash_table, size_t bucket_index, HashTableNode* node)
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
        

        const_iterator& operator++() {
            if (_node) {
                _node = _node->next;
            }
            if (!_node) {
                _bucket_index++;
                _null_node_advance_to_next_valid_bucket(); // 
            }
            return *this;
        }


        const_iterator operator++(int) {
            const_iterator tmp = *this;
            ++(*this);
            return tmp;
        }


        bool operator==(const const_iterator& other) const {
            return _node == other._node && _hash_table == other._hash_table;
        }


        bool operator!=(const const_iterator& other) const {
            return !(*this == other);
        }

    private:

        const hash_table_st_chain* _hash_table;

        size_t _bucket_index;

        HashTableNode* _node;

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
    * 迭代器
    * 
    // for(auto it = iterator.begin(); it != iterator.end(); ++it)
    // 上述是迭代器的用法. 迭代器 iterator类 本质是对 "迭代产出对象" it 的引用, it 是 iterator 缩写.
    // 即一个迭代器类经过 begin 构造为迭代器对象 it 之后, it 就一直是该迭代器的引用, 迭代器内部不同的状态引向不同it结果
    // 哈希表迭代器, 输出 k-v. 对 it 解引用 *it 即得到想要的输出
    */
    class iterator {

    // 哈希表的迭代器应该返回所有 node 的 key-value. 所以要遍历所有 buckets 的所有 nodes
    public:

        // 迭代器的构造, 应该满足能准确表达构造 begin 状态, 和 end 状态. 中间线性迁移交给 ++ 操作
        /*
        * @param hash_table: 本哈希表指针
        * @param bucket_index: for begin: 0; for end: 本哈希表的_capacity
        * @param node: for begin: nullptr; for end: nullptr
        * 
        * 上述三个属性决定了本迭代器的状态, 然后决定了不同的迭代产出
        * 行为: begin(this哈希表指针, 0, nullptr)初始化下, 成功自定位到first valid bucket状态
        *       end(this哈希表指针, _capacity, nullptr)下成功定位到 ++ 操作符的临界退出点
        */
        // for begin: _node = nullptr, _bucket_index=0 开始寻找第一个valid bucket
        // for end: _node = nullptr, _bucket_index=_capacity, 正好是迭代结束后的临界点
        iterator(hash_table_st_chain* hash_table, size_t bucket_index, HashTableNode* node)
            :_hash_table(hash_table),
            _bucket_index(bucket_index),
            _node(node)
        {
            _null_node_advance_to_next_valid_bucket();
        }

        // 对迭代器的解引用 *it --> 返回 k-v pair. 注意在外面不能引用接收, 即 pair& p = *it 是非法的
        // 只能 pair p = *it; 这样 p 是两个引用组成的 pair, 或 auto&& [k, v] = *it; C++17的万能引用
        std::pair<const TYPE_K&, TYPE_V&> operator*() const {
            return {_node->key, _node->value}; // 返回 pair(key, value)临时对象
        }
        
        // C++/C 风格: 前置自增: 返回改变后的对象自身(引用)；后置自增：对象改变后，返回原值副本
        // 对迭代器的前置自增（自增自身, 返回自增后新值引用） ++it --> 下一个状态的迭代器
        iterator& operator++() {
            if (_node) {
                _node = _node->next; // 如果当前 _node 仍然在某链表里, move to next
            }
            // 如果 _node 为空, 不论是next为空, 还是本来就空, 说明当前桶已经遍历完了
            if (!_node) {
                _bucket_index++; // move to next bucket
                _null_node_advance_to_next_valid_bucket(); // 
            }
            return *this; // this是本对象指针, *this就是返回本对象
        }

        // 对迭代器的后置自增（自增自身, 返回自增前原值副本） it++ --> 下一个状态的迭代器
        iterator operator++(int) {
            iterator tmp = *this;
            ++(*this);
            return tmp; // 返回原值副本
        }

        // 给出两个迭代器状态是否相等的判决方法: 稳态下判断 _node 就够了, 因为节点已经蕴含了桶信息
        bool operator==(const iterator& other) const {
            return _node == other._node && _hash_table == other._hash_table;
        }

        // 给出两个迭代器状态是否不相等的判决方法, 必须是 operator == 操作的反面
        bool operator!=(const iterator& other) const {
            return !(*this == other); // this是本对象指针, *this就是返回本对象
        }

    private:

        // 迭代器所迭代的容器, 在这里是哈希表. 从这里得到bucket/node等内部结构
        hash_table_st_chain* _hash_table;

        // 遍历哈希表的所有桶, 0 -> _capacity-1
        size_t _bucket_index;

        // 遍历所有桶的所有node
        HashTableNode* _node;

        // 当 _node 沿着 _bucket 链表移动到 nullptr, 亦或是初始化为 nullptr, 需要"跳步"到next valid bucket链表头

        // 此跳步操作, 只在 _node 为空时才会执行
        // 执行结果1: _node 跳转到 next valid bucket head, _bucket_index 正确为该 valid bucket
        // 执行结果2: _node 仍然为空, _bucket_index = hashtable capacity
        void _null_node_advance_to_next_valid_bucket() {
            // 当前 _node 为 nullptr, 且当前 _bucket_index 尚未穷尽
            while (!_node && _bucket_index < _hash_table->_capacity) {
                // 哈希表取出_table内部属性, 再取出当前 bucket 链表头作为 potential next node
                _node = (_hash_table->_table)[_bucket_index];

                if (_node) break; // 如果 _node 不为 nullptr, 说明跳步 bucket 成功了, break

                // 如果 _node 仍然是 null, 说明 _bucket_index 对应桶是空的. 尝试下一个桶
                _bucket_index++;
            }
        }

    };  // end of iterator definition

    // hash_table_st_chain 类对象 hashtable 调用 begin 方法, 返回一个迭代器
    // begin 方法返回的迭代器应该处于 begin 的状态, 即指向 first it
    // .begin 方法返回的是 iterator 对象, 故同一张哈希表, 多次调用会返回不同的 iterator 对象.
    iterator begin() {
        return iterator(this, 0, nullptr);
    }

    // hash_table_st_chain 类对象 hashtable 调用 end 方法, 返回一个迭代器
    // end 方法返回的迭代器应该处于 end 的临界状态, 即刚结束迭代的 状态
    iterator end() {
        return iterator(this, _capacity, nullptr);
    }

}; // end of hash_table_st_chain definition




#endif