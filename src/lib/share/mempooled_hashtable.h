// mempooled_hashtable.h
// 哈希表的 node* 指针数组适合放在内存池外, 而 nodes 适合放在内存池里
// 这是因为 哈希表涉及到扩容(rehash), 而扩容后旧数组若在内存池里, 则无法回收复用(内存池reset之前). 放在系统内存则可以由系统立即回收复用.

// 内存池上的哈希表由两部分组成: nodes 和 buckets(链表头node指针数组). 其中 nodes 类型是 HashTableNode, 在insert时逐一分配在内存池上
// 而 buckets 类型是指针数组, 分配在内存池之外, 由创建方式分配内存, 即:
// 方法1: HashTable* map = new HashTable(capacity, &mempool); 此时 buckets 分配在 堆内存 上, 由new/delete手动管理哈希表的生命周期
// 方法2: HashTable map(capacity, &mempool); 此时 buckets 分配在 栈内存 上, 由函数调用自动管理哈希表的生命周期
// 这样的好处是 rehash 后原buckets相关空间可以即时被系统回收.

// 推荐方法1, 且将哈希表指针存储在 静态区. 这样可以全程手动控制哈希表的生命周期, 且资源做到最大程度的可复用和即时回收.

#ifndef MEMPOOLED_HASHTABLE_H
#define MEMPOOLED_HASHTABLE_H


#include <functional>
#include <cstddef>
#include <type_traits>
#include <cstring>
#include <vector>
#include <new>
#include <stdexcept>

template <typename TYPE_K, typename TYPE_V, typename TYPE_MEMPOOL, typename HASH_FUNC = std::hash<TYPE_K>>
class pooled_hashtable {

private:

    // 哈希表的 node. node 会根据 hash(key) 进入 bucket, 在 bucket 中形成一个链表
    struct HashTableNode {
        TYPE_K key; // 键
        TYPE_V value; // 值
        HashTableNode* next; // bucket链表，用于解决哈希冲突
        HashTableNode* free_next = nullptr; // free空闲链表，用于将 poped nodes 链接之后再析构, 供给insert/upsert等方法复用地址
        // 提供placement new 构造支持
        
        // RULE of 5

    // RULE 1: 析构函数: ~ClassName(), 负责释放资源(if有资源)
        /*
        HashTableNode 管理对象 TYPE_K key, TYPE_V value, 以及指针 next 和 free_next
        对于一个 HashTableNode 对象实例(节点)而言, 
             1. 它不应该应该去管理 next & free_next 的资源: 它们的用途仅仅是链表寻址而已. 所以 next & free_next 无需析构
             2. 节点存储有 key & value. 
        如果key/value是内存型(值类型/sso/memcpy可复制), 那么key/value将进入sizeof统计，并全部存放在内存池上
             --> 对于这类, 无需析构函数
        如果key/value是资源管理型(句柄/非平凡析构), 那么key/value将仅有控制块(指向资源的内部指针)进入sizeof统计,
        所以只有key/value对象本身(控制块)存放在内存池上, 其真正的数据存放在控制块内部指针指向的系统堆内存上
             --> 对于这类, 在HashTableNode析构时, key 和 value 的析构函数 会被依次调用
        */
        // HashNodeTable 的析构函数 --> 只要 TYPE_K 和 TYPE_V 作为 资源管理类型时，实现了完善的析构, 那么 哈希表节点的析构就无需手动编写

    // RULE 2: 拷贝构造函数: ClassName(const ClassName& other), 负责深拷贝资源(if有资源, 分配新内存, 深拷贝资源)
        /*
        k/v 类型都是 const TYPE&, 不可变动引用, 那么在用 k / v 初始化 key / value 时, 会分别触发 TYPE_K / TYPE_V 的 构造函数
        由于 k / v 都是不可变动引用, 无法移动掏空源对象, 故都会触发 TYPE_K 和 TYPE_V 的拷贝构造 --> 发生 k 和 v 的拷贝
        */
        // HashNodeTable 的拷贝构造函数 --> 只要 TYPE_K 和 TYPE_V 实现了完善的拷贝构造函数, 那么哈希表节点的拷贝构造就是用 const& 类型去触发成员构造即可
        HashTableNode(const TYPE_K& k, const TYPE_V& v, HashTableNode* ptr): key(k), value(v), next(ptr) {}

    // RULE 3: 拷贝赋值运算符: ClassName& operator=(const ClassName& other), 负责自赋值(if有资源, 释放旧资源, 深拷贝新资源)
        /*
        HashTableNode 应该满足 unique 性质: 不应该允许拷贝赋值 搞出两个一模一样的 节点，这没有意义
        */
        // 禁止 哈希表节点 的 拷贝赋值
        HashTableNode& operator=(const HashTableNode& other) = delete;

        // &&符号在 模板函数 中代表 万能引用: 
        //      万能引用&&: 如果实参是左值/常左值, 那函数参数推导为 左值引用/常左值引用; 如果实参是右值(移动/临时), 那函数参数推导为 右值引用
        // 配合 std::forward<TYPE>(arg) 完美转发: 在函数内部保持 std::forward<TYPE>(arg) 为推导出来的类型（引用/常引用/右值引用）

        // &&符号在 非模板函数 中代表 右值引用
    // RULE 4: 移动构造函数: ClassName(ClassName&& other), 负责窃取资源(if有资源, 将other的资源指针复制过来, 置空other的资源指针), 避免拷贝开销
        /*
        对于 资源管理类型(非平凡析构类型), 移动构造是刚需:
        可能1. 用临时资源作为参数去构造对象, 比如 std::string("hello") --> TYPE_K(std::string('hello'))
               当然了, 由于 C++ 允许 const& 参数去 const引用 临时资源(并延长临时资源的生命周期), 所以即使没有 移动构造函数, 临时资源还是可以作为参数触发拷贝构造
        可能2. 用非常巨大的源对象去构造新对象，且不在乎构造后的源对象，比如 std::string s = "超长文本" --> TYPE_K(s);
        HashTableNode在构造时，成员中 key / value 确实都有可能是 巨大复杂对象or临时资源, 所以对于key/value, 支持移动构造是有必要的；对于 节点指针, 浅拷贝即可
        */
        // HashNodeTable 的移动构造函数 --> 只要 TYPE_K 和 TYPE_V 实现了完善的移动构造函数, 那么哈希表节点的移动构造就是 保持右值类型 std::move 去触发成员构造即可
        HashTableNode(TYPE_K&& k, TYPE_V&& v, HashTableNode* ptr): key(std::move(k)), value(std::move(v)), next(ptr) {}
        // 既然 k 已经是 右值引用 类型 TYPE_K&&, 为什么要用 std::move(k) 保持右值？
        // 答: 旧版本 C++ 在函数内部会把带名字的变量(具名变量)视为左值. 新版本引入移动语义后, std::move保持兼容性

        // 哈希表node 在 atomic_upsert 方法里有一个比较特殊的情形需要重载: key实参右值以移动/临时, 而default_value作为重复使用的对象, 必须const&
        HashTableNode(TYPE_K&& k, const TYPE_V& v, HashTableNode* ptr): key(std::move(k)), value(v), next(ptr) {}

    // RULE 5: 移动赋值运算符: ClassName& operator=(ClassName&& other), 负责窃取置换资源(if有资源, 释放旧资源, 窃取新资源)
        /*
        HashTableNode Node 与 Node 之间, 除了索引没有交互的必要, 故禁止 移动赋值. 这没有意义
        */
        HashTableNode& operator=(HashTableNode&& other) = delete;
    };

    // 如果 node 存在非平凡析构对象, 那么对 HashTableNode 显式调用析构
    void destroy_node(HashTableNode* node) {
        if constexpr (!std::is_trivially_destructible<HashTableNode>::value) {
            node->~HashTableNode(); // HashTableNode 没有显式定义 析构函数, 编译器会生成默认的析构函数: 其会逆序递归析构所有需要析构的成员
        }
    }

    size_t _capacity; // 哈希表的容量, bucket数量

    const float _max_load_factor = 0.75f; // 默认最大负载因子. 当 node 数量/_capacity 超过时, 触发扩容

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
        // calloc: 从堆内存 分配 n 个 节点指针 的空间, 并零初始化, 避免 .resize 的逐元素置空
        // 本质是 node节点指针数组
        _table = static_cast<HashTableNode**>(std::calloc(n, sizeof(HashTableNode*)));
        if (!_table) throw std::bad_alloc();
    }

    // 释放节点指针数组(位于堆内存)，相当于 vector.clear(). 但节点内存(位于内存池)并没有释放，由mempool管理
    void free_table_ptrs() noexcept {
        std::free(_table);
        _table = nullptr;
    }

    // 空闲 free 链表: 链起所有 析构后的 poped nodes
    HashTableNode* _free_nodes_head = nullptr;

    // 指针传入内存池（模板方式传入。用纯虚类接口的方式传入过不了编译器）
    TYPE_MEMPOOL* _pool; // void* allocate(size_t size)

    // 成员遍历哈希器, 定义了 operator() 即可供函数式调用 _hasher(key)
    HASH_FUNC _hasher;

    // 使用哈希器, 对 TYPE_K 类型的输入 key, 作hash算法, 返回值类型必须是 size_t 与 _capacity 对齐；入参签名必须是 const TYPE_K&，这样无论实参是左右值都可以接收
    size_t hash(const TYPE_K& key) const {
        return _hasher(key);
    }

    /*
    * 扩容 rehash
    * @param new_capacity
    * 
    * 行为:对每一个node重新计算bucket, 然后将其重新挂载到新的bucket链表的头部. rehash并没有改变 HashTable node 的存储位置, 仍然在 mempool 上
    * 只是 node 的指针指向改变, 以及存储 链表头node 指针的 指针数组改变
    * 链表头node指针数组就是哈希表本身, 它不是在 mempool 上分配, 而是由机器 std::calloc 从系统堆内存分配, 由 new/delete 语句管理生命周期.
    */
    void rehash(size_t new_capacity) {
        // 初始化一个新的 table
        HashTableNode** _new_table = static_cast<HashTableNode**>(std::calloc(new_capacity, sizeof(HashTableNode*)));
        if (!_new_table) throw std::bad_alloc();

        // 重新计算 _size, 为缩容式 rehash 留下余地. 不过缩容式rehash必要性不大: 避免频繁rehash
        size_t actual_node_count = 0;

        // for (size_t old_index: _occupied_indices) // 遍历非空桶
        for (size_t old_index = 0; old_index < _capacity; ++old_index) // 遍历所有桶
        {
            HashTableNode* curr = _table[old_index];
            while (curr) { // 当前node非空
                HashTableNode* next = curr->next; // 先取出next node
                size_t new_index = hash(curr->key) % new_capacity; // 计算得出新bucket

                // 头插到新bucket
                curr->next = _new_table[new_index]; // 当前node挂载到新bucket链表头
                _new_table[new_index] = curr; // 更新确认新bucket的链表头

                // _free_nodes_head 不需要变动

                // 更新计数
                ++actual_node_count;

                curr = next; // 遍历下一个node
            }
        }

        // 切换
        std::free(_table);
        _table = _new_table;
        _capacity = new_capacity;
        _size = actual_node_count;
        // _free_nodes_head 不需要变动
    }


public:

    explicit pooled_hashtable(const HASH_FUNC& hasher, size_t capacity, TYPE_MEMPOOL* pool):
        _hasher(hasher), // 这里哈希器采用参数传入的实现了 operator()支持函数式调用hasher(key)的结构体
        _capacity(capacity),
        _pool(pool)
    {
        alloc_table_ptrs(_capacity);
    }

    explicit pooled_hashtable(size_t capacity, TYPE_MEMPOOL* pool):
        _hasher(), // 这里哈希器采用模板的默认构造 std::hash<TYPE_K>
        _capacity(capacity),
        _pool(pool)
    {
        alloc_table_ptrs(_capacity);
    }

    // 析构函数, 会调用 destroy 方法来 析构 所有 HashTableNode 中需要显式析构的部分, 释放 buckets数组 _table并置空, _size 和 _capacity 置0
    // 但不负责内存释放. 由内存池在外部统一释放
    ~pooled_hashtable() {
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
    * 语义等同于 std::unordered_map.insert_or_assign: 有则更新，无则插入
    * @param key: 可能是左值/常左值, 此时 key 类型为 TYPE_K&/const TYPE_K&, 源对象不会被掏空; 也可能是右值(临时/移动), 此时 key 类型为 TYPE_K&&, 源对象会被掏空
    * @param value: 同 key
    * key-pair 组成的 HashTableNode 在创建时, 如果 key / value 是右值引用, 那么可以调用 节点的移动构造 来节省拷贝成本
    * ---> 模板函数
    * @return 如果插入或更新成功, 返回true; 如果内存分配失败返回false
    * 
    * 行为: 若 key 已经存在, 则更新对应的 value; 否则新建节点插入. 插入后检查是否需要扩容
    */
    template <typename K, typename V>
    bool insert(K&& key, V&& value) {
        if (_capacity == 0 || !_table) return false;

        // 计算 bucket index. hash 的函数签名是 const TYPE_K&, 兼容key的左值/右值引用情况 且保证 源key 不会被改变
        size_t index = hash(key) % _capacity;

        // 首先查找 key 是否已经存在. 若 key 存在, 修改原 value 到 新value
        for (HashTableNode* cur = _table[index]; cur; cur = cur->next) {
            if (cur->key == key) { // 在函数体内部, key 是具名变量, 所以不管其函数参数签名是 左/右值引用, 具名变量都被视为左值. ==操作符应该接受const&不改变操作数.key安全
                // 如果用value(具名变量作为左值), 会触发TYPE_V的拷贝赋值. 但语义上若value在参数签名处为 右值引用时, 调用本意应该是移动构造
                cur->value = std::forward<V>(value); // 用std::forward完美转发, 保持 value 的右值语义(如果最开始是右值), 得以触发TYPE_V的移动赋值(如果有)
                return true;
            }
        }

        // 如果执行到这里, 说明要么 _table[index] 是 nullptr, 要么 _table[index] 链表里没有 key

        // 那么就要执行新建节点, 并将新节点放到 _table[index] 这个bucket的头部
        HashTableNode* new_node;

        if (_free_nodes_head) { // 首先复用 _free_nodes_head 里的地址, 如果存在
            // 直接复用 _free_nodes_head 地址: 
            new_node = _free_nodes_head; // 这里 new_node 指向的地址已经被析构

            // 更新 _free_nodes_head
            // 尽管 new_node 指向的地址已经析构, 但->是纯粹的偏移操作, 允许执行读取free_next来更新. 当然free_next也是析构后的地址
            _free_nodes_head = std::launder(new_node)->free_next;
            // new_node指向的地址已析构, 有些编译器会警告这种读取"析构后的地址的偏移"
            // 用 launder 告诉编译器: 虽然 new_node 这块对象死了, 但数据还在, 我要读取

            // 在 new_node指向的地址上(已析构), placement new 构造, 并用头插法在构造时直接把该index代表的bucket插入new_node->next
            new(new_node) HashTableNode{std::forward<K>(key), std::forward<V>(value), _table[index]};
            // 完美转发以保持key和value的 左/右 值引用性质, 才能触发对应的 HashTableNode 构造函数(左(常)值引用-->拷贝, 右值引用-->移动)
            // 如果传入的是右值引用，那么源对象会被掏空. 这样调用的本意就是转移资源，所以不介意源被掏空.

        }
        else { // 如果没有 _free_nodes_head 可复用地址
            // 在 内存池 上分配新内存给新节点, raw_mem 内存

            void* raw_mem = _pool->allocate(sizeof(HashTableNode));
            if (!raw_mem) return false; // 如果内存分配失败

            // placement new 构造, 头插法 直接在构造时把 bucket 插入 new_node->next
            new_node = new(raw_mem) HashTableNode{std::forward<K>(key), std::forward<V>(value), _table[index]};
            // 完美转发以保持key和value的 左/右 值引用性质, 才能触发对应的 HashTableNode 构造函数(左(常)值引用-->拷贝, 右值引用-->移动)
        }

        _table[index] = new_node;
    
        // node数量自加1. 原子线程安全
        ++_size;

        // 单线程, 只需检查是否满足负载因子，触发扩容
        if (_size >= _capacity*_max_load_factor) {
            rehash( _capacity*2 ); // 扩容为两倍
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
    template <typename K, typename FUNC> // 类额外的模板参数
    bool atomic_upsert(K&& key, FUNC&& updater, const TYPE_V& default_val) {
        if (_capacity == 0 || !_table) return false;

        // 基本照搬 insert 逻辑, 除了修改节点value/插入新节点时, 分别使用updater/default_val来更新/插入
        size_t index = hash(key) % _capacity;

        // 在该bucket中遍历寻找, 以尝试执行 update 逻辑
        for (HashTableNode* cur = _table[index]; cur; cur = cur->next) {
            if (cur->key == key) {
                std::forward<FUNC>(updater)(cur->value); // 用forward 完美转发 updater(作为临时资源的updater) 成函数类型
                return true;
            }
        }
        
        // 如果执行到这里, 说明要么 _table[index] 是 nullptr, 要么 _table[index] 链表里没有 key, 无法执行 update 逻辑
        // 执行 insert 逻辑. 那么就要执行新建节点, 并将新节点放到 _table[index] 这个bucket的头部
        HashTableNode* new_node;

        if (_free_nodes_head) { // 首先复用 _free_nodes_head 里的地址, 如果存在
            // 直接复用 _free_nodes_head 地址: 
            new_node = _free_nodes_head; // 这里 new_node 指向的地址已经被析构

            // 更新 _free_nodes_head
            // 尽管 new_node 指向的地址已经析构, 但->是纯粹的偏移操作, 允许执行读取free_next来更新. 当然free_next也是析构后的地址
            _free_nodes_head = std::launder(new_node)->free_next;
            //new_node指向的地址已析构, 有些编译器会警告这种读取"析构后的地址的偏移"
            // 用 launder 告诉编译器: 虽然 new_node 这块对象死了, 但数据还在, 我要读取

            // 在 new_node指向的地址上(已析构), placement new 构造, 并用头插法在构造时直接把该index代表的bucket插入new_node->next
            new(new_node) HashTableNode{std::forward<K>(key), default_val, _table[index]};

        }
        else { // 如果没有 _free_nodes_head 可复用地址
            // 在 内存池 上分配新内存给新节点, raw_mem 内存

            void* raw_mem = _pool->allocate(sizeof(HashTableNode));
            if (!raw_mem) return false; // 如果内存分配失败

            // placement new 构造, 头插法 直接在构造时把 bucket 插入 new_node->next
            new_node = new(raw_mem) HashTableNode{std::forward<K>(key), default_val, _table[index]};
        }

        _table[index] = new_node;
        
        // node数量自加1. 原子线程安全
        _size++;

        if (_size >= _capacity*_max_load_factor) {
            rehash( _capacity*2 );
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
        if (_capacity == 0 || !_table) return false;

        // 计算 bucket index
        size_t index = hash(key) % _capacity;

        // 遍历查询 key
        
        // 若 key-hash 不存在, 直接返回 false 结束
        HashTableNode* head = _table[index];
        if (!head) return false;

        // 若 key-hash 存在, 遍历该链表以查询 key
        HashTableNode* parent = head; // 为了"删除"节点, 需要跟随保留父节点指针

        while (head) {
            if (head->key == key) {
                // 获取 value
                value = head->value;

                // 摘除 node
                if (!parent) { // parent为空, 说明头节点head就是待删除节点
                    _table[index] = nullptr; // 直接置空指针摘除head
                }
                else { // 如果 parent 不为空, 说明待删节点head不是头节点
                    parent->next = head->next; // parent 一定不是空指针: next重挂, 从而 head 从链表中脱离
                }

                // 防御性编程 置空 head 的 next以防止非法访问
                head->next = nullptr;

                // node数量自减1. 原子线程安全
                --_size;

                // 把 head 挂到 free_list 上, 然后析构 head. 这样该地址可被 insert/upsert 等插入方法复用
                head->free_next = _free_nodes_head;
                // _free_nodes_head 是 Node* 类链表头, 其自身以及其->free_next 指向的是析构后的nodes的地址们.
                _free_nodes_head = head;
                destroy_node(head); // _free_nodes_head 指向的地址被析构了

                return true;
            }
            parent = head;
            head = head->next;
        }
        
        // 如果执行到这里, 说明从 head 遍历到 nullptr 都没能查找到 key. 那么这个是不应该的: key-hash在此index
        throw std::runtime_error("Error in hashtable pop");
    }

    /*
    * 哈希表自身作为 链表头node指针数组 构建在 堆内存, node全部构建在内存池 
    * 哈希表不应该负责 内存池 的 复位or销毁
    * 内存池本身是只可以 整体复用/整体销毁，不可精确销毁单次allocate的内存
    * 哈希表的"清空"：内存池上的node全部析构, 不再可访问, 但其分配的内存不会在这里被 复位or销毁. 链表头node指针数组全部置空
    * 由于保持了 bucket结构(堆内存上的链表头node指针数组, 全部是nullptr) 和 内存池, 故 reset 内存池之后, 本哈希表即可重新复用(insert/upsert node)
    */
    void clear() {
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

            _table[index] = nullptr; // _table指针数组(buckets)保持结构.
        }

        _free_nodes_head = nullptr; // 全表clear时置空 空闲链表, 等待 reset 内存池全表复用而不是node地址复用
        _size = 0;

    }

    // clear 不破坏表结构, 即 bucket 数组仍然存在. destroy 在 clear 基础上, 释放 bucket 数组 _table, _capacity置0
    // destroy 之后 哈希表不可复用. 但是所使用过的内存未释放, 等待mempool在外部统一释放
    void destroy() {
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

        // 释放 节点指针(桶)数组, 置空 _table. 所有桶/节点不再可访问. 但内存尚未释放, 等待内存池操作
        free_table_ptrs();

        _size = 0; // node数量置0
        _capacity = 0; // _capacity 置零
        _free_nodes_head = nullptr; // 空闲链表置空
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

        const_iterator(const pooled_hashtable* hash_table, size_t bucket_index, HashTableNode* node)
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

        const pooled_hashtable* _hash_table;

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
        iterator(pooled_hashtable* hash_table, size_t bucket_index, HashTableNode* node)
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
        pooled_hashtable* _hash_table;

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

    // pooled_hashtable 类对象 hashtable 调用 begin 方法, 返回一个迭代器
    // begin 方法返回的迭代器应该处于 begin 的状态, 即指向 first it
    // .begin 方法返回的是 iterator 对象, 故同一张哈希表, 多次调用会返回不同的 iterator 对象.
    iterator begin() {
        return iterator(this, 0, nullptr);
    }

    // pooled_hashtable 类对象 hashtable 调用 end 方法, 返回一个迭代器
    // end 方法返回的迭代器应该处于 end 的临界状态, 即刚结束迭代的 状态
    iterator end() {
        return iterator(this, _capacity, nullptr);
    }

}; // end of pooled_hashtable definition




#endif