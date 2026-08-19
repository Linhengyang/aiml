// mempooled_hashtable_iterators.inl
// 为 mempooled_hashtable 提供各种性质的遍历器

#pragma once
// ================= 嵌套类实现区 =================


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



/*
* 不加锁、线程不安全的 只读迭代器
*/
template <typename TYPE_K, typename TYPE_V, typename TYPE_MEMPOOL, typename HASH_FUNC>
pooled_concurrent_hashtable<TYPE_K, TYPE_V, TYPE_MEMPOOL, HASH_FUNC>::unsafe_const_iterator::unsafe_const_iterator(const pooled_concurrent_hashtable* hash_table, size_t bucket_index, HashTableNode* node)
        :_hash_table(hash_table),
        _bucket_index(bucket_index),
        _node(node)
{
    _null_node_advance_to_next_valid_bucket();
}


// *it 迭代器对象解引用 --> 只读返回
template <typename TYPE_K, typename TYPE_V, typename TYPE_MEMPOOL, typename HASH_FUNC>
auto pooled_concurrent_hashtable<TYPE_K, TYPE_V, TYPE_MEMPOOL, HASH_FUNC>::unsafe_const_iterator::operator*() const
    -> ConstProxy
{
    // 返回 ConstProxy(key, value)临时对象: 是一个代理类型
    return ConstProxy{_node->key, _node->value};
}


// ++it 迭代器对象自增后返回自身引用. 使用尾置返回类型
template <typename TYPE_K, typename TYPE_V, typename TYPE_MEMPOOL, typename HASH_FUNC>
auto pooled_concurrent_hashtable<TYPE_K, TYPE_V, TYPE_MEMPOOL, HASH_FUNC>::unsafe_const_iterator::operator++()
    -> unsafe_const_iterator&
{
    if (_node) {
        _node = _node->next;
    }
    if (!_node) {
        _bucket_index++;
        _null_node_advance_to_next_valid_bucket();
    }
    return *this;
}


// it++ 迭代器对象自增后, 返回自增前的自身拷贝. 使用尾置返回类型
template <typename TYPE_K, typename TYPE_V, typename TYPE_MEMPOOL, typename HASH_FUNC>
auto pooled_concurrent_hashtable<TYPE_K, TYPE_V, TYPE_MEMPOOL, HASH_FUNC>::unsafe_const_iterator::operator++(int)
    -> unsafe_const_iterator
{
    unsafe_const_iterator tmp = *this;
    ++(*this);
    return tmp;
}


// 返回类型（第一个 pooled_hashtable<...>::const_iterator）：此时编译器还没有进入 pooled_hashtable 或 const_iterator 的作用域（因为它在 :: 之前）。所以必须使用完全限定名
// 参数列表（const const_iterator& other）：此时编译器已经进入了 const_iterator 的作用域（在 :: 之后）。在类作用域内，可以直接使用类名，所以不需要加前缀
// 迭代器的 == 相等判断 用于是否结束状态
template <typename TYPE_K, typename TYPE_V, typename TYPE_MEMPOOL, typename HASH_FUNC>
bool pooled_concurrent_hashtable<TYPE_K, TYPE_V, TYPE_MEMPOOL, HASH_FUNC>::unsafe_const_iterator::operator==(const unsafe_const_iterator& other) const
{
    return _node == other._node && _hash_table == other._hash_table;
}


// 迭代器的 != 不等判断
template <typename TYPE_K, typename TYPE_V, typename TYPE_MEMPOOL, typename HASH_FUNC>
bool pooled_concurrent_hashtable<TYPE_K, TYPE_V, TYPE_MEMPOOL, HASH_FUNC>::unsafe_const_iterator::operator!=(const unsafe_const_iterator& other) const
{
    return !(*this == other);
}


// 迭代器的关键私有函数: 找到下一个(第一个)有效node
template <typename TYPE_K, typename TYPE_V, typename TYPE_MEMPOOL, typename HASH_FUNC>
void pooled_concurrent_hashtable<TYPE_K, TYPE_V, TYPE_MEMPOOL, HASH_FUNC>::unsafe_const_iterator::_null_node_advance_to_next_valid_bucket()
{
    while (!_node && _bucket_index < _hash_table->_capacity) {
        _node = (_hash_table->_table)[_bucket_index];
        if (_node) break;
        _bucket_index++;
    }
}






/*
* 不加锁、线程不安全的 可变迭代器
*/
template <typename TYPE_K, typename TYPE_V, typename TYPE_MEMPOOL, typename HASH_FUNC>
pooled_concurrent_hashtable<TYPE_K, TYPE_V, TYPE_MEMPOOL, HASH_FUNC>::unsafe_iterator::unsafe_iterator(pooled_concurrent_hashtable* hash_table, size_t bucket_index, HashTableNode* node)
        :_hash_table(hash_table),
        _bucket_index(bucket_index),
        _node(node)
{
    _null_node_advance_to_next_valid_bucket();
}


// *it 迭代器对象解引用 --> 只读返回
template <typename TYPE_K, typename TYPE_V, typename TYPE_MEMPOOL, typename HASH_FUNC>
auto pooled_concurrent_hashtable<TYPE_K, TYPE_V, TYPE_MEMPOOL, HASH_FUNC>::unsafe_iterator::operator*() const
    -> MutableProxy
{
    // 返回 MutableProxy(key, value)临时对象: 是一个代理类型
    return MutableProxy{_node->key, _node->value};
}


// ++it 迭代器对象自增后返回自身引用. 使用尾置返回类型
template <typename TYPE_K, typename TYPE_V, typename TYPE_MEMPOOL, typename HASH_FUNC>
auto pooled_concurrent_hashtable<TYPE_K, TYPE_V, TYPE_MEMPOOL, HASH_FUNC>::unsafe_iterator::operator++()
    -> unsafe_iterator&
{
    if (_node) {
        _node = _node->next;
    }
    if (!_node) {
        _bucket_index++;
        _null_node_advance_to_next_valid_bucket();
    }
    return *this;
}


// it++ 迭代器对象自增后, 返回自增前的自身拷贝. 使用尾置返回类型
template <typename TYPE_K, typename TYPE_V, typename TYPE_MEMPOOL, typename HASH_FUNC>
auto pooled_concurrent_hashtable<TYPE_K, TYPE_V, TYPE_MEMPOOL, HASH_FUNC>::unsafe_iterator::operator++(int)
    -> unsafe_iterator
{
    unsafe_iterator tmp = *this;
    ++(*this);
    return tmp;
}


// 返回类型（第一个 pooled_hashtable<...>::const_iterator）：此时编译器还没有进入 pooled_hashtable 或 const_iterator 的作用域（因为它在 :: 之前）。所以必须使用完全限定名
// 参数列表（const const_iterator& other）：此时编译器已经进入了 const_iterator 的作用域（在 :: 之后）。在类作用域内，可以直接使用类名，所以不需要加前缀
// 迭代器的 == 相等判断 用于是否结束状态
template <typename TYPE_K, typename TYPE_V, typename TYPE_MEMPOOL, typename HASH_FUNC>
bool pooled_concurrent_hashtable<TYPE_K, TYPE_V, TYPE_MEMPOOL, HASH_FUNC>::unsafe_iterator::operator==(const unsafe_iterator& other) const
{
    return _node == other._node && _hash_table == other._hash_table;
}


// 迭代器的 != 不等判断
template <typename TYPE_K, typename TYPE_V, typename TYPE_MEMPOOL, typename HASH_FUNC>
bool pooled_concurrent_hashtable<TYPE_K, TYPE_V, TYPE_MEMPOOL, HASH_FUNC>::unsafe_iterator::operator!=(const unsafe_iterator& other) const
{
    return !(*this == other);
}


// 迭代器的关键私有函数: 找到下一个(第一个)有效node
template <typename TYPE_K, typename TYPE_V, typename TYPE_MEMPOOL, typename HASH_FUNC>
void pooled_concurrent_hashtable<TYPE_K, TYPE_V, TYPE_MEMPOOL, HASH_FUNC>::unsafe_iterator::_null_node_advance_to_next_valid_bucket()
{
    while (!_node && _bucket_index < _hash_table->_capacity) {
        _node = (_hash_table->_table)[_bucket_index];
        if (_node) break;
        _bucket_index++;
    }
}


template <typename TYPE_K, typename TYPE_V, typename TYPE_MEMPOOL, typename HASH_FUNC>
std::vector<TYPE_K> pooled_concurrent_hashtable<TYPE_K, TYPE_V, TYPE_MEMPOOL, HASH_FUNC>::get_readonly_keys() const
{
    std::vector<TYPE_K> keys_snapshot;
    {
        // 上 表读锁: 要排除 rehash & clear 等需要独占(写锁)表锁的行为
        std::shared_lock<std::shared_mutex> _lock_table_from_rehash_clear_(_table_mutex);
        keys_snapshot.reserve( size() ); // 预设大小

        // 遍历所有 bucket
        for (size_t i = 0; i < _capacity; ++i) {
            // 上 桶(条带)读锁: 要排除 insert/atomic_upsert/pop 等需要独占(写锁)桶锁的行为
            std::shared_lock<std::shared_mutex> _lock_from_insert_(bucket_lock(i));

            for (HashTableNode* node = _table[i]; node; node = node->next) {
                keys_snapshot.push_back(node->key);
            }
        }
        // 该单次循环结束时 释放对应的共享桶(条带)锁
    }
    // 释放共享表锁
    return keys_snapshot;
}