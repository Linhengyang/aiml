// mempooled_hashtable_iterators.inl
// 为 mempooled_hashtable 提供各种性质的遍历器

#pragma once
// ================= 嵌套类实现区 =================


// drain_iterator: 移动语义下的 哈希表 迭代器, 把所有 节点node 的 key-value 都用移动的方式转移出去, 即:
// std::pair<K, V> operator*() {
//     return {std::move(k), std::move(v)};
//     或
//     return std::make_pair(std::move(k), std::move(v));
// }
// 在外部使用C++17结构化绑定 auto&& [k, v] = *it; // 外部 k 和 v 移动承接 迭代器解引用返回的key-value资源.


// drain_iterator 必然是 InputIterator(阅后即焚类型, 只迭代一次). 不过针对要不要暴露迭代器, 有两种设计:

// 设计1: 不要把迭代器暴露出来, 用一个 哈希表的成员函数, 来封装迭代的过程
// for (auto&& [k, v]: map.drain()) { // drain方法返回一个 drain range, 内部定义好 begin 和 end, 就能在 for : 语句中自动*解引用和++移动
//     code using std::move(k) & std::move(v) to keep them in move // kv已经是具名变量, 所以要用std::move去保持移动语义来触发移动构造/赋值
// }

// 设计2: 把迭代器暴露出来, *和++解耦 迭代中允许解引用多次(幂等), 用一个缓存, 承接*的结果
// for (auto it = map.drain_begin(); it != map.drain_end(); ++it) {
//     auto&& [k, v] = *it;
//     code using std::move(k) & std::move(v) to keep them in move 
// }

// --> 建议设计1. 设计2中, 加了缓存 std::pair<K, V> _cache 之后, 它不再只是个轻量级的引用/指针包装，而是成了一个持有完整K和V的胖对象.
// --> 外部变量承接, 一定使用 auto&& 即 C++17结构化绑定 写法. 编译器会搞定一切.


// 额外的设计
// 1. 返回代理类型drainProxy (本质是值, 但是尽量模拟引用, 且要把潜在的引发深拷贝的操作禁用，强制必须是移动使用这个*it返回的值
//    此外, 代理类型使得使用可以更明确: std::pair<K, V>.first --> drainProxy.key, std::pair<K, V>.second --> drainProxy.value
// 2. 破坏式清空clean_up兜底设计: 采用设计1之后，drainIterator就像哈希表的rehash过程一样, 是破坏性的, 如果遍历中因为某些原因break掉了, 这里也对应这两种设计:
//    设计1: 中途break之后, 哈希表剩下的部分也全部释放清空掉(但内存池reset还是交给内存池来做). 这样在drain的过程中就无需维护size / buckets数组 等哈希表的内部状态
//    ---> drain_iterator析构时要执行 cleaup_remaining
//    设计2: 支持部分node移动转移, 也就是说哈希表剩下的部分仍然保持一个有效完整的哈希表状态. 这样在drain过程中需要细心维护哈希表的所有内部状态, 好处是可以支持条件性node移动
//    但不管怎么样, 都要求 哈希表在 drain遍历之后, 处于 "空但有效, 允许重新insert节点" 的状态
// 3. 把 drainIterator / drainProxy / drainRange 设计成 哈希表 的嵌套类, 但是拆分实现. 给哈希表添加 drain() 成员方法封装使用 drain_iterator
// 4. drain后这些node的内存池地址, 是该进入free_list等待复用, 还是直接free_list也置空, 整个表全部重新置初始态? 
//    --> ARENA内存池支持reset, 全表置初始态是更好的选择. 避免free_list膨胀, 新插入的node排列也更紧凑






/*
* 只读迭代器
* 
* 用法: 单一线程下 for(auto it = hash_table.cbegin(); it != hash_table.cend(); ++it) {auto [k, v] = *it; <...code...>}
*/

// template <...> 这一行是 C++ 语法硬性规定的，无法省略

// 构造函数. 默认模板参数（typename HASH_FUNC = std::hash<TYPE_K>）只能在模板的第一次声明中出现一次（通常是在 .h 文件的类定义处）。
template <typename TYPE_K, typename TYPE_V, typename TYPE_MEMPOOL, typename HASH_FUNC>
pooled_hashtable<TYPE_K, TYPE_V, TYPE_MEMPOOL, HASH_FUNC>::const_iterator::const_iterator(const pooled_hashtable* hash_table, size_t bucket_index, HashTableNode* node)
        :_hash_table(hash_table),
        _bucket_index(bucket_index),
        _node(node)
{
    _null_node_advance_to_next_valid_bucket();
}


// *it 迭代器对象解引用 --> 只读返回
template <typename TYPE_K, typename TYPE_V, typename TYPE_MEMPOOL, typename HASH_FUNC>
std::pair<const TYPE_K&, const TYPE_V&> pooled_hashtable<TYPE_K, TYPE_V, TYPE_MEMPOOL, HASH_FUNC>::const_iterator::operator*() const
{
    // 返回 pair(key, value)临时对象
    return {_node->key, _node->value};
}


// ++it 迭代器对象自增后返回自身引用. 使用尾置返回类型
template <typename TYPE_K, typename TYPE_V, typename TYPE_MEMPOOL, typename HASH_FUNC>
auto pooled_hashtable<TYPE_K, TYPE_V, TYPE_MEMPOOL, HASH_FUNC>::const_iterator::operator++()
    -> const_iterator&
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
auto pooled_hashtable<TYPE_K, TYPE_V, TYPE_MEMPOOL, HASH_FUNC>::const_iterator::operator++(int)
    -> const_iterator
{
    const_iterator tmp = *this;
    ++(*this);
    return tmp;
}


// 返回类型（第一个 pooled_hashtable<...>::const_iterator）：此时编译器还没有进入 pooled_hashtable 或 const_iterator 的作用域（因为它在 :: 之前）。所以必须使用完全限定名
// 参数列表（const const_iterator& other）：此时编译器已经进入了 const_iterator 的作用域（在 :: 之后）。在类作用域内，可以直接使用类名，所以不需要加前缀
// 迭代器的 == 相等判断 用于是否结束状态
template <typename TYPE_K, typename TYPE_V, typename TYPE_MEMPOOL, typename HASH_FUNC>
bool pooled_hashtable<TYPE_K, TYPE_V, TYPE_MEMPOOL, HASH_FUNC>::const_iterator::operator==(const const_iterator& other) const
{
    return _node == other._node && _hash_table == other._hash_table;
}



// 迭代器的 != 不等判断
template <typename TYPE_K, typename TYPE_V, typename TYPE_MEMPOOL, typename HASH_FUNC>
bool pooled_hashtable<TYPE_K, TYPE_V, TYPE_MEMPOOL, HASH_FUNC>::const_iterator::operator!=(const const_iterator& other) const
{
    return !(*this == other);
}


// 迭代器的关键私有函数: 找到下一个(第一个)有效node
template <typename TYPE_K, typename TYPE_V, typename TYPE_MEMPOOL, typename HASH_FUNC>
void pooled_hashtable<TYPE_K, TYPE_V, TYPE_MEMPOOL, HASH_FUNC>::const_iterator::_null_node_advance_to_next_valid_bucket()
{
    while (!_node && _bucket_index < _hash_table->_capacity) {
        _node = (_hash_table->_table)[_bucket_index];
        if (_node) break;
        _bucket_index++;
    }
}




/*
* 迭代器: iterator类 本质是对 "迭代产出对象" it 的引用, it 是 iterator 缩写. value可修改
* 
* 用法: for(auto it = iterator.begin(); it != iterator.end(); ++it)

一个迭代器类经过 begin 构造为迭代器对象 it 之后, it 就一直是该迭代器的引用, 迭代器内部不同的状态引向不同it结果
哈希表迭代器, 输出 k-v. 对 it 解引用 *it 即得到想要的输出. 迭代器的构造, 应该满足能准确表达构造 begin 状态, 和 end 状态. 中间线性迁移交给 ++ 操作
*/


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
template <typename TYPE_K, typename TYPE_V, typename TYPE_MEMPOOL, typename HASH_FUNC>
pooled_hashtable<TYPE_K, TYPE_V, TYPE_MEMPOOL, HASH_FUNC>::iterator::iterator(pooled_hashtable* hash_table, size_t bucket_index, HashTableNode* node)
    :_hash_table(hash_table),
    _bucket_index(bucket_index),
    _node(node)
{
    _null_node_advance_to_next_valid_bucket();
}


// 对迭代器的解引用 *it --> 返回 k-v pair. 注意这里返回的是代理类型, 在外面不能引用接收, 即 pair& p = *it 是非法的
// 只能 pair p = *it; 这样 p 是两个引用组成的 pair, 或 auto&& [k, v] = *it; C++17的万能引用(结构化绑定)
// 这样设计下来, 返回类型是个代理类型: 即本质是个值, 但试图是引用. 所以在外部只能用值作为承接变量
template <typename TYPE_K, typename TYPE_V, typename TYPE_MEMPOOL, typename HASH_FUNC>
std::pair<const TYPE_K&, TYPE_V&> pooled_hashtable<TYPE_K, TYPE_V, TYPE_MEMPOOL, HASH_FUNC>::iterator::operator*() const
{
    // 返回 pair(key, value)临时对象
    return {_node->key, _node->value};
}


// C++/C 风格: 前置自增: 返回改变后的对象自身(引用)；后置自增：对象改变后，返回原值副本

// 对迭代器的前置自增（自增自身, 返回自增后新值引用） ++it --> 下一个状态的迭代器
template <typename TYPE_K, typename TYPE_V, typename TYPE_MEMPOOL, typename HASH_FUNC>
auto pooled_hashtable<TYPE_K, TYPE_V, TYPE_MEMPOOL, HASH_FUNC>::iterator::operator++()
    -> iterator&
{
    if (_node) {
        _node = _node->next; // 如果当前 _node 仍然在某链表里, move to next
    }
    // 如果 _node 为空, 不论是next为空, 还是本来就空, 说明当前桶已经遍历完了
    if (!_node) {
        _bucket_index++;
        _null_node_advance_to_next_valid_bucket();
    }
    return *this;
}


// 对迭代器的后置自增（自增自身, 返回自增前原值副本） it++ --> 下一个状态的迭代器
template <typename TYPE_K, typename TYPE_V, typename TYPE_MEMPOOL, typename HASH_FUNC>
auto pooled_hashtable<TYPE_K, TYPE_V, TYPE_MEMPOOL, HASH_FUNC>::iterator::operator++(int)
    -> iterator
{
    iterator tmp = *this;
    ++(*this);
    // 返回原值副本
    return tmp;
}


// 给出两个迭代器状态是否相等的判决方法: 稳态下判断 _node 就够了, 因为节点已经蕴含了桶信息
template <typename TYPE_K, typename TYPE_V, typename TYPE_MEMPOOL, typename HASH_FUNC>
bool pooled_hashtable<TYPE_K, TYPE_V, TYPE_MEMPOOL, HASH_FUNC>::iterator::operator==(const iterator& other) const {
    return _node == other._node && _hash_table == other._hash_table;
}


// 给出两个迭代器状态是否不相等的判决方法, 必须是 operator == 操作的反面
template <typename TYPE_K, typename TYPE_V, typename TYPE_MEMPOOL, typename HASH_FUNC>
bool pooled_hashtable<TYPE_K, TYPE_V, TYPE_MEMPOOL, HASH_FUNC>::iterator::operator!=(const iterator& other) const {
    return !(*this == other);
}


// 当 _node 沿着 _bucket 链表移动到 nullptr, 亦或是初始化为 nullptr, 需要"跳步"到next valid bucket链表头

// 此跳步操作, 只在 _node 为空时才会执行
// 执行结果1: _node 跳转到 next valid bucket head, _bucket_index 正确为该 valid bucket
// 执行结果2: _node 仍然为空, _bucket_index = hashtable capacity
template <typename TYPE_K, typename TYPE_V, typename TYPE_MEMPOOL, typename HASH_FUNC>
void pooled_hashtable<TYPE_K, TYPE_V, TYPE_MEMPOOL, HASH_FUNC>::iterator::_null_node_advance_to_next_valid_bucket() {
    // 当前 _node 为 nullptr, 且当前 _bucket_index 尚未穷尽
    while (!_node && _bucket_index < _hash_table->_capacity) {
        // 哈希表取出_table内部属性, 再取出当前 bucket 链表头作为 potential next node
        _node = (_hash_table->_table)[_bucket_index];

        if (_node) break; // 如果 _node 不为 nullptr, 说明跳步 bucket 成功了, break

        // 如果 _node 仍然是 null, 说明 _bucket_index 对应桶是空的. 尝试下一个桶
        _bucket_index++;
    }
}