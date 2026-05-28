// iterators.h
// 为各种数据结构提供的各种性质的遍历器. 以友元的方式 插入 各数据结构 使用

// drain_iterator: 移动语义下的 哈希表 迭代器, 把所有 节点node 的 key-value 都用移动的方式转移出去
// 即 auto&& [k, v] = *it; // 外部 k 和 v 移动承接 迭代器解引用返回的key-value资源

std::pair<K, V> operator*() {
    return {std::move(k), std::move(v)};
    或
    return std::make_pair(std::move(k), std::move(v));
}

// 设计1: InputIterator, 阅后即焚类型, 只迭代一次, 且迭代中只能读取一次 ---> 用一个 哈希表的成员函数, 来封装迭代的过程, 不要把迭代器暴露出来
// for (auto&& [k, v]: map.drain()) { // 已经从 drain_iterator 中转移出 kv
//     code using std::move(k) & std::move(v) to keep them in move // kv已经是具名变量, 所以要用std::move去保持移动语义来触发移动构造/赋值
// }

// 设计2: InputIterator, 尽管也是只能迭代一次, 但是要把迭代器暴露出来, *和++解耦m 迭代中允许解引用多次(幂等) ---> 用一个缓存, 承接*的结果
// for (auto it = map.drain_begin(); it != map.drain_end(); ++it) {
//     auto&& [k, v] = *it;
//     code using std::move(k) & std::move(v) to keep them in move 
// }