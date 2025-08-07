// mempool_counter.h

#ifndef MEMPOOL_COUNTER_H
#define MEMPOOL_COUNTER_H


#include <vector>
#include <functional>
#include <cstddef>
#include <type_traits> // std::conditional
#include <utility> // std::forward
#include "mempool_hash_table_st.h"
#include "mempool_hash_table_mt.h"


template<typename TYPE_K, bool threadsafe, typename TYPE_MEMPOOL>
using selected_hash_table = typename std::conditional<
    threadsafe,
    hash_table_mt_chain<TYPE_K, uint64_t, TYPE_MEMPOOL>,
    hash_table_st_chain<TYPE_K, uint64_t, TYPE_MEMPOOL>
>::type;



template<typename TYPE_K, bool threadsafe, typename TYPE_MEMPOOL>
class counter {

private:

    // 计数器 counter 和 哈希表是 has-a 的关系：组合而非继承
    // counter 内部就是一个 hash table, 其中key是待输入的TYPE_K, value 就是 uint64 以统计非负频次
    using hash_table = selected_hash_table<TYPE_K, threadsafe, TYPE_MEMPOOL>; // 根据 threadsafe 确定使用哪个hashtable

    hash_table _hash_table;

public:

    // counter 的构造函数: 触发内部hashtable的构造函数, 预设bucket数量为capacity
    explicit counter(size_t capacity, TYPE_MEMPOOL* pool): _hash_table(capacity, pool) {}

    // counter 的析构函数: 用隐式析构即可：自动调用成员变量_hash_table的析构函数

    // 函数调用操作符: 支持 counter(key)
    void operator()(const TYPE_K& key) {
        increment(key);
    }

    // 右值引用: counter(const TYPE_K& key) 可能引发拷贝, 当 TYPE_K 是复杂类型时, 移动语义提升性能
    void operator()(TYPE_K&& key) {
        increment(std::forward<TYPE_K>(key)); // 直接转发, 移动语义, 避免拷贝
    }

    /*
    * value 自增 1
    * @param key
    * @return void
    * 
    * 行为: 若 key 已经存在, 则对应的 value 自增 1; 否则新建节点插入, value = 1
    * 两次查找: 由于hash缓存的存在, 每个bucket不会太长(冲突不会严重), 两次查找的开销很低.
    * get时共享表锁+共享桶锁, insert时共享表锁+独占桶锁, 为了线程安全, 使用原子更新/插入的 atomic_upsert方法
    */
    void increment(const TYPE_K& key) {
        // C++ 的lambda表达式: uint64_t& 表示 lambda function 需要个该类型的参数, []表示lambda函数体内没用到其他变量
        // 若[&], 表示lambda函数体内涉及到的其他变量, 以引用的方式传入; 若[=], 表示以值得方式传入
        // C++的lambda表达式: 显示定义一个函数  std::function<void(int&)> f = [](int& x) {x+=1;};
        // 这里 void 是 return type, int& 是input arg type. 这里 f 是一个左值. 如果省去 std::function声明部分, 那就是右值
        // 这里 [](auto& value) { value += 1;} 就是一个右值临时函数, 意思是引用 value 并变化它的值.
        // atomic_upsert 内部: 
        //      std::forward<Func>(updater)(node->value); // 用forward 完美转发 updater
        // 内部 node->value 被 lambda 函数引用, 并修改值.
        _hash_table.atomic_upsert(key, [](auto& value) { value += 1;}, 1);
    }

    // 暴露哈希表的clear方法. 析构哈希表, 但不release/reset哈希表所占的内存池空间, 同时表结构也不会变化
    void clear() {
        _hash_table.clear();
    }

    // 暴露哈希表的size方法
    size_t size() const {
        return _hash_table.size();
    }

    // 暴露哈希表的迭代器以支持迭代输出计数的结果
    auto begin() { return _hash_table.begin(); }
    auto end() { return _hash_table.end(); }

    // 暴露哈希表的只读迭代器以支持迭代输出计数的结果
    auto cbegin() const { return _hash_table.cbegin(); }
    auto cend() const { return _hash_table.cend(); }

}; // end of counter definition



#endif