// heap.h
// 动态扩容的堆(数组二叉树)适合用 vector 实现, 全部放在系统内存, 不要使用内存池.
// 堆的性质(只支持同一时刻单元素操作, 复杂度极高logN) 意味着并发之下堆的标准使用方法就是 封装一个全局互斥锁mutex -> threadsafe_heap

// 无论是 大顶堆(maxHeap, 父节点>子节点) 还是 小顶堆(minHeap, 父节点<子节点), 都可以归纳为 父节点优先级 > 子节点优先级, 只不过
// 大顶堆大者优先级高 a < b 时, b 优先级高; 小顶堆小者优先级高 a > b 时, b 优先级高
// --> 优先级比较器 compare(a, b): 满足性质 当 b 优先级更高时返回True


// 二叉堆 --> std::priority_queue from <queue>, 模板参数 std::priority_queue<
// T,                          // 元素类型
// Container = std::vector<T>, // 底层容器class, 必须支持random access即index
// Compare = std::less<T>      // 优先级比较器 Compare(a, b) 满足当 b 优先级高于 a 时 返回True
// >


// 关于 优先级比较器 Compare: 必须是一个类型参数, 必须是一个 类型(必须是函数对象Functor而不是函数指针).
// C++里定义的普通函数只是一个函数指针, 函数指针无法内联, 每次使用都要切实在内存里跳转(即函数调用开销)
// 而函数对象是“把一个函数逻辑包装成类型”: 它的实例可以像函数一样调用:

// T_Comparator cmp; // 创建一个可调用对象
// bool result = cmp(a, b); // 像调用函数一样

// 自定义比较器方式一: 结构体包装+重载()定义Functor, 在构造函数中输入比较器函数体实例. 如果函数体不需要状态, 那么可以省略构造参数默认构造
// struct T_Comparator {
//     bool operator()(const T& a, const T& b){
//         return a.priority < b.priority;
//     }
// }
// std::priority_queue<T, std::vector<T>, T_Comparator> my_heap; // 省略构造参数默认构造


// struct T_Comparator {
//     int mode; // 0 for mode0, 1 for mode1
//     bool operator()(const T& a, const T& b){
//         if(mode==0) return a.priority < b.priority;
//         else return a.priority > b.priority;
//     }
// }
// T_Comparator cmp(1); // 实例化得到函数实例
// std::priority_queue<T, std::vector<T>, T_Comparator> my_heap(cmp); // 构造函数里输入比较器实例




// 自定义比较器方式二: decltype推断+lambda定义Functor(语法糖), 在构造函数中要输入比较器函数体实例(lambda函数即可)
// auto cmp = [](const T& a, const T& b) {
//     return a.priority < b.priority;
// }
// std::priority_queue<T, std::vector<T>, decltype(cmp)> my_heap(cmp); // 构造函数里输入比较器实例




// 上述是空堆的定义, 也就是说初始化一个空heap, 后面使用 push 和 top 来维护. N个push插入建堆的复杂度为 O(N*logN)
// 如果是要根据给定的 数据容器 生成一个heap 即 heapify(复杂度O(N)), 那么priority_queue 需要使用不同的 构造函数 来初始化heap:

// 考虑原始数据 std::vector<T> init_data = {...};

// 构造方式一: 迭代器范围(拷贝)构造, 即:
// std::priority_queue<T, T_Compare> pq(init_data.begin(), init_data.end(), cmp);
// 复杂度: O(N) for heapify + O(N) for copy, 副作用: init_data原数据被保留 --> 本质是迭代原容器所有元素 拷贝 到堆的底层容器, 再执行 O(N) 的heapify算法

// 构造方式二: 原容器移动传入构造, 即:
// std::priority_queue<T, T_Compare> pq(std::move(init_data), cmp);
// 复杂度: O(N) for heapify + O(1) for move, 副作用: init_data被移动掏空 --> 本质是pq底层容器 O(1) 接管init_data, 再执行 O(N) 的heapify算法

// 以上两种就是最典型的 already-data --heapify--> priority_queue 的方式
// 如果是需要 原始数据容器 in-place 堆化，那么请使用 std::make_heap from <algorithm>





// 关于堆的泛型: std::priority_queue 针对三个地方作了泛型, 除了上文阐述的 元素T、严格弱序比较器T_comparator, 还有堆的底层容器 container<T>
// 堆要求这个底层容器有类似数组的性质, 即支持 随机访问 --> 支持 偏移相加定位 / index取值 / 位置相减算偏移 等操作, 并且要求对 偏移 diff 有明确的定义
// 典型的底层容器用 vector 即可.   <-- 事实上由于把 堆(动态数组二叉树) 放在系统内存, 所以vector已经是完美的底层容器. 此结论对于其他 八叉堆 等自定义堆类数据结构也成立



// 八叉堆 octanary heap / 8-ary heap
#ifndef HEAP_H
#define HEAP_H


#include <optional>
#include <queue>


template<typename TYPE_NODE, typename NODE_COMPARATOR>
class octanary_heap {

private:

    std::vector<TYPE_NODE> _container;
    /*
    TODO
    */

public:

    // 初始化一个空堆, 等待 push 入 node
    explicit octanary_heap(const NODE_COMPARATOR& compare)
    {
        //TODO
    }

    // 迭代器范围构造, 拷贝解引用结果 到堆的底层容器, 再执行 O(N) 的heapify算法
    template <typename InputIterator>
    octanary_heap(InputIterator first, InputIterator last, const NODE_COMPARATOR& compare)
    {
        //TODO
    }

    // 按值传参: 实参可以是右值, 那么形参会以移动构造的方式在函数边界生成, 然后在函数初始化列表及内部继续以std::move(形参)的方式移动(此后形参不再可用); 实参可以是左值, 那么形参会以拷贝构造的方式在函数边界生成
    // 右值引用传参: 实参必须是右值, 形参并不存在(不会在函数边界真实生成对象, 只作为一个引用), 在函数初始化列表以及内部以 std::move(形参) 的方式移动(因为一旦具名, 它就成了左值)
    // 模板+完美转发: 本质是按值传参的极致优化版(形参被直接优化掉了, 不生成对象, 只是引用), 在函数体内部以 std::forward<U>(形参) 的方式保持实参的属性，透传给底层. 实参即能是左值，也可以是右值
    // 左值引用: 实参必须是左值, 需要修改实参
    // 常引用: 实参可以是左值(此时形参是左值的常引用, 零拷贝只读观察, 不修改不拿走), 也可以是右值(此时形参是临时值的常引用, 延长临时值的生命周期至本full-expression结束). 并且由于拷贝构造/赋值函数的签名一律为常引用, 所以常引用类型的形参在函数内部经常会引发对象的拷贝构造/赋值方法, 从而拷贝储存了实参(无论是左值还是右值)
    
    // 强调明确只接受右值, 强制消耗掉传入的vector容器（sink语义）, 执行O(N)heapify堆化. 具备极致性能
    explicit octanary_heap(std::vector<TYPE_NODE>&& data, const NODE_COMPARATOR& compare):
        _container(std::move(data)), // 这里触发 _data(vector) 的移动构造, 窃取外部实参的所有资源. 这种窃取是O(1)的, 效率极高, 不随data大小和长度改变
    {
        // TODO
    }

    size_t size() const 
    {
        return _container.size();
    }

    bool empty() const noexcept
    {
        return _container.empty();
    }

    // top方法不检查空, 在外部先用 .empty方法检测是否为空, 再使用top方法窥探堆顶
    const TYPE_NODE& top() const
    {
        // 可以加一些防止空的报错选项或assert断言
        assert(!_container.empty() && "top() called on empty heap");

        return _container.front();
    }

    // 将一个 new node 推入 堆底, 然后上浮至合适位置. 实参可以是左值, 也可以是右值. 左值实参会拷贝构造形参new_node, 移动右值的实参会移动构造形参new_node, 更高效
    void push(TYPE_NODE new_node)
    {
        _container.emplace_back(std::move(new_node));
        // TODO
    }

    // 函数返回, 站在避免拷贝的角度, 移动+按值返回已经能满足100%场景
    // 返回右值引用 / 返回指针: 都是基本不考虑的选项, 前者只有在一些底层库还有丁点黑魔法用处, 后者基本用于可空返回/多态/兼容C等小众地方
    // 返回左值引用: 是返回一个长生命周期(不随函数调用而结束)的对象并允许修改
    // 返回常左值引用: 为了提供一个零拷贝的只读长效变量对象(比如成员变量)的途径
    // 按值返回: 返回类型 T, 是最核心的返回范式, 且C++11之后的编译器能通过 二级优化方式完全实现 零拷贝
    //   编译器首先尝试的是一级优化: NRVO & RVO
    //   RVO: 对于纯右值结果返回, 直接构造在外部承接对象的内存地址上
    //   NRVO: 对具名变量结果返回, 编译器会直接把外部承接变量 和 要返回的具名变量 直接别名处理(重定向), 使得具名变量在被构建时(无论是拷贝还是移动), 就直接构造在了外部承接变量的地址上, 从而直接免去了return开销
    //         NRVO的触发条件非常严格, 即直接写变量名 return XXX; 这样, 不能有什么std::move修饰， 抑或是过于复杂的条件判断使得编译器无法判断. 出现这些情况后, 编译器就会放弃NRVO, 走二级优化隐式移动
    //   编译器如果NRVO/RVO失败, 则会使用二级优化: 隐式移动返回右值, 从而触发外部承接对象的移动构造
    
    // 综上: 若希望零拷贝, 返回类型写成 按值返回 即可, 重点是函数内部要用 移动语义等 尽量实现零拷贝, 返回这里编译器几乎能处理一切.

    // pop出堆顶. 如果堆为空, 则raise error. 这个方法没有返回 std::optional<TYPE_NODE> 好.
    /*
    TYPE_NODE pop()
    {
        if (_container.empty()) {
            raise std::runtime_error("Empty Heap");
        }
        // TODO, swap first & last
        TYPE_NODE top_node = std::move(_container.back()); // vecotor.back() 返回最后一个元素的引用, 移动语义窃取并掏空它到 top_node, 最后一个元素有效但unspecified
        _container.pop_back(); // 安全析构并删除最后一个valid but unspecified末尾node
        return top_node; // 触发NRVO
    }
    */

    // 堆顶被pop 且存入到 node 里. 如果成功返回 true, 失败则返回 false
    bool pop(TYPE_NODE& node)
    {
        if (_container.empty()) {
            return false;
        }

        // TODO, swap first & last
        node = std::move(_container.back());
        _container.pop_back();
        return true;
    }

    // pop出堆顶. 结果可空. optional容器不影响 按值返回的 NRVO/RVO/隐式移动 优化
    std::optional<TYPE_NODE> pop()
    {
        // 检查是否空堆. 如果是, 返回 nullopt
        if (_container.empty()) {
            return std::nullopt;
        }

        // TODO, swap first & last
        TYPE_NODE top_node = std::move(_container.back());
        _container.pop_back();
        return top_node;
    }

}; // end of octanary_heap definition


#endif