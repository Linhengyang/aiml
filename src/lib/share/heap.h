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

// 构造方式一: 迭代器传入构造, 即:
// std::priority_queue<T, T_Compare> pq(init_data.begin(), init_data.end(), cmp);
// 复杂度: O(N) for heapify + O(N) for copy, 副作用: init_data原数据被保留 --> 本质是迭代原容器所有元素 拷贝 到堆的底层容器, 再执行 O(N) 的heapify算法

// 构造方式二: 原容器移动传入构造, 即:
// std::priority_queue<T, T_Compare> pq(cmp, std::move(init_data));
// 复杂度: O(N) for heapify + O(1) for move, 副作用: init_data被移动掏空 --> 本质是pq底层容器 O(1) 接管init_data, 再执行 O(N) 的heapify算法

// 以上两种就是最典型的 already-data --heapify--> priority_queue 的方式
// 如果是需要 原始数据容器 in-place 堆化，那么请使用 std::make_heap from <algorithm>