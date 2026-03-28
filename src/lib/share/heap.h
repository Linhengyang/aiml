// heap.h
// 动态扩容的堆(数组二叉树)适合全部放在系统内存, 不要使用内存池.

// 由 class T 构成的 vector(array) binary tree. Method comp 构成优先级函数, 优先级最高的在堆顶
// 当极小堆时, 优先级函数 comp 即 小于函数, 越小优先级越高
// 当极大堆时, 优先级函数 comp 即 大于函数, 越大优先级越高