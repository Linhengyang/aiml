// memory_pool.h


// TODO: memory_pool 的 全局单例模式要去掉. 为了支持 hash table 的复用, hash table 所使用的 memory pool 应该是和它自身单一绑定的
// 这样，一个hash table clear 自身时, reset 自身的 memory pool, 才使得这个hash table 可以复用. 不然 reset 会干扰其他所有分配在 memory pool上的对象
