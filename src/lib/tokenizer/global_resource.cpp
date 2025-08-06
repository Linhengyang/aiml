// global_resource.cpp
#include "global_resource.h"
#include "memory_pool_global.h"

// 初始化为 nullptr
counter<counter_key_type, false>* global_counter_st = nullptr;
counter<counter_key_type, true>* global_counter_mt = nullptr;


void init_global_counter(size_t capacity) {
    global_mempool* pool = &global_mempool::get(4096, 8);

    if (!global_counter_st) {
        global_counter_st = new counter<counter_key_type, false>{capacity, pool};
    }
    if (!global_counter_mt) {
        global_counter_mt = new counter<counter_key_type, true>(capacity, pool);
    }
}