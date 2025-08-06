// global_resource.cpp
#include "global_resource.h"
#include "memory_pool_global.h"
#include <cstddef>
#include "mempool_counter.h"

// 初始化为 nullptr
counter<counter_key_type, false>* global_counter_st = nullptr;
counter<counter_key_type, true>* global_counter_mt = nullptr;

void init_global_counter(size_t capacity) {
    mempool_interface* pool = &global_mempool::get();
    
    if (!global_counter_st) {
        global_counter_st = new counter<counter_key_type, false>(capacity, pool);
    }
    if (!global_counter_mt) {
        global_counter_mt = new counter<counter_key_type, true>(capacity, pool);
    }
}
