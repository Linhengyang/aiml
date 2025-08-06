// global_resource.cpp
#include <cstddef>
#include "global_resource.h"
#include "memory_pool_global.h"
#include "memory_pool.h"
#include "mempool_counter.h"
#include "mempool_hash_table_mt.h"
#include "mempool_hash_table_st.h"
#include "interface_memory_pool.h"

// 初始化为 nullptr
counter<counter_key_type, false>* global_counter_st = nullptr;
counter<counter_key_type, true>* global_counter_mt = nullptr;

void init_global_counter(size_t capacity) {
    auto pool = new mempool(4096, 8);
    mempool_interface* pool_ptr = static_cast<mempool_interface*>(pool);

    hash_table_st_chain<counter_key_type, uint64_t> hashtable(capacity, pool_ptr);

    if (!global_counter_st) {
        global_counter_st = new counter<counter_key_type, false>(capacity, pool);
    }
    if (!global_counter_mt) {
        global_counter_mt = new counter<counter_key_type, true>(capacity, pool);
    }
}
