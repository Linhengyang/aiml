// global_resource.h

// 在这里管理 tokenizer 项目要用到的所有全局资源：全局内存池，以及依附于之上的 counter

#pragma once
#include "mempool_counter.h"
#include "memory_pool_global.h"



// 全局可复用 counter

using counter_key_type = std::pair<uint16_t, uint16_t>;

extern counter<counter_key_type, false>* global_counter_st;
extern counter<counter_key_type, true>* global_counter_mt;

void init_global_counter(size_t capacity);
void reset_global_counter();
