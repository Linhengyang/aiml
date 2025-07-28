// hashtable_openaddressing.h

#ifndef HASH_TABLE_OPEN_ADDRESSING_H
#define HASH_TABLE_OPEN_ADDRESSING_H

#include <cstddef>
#include <cstdint>
#include <vector>
#include <atomic>
#include <shared_mutex>

struct hash_key {
    uint16_t L;
    uint16_t R;
};


struct hash_entry {
    hash_key key;
    std::atomic<uint64_t> count{0};
    bool used = false;
    mutable std::shared_mutex mutex; // 共享互斥锁 支持读写分离
};



class hashtable_openaddressing {

private:
    std::vector<hash_entry> table;
    size_t capacity;
    size_t size; // 当前元素数量
    float load_factor_threshold = 0.7f;

    size_t hash(uint16_t L, uint16_t R) const {
        return ((size_t)L * 123456791 + (size_t)R) % capacity;
    }

public:
    hashtable_openaddressing(size_t init_capacity = 1024);
    ~hashtable_openaddressing();
};

hashtable_openaddressing::hashtable_openaddressing(size_t init_capacity = 1024)
{
}

hashtable_openaddressing::~hashtable_openaddressing()
{
}



#endif

