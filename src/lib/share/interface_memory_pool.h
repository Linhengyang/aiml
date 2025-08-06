// interface_memory_pool.h

#include <cstddef>

class mempool_interface {

public:
    virtual void* allocate(size_t size) = 0;
    virtual void dealloc_large(void* ptr) = 0;
    virtual void shrink(size_t max_num) = 0;
    virtual void reset() = 0;
    virtual void release() = 0;
    virtual ~mempool_interface() = default;

};