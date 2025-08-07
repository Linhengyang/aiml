#include <vector>
#include <shared_mutex>

constexpr size_t CACHE_LINE_SIZE = 64;

template <typename TYPE_LOCK>
struct padded_mutex {
    
    // alignas, C++11引入的关键字, 指定变量的内存对齐方式
    alignas(CACHE_LINE_SIZE) TYPE_LOCK lock; // alignas强制TYPE_LOCK类变量 lock 按64字节内存对齐

    // padding数组, 使得当sizeof(TYPE_LOCK)小于 CACHE_LINE_SIZE 时, lock占据+padding部分正好占满一个完整的cache line.
    // 当 sizeof(TYPE_LOCK)大于 CACHE_LINE_SIZE 时, 前面alignas 对齐就够了. 此时pad至少1以满足部分编译器的要求
    char padding[CACHE_LINE_SIZE - sizeof(TYPE_LOCK) > 0 ? CACHE_LINE_SIZE - sizeof(TYPE_LOCK) : 1];

    padded_mutex() = default; // padded_mutex 要用在桶锁vector中，而vector需要元素满足拷贝构造
    padded_mutex(const padded_mutex&) = delete; // 禁止拷贝
    padded_mutex& operator=(const padded_mutex&) = delete; // 禁止赋值
};

int fail() {
    std::vector<padded_mutex<std::shared_mutex>> _bucket_mutexs;
    _bucket_mutexs.resize(10); // 编译不通过
}

int main() {
    std::vector<padded_mutex<std::shared_mutex>> _bucket_mutexs(10); // 编译通过
}