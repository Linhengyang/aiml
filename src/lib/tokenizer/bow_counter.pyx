# distutils: language = c++
# cython: language_level=3, boundscheck=False, wraparound=False

from libcpp.unordered_map cimport unordered_map
from libcpp.utility cimport pair
from libcpp.string cimport string
from libcpp.string_view cimport string_view # <-- C++17 特性

from cython.operator cimport dereference as deref, preincrement as inc
from cpython.bytes cimport PyBytes_AsString, PyBytes_GET_SIZE, PyBytes_FromStringAndSize
from cpython.dict cimport PyDict_SetItem
from cpython.long cimport PyLong_FromLongLong


# cdef声明: 静态类型编译时声明
# 对于def函数(而不是cpdef/cdef,它们有额外的准则), C/Cpp类型变量必须要cdef声明在函数块的开头, 不能在for循环或其他控制流语句内部
# 因为for循环或其他控制流语句内部 引入了 Python运行时, 此时是没办法执行 C/Cpp编译时的. 所以必须要在 Python运行时 之外执行 C/Cpp编译时.

# Cython是一种 C/Cpp & Python 混写的语言, 解释器会将其彻底翻译成 C代码来执行.
# 对于 C/Cpp代码, 解释器直接执行 C/Cpp编译时;
# 对于 Py代码, 解释器首先尽量尝试翻译成静态 C/Cpp代码, 这里需要一些预先写好的C类型辅助类, 比如 PyObject* 等; 
#   实在无法静态翻译的, 引入 Python运行时(同时引入了各种Python字典/fast locals数组等开销)

# --> Cython代码总结: C/Cpp部分不要沾染Python, 纯Python移到外部Py脚本, py-cpp边界做好高效转换
# --> 尽量避免Python运行时: 尽量写纯C/Cpp代码，利用C/Cpp执行计算
# --> 如果实在不得已要用Python运行时（比如这里的match，利用了regex的finditer函数），要做到高效转换（提前cdef是告诉Cython编译器生成纯Cpp代码的唯一方式）:
#  在 Python运行时之外执行 C/Cpp类型的cdef声明（如果不cdef, Cython会把变量当成Python对象，变量本身会通过 fast locals数组或Python字典访问，变量之间的运算会触发 Python C API）
#  而 执行高效转换后得到的 C/Cpp类型, 则可以依靠丰富的 Cpp零成本抽象，实现高性能计算



def _bytes_chunk_count(bytes text_bytes, object compiled_regex):
    """
    text_bytes: 已经 utf-8 编码的整个 batch 文本 (b'\n'.join(...))
    compiled_regex: 正则表达式字符串编译后缓存, 即 re.compile 返回的对象. 具备 .finditer 方法
    返回：dict of {bytes: int}
    """
    
    # 1. 获取底层 C 常指针，避免 Python 切片开销
    cdef const char* buffer = PyBytes_AsString(text_bytes)
    cdef Py_ssize_t total_len = PyBytes_GET_SIZE(text_bytes)

    # 2. C++ 计数器, 和 可复用 token_str 提前cdef声明
    cdef unordered_map[string, unsigned long long] local_map
    cdef string token_str

    cdef Py_ssize_t start
    cdef Py_ssize_t end
    # 虽然 match 是 Py对象, 但是这样提前声明，有助于 Cython编译器生成清晰的 C 代码（Cython3将默认 PyObject* 走 fast locals数组而不是Python字典）
    cdef object match
    
    # 3. 迭代匹配的结果 (finditer 不会像 findall 那样一次性生成列表. finditer提供了两种重载, 参数string可以是str, 也可以是ReadableBuffer. 这里是后一种)
    # 注意：这里仍然会创建 Match py对象，但避免了创建子串 bytes 对象
    for match in compiled_regex.finditer(text_bytes):
        # 只获取 匹配到的跨度
        start = <Py_ssize_t>match.start()
        end = <Py_ssize_t>match.end()
        
        # 安全校验
        if start < 0 or end > total_len or start >= end:
            continue
            
        # 4. 直接从 buffer 构造 std::string，无 Python 对象分配. 但是应该尽量减少 buffer bytes的拷贝, 最好只发生一次: buffer -> map's key
        token_str.assign(buffer + start, end - start)  # .assign(const char* s, size_t n)复用内存地址(自动扩容), 摊销下存在 0次malloc + 1次memcpy
        # 这里若使用 string(const char* s, size_t n) 构造函数, 则存在 1次 malloc + 1次memcpy --> 降低了 1次malloc开销

        # 若存在插入新节点, 则存在 1次malloc + 1次memcpy; 若非新节点, 则 0次malloc + 0次memcpy
        local_map[token_str] += 1
        # buffer_bytes --copy--> token_str --copy--> local_map.keys, 存在 num_whole_words + num_unique_words 次 memcpy开销, 但优点是 生命周期解耦
        
    # 5. 将 C++ map 转换回 Python dict (仅在最后发生一次)
    cdef dict result = {}
    cdef unordered_map[string, unsigned long long].iterator it = local_map.begin()
    cdef string key
    cdef unsigned long long val
    
    while it != local_map.end():
        key = deref(it).first
        val = deref(it).second
        # 构造 Python bytes 对象 (这是必须的，因为要返回给 Python). Cython会调用内置的 std::string 到 Python bytes 的自动转换
        py_key = bytes(key)
        # 其实也可以使用 PyBytes_FromStringAndSize(const char*, size_t). 不过这样要取 it->first 的 data() 和 size() 两个string成员方法的返回结果

        # Cython自动完成 C类型 到 Py对象的转换
        result[py_key] = val
        inc(it) # 迭代器前置自增
        
    return result





def bytes_chunk_count(bytes text_bytes, object compiled_regex):
    """
    text_bytes: 已经 utf-8 编码的整个 batch 文本 (b'\n'.join(...))
    compiled_regex: 正则表达式字符串编译后缓存, 即 re.compile 返回的对象. 具备 .finditer 方法
    返回：dict of {bytes: int}
    """
    
    # 1. 获取底层 C 常指针，避免 Python 切片开销
    cdef const char* buffer = PyBytes_AsString(text_bytes)
    cdef Py_ssize_t total_len = PyBytes_GET_SIZE(text_bytes)

    # 2. C++ 计数器 但这里使用 string_view 作为 key
    cdef unordered_map[string_view, unsigned long long] local_map
    cdef string_view token_view

    cdef Py_ssize_t start, end
    # 虽然 match 是 Py对象, 但是这样提前声明，有助于 Cython编译器生成清晰的 C 代码（Cython3将默认 PyObject* 走 fast locals数组而不是Python字典）
    cdef object match
    
    # 3. 核心循环(Python运行时与Cpp交互)
    for match in compiled_regex.finditer(text_bytes):
        start = <Py_ssize_t>match.start()
        end = <Py_ssize_t>match.end()
        
        if start < 0 or end > total_len or start >= end:
            continue
            
        # 4. 构造 string_view: 本质只有指针赋值和长度赋值, 0次 malloc + 0次memcpy
        token_view = string_view(buffer + start, end - start)

        # 若存在插入新节点, 则key只存ptr和length, hash计算发生在buffer_bytes上, 从而存在 0次malloc + 0次memcpy; 若非插入, 则 0次malloc + 0次memcpy
        local_map[token_view] += 1
        # buffer_bytes --ref--> token_view --hash--> local_map.keys, 完美做到零开销: 全程 0 malloc + 0 memcpy
        # 缺点是生命周期耦合 --> 为了保证 local_map 有效, buffer_bytes必须在 return前有效不能被gc --> Cython引用计数机制保证, 只要传进入Cython侧, 就能保证其在cython函数return前不被gc
        
    # 5. 将 C++ map 转换回 Python dict
    cdef dict result = {}
    cdef unordered_map[string_view, unsigned long long].iterator it = local_map.begin()

    while it != local_map.end():
        # 使用 CPython C API 直接从 const char* 和 length 构造 Python bytes 对象（string_view必须如此, 因为cython没有从string_view到python bytes对象的自动转换）
        # 所以必须要使用 利用 PyBytes_FromStringAndSize 这个 CPython C API 构造 Python bytes
        # string_view类有类似string类的两个成员方法, 其中 .data() 返回 const char*, size() 返回 size_t
        py_key = PyBytes_FromStringAndSize(it->first.data(), it->first.size())
        
        # Cython自动完成 C类型 到 Py对象的转换
        result[py_key] = it->second
        inc(it) # 迭代器前置自增
        
    return result