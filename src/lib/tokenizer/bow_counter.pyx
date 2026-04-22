# distutils: language = c++
# cython: language_level=3, boundscheck=False, wraparound=False

from libcpp.unordered_map cimport unordered_map
from libcpp.string cimport string
from cpython.bytes cimport PyBytes_AsString, PyBytes_GET_SIZE
from cpython.dict cimport PyDict_SetItem
from cpython.long cimport PyLong_FromLongLong



def bow_chunk_count_bytes(bytes text_bytes, object compiled_regex):
    """
    text_bytes: 已经 utf-8 编码的整个 batch 文本 (b'\n'.join(...))
    compiled_regex: 正则表达式字符串编译后缓存
    返回：dict of {bytes: int}
    """
    
    # 1. 获取底层 C 指针，避免 Python 切片开销
    cdef const char* buffer = PyBytes_AsString(text_bytes)
    cdef Py_ssize_t total_len = PyBytes_GET_SIZE(text_bytes)
    
    # 2. C++ 计数器
    cdef unordered_map[string, unsigned long long] local_map
    cdef string token_str
    
    # 3. 迭代匹配 (finditer 不会像 findall 那样一次性生成列表)
    # 注意：这里仍然会创建 Match py对象，但避免了创建子串 bytes 对象
    for match in compiled_regex.finditer(text_bytes):
        cdef int start = match.start()
        cdef int end = match.end()
        
        # 安全校验
        if start < 0 or end > total_len or start >= end:
            continue
            
        # 4. 直接从 buffer 构造 std::string，无 Python 对象分配
        # string(const char* s, size_t n) 构造函数
        token_str = string(buffer + start, end - start)
        
        local_map[token_str] += 1
        
    # 5. 将 C++ map 转换回 Python dict (仅在最后发生一次)
    cdef dict result = {}
    cdef unordered_map[string, unsigned long long].iterator it = local_map.begin()
    cdef string key
    cdef unsigned long long val
    
    while it != local_map.end():
        key = it.first
        val = it.second
        # 构造 Python bytes 对象 (这是必须的，因为要返回给 Python)
        # 但此时只针对唯一词表，而非所有 token
        py_key = bytes(key) 
        result[py_key] = val
        ++it
        
    return result