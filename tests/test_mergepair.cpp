#include <iostream>
#include <stdexcept>
#include "memory_pool.h"
#include "merge_pair.h"


int main() {
    try {
        // 创建内存池
        init_memory_pool(64, 16);
        std::cout << "memory pool created" << "\n";

        // 模拟输入：tokens_flat = [1, 2, 3, 1, 2, 3], 分成2段：[1,2,3], [1,2,3]
        int tokens_flat[] = {1, 2, 3, 1, 2, 3};
        long offsets[] = {0, 3, 6};  // 两个 chunk，长度分别为 3
        size_t num_chunks = 2;

        int pair_L = 1;
        int pair_R = 2;
        int new_token = 99;
        
        // 调用目标函数
        return_bundle result = c_merge_pair_batch(
            tokens_flat,
            offsets,
            num_chunks,
            pair_L,
            pair_R,
            new_token
        );

        // 打印 output_tokens_flat
        std::cout << "output_tokens_flat: ";
        for (size_t i = 0; i < offsets[num_chunks]; ++i) {
            std::cout << result.output_tokens_flat_ptr[i] << " ";
        }
        std::cout << "\n";

        // 打印 output_filter
        std::cout << "output_filter: ";
        for (size_t i = 0; i < offsets[num_chunks]; ++i) {
            std::cout << result.output_filter_ptr[i] << " ";
        }
        std::cout << "\n";

        // 打印 output_tokens_lens
        std::cout << "output_tokens_lens: ";
        for (size_t i = 0; i < num_chunks; ++i) {
            std::cout << result.output_tokens_lens_ptr[i] << " ";
        }
        std::cout << "\n";

        // 清理内存池
        release_memory_pool();
        std::cout << "memory pool released" << "\n";

    } catch (const std::exception& ex) {
        std::cerr << "Exception caught: " << ex.what() << std::endl;
        return 1;
    }

    return 0;
}
