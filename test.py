# test.py
import collections
import typing as t
import os
import time
import random
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed

# # --- 内部函数 (在每个进程内由线程执行) ---
# def process_single_line(line_data: str) -> str:
#     """
#     模拟 CPU 密集型计算任务。
#     这个函数会在每个进程内部被调用。
#     """
#     # print(f"  线程 {os.getpid()}: 正在计算 '{line_data[:15]}...'")
#     # 模拟 CPU 计算（增加循环次数以体现CPU密集型）
#     result = line_data.upper()
#     for _ in range(1_000_000): # 增加计算量
#         _ = result * random.random()
#     # print(f"  线程 {os.getpid()}: 完成计算 '{line_data[:15]}...'")
#     return result + "_PROCESSED"

# def read_and_process_file_with_threads(file_path: str) -> list[str]:
#     """
#     这个函数在独立的进程中运行。
#     它负责读取一个文件，并利用多线程并行处理文件中的每一行。
#     """
#     processed_results_for_file = []
#     print(f"进程 {os.getpid()}: 开始处理文件 '{file_path}'")

#     # 模拟文件内容
#     # 在实际应用中，这里会打开 file_path 并逐行读取
#     dummy_file_lines = [f"Data from {file_path}, line {i}" for i in range(1, 10)]

#     # 在当前进程内部使用 ThreadPoolExecutor 来处理每一行（模拟 I/O 密集型读取和并行处理）
#     # max_workers 可以根据文件中的行数和 I/O 瓶颈调整
#     with ThreadPoolExecutor(max_workers=5) as thread_executor:
#         # submit tasks to threads
#         future_to_line = {thread_executor.submit(process_single_line, line): line for line in dummy_file_lines}

#         # collect results as they complete
#         for future in as_completed(future_to_line):
#             line = future_to_line[future]
#             try:
#                 processed_line = future.result()
#                 processed_results_for_file.append(processed_line)
#             except Exception as exc:
#                 print(f"进程 {os.getpid()}: 行 '{line[:15]}...' 生成了一个异常: {exc}")
    
#     print(f"进程 {os.getpid()}: 完成处理文件 '{file_path}'。")
#     return processed_results_for_file

# # --- 外部主函数 (在主进程中执行) ---
# def main():
#     # 模拟多个文件路径，每个文件代表一个“大任务”
#     # 这些文件将由不同的进程处理
#     file_paths = [f"file_{i}.txt" for i in range(1, 5)] # 4个文件

#     print("--- 启动多进程处理 (每个进程内部有多线程) ---")
#     start_time = time.time()

#     # 使用 ProcessPoolExecutor 来并行处理不同的文件
#     # max_workers 通常设置为 CPU 核心数，以获得最佳 CPU 密集型任务性能
#     with ProcessPoolExecutor(max_workers=len(file_paths)) as process_executor: # 或 os.cpu_count()
#         # 将 read_and_process_file_with_threads 函数应用到每个 file_path
#         # results_iterator 会返回一个迭代器，其中每个元素是对应进程返回的结果列表
#         future_to_filepath = {process_executor.submit(read_and_process_file_with_threads, path): path for path in file_paths}

#         all_results = []
#         # 收集所有进程的结果
#         for future in as_completed(future_to_filepath):
#             filepath = future_to_filepath[future]
#             try:
#                 file_results = future.result()
#                 all_results.extend(file_results)
#             except Exception as exc:
#                 print(f"文件 '{filepath}' 的处理过程中发生异常: {exc}")

#     end_time = time.time()
#     print(f"\n所有任务完成。总耗时: {end_time - start_time:.4f} 秒")
#     print(f"收集到的总结果数量: {len(all_results)}")
#     # print(f"部分结果: {all_results[:10]}...") # 打印部分结果以检查

# if __name__ == "__main__":
#     main()




import collections
import typing as t
import os
import tempfile
import pickle # 用于序列化数据到文件
from concurrent.futures import ProcessPoolExecutor, as_completed, Future

# --- 辅助函数（假设已定义或根据需求实现）---
def get_pair_counts_for_chunk(tokens: t.List[int]) -> t.Dict[tuple[int, int], int]:
    """
    计算单个 tokens 块的 pair 计数，并返回本地字典。
    这个函数会在分布式/并行环境中执行。
    """
    p_counts_local: t.Dict[tuple[int, int], int] = collections.defaultdict(int)
    for i in range(len(tokens) - 1):
        pair = (tokens[i], tokens[i+1])
        p_counts_local[pair] += 1
    return p_counts_local

def merge_pair(tokens: t.List[int], pair_to_merge: tuple[int, int], new_token_id: int) -> t.List[int]:
    """
    在单个 token 块中执行合并操作。
    这个函数会在分布式/并行环境中执行。
    """
    merged_tokens = []
    i = 0
    while i < len(tokens):
        if i + 1 < len(tokens) and (tokens[i], tokens[i+1]) == pair_to_merge:
            merged_tokens.append(new_token_id)
            i += 2 # Skip both tokens that were merged
        else:
            merged_tokens.append(tokens[i])
            i += 1
    return merged_tokens

def raise_run_out_corpus_error(rank: int, num_specials: int):
    raise ValueError(f"Corpus exhausted at rank {rank} (specials: {num_specials})")

# --- 重构后的 bpe_single_merge 核心逻辑 ---
# 注意：这个函数本身不再直接执行并发，而是定义了流程
# 并行执行的协调由外部的 `run_bpe_single_merge_distributed` 函数负责
def bpe_single_merge_core_logic(
    rank: int,
    all_pair_counts: t.List[t.Dict[tuple[int, int], int]], # 聚合后的所有 pair 计数
    num_specials: int
) -> t.Tuple[tuple[int, int], int, int]:
    """
    BPE 单次合并的核心逻辑，负责找出最佳合并对和生成新 token ID。
    这部分是串行的，由主进程执行。
    """
    p_counts_aggregated: t.Dict[tuple[int, int], int] = collections.defaultdict(int)
    for partial_counts in all_pair_counts:
        for pair, count in partial_counts.items():
            p_counts_aggregated[pair] += count

    if not p_counts_aggregated:
        raise_run_out_corpus_error(rank, num_specials)
    
    occur_most_pair: tuple[int, int] = max(p_counts_aggregated, key=p_counts_aggregated.get)
    occurance = p_counts_aggregated[occur_most_pair]
    new_token: int = rank + 256 # 从 256 开始分配新的 token ID

    return occur_most_pair, occurance, new_token

# --- 外部包装函数：协调分布式加速 ---
def run_bpe_single_merge_distributed(
    rank: int,
    tokens_chunks_iterator: t.Iterable[t.List[int]], # 现在是一个可重复遍历的迭代器或列表
                                                   # 或者是文件路径列表，代表分块的 token 数据
    num_specials: int,
    process_executor: ProcessPoolExecutor, # 外部传入的进程池
    temp_dir: str = None # 用于存储临时文件的目录
) -> t.Generator[t.Union[t.Tuple[tuple[int, int], int, int], t.List[int]], None, None]:
    """
    协调 BPE 单次合并的分布式加速过程。
    rank: 当前合并的等级 (0, 1, 2...)
    tokens_chunks_iterator: 原始 token 块的可迭代对象。
                            如果数据量大，它应该代表分块的数据源 (如文件路径列表)。
    num_specials: 特殊 token 的数量。
    process_executor: 预先创建的 ProcessPoolExecutor 实例。
    temp_dir: 临时文件存放目录，用于 out-of-core 场景。
    """
    print(f"\n--- BPE 单次合并开始 (Rank: {rank}) ---")

    # 如果 tokens_chunks_iterator 是一个生成器，我们需要将其物化到内存或磁盘
    # 以便第二次遍历，但因为你的注释说 `tmp_store = []`，我们先假定它能进内存
    # 真正的分布式需要写入临时文件
    
    # 模拟 `tmp_store` 的构建，同时启动并行 `get_pair_counts`
    # 在真正的分布式场景下，这里是将 tokens_generator 写入多个临时文件，
    # 并让 worker 读取这些文件计算 counts
    
    # 阶段 1: 可分布式加速，写入密集（如果不是列表），计算密集（p_counts）
    # 同时收集所有 `tokens` 以备第二阶段使用。
    # 这里的 tokens_chunks_iterator 可能是来自上一次合并的输出。
    
    futures_counts: t.List[Future] = []
    
    # 为了避免在内存中存储 tmp_store，并在计算 p_counts 的同时处理，
    # 我们假设 tokens_chunks_iterator 是一个可重复遍历的列表或 Dask Bag 等。
    # 如果它是一次性迭代器，那你的 `tmp_store` 方案是必须的，
    # 但 `tmp_store` 又会导致内存问题。
    # 更现实的 out-of-core 方案是：
    # 第一次：将 tokens_generator 写入多个临时文件。
    # 第二次：让 worker 读取这些临时文件并计算 p_counts。
    # 第三次：再让 worker 读取这些临时文件并执行 merge_pair。
    
    # 为了演示，我们假设 `tokens_chunks_iterator` 是一个列表，它已在外部准备好，
    # 或者这个函数负责将其物化到临时文件。
    
    # 实际应用中，如果 tokens_chunks_iterator 是一个巨大的、一次性的生成器，
    # 你会在这里将其写入到多个分块的临时文件中，然后提交文件路径给进程池。
    # 为了代码简化，我们直接传递 `tokens` 块。
    
    all_chunks_for_pass2 = [] # 这个变量会在第一阶段被填充，用于第二阶段的merge
                               # 如果数据巨大，这里需要是临时文件路径列表
    
    print(f"阶段 1: 提交 {len(list(tokens_chunks_iterator))} 个 Pair 计数任务...") # 消耗迭代器一次，需注意
    
    # 重新获取迭代器，因为上面的 len() 可能已经耗尽了它
    # 或者假设 tokens_chunks_iterator 总是可重复的，例如是一个列表
    # 如果它是一个一次性迭代器，你需要在外面 `list()` 或者写入文件
    tokens_chunks_for_pass1 = list(tokens_chunks_iterator) # 这里是内存瓶颈，如果数据量大，需要改为写入临时文件
    
    for idx, tokens_chunk in enumerate(tokens_chunks_for_pass1):
        # 提交 pair 计数任务到进程池
        futures_counts.append(process_executor.submit(get_pair_counts_for_chunk, tokens_chunk))
        # 同时，将 tokens_chunk 保存下来，供后续的 merge 阶段使用
        all_chunks_for_pass2.append(tokens_chunk) # 同样，如果数据大，这里应是保存到文件并记录文件路径

    aggregated_pair_counts: t.List[t.Dict[tuple[int, int], int]] = []
    print("阶段 1: 收集 Pair 计数结果...")
    for future in as_completed(futures_counts):
        try:
            partial_counts = future.result()
            aggregated_pair_counts.append(partial_counts)
        except Exception as exc:
            print(f"计算 pair 计数时发生错误: {exc}")
            # 根据错误处理策略决定是否继续

    # 阶段 2: 单一行为，主进程该做的事情
    print("阶段 2: 计算最佳合并对...")
    occur_most_pair, occurance, new_token = bpe_single_merge_core_logic(
        rank, aggregated_pair_counts, num_specials
    )
    yield occur_most_pair, occurance, new_token # first yield, 返回合并规则

    # 阶段 3: 可分布式加速，读取密集 + 计算密集，顺序不重要
    # yield remain as new tokens_generator
    futures_merges: t.List[Future] = []
    
    print("阶段 3: 提交合并任务...")
    for idx, tokens_chunk in enumerate(all_chunks_for_pass2): # 从之前保存的块中读取
        # 提交合并任务到进程池
        # partial 可以在这里用，但为了示例清晰，直接传递参数
        futures_merges.append(
            process_executor.submit(merge_pair, tokens_chunk, occur_most_pair, new_token)
        )
    
    # 遍历合并结果，并 yield 出新的 token 序列
    print("阶段 3: 收集合并结果并返回新的 tokens_generator...")
    # 注意: as_completed 不保证顺序，但你的注释说“顺序不重要”
    # 如果顺序重要，需要使用一个按照原始索引排序的机制 (如 futures_list[idx].result())
    for future in as_completed(futures_merges):
        try:
            merged_tokens_chunk = future.result()
            yield merged_tokens_chunk # yield each merged chunk
        except Exception as exc:
            print(f"合并 tokens 块时发生错误: {exc}")
            yield [] # 返回空列表或根据错误处理策略

# --- 完整的主 BPE 训练循环 (模拟外层超高次数循环) ---
def full_bpe_training_loop(
    initial_chunks: t.List[t.List[int]], # 初始化的 token 块列表
    num_merges: int,
    num_specials: int = 0
):
    print("--- 启动 BPE 完整训练流程 ---")
    current_tokens_chunks = list(initial_chunks) # 初始的 tokens 数据

    # 在最外层循环外部一次性创建进程池
    with ProcessPoolExecutor(max_workers=os.cpu_count()) as executor:
        for rank in range(num_merges):
            print(f"\n>>>> BPE 合并迭代 {rank + 1}/{num_merges} <<<<")
            
            # 调用协调函数，它会并行化内部的计算和合并步骤
            # 并返回合并规则和新的 tokens_generator
            bpe_merge_generator = run_bpe_single_merge_distributed(
                rank,
                current_tokens_chunks, # 传递当前轮的 token 数据
                num_specials,
                executor # 传入进程池
            )
            
            # 获取合并规则 (第一次 yield)
            try:
                occur_most_pair, occurance, new_token = next(bpe_merge_generator)
                print(f"  已获得合并规则: ({occur_most_pair}) -> {new_token}, 出现次数: {occurance}")
            except StopIteration:
                print("  生成器提前耗尽，没有新的合并规则。")
                break # 没有更多可合并的了

            # 更新当前轮的 tokens_chunks，为下一轮做准备
            # 这里需要消耗掉 bpe_merge_generator 剩余的部分 (即合并后的 tokens 块)
            # 这又是内存瓶颈，如果数据巨大，需要将 `yield merged_tokens_chunk` 写入文件
            # 并在下一轮从这些文件读取。
            print("  收集合并后的 tokens 块，准备下一轮...")
            new_current_tokens_chunks = []
            for merged_chunk in bpe_merge_generator:
                new_current_tokens_chunks.append(merged_chunk)
            current_tokens_chunks = new_current_tokens_chunks
            print(f"  本轮合并后，tokens 块总数: {len(current_tokens_chunks)}")

            # 如果所有块都变空或没有新的 token 了，可以提前停止
            if not any(current_tokens_chunks): # 如果所有块都为空列表
                print("所有 token 块已耗尽，提前停止。")
                break

    print("\n--- BPE 训练完成 ---")
    # 最终的 current_tokens_chunks 就是完全合并后的结果
    return current_tokens_chunks


# --- 运行示例 ---
if __name__ == "__main__":
    # 模拟一些初始的 tokens 块数据
    # 假设每个内部列表是一个分词后的文本块，已经编码成整数
    initial_data = [
        [ord('a'), ord('b'), ord('c'), ord('a'), ord('b')],
        [ord('b'), ord('c'), ord('a'), ord('a'), ord('b'), ord('c')],
        [ord('x'), ord('y'), ord('z'), ord('a'), ord('b'), ord('c'), ord('x'), ord('y')],
        [ord('a'), ord('a'), ord('b'), ord('b'), ord('c'), ord('c')]
    ]
    
    # 增加数据量以体现并行化优势，每个数据块重复多次
    large_initial_data = []
    for _ in range(500): # 增加块的数量
        large_initial_data.extend([list(chunk) for chunk in initial_data])


    start_time = time.time()
    final_merged_chunks = full_bpe_training_loop(large_initial_data, num_merges=10) # 进行 10 次合并
    end_time = time.time()

    print(f"\n总执行时间: {end_time - start_time:.4f} 秒")
    # 可以打印一些最终结果的片段
    # print(f"最终合并后的部分块: {final_merged_chunks[:2]}")