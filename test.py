# test.py
import collections
import typing as t
import os
import time
import random
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed

# --- 内部函数 (在每个进程内由线程执行) ---
def process_single_line(line_data: str) -> str:
    """
    模拟 CPU 密集型计算任务。
    这个函数会在每个进程内部被调用。
    """
    # print(f"  线程 {os.getpid()}: 正在计算 '{line_data[:15]}...'")
    # 模拟 CPU 计算（增加循环次数以体现CPU密集型）
    result = line_data.upper()
    for _ in range(1_000_000): # 增加计算量
        _ = result * random.random()
    # print(f"  线程 {os.getpid()}: 完成计算 '{line_data[:15]}...'")
    return result + "_PROCESSED"

def read_and_process_file_with_threads(file_path: str) -> list[str]:
    """
    这个函数在独立的进程中运行。
    它负责读取一个文件，并利用多线程并行处理文件中的每一行。
    """
    processed_results_for_file = []
    print(f"进程 {os.getpid()}: 开始处理文件 '{file_path}'")

    # 模拟文件内容
    # 在实际应用中，这里会打开 file_path 并逐行读取
    dummy_file_lines = [f"Data from {file_path}, line {i}" for i in range(1, 10)]

    # 在当前进程内部使用 ThreadPoolExecutor 来处理每一行（模拟 I/O 密集型读取和并行处理）
    # max_workers 可以根据文件中的行数和 I/O 瓶颈调整
    with ThreadPoolExecutor(max_workers=5) as thread_executor:
        # submit tasks to threads
        future_to_line = {thread_executor.submit(process_single_line, line): line for line in dummy_file_lines}

        # collect results as they complete
        for future in as_completed(future_to_line):
            line = future_to_line[future]
            try:
                processed_line = future.result()
                processed_results_for_file.append(processed_line)
            except Exception as exc:
                print(f"进程 {os.getpid()}: 行 '{line[:15]}...' 生成了一个异常: {exc}")
    
    print(f"进程 {os.getpid()}: 完成处理文件 '{file_path}'。")
    return processed_results_for_file

# --- 外部主函数 (在主进程中执行) ---
def main():
    # 模拟多个文件路径，每个文件代表一个“大任务”
    # 这些文件将由不同的进程处理
    file_paths = [f"file_{i}.txt" for i in range(1, 5)] # 4个文件

    print("--- 启动多进程处理 (每个进程内部有多线程) ---")
    start_time = time.time()

    # 使用 ProcessPoolExecutor 来并行处理不同的文件
    # max_workers 通常设置为 CPU 核心数，以获得最佳 CPU 密集型任务性能
    with ProcessPoolExecutor(max_workers=len(file_paths)) as process_executor: # 或 os.cpu_count()
        # 将 read_and_process_file_with_threads 函数应用到每个 file_path
        # results_iterator 会返回一个迭代器，其中每个元素是对应进程返回的结果列表
        future_to_filepath = {process_executor.submit(read_and_process_file_with_threads, path): path for path in file_paths}

        all_results = []
        # 收集所有进程的结果
        for future in as_completed(future_to_filepath):
            filepath = future_to_filepath[future]
            try:
                file_results = future.result()
                all_results.extend(file_results)
            except Exception as exc:
                print(f"文件 '{filepath}' 的处理过程中发生异常: {exc}")

    end_time = time.time()
    print(f"\n所有任务完成。总耗时: {end_time - start_time:.4f} 秒")
    print(f"收集到的总结果数量: {len(all_results)}")
    # print(f"部分结果: {all_results[:10]}...") # 打印部分结果以检查

if __name__ == "__main__":
    main()
