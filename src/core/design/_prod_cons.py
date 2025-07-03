# test.py
import collections
import typing as t
import os
import time
import random
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed

import asyncio
import time
import asyncio
import collections
import random
import time
import os

# --- 模拟 I/O 操作 和 计算函数 ---

async def simulate_fetch_tokens_from_IO(chunk_id: int, num_tokens: int = 1000):
    """
    模拟从 I/O 异步获取 tokens 块。
    实际中可能是从大文件读取一部分。
    """
    print(f"[Fetch {chunk_id}] 开始从 I/O 获取...")
    await asyncio.sleep(random.uniform(0.1, 0.5)) # 模拟 I/O 等待
    # 模拟生成 token 列表
    tokens = [random.randint(0, 255) for _ in range(num_tokens)]
    print(f"[Fetch {chunk_id}] 获取完成，包含 {len(tokens)} 个 tokens。")
    return tokens

def get_p_counts(tokens: list[int]) -> dict[tuple[int, int], int]:
    """
    同步函数：计算 tokens 块的 Pair 频率。
    这是一个 CPU 密集型任务，通常需要在 ProcessPoolExecutor 中运行。
    为了简化异步演示，这里先假设它足够快，或者在单个协程中运行。
    在实际 BPE 中，这部分可能需要通过 asyncio.to_thread 或 run_in_executor 桥接到进程池。
    """
    # print(f"[Compute] 计算 Pair 频率 (tokens: {len(tokens)})")
    p_counts = collections.defaultdict(int)
    for i in range(len(tokens) - 1):
        pair = (tokens[i], tokens[i+1])
        p_counts[pair] += 1
    # time.sleep(0.01) # 模拟少量计算时间
    return dict(p_counts) # 转换为普通 dict 方便传递

# --- 核心协程：生产者 ---

async def tokens_generator(queue: asyncio.Queue, total_chunks: int):
    """
    生产者协程：不断获取 tokens，并放入队列。
    """
    for i in range(total_chunks):
        next_tokens = await simulate_fetch_tokens_from_IO(i) # 模拟从 I/O 获取
        await queue.put(next_tokens) # 将 tokens 放入队列，如果队列满则暂停等待
        print(f"[Generator] 放入 chunk {i} ({len(next_tokens)} tokens) 到队列。")
    
    # 所有 tokens 获取完毕后，放入一个特殊的“哨兵”值，通知消费者没有更多数据了
    await queue.put(None) 
    print("[Generator] 所有 tokens 已放入队列，发送结束信号。")

# --- 核心协程：消费者 ---

async def token_processor_consumer(
    queue: asyncio.Queue,
    shared_stored_tokens: list[list[int]], # 共享列表，直接在内存中修改
    shared_agg_p_counts: collections.defaultdict # 共享字典，直接在内存中修改
):
    """
    消费者协程：从队列中取出 tokens，处理并聚合。
    """                           
    instance_id = random.randint(1000, 9999)
    print(f"[Consumer {instance_id}] 消费者启动。")
    agg_lock = asyncio.Lock()

    while True:
        tokens_chunk = await queue.get() # 从队列取出 tokens，如果队列空则暂停等待

        if tokens_chunk is None:
            # 收到结束信号，将信号重新放入队列供其他消费者使用（如果多个消费者的话）
            await queue.put(None) 
            print(f"[Consumer {instance_id}] 收到结束信号，退出。")
            break
        
        print(f"[Consumer {instance_id}] 从队列取出 {len(tokens_chunk)} 个 tokens，开始处理...")

        # 1. append tokens to stored_tokens
        shared_stored_tokens.append(tokens_chunk)
        print(f"[Consumer {instance_id}] 已追加 {len(tokens_chunk)} tokens 到 stored_tokens。当前总块数: {len(shared_stored_tokens)}")

        # 2. apply function get_p_counts and aggregate
        # 注意：get_p_counts 是一个同步的 CPU 密集型函数。
        # 如果它很慢，并且你的程序需要并行处理多个这样的计算，
        # 你应该使用 asyncio.to_thread 或 run_in_executor 将其放到线程池或进程池中。
        # 这里为了演示方便，直接调用它。
        
        # CPU 密集型计算的最佳实践：使用 asyncio.to_thread (Python 3.9+)
        # 或 loop.run_in_executor(None, get_p_counts, tokens_chunk)
        partial_p_counts = await asyncio.to_thread(get_p_counts, tokens_chunk) # 推荐方式
        
        # 聚合 p_counts：注意，如果是多个消费者并发聚合同一个共享字典，
        # 需要加锁或使用专门的并发数据结构来避免竞态条件。
        # 在异步单线程模型中，如果只有一个消费者，则不需要。
        # 如果是多个消费者，且 agg_p_counts 是共享内存/变量，则需要同步机制。
        # 对于 defaultdict 这种可变对象，直接在协程中修改在单事件循环中是安全的。
        # 但如果 get_p_counts 在 run_in_executor (特别是在 ProcessPoolExecutor) 中，
        # 那么 partial_p_counts 是子进程的结果，需要收集。
        
        # 简单聚合，假定单消费者或聚合操作原子性
        async with agg_lock:
            for k, v in partial_p_counts.items():
                shared_agg_p_counts[k] += v
        
        print(f"[Consumer {instance_id}] 已聚合 Pair 频率。")
        queue.task_done() # 通知队列此任务已处理完成
        
# --- 主协程：协调器 ---

async def main_bpe_pipeline(total_fetch_chunks: int, num_consumers: int = 1):
    """
    BPE 异步处理管道的主入口。
    """
    print(f"--- BPE 异步管道启动 --- (总共 {total_fetch_chunks} 个 chunks, {num_consumers} 个消费者)")
    
    # 共享数据结构（在单进程异步模型中，这些可以直接共享）
    # 注意：如果 get_p_counts 在 ProcessPoolExecutor 中运行，
    # 那么 agg_p_counts 就不能直接是内存中的 defaultdict，需要通过主进程聚合。
    stored_tokens_list: list[list[int]] = []
    # 使用 defaultdict 更方便聚合
    aggregated_pair_counts = collections.defaultdict(int) 
    
    # 创建异步队列
    # maxsize 可以控制队列大小，防止生产者生产过快，消费者来不及处理导致内存爆炸
    tokens_queue = asyncio.Queue(maxsize=10) # 队列最大可放10个 chunk

    start_time = time.time()

    # 启动生产者和消费者
    producer_task = asyncio.create_task(tokens_generator(tokens_queue, total_fetch_chunks))
    
    consumer_tasks = []
    for _ in range(num_consumers):
        consumer_tasks.append(
            asyncio.create_task(token_processor_consumer(
                tokens_queue,
                stored_tokens_list,
                aggregated_pair_counts
            ))
        )

    # 等待生产者完成所有任务
    await producer_task
    
    # 等待队列中的所有任务都被消费者处理完成
    # 这非常重要，它确保所有 put 进队列的 None/数据都被消费
    await tokens_queue.join() 
    
    # 等待所有消费者任务安全退出
    await asyncio.gather(*consumer_tasks) # 或者使用 asyncio.wait(consumer_tasks)

    end_time = time.time()
    
    print(f"\n--- BPE 异步管道完成 ---")
    print(f"总耗时: {end_time - start_time:.2f} 秒")
    print(f"最终存储的 tokens 块数量: {len(stored_tokens_list)}")
    print(f"聚合的 Pair 总数: {len(aggregated_pair_counts)}")
    # print(f"部分聚合结果: {list(aggregated_pair_counts.items())[:5]}...")

# --- 程序入口 ---
if __name__ == "__main__":
    # 模拟获取 20 个 tokens 块，并使用 3 个消费者并发处理
    asyncio.run(main_bpe_pipeline(total_fetch_chunks=20, num_consumers=3))
