# producer_consumer design pattern

import asyncio
import typing as t

# 异步的生产者-消费这模式，是通过一个queue连接，但解耦了 生产数据 和 消费数据 两个步骤之间的依赖关系，使得
# 生产和消费两个异步task可以自顾自运行

# 同步的 generator --> 异步put的queue
async def async_queue_get(yield_generator:t.Generator, queue:asyncio.Queue):
    '''
    异步读取 generator 的导出结果, 并放入 queue
    '''
    async for product in yield_generator:
        await queue.put(product) # 异步地放入 queue: 当 queue 满的时候，要等待空间

    await queue.put(None) # 生产结束的信号



# 单消费者: queue 中的 output 全部由一个消费任务消费
async def async_queue_process(
        queue:asyncio.Queue,
        process_fc:t.Callable,
        executor,
        collector:t.Callable|None=None,
        *args, **kwargs):
    '''
    异步处理 queue 中的导出结果.
    处理函数 process_fc 是同步的, 且可并行. 它的第一个位置参数必须是队列中的 product
    executor 是线程池concurrent.futures.ThreadPoolExecutor/进程池concurrent.futures.ProcessPoolExecutor
    collector(if not None)是结果收集器, 应该根据executor有不同的设计, 分别保证线程安全/内存可共享

    若executor是线程池, 那么collector的设计如下:
    shared_container = []
    lock = asyncio.Lock()
    async def collector(result):
        async with lock: // 锁保证线程安全
            shared_container.append(result)

    若executor是进程池, 那么collector的设计如下:
    from multiprocessing import Manager, get_context
    manager = get_context("spawn").Manager() 或者是 Manager()
    shared_container = manager.list() // 由一个进程管理的跨进程共享
    async def collector(result):
        shared_container.append(result)
    '''
    loop = asyncio.get_event_loop()

    while True:
        product = await queue.get()

        if product is None: # 收到结束信号
            break
        
        results = await loop.run_in_executor(executor, process_fc, product, *args, **kwargs)

        # 聚合逻辑
        if collector:
            collector(results)
    
    return results



# 主协程: 建立队列，启动异步生产者-消费者任务
async def main_pipeline_producter_consumer(
        yield_generator:t.Generator,
        process_fc:t.Callable,
        executor,
        max_queue_size:int, 
        *args, **kwargs):
    # 创建队列
    queue = asyncio.Queue(max_queue_size)

    producer_task = asyncio.create_task(async_queue_get(yield_generator, queue))

    consumer_task = asyncio.create_task(async_queue_process(queue, executor, process_fc, *args, **kwargs))

    await producer_task
    results = await consumer_task

    return results