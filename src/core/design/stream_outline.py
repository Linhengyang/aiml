# customized design pattern for sync stream process

from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, wait, FIRST_COMPLETED
import typing as t

# 同步的流水线处理, 很多地方使用 列表生成式+生成器, 即
# [process_fn(item) for item in data_gen]
# 并发执行下是这样: 
# futures = [executor.submit(process_fn, item) for item in data_gen]
# for future in as_completed(futures):
#     result = future.result()
#     result_handler_fn(result)

# 但列表生成式没有背压设计, 即 for item in data_gen 会在主线程中, 无视下游的处理消费能力, 一直消费data_gen
# 若下游的处理消费能力跟不上, 那么就会使得 item 大量积压在内存中, 造成内存爆炸. 即 同步for-loop不作任何协调暂停
# 真正的背压设计只存在在 asyncio, RxPy, Streams 和 kafka-like queue. 这里提供一个可以pending消费的同步流式处理


def stream_parallel_process_with_pending(
    executor: ProcessPoolExecutor|ThreadPoolExecutor,
    data_gen: t.Generator,
    process_fn: t.Callable[[t.Any], t.Any],
    result_handler: t.Callable[[t.Any], None],
    max_pending: int = 8,
    *args # 提交给 process_fn 的其他参数
    ):
    '''
    流式并发处理生成器中的任务, 控制内存占用

    Args:
        executor: ProcessPoolExecutor / ThreadPoolExecutor
        data_gen: generator, 逐批生成任务数据
        process_fn: 对每一批的处理函数(子进程中执行)
        result_handler: 对任务结果的处理(在主线程中执行)
        max_pending: 最大挂起任务数(根据内存控制)
    '''
    futures = set()

    for item in data_gen:
        # 提交任务
        future = executor.submit(process_fn, item, *args)
        futures.add(future)

        # 当任务队列长度达到 max_pending 时, 暂停提交任务, 等到部分已提交的任务完成并处理
        if len(futures) >= max_pending:
            # wait会返回已经处理好的 futures(作为done) 和 尚未处理好的 futures(作为futures)
            done, futures = wait(futures, return_when=FIRST_COMPLETED)
            for f in done:
                result = f.result()
                result_handler(result)

    # 收尾, 处理剩下的任务
    for f in futures:
        result = f.result()
        result_handler(result)