import typing as t
import os
from collections import Counter
import regex as re
import pyarrow as pa
import pyarrow.parquet as pq
from concurrent.futures import ProcessPoolExecutor
from ..parquet.sharding import shard_pq_to_ds
from ...common.stream_control import stream_parallel_process_with_pending

# 0 读取/写入 parquet文件 的接口都用 Python 处理 -- 避免在Cython/C++侧引入复杂的 parquet 依赖

# 1 多进程并行对每个 shard 执行 worker:
# 2.map.worker:
#   初始化一个local_counter, 读取parquet文件, iter_batches循环, 对每一个batch执行预切分成 words utf-8 编码后再local_counter计数
#   worker部分可以使用 cython/C++ 加速: 使用 C++ 的unordered_map计数, local_counter计数完毕后, 遍历之至python dict对象.
# 2.reduce.collector:
#   使用Counter.update方法, 把map.worker得到的local_counter汇总 --> global_counter(python dict对象)
# 3 落盘 global_counter 的keys和values为parquet两列

# 一: shard只作分片, 不作任何其他操作. 预切分和utf-8编码都 放在拿到batch后、counter计数前.
#   原因: parquet存储超多个独立字符串对象word造成压缩率下降, 而拿到batch后一股脑切分+编码+计数, 紧凑高效，IO效率高，原始数据只读取一次.
# 二：无论输出格式是binary/string/integer，在计数前都将word作utf-8编码成bytes, 使用bytes作为key计数.
#   原因: bytes哈希更快, 只需要编码一次，在python侧bytes更紧凑; string哈希慢，且每次计数都要处理编码后再哈希，在python侧str对象开销更大
#   建议把utf-8编码(string-->bytes)操作放在cython/C++侧, 这样比python侧编码更快更紧凑.
#   计数统一使用bytes类型, 在python侧拿到global_counter之后落盘前, 根据输出类型统一转换成目标类型（整数/bytes/字符）

def bow_worker(pq_fpath, text_colname, ENDOFTEXT, split_pattern):
    '''
    执行BoW的进程: 分发得到 parquet 文件地址, 遍历 batch, 对每一个 batch 执行 split / count, 累积更新至 local_counter
    '''
    local_counter = Counter()
    pf = pq.ParquetFile(pq_fpath)

    for batch in pf.iter_batches(batch_size = 65536, columns=[text_colname]):
        batch_text = batch[text_colname].to_pylist()
        # accelerate by Cython/C++: 
        # input py-obj: list of text(string)
        batch_counts = split_count_batch(ENDOFTEXT.join( batch_text ), split_pattern) # list of strings ->预切分/编码/计数-> dict of {bytes: uint64} 
        # output py-obj: dict of {bytes: uint64} 
        local_counter.update(batch_counts)
    
    return local_counter



def get_BoW(
    corpora: t.List[str]|str,
    column: str|None,
    ENDOFTEXT: str,
    split_pattern: str,
    word_format: t.Literal['string', 'binary', 'u32list'],
    bow_save_path: str|None,
    bow_save_colnames: tuple[str, str]|None = ['word', 'freq'],
    compression: str = 'snappy') -> tuple|None:
    '''
    args:
    :param corpora: 语料. 可以是 1. parquet文件列表  2. 语料文本
    :param column: 语料 corpora 中代表文本的 parquet 列名. 若column is None, 说明corpora是语料文本而不是路径
    :split_pattern: 预切分文本的正则表达式
    :word_format: 输出保存的 BoW parquet文件中 word 的格式. string / bytes / list of uint32
    :bow_save_path: BoW保存为 parquet文件的 路径. 如果None, 说明不需要保存, 直接返回 BoW as tuple of words & freqs
    :bow_save_colnames: 如果保存BoW为parquet, 那么 save_colnames 是parquet文件的列名

    returns: 生成corpora语料的词袋BoW
        save_path is None, 那么返回 tuple of np.ndarray 形式的BoW (words, freqs); else via save_path, 那么保存BoW为parquet, 一列words & 一列freqs
        word_format指定了word的格式: 字符串 / 二进制字节 / u32序列
    '''
    if isinstance(corpora, str):
        # 若 column not None 且 corpora 作为路径 --> 组装成list
        if column is not None and os.path.isfile(corpora) and os.path.exists(corpora) and corpora.endswith('.parquet'):
            corpora = [corpora]
        # 若 column is None, corpora 作为语料文本 --> 保存成parquet文件
        else:
            if not corpora.endswith(ENDOFTEXT):
                corpora = corpora + ENDOFTEXT
            chunks_str = re.findall(split_pattern, corpora) # list of tokens(string)
            BoW = Counter(chunks_str)
            words, freqs = BoW.keys(), list( BoW.values() )
            if word_format == 'binary':
                words = [word.encode('utf-8') for word in words] # list of bytes
                word_dtype = pa.binary()
            elif word_format == 'u32list':
                words = [list(word.encode('utf-8')) for word in words] # list of int_lists
                word_dtype = pa.large_list(pa.field('token', pa.uint32()))
            else:
                words = list(words) # list of strings
                word_dtype = pa.string()
            
            if not bow_save_path:
                return words, freqs
            
            BoW_schema = pa.schema([
                pa.field(bow_save_colnames[0], word_dtype),
                pa.field(bow_save_colnames[1], pa.uint64()),
            ])
            table = pa.Table.from_pydict({
                bow_save_colnames[0]: words,
                bow_save_colnames[1]: freqs,
            }, schema = BoW_schema)

            pq.write_table(table, bow_save_path, compression = compression)
            return

    # 0 对 corpora 执行 sharding 分片, 直接存储在 BoW 所在文件夹
    shards_dir = os.path.dirname(bow_save_path)
    shard_pq_to_ds(corpora, shards_dir, shard_size_mb = 2048, compression = compression, write_metadata=False)
    shards = [ os.path.join(shards_dir, pf) for pf in os.listdir(shards_dir) if pf.endswith('.parquet')]

    # 1 进程池并行对每个 shard 执行 worker, 主进程收集 worker 的 local_BoW 至 global_BoW
    global_BoW = Counter()

    with ProcessPoolExecutor() as executor:
        # map.worker: 可以使用 cython/C++ 加速: 使用 C++ 的unordered_map计数, local_BoW计数完毕后, 遍历之至python dict对象.
        stream_parallel_process_with_pending(
            executor = executor,
            data_gen = iter(shards),
            process_fn = bow_worker,  # 初始化一个local_BoW, 读取parquet文件后 iter_batches遍历之, 对batch作切分-编码-计数->local_BoW
            result_handler = global_BoW.update, # 在父进程 global_BoW.update(local_BoW) 累积收集分片的 BoW
            max_pending = 12,
            process_args = (column, ENDOFTEXT, split_pattern)
        )

    # 2 输出格式转换: 这里 global_BoW 的 words 类型是 binary bytes
    words, freqs = global_BoW.keys(), list( global_BoW.values() )
    if word_format == 'binary':
        words = list(words) # list of bytes
        word_dtype = pa.binary()
    elif word_format == 'u32list':
        words = [list(word) for word in words] # list of u32_lists
        word_dtype = pa.large_list(pa.field('token', pa.uint32()))
    else:
        words = [word.decode('utf-8') for word in words] # list of strings
        word_dtype = pa.string()

    # 3 落盘 global_BoW 的 words 和 freqs 为 parquet
    if not bow_save_path:
        return words, freqs
    
    BoW_schema = pa.schema([
        pa.field(bow_save_colnames[0], word_dtype),
        pa.field(bow_save_colnames[1], pa.uint64()),
    ])
    table = pa.Table.from_pydict({
        bow_save_colnames[0]: words,
        bow_save_colnames[1]: freqs,
    }, schema = BoW_schema)

    pq.write_table(table, bow_save_path, compression = compression)
    return