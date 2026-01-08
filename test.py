# test.py
import pyarrow as pa
import pyarrow.parquet as pq
import pyarrow.dataset as ds
import regex as re
import os
import typing as t
from src.core.utils.file.folder_op import clean_folder

valid_pq = '../../data/TinyStories/raw/validation.parquet'
train_pq = '../../data/TinyStories/raw/train.parquet'
raw_pq_dir = '../../data/TinyStories/raw/'

token_dtype = pa.uint16()
tokens_schema = pa.schema([
    pa.field( 'tokens', pa.large_list(pa.field('token', token_dtype)) ),
    ])
p_counts_schema = pa.schema([
    pa.field('L', token_dtype),
    pa.field('R', token_dtype),
    pa.field('counts', pa.uint64()),
    ])
buffer_dir = '../cache/temp/buffer'
buffer_tokens_dir = os.path.join(buffer_dir, 'tokens')
buffer_pcounts_dir = os.path.join(buffer_dir, 'p_counts')

GPT4_TOKENIZER_REGEX = \
    r"""'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]++[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+"""


ENDOFTEXT = '<|endoftext|>'

def encode_to_ints(s:str, encoding='utf-8') -> t.List[int]:
    return list( s.encode(encoding) )


def text_to_tokens_pa_table(pre_split_pat, text):
    if not text.endswith(ENDOFTEXT):
        text = text + ENDOFTEXT
    chunks_str = re.findall(pre_split_pat, text) # list of tokens(string)
    
    # list of list of integers(every list of integers as tokens)
    chunks_tokens = [encode_to_ints(chunk) for chunk in chunks_str]
    
    # 创建 pa table
    batch_table = pa.Table.from_pydict({tokens_schema[0].name: chunks_tokens}, tokens_schema)
    return batch_table


def yield_parquet_batch(file_path: str, batch_size: int, columns: t.List[str]|None=None):
    """
    从 Parquet 文件中分批读取数据。

    Args:
        file_path (str): Parquet 文件的路径。
        batch_size (int): 每次读取的行数。
        columns: 每次读取的列集（列名列表）。输入代表读取全部列

    Yields:
        parquet.RecordBatch: 包含当前批次数据的 pyarrow record batch
        use __getitem__(field name) to get column data
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Error: File '{file_path}' not found.")
    
    parquet_file = pq.ParquetFile(file_path)
    for batch in parquet_file.iter_batches(batch_size=batch_size, columns=columns):
        yield batch



if __name__ == "__main__":
    # step i, i = 0 <-> num_merges-1  --->  step i+1

    # parquet dataset as /tokens/i  ---> shard read batch
    #   ---> map to executor to compute ---> part_pair_count
    # ---> write independently with metadata --> parquet dataset as /p_counts/i

    # --aggregate parquet dataset /p_counts/i ---> to_merge_pair, new_token

    # parquet dataset as /tokens/i ---> shard read batch
    #   ---> map to executor to compute, along with to_merge_pair, new_token ---> part_merged_tokens
    # ---> write independently with metadata --> parquet dataset as /tokens/i+1


    # 性能分析
    # 1. 从 raw parquet files 生成 parquet dataset of /tokens/0
    # 第一步控制了 文件数量, 后续所有 p_counts/i 和 tokens/i+1 的文件数量就都被确定了
    # num_batches = num_rows_after_chunk / batch_size   所以第0步, 复用原有代码, pre-chunk, byte-init-tokenizer raw data

    # 所以batch_size非常重要: 具体确定步骤如下:
    # 1.pre_chunk-init_tokenize all parquet corpus files. get the total_num_rows_after_chunk
    # 2.确定 batch_size/num_batches : 控制 dataset 内部单个 parquet 文件大小在 256MB - 1GB <-- dataset并发读取, 单个pq文件大小比控制pq文件总数更重要
    #   avg_num_tokens_per_chunk, 英文约为5-10, 中文约为60.
    #   token_bytes, 小词表(大小<=65536)为2, 全尺寸词表(大小>65536)为4
    #   single paritial(from batch) file size: 
    #       tokens = batch_size * avg_num_tokens_per_chunk * token_bytes
    #       pcounts <= batch_size * vocab_size^2 * (2*token_bytes + 8)
    #   由于 pcounts 的 生成完全是由客观情况决定的, 且 pcounts dataset 只涉及 并发单文件写入 和 聚合统计 过程, 不涉及并发读取, 所以就不控制其大小了
    #   ----> 控制 batch_size * avg_num_tokens_per_chunk * token_bytes 在 256MB ~ 1024MB 之间. 中/英 语料 avg_num_tokens_per_chunk 分别取64/8
    #       for英文&小词表, batch_size=(16~64)M, for英文&大词表, batch_size=(8~32)M,
    #       for中文&小词表, batch_size=(2~8)M, for中文&大词表, batch_size=(1~4)M

    # 3.确定写入的 row_group_size: 由于 batch_size 完整地落在 valid row group size 区间内, 直接让 row_group_size = batch_size 是有利于并发的
    # 4.确定工作进程/线程数量: 根据 总可用内存, 和 batch_size 大小的批数据 单任务计算 所需的内存, 得到 任务并行数目
    #   merge任务:
    #     input:
    #       tokens_flat:        num_tokens个uint16/uint32 = batch_size * avg_num_tokens_per_chunk * 2/4 bytes
    #       offsets:            batch_size个int64         = batch_size * 8 bytes
    #     output:
    #       merged_tokens_lens: batch_size个int64         = batch_size * 8 bytes
    #       filter:             num_tokens个bool          = batch_size * avg_num_tokens_per_chunk * 1 bytes
    #       merged_tokens_flat: num_tokens个uint16/uint32 = batch_size * avg_num_tokens_per_chunk * 2/4 bytes
    #   --> merge任务总结:
    #       for英文&小词表, 56*batch_size(内存池32*batch_size) bytes; for英文&大词表, 88*batch_size(内存池48*batch_size) bytes
    #       for中文&小词表, 336*batch_size(内存池200*batch_size) bytes; for中文&大词表, 592*batch_size(内存池328*batch_size) bytes
    #   count任务:
    #     input:
    #       L_tokens:        num_tokens个uint16/uint32 = batch_size * avg_num_tokens_per_chunk * 2/4 bytes
    #       R_tokens:        num_tokens个uint16/uint32 = batch_size * avg_num_tokens_per_chunk * 2/4 bytes
    #     output:
    #       keys:            num_tokens个uint32/uint64 = batch_size * avg_num_tokens_per_chunk * 4/8 bytes
    #       L_uniqs:         num_tokens个uint32/uint64 = batch_size * avg_num_tokens_per_chunk * 2/4 bytes
    #       R_uniqs:         num_tokens个uint32/uint64 = batch_size * avg_num_tokens_per_chunk * 2/4 bytes
    #       counts:          num_tokens个uint64 = batch_size * avg_num_tokens_per_chunk * 8 bytes
    #   --> count任务总结:
    #       for英文&小词表, 160*batch_size(内存池128*batch_size) bytes; for英文&大词表, 256*batch_size(内存池192*batch_size) bytes
    #       for中文&小词表, 1280*batch_size(内存池1024*batch_size) bytes; for中文&大词表, 2048*batch_size(内存池1536*batch_size) bytes
    #   显然 count任务 所需的内存 远大于 merge任务. 由于内存池复用, 只需考虑 count任务.
    #   单工作线程/进程任务计算 所需内存 如下:
    #       for英文&小词表, 2.5~10GB(内存池2~8GB); for英文&大词表, 2~8GB(内存池1.5~6GB)
    #       for中文&小词表, 2.5~10GB(内存池2~8GB); for中文&大词表, 2~8GB(内存池1.5~6GB)

    # 5.以本机 64GB 总内存, 80% 内存水位 即 51.2GB 可用工作内存来评估:
    #   英文&小词表: batch_size = 16M, num_wokers = 20; batch_size = 32M, num_wokers = 10; batch_size = 48M, num_wokers = 7; batch_size = 64M, num_wokers = 5
    #   英文&大词表: batch_size = 8M, num_wokers = 25; batch_size = 16M, num_wokers = 12; batch_size = 24M, num_wokers = 8; batch_size = 32M, num_wokers = 6
    #   中文&小词表: batch_size = 2M, num_wokers = 20; batch_size = 4M, num_wokers = 10; batch_size = 6M, num_wokers = 7; batch_size = 8M, num_wokers = 5
    #   中文&大词表: batch_size = 1M, num_wokers = 25; batch_size = 2M, num_wokers = 12; batch_size = 3M, num_wokers = 8; batch_size = 4M, num_wokers = 6

    batch_size = 32 * 1024 * 1024 # 当下是 英文语料+小词表, 选用 32M 作为 batch_size
    row_group_size = batch_size

    # 遍历每个 text parquet file, 分批 转换成 一个 tokens(uint16) parquet file: init_tokens_pq
    init_tokens_pq = os.path.join(buffer_dir, 'byte_tokens.parquet')


    # corpus_files = os.listdir(raw_pq_dir)
    # text_colnames = ['text', ]*len(corpus_files)

    # with pq.ParquetWriter(init_tokens_pq, tokens_schema) as writer:
    #     for (corpus_pq, text_col) in zip(corpus_files, text_colnames):
    #         corpus_path = os.path.join(raw_pq_dir, corpus_pq)

    #         corpus_batch_iter = yield_parquet_batch(corpus_path, batch_size, [text_col])
    #         for k, batch in enumerate(corpus_batch_iter):
    #             text = ENDOFTEXT.join( batch[text_col].to_pylist() )
    #             batch_table = text_to_tokens_pa_table(GPT4_TOKENIZER_REGEX, text)
    #             writer.write_table(batch_table, row_group_size)
    
    # # 从 init_tokens(parquet文件) 到 init dataset: tokens/0
    # def _init_tokens_dataset(init_tokens_pq):
    #     # 0. 检查 总的batch数量不能超过 10000
    #     num_total_rows = pq.read_metadata(init_tokens_pq).num_rows
    #     assert num_total_rows // batch_size < 10000, \
    #         f'batch_size {batch_size} too small for parquet file {init_tokens_pq}, which leads more than 10000 fragments in dataset.'
        
    #     print(f'initalizing tokens dataset at merge 0: {num_total_rows // batch_size} fragments with batch_size {batch_size}')

    #     # 1. 创建并清空 init_tokens_ds
    #     init_tokens_ds = os.path.join(buffer_tokens_dir, '0')
    #     os.makedirs(init_tokens_ds, exist_ok=True)

    #     # 清空但保留 init_tokens_ds
    #     clean_folder(init_tokens_ds, method='all', keep=True)

    #     # 2. 写入 common_metadata
    #     pq.write_metadata(schema=tokens_schema, where=os.path.join(init_tokens_ds, "_common_metadata"))

    #     # 3. 以 batch_size 为 批大小, 遍历 init_tokens_pq, 并将 batch data 作为 fragment 写入 init_tokens_ds
    #     # TODO: 改造成多线程/多进程
    #     for i, batch in enumerate( yield_parquet_batch(init_tokens_pq, batch_size, [tokens_schema[0].name]) ):
    #         b_table = pa.Table.from_batches([batch], schema=tokens_schema)
    #         b_path = os.path.join(init_tokens_ds, f'part-{i:04d}.parquet')
    #         pq.write_table(b_table, b_path)
        
    #     # 4. 写入完整 metadata --> 跳过

    #     return init_tokens_ds
    

    # tokens_ds_path = _init_tokens_dataset(init_tokens_pq)
    tokens_ds = ds.dataset( os.path.join(buffer_tokens_dir, '0') )
    fragments = list(tokens_ds.get_fragments())
    

    # count map 任务: 多进程版本, 分发 tokens dataset里的 fragments parquet文件路径给 process_pool
    # fragment_paths = [f.path for f in fragments]
    # map(_task, fragment_paths, *args, **kwargs)
    def _map2process_read_count_write(fragment_path, count_func, save_dir):
        # parquet path 直接读成 recordBatch
        # if recordBatch 空, 直接结束 本任务
        # 执行 count_pair_batch_c_extension, 生成 p_counts batch
        # 组装 p_counts batch to table, write p_counts table to save dir with same partID
        pass

    # count map 任务:
    # 多线程版本, 为了绕开GIL, 应该在主线程处理dataset里的 fragments(arrow对象) 成non-GIL 函数 tls_count_non_gil_func 的输入 tokens_flat, offsets
    # 然后主线程应该收集 tls_count_non_gil_func 的输出 b_pcounts 作聚合
    def _map2thread_read_count_write(fragment, tls_count_non_gil_func, save_dir):
        # fragment 只能读成 table, table无法像 recordBatch 一样直接读 values 和 offsets, 需要先拼接chunk
        table = fragment.to_table()
        tokens_col = table.column(0) # 只有一列
        if tokens_col.num_chunks == 1:
            arr = tokens_col.chunk(0)
        else:
            arr = tokens_col.combine_chunks() # 合并所有chunk
        
        # 取到 tokens_flat / offsets as numpy arr, b_order from fragment
        tokens_flat = arr.values.to_numpy()
        offsets = arr.offsets.to_numpy()
        b_order = None #TODO: 从 fragment 中解析出 

        # 计算 pair count: b_pcounts tuple of L, R, counts arrays
        b_pcounts = tls_count_non_gil_func(tokens_flat, offsets)



    # count reduce 任务: 聚合统计 p_counts dataset，得到 max_occur_pairs

    # merge map 任务: 多进程版本, 分发 tokens dataset里的 fragments parquet文件路径给 process_pool
