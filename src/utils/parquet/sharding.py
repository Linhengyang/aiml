# sharding.py
# parquet大文件并行处理是刚需, pq.parquetFile对象既不可序列化（无法多进程分发），又线程不安全（无法多线程共享）.
# sharding分片之后, 即可以作为独立的parquet文件, 又可以放一起作为pq.dataset。前者可以多进程读取pq文件, 后者可以多线程读取fragment


import os
import typing as t
import pyarrow.parquet as pq
import pyarrow as pa
from pyarrow import dataset as ds
from ..file_utils.remove import clean_folder


def conformity_schema(pq_files: t.List[str]) -> pa.Schema:
    '''
    验证所有 parquet 文件的 schema 是否一致
    args: pq_files, parquet文件路径列表
    return: 一致的schema
    raise: value error 如果schema不一致
    '''
    if not pq_files:
        raise ValueError('no parquet files provided')
    
    reference_schema = pq.read_schema(pq_files[0])
    for i, pf in enumerate(pq_files[1:]):
        current_schema = pq.read_schema(pf)
        if not reference_schema.equals(current_schema, check_metadata=False):
            raise ValueError(
                f'schema mismatch\n'
                f'File 0: {pq_files[0]} with {reference_schema}\n'
                f'File {i}: {pf} with {current_schema}\n'
            )
    
    return reference_schema


def shard_pq_to_ds(
        pq_files: t.List[str],
        save_dir: str,
        shard_row_size: int|None = None,
        shard_size_mb: int|None = None,
        compression: str = 'snappy',
        write_metadat: bool = True):
    '''
    将schema 一致的 Parquet文件 分片为一个 Parquet Dataset
    args:
        pq_files: Parquet文件路径列表, 要求schema一致
        save_dir: 输出的 pq.Dataset目录, 内部含有 'fname-part-index.parquet' 的分片文件
        shard_row_size: 每个分片的行数(与shard_size_mb二选一)
        shard_size_mb: 每个分片的目标大小(与shard_row_size二选一)
        compression: 压缩算法('snappy', 'gzip', 'zstd', 'none')
    '''
    if not pq_files:
        raise ValueError('no parquet files provided')
    if (shard_row_size is None) == (shard_size_mb is None):
        raise ValueError(f'args `shard_row_size` and `shard_size_mb` must be chosen 1 from 2')
    
    # 0. 检查 parquet文件的 schema 一致. 统计总行数 / 一致schema / 总size
    num_total_rows, total_size = 0, 0
    reference_schema = pq.read_schema(pq_files[0])
    for pf in pq_files:
        # 检查 schema 一致性
        curr_schema = pq.read_schema(pf)
        if not reference_schema.equals(curr_schema, check_metadata=False):
            raise ValueError(
                f'schema mismatch\n'
                f'File 0: {pq_files[0]} with {reference_schema}\n'
                f'File {i}: {pf} with {curr_schema}\n'
            )
        # 累加行数
        num_total_rows += pq.read_metadata(pf).num_rows
        # 累加文件大小(如果需要 shard_size_mb 来限制 shard)
        if shard_size_mb:
            total_size += os.path.getsize(pf)
    bytes_per_row = total_size // num_total_rows

    # 1. 确定 shard_row_size. 检查 总shard数量不能超过 10000
    if shard_size_mb:
        shard_row_size = max(1024, shard_size_mb*1024*1024 // bytes_per_row)
    assert num_total_rows // shard_row_size < 10000, \
        f'shard limitation {str(shard_size_mb)+'mb/shard' if shard_size_mb else shard_row_size + 'rows/shard'} too small'\
        f'which leads more than 10000 fragments in dataset.'
    
    # 1. 创建并保证清空 save_dir
    os.makedirs(save_dir, exist_ok=True)
    clean_folder(save_dir, method='all', keep=True)

    # 2. 写入 common_metadata
    pq.write_metadata(reference_schema, os.path.join(save_dir, "_common_metadata"))

    # 3. 分片写入
    for pf in pq_files:
        pq_file = pq.ParquetFile(pf)
        for i, batch in enumerate( pq_file.iter_batches(shard_row_size) ):
            if batch.num_rows == 0:
                continue
            shard_data = pa.Table.from_batches([batch], reference_schema)
            shard_path = os.path.join(save_dir, f'{os.path.basename(pf)}-part-{i:04d}.parquet')
            pq.write_table(shard_data, shard_path, compression = compression, write_statistics = True)
    
    # 4. 写入完整 metadata
    if write_metadat:
        # 扫描所有分片, 收集完整元数据
        shards_files = [os.path.join(save_dir, fname) for fname in os.listdir(save_dir) if fname.endswith('.parquet')]
        
        _metadata_list = []
        for pf in shards_files:
            pq_file = pq.ParquetFile(pf)
            _metadata_list.append( pq_file.metadata )
        merged_metadata = pq.merge_parquet_metadat(_metadata_list)

        pq.write_metadata(reference_schema, os.path.join(save_dir, "_metadata"), metadata_collector = merged_metadata)