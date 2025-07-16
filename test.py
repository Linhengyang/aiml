# test.py
import sys
import typing as t
import pyarrow.parquet as pq
import pyarrow as pa
import pyarrow.compute as pc
import os
import regex as re
import functools
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed, Future
import pandas as pd
# from src.core.utils.file.parquet_io import build_pyarrow_table_from_row_data

cache_play = '../cache/playground/'





if __name__ == "__main__":
    # sample_pq = os.path.join(cache_play, "sample_raw.parquet")
    # sample_tokens_pq = os.path.join(cache_play, "sample_tokens.parquet")
    # batch_size = 3

    # # 测试 parquet write 是否覆盖原文件
    # pd.read_parquet(sample_tokens_pq)
    # p_counts_1 = {(1,2):10, (3,4):20}
    # p_counts_2 = {(1,2):15, (3,4):5, (6,7):1}
    # p_counts_3 = {}
    p_counts_schema = pa.schema([
            pa.field('L', pa.int32()),
            pa.field('R', pa.int32()),
            pa.field('counts', pa.int64()),])
    # p_counts = [p_counts_1, p_counts_2, p_counts_3]
    # for i, part_pcounts in enumerate(p_counts):
    #     datapoints = [ [l, r, count] for (l, r), count in part_pcounts.items() ]
    #     with pq.ParquetWriter(f'test_pcounts_{i}.parquet', p_counts_schema) as writer:
    #         writer.write_table( build_pyarrow_table_from_row_data(datapoints, p_counts_schema) )
    
    # # read partial p_counts
    # # agg_p_counts = pa.Table.from_pydict({field.name:[] for field in p_counts_schema}, p_counts_schema)
    # p_counts_collect = []
    # p_counts_pq = [f'test_pcounts_{i}.parquet' for i in range(3)]
    # for i, p_counts_pq in enumerate(p_counts_pq):
    #     part_p_counts = pq.read_table(p_counts_pq) # pa table of (l,r), count
    #     p_counts_collect.append( part_p_counts )
    # p_counts = pa.concat_tables( p_counts_collect )
    # agg_p_counts = p_counts.group_by(['L', 'R']).aggregate([('counts', 'sum')]) # counts 列 --> counts_sum 列
    # # print(agg_p_counts)

    # filter_mask = pc.equal( agg_p_counts['counts_sum'], pc.max(agg_p_counts['counts_sum']).as_py() )
    # occur_most_row = agg_p_counts.filter(filter_mask).slice(0, 1)
    # print(occur_most_row)
    # occur_most_pair = (occur_most_row['L'][0].as_py(), occur_most_row['R'][0].as_py())
    # occurence = occur_most_row['counts_sum'][0].as_py()
    # print(occur_most_pair, occurence)
    # data = {'a':[1,2,3], 'b':[4,5,6]}
    # schema = pa.schema([pa.field('a', pa.int32()), pa.field('R', pa.int32())])
    # with pq.ParquetWriter(f'test_data.parquet', schema) as writer:
    #     writer.write_table(pa.Table.from_pydict(data, schema))
    # import pandas as pd
    # df = pd.read_parquet('test_data.parquet')
    # print(df)
    p_counts = {(1,2):15, (3,4):5, (6,7):1}
    col1, col2 = list( zip(*p_counts.keys()) )
    counts = list(p_counts.values())
    print(col1)
    print(col2)
    print(counts)


