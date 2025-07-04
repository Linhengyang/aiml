import os
import typing as t
import pyarrow.parquet as pq
import pyarrow as pa
import os
import regex as re
from collections import defaultdict

def yield_parquet_batch(file_path: str, batch_size: int, columns: t.List[str]):
    """
    从 Parquet 文件中分批读取数据。

    Args:
        file_path (str): Parquet 文件的路径。
        batch_size (int): 每次读取的行数。
        columns: 每次读取的列集（列名列表）

    Yields:
        parquet.RecordBatch: 包含当前批次数据的 pyarrow record batch
        use __getitem__(field name) to get column data
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Error: File '{file_path}' not found.")
    
    parquet_file = pq.ParquetFile(file_path)
    for batch in parquet_file.iter_batches(batch_size=batch_size, columns=columns):
        yield batch


# sample1   sample2
# [[1,2,3], [2,3,4,5],...]

# def build_pyarrow_table(row_data:t.List, schema):
#     # row_data is data in row: len(row_data) = sample_size, row_data[0] is first sample, len(row_data[0]) is num_field
#     # to build table, get offsets and values
#     offset, current_offset = [0], 0
#     for row in row_data:
#         current_offset += len(row)
#         offset.append(current_offset)

#     values_array = pa.array([v for row in row_data for v in row], type=schema.field(0).type.value_type)
#     list_array = pa.ListArray.from_arrays(offset, values_array)

#     data_table = pa.Table.from_arrays([list_array], schema=schema)
#     return data_table


def build_pyarrow_table_from_row_data(row_data:t.List, schema):
    # row_data is data in row: len(row_data) = sample_size, row_data[0] is first sample, len(row_data[0]) is num_field
    num_fields = len(schema)
    col_data =[[] for _ in range(num_fields)] # [[col1_data], [col2_data],...]
    offsets = {} # {field j1: [0, 3, 6, ..],  field j2: [0, 4, 8, 12, ..]}

    for i, row in enumerate(row_data): # row can be [ 1, 3.14, 'A', [3,2,1,0] ] for 4 fields
        assert len(row) == num_fields, f'row {i} length not match with schema num_fields {num_fields}'

        for j in range(num_fields):
            if isinstance(row[j], t.List|t.Tuple): # 如果 field j 是一个序列 # 尚未测试 空序列
                try:
                    offsets[j].append( offsets[j][-1] + len(row[j]) )
                except KeyError:
                    offsets[j] = [0, len(row[j])]

                col_data[j].extend( row[j] )
            else: # 如果 field j 是一个标量
                col_data[j].append(row[j])

    for j in range(num_fields):
        if j in offsets:
            values_array = pa.array(col_data[j], type=schema.field(j).type.value_type)
            col_array = pa.ListArray.from_arrays(offsets[j], values_array)
        else:
            col_array = pa.array(col_data[j])
        col_data[j] = col_array

    data_table = pa.Table.from_arrays(col_data, schema=schema)
    return data_table













if __name__ == "__main__":
    row1 = [ 1, 3.14, 'A', [3,2,1,0] ]
    row2 = [ -1, 0.00, 'C', [0,1,0] ]
    row3 = [0, -4.89, 'Y', [1,]]
    row_data = [row1, row2, row3]
    list_of_ints_type = pa.list_(pa.field('token', pa.int32()))
    schema = pa.schema([ pa.field('int', pa.int32()), pa.field('float', pa.float32()), pa.field('char', pa.string()), pa.field('tokens', list_of_ints_type) ])
    print( build_pyarrow_table_from_row_data(row_data, schema) )