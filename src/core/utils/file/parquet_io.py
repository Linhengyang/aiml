import os
import typing as t
import pyarrow.parquet as pq
import pyarrow as pa
import os





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




def build_pa_table_from_row(row_data:t.List, schema):
    '''
    row_data is list of data points(row):
        len(row_data) = sample_size, row_data[0] is the first sample, len(row_data[0]) is num_field
    schema must match with data points from row_data
    '''
    num_fields = len(schema)
    col_data =[[] for _ in range(num_fields)] # transfer row data to col data. [[col1_data], [col2_data],...]
    offsets = {} # for possible listarray col data. should be {field j1: [0, 3, 6, ..],  field j2: [0, 4, 8, 12, ..]}

    for i, row in enumerate(row_data):
        # row e.g, 4 fields: [ 9, 3.14, 'A', [3,2,1,0] ]
        assert len(row) == num_fields, f'row {i} length {len(row)} not match with schema num_fields {num_fields}'

        for j in range(num_fields):
            if isinstance(row[j], list|tuple): # 如果 field j 是一个向量 #TODO 尚未测试 空向量
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
    
    return pa.Table.from_arrays(col_data, schema=schema)



def build_pa_table(data:t.List|t.Dict, schema):
    if isinstance(data, list):
        pa_table = build_pa_table_from_row(data, schema)
    elif isinstance(data, dict):
        pa_table = pa.Table.from_pydict(data, schema)
    else:
        raise TypeError(f'data should be list for row_data, or dict for column_data.')
    return pa_table



































































if __name__ == "__main__":
    row1 = [ 1, 3.14, 'A', [3,2,1,0] ]
    row2 = [ -1, 0.00, 'C', [0,1,0] ]
    row3 = [0, -4.89, 'Y', [1,]]
    row_data = [row1, row2, row3]
    list_of_ints_type = pa.list_(pa.field('token', pa.int32()))
    schema = pa.schema([ 
        pa.field('int', pa.int32()), 
        pa.field('float', pa.float32()), 
        pa.field('char', pa.string()), 
        pa.field('tokens', list_of_ints_type) 
        ])
    print( build_pa_table_from_row(row_data, schema) )