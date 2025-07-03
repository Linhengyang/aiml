# test.py
from src.core.design.producer_consumer import *

import pandas as pd
import pyarrow.parquet as pq
import os

def create_sample_parquet_file(file_path: str, num_rows: int = 10000):
    """
    创建一个大的示例 Parquet 文件用于演示。
    """
    print(f"Creating a sample Parquet file with {num_rows} rows...")
    data = {
        'id': range(num_rows),
        'name': [f'User_{i}' for i in range(num_rows)],
        'value': [i * 0.1 for i in range(num_rows)],
        'category': ['A', 'B', 'C', 'D'] * (num_rows // 4)
    }
    df = pd.DataFrame(data)
    df.to_parquet(file_path, index=False)
    print(f"Sample file '{file_path}' created successfully.")

def read_parquet_in_batches(file_path: str, batch_size: int = 1000):
    """
    从 Parquet 文件中分批读取数据。

    Args:
        file_path (str): Parquet 文件的路径。
        batch_size (int): 每次读取的行数。

    Yields:
        pd.DataFrame: 包含当前批次数据的 Pandas DataFrame。
    """
    if not os.path.exists(file_path):
        print(f"Error: File '{file_path}' not found.")
        return

    print(f"\nStarting to read '{file_path}' in batches of {batch_size} rows...")
    
    # 使用 pyarrow.parquet.ParquetFile 打开文件
    parquet_file = pq.ParquetFile(file_path)
    
    # 遍历文件的所有行组 (Row Groups)
    # Parquet 文件内部数据按 Row Group 组织，每个 Row Group 包含一部分行
    # iter_batches 方法可以让你从 Row Group 中进一步分批读取 RecordBatch
    for i, batch in enumerate(parquet_file.iter_batches(batch_size=batch_size)):
        # 将 PyArrow RecordBatch 转换为 Pandas DataFrame
        df_batch = batch.to_pandas()
        print(f"  Read batch {i+1}: {len(df_batch)} rows.")
        yield df_batch # 返回当前批次的 DataFrame

    print(f"Finished reading all batches from '{file_path}'.")


if __name__ == "__main__":
    test_file = 'large_sample.parquet'
    
    # 1. 创建一个大的示例文件
    create_sample_parquet_file(test_file, num_rows=25345) # 使用一个不是 batch_size 倍数的行数

    # 2. 以每次 5000 行的方式读取文件
    batch_size = 5000
    for i, df_chunk in enumerate(read_parquet_in_batches(test_file, batch_size)):
        print(f"Processing chunk {i+1}: First 3 rows of this chunk:\n{df_chunk.head(3)}")
        # 在这里你可以对 df_chunk 进行任何处理，例如：
        # - 写入数据库
        # - 进行聚合计算
        # - 过滤并保存到新文件
        # - 发送到其他服务
        
        # 模拟一些处理时间
        # import time
        # time.sleep(0.1)

    print("\nAll chunks processed!")

    # 3. 清理示例文件
    os.remove(test_file)
    print(f"Cleaned up '{test_file}'.")