# test.py
import pyarrow as pa
import pyarrow.parquet as pq
import pyarrow.dataset as ds
import regex as re
import os
import typing as t
import sys

valid_pq = '../../data/TinyStories/raw/validation.parquet'
train_pq = '../../data/TinyStories/raw/train.parquet'
raw_pq_dir = '../../data/TinyStories/raw/'




if __name__ == "__main__":
    print(sys.path)