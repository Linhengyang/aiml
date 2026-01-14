# test.py
import pyarrow as pa
import pyarrow.parquet as pq
import pyarrow.dataset as ds
import regex as re
import os
import typing as t
from src.core.utils.file.folder_op import clean_folder
from src.core.utils.text.tokenizer import bufferBBPE_u16Tokenizer

valid_pq = '../../data/TinyStories/raw/validation.parquet'
train_pq = '../../data/TinyStories/raw/train.parquet'
raw_pq_dir = '../../data/TinyStories/raw/'




if __name__ == "__main__":
    pass