import yaml
import typing as t
import pyarrow.parquet as pq
import pyarrow as pa
import os
import regex as re





configs = yaml.load(open('src/apps/bpe_build/configs.yaml', 'rb'), Loader=yaml.FullLoader)
train_pq = configs['train_pq']
valid_pq = configs['valid_pq']