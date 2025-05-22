# run some simple scripts
import pandas as pd
import os
resource = '../../resource/'







from datasets import load_dataset # type: ignore

wikitext2 = '../../data/WikiText2/raw/'

ds = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", cache_dir=resource)

ds['train'].to_parquet(os.path.join(wikitext2, "train.parquet"))

ds['test'].to_parquet(os.path.join(wikitext2, "test.parquet"))

ds['validation'].to_parquet(os.path.join(wikitext2, "validation.parquet"))


