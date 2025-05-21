# run some simple scripts
import pandas as pd

wikitext2 = '../../data/WikiText2/test2.parquet'

try:
    df = pd.read_parquet(wikitext2)
    print("Successfully read Parquet file:")
    print(df.head())
    print(f"\nDataFrame shape: {df.shape}")
    print(f"\nDataFrame columns: {df.columns.tolist()}")

except FileNotFoundError:
    print(f"Error: The file '{wikitext2}' was not found.")
except Exception as e:
    print(f"An error occurred: {e}")
