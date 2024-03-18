import zipfile 
import polars as pl
from pathlib import Path

for path in Path("datasets").glob("*/train.csv"):
    name = path.parts[-2]
    print(name)
    pl.read_csv(path).write_parquet(f"datasets/{name}.parquet")

# for path in Path(".").glob("playground-*.zip"):
#     with zipfile.ZipFile(path, 'r') as zip_ref:
#         zip_ref.extractall(f"datasets/{path.stem}/")
