import ssl
import polars as pl 
from pathlib import Path 
import urllib.request


BENCHY_DIR = Path.home() / ".benchy"


def download_and_open_datasette(url:str, name:str, data_home:Path, force:bool=False, cleanup=lambda d: d):
    # This is a temporary workaround for mac. 
    # https://stackoverflow.com/a/28052583
    original = ssl._create_default_https_context
    ssl._create_default_https_context = ssl._create_unverified_context

    # Normal behavior from here
    Path(data_home).mkdir(parents=True, exist_ok=True)
    csv_path = data_home / f"{name}.csv"
    pq_path = data_home / f"{name}.parquet"
    if force or not pq_path.exists():
        urllib.request.urlretrieve(url, csv_path)
        pl.read_csv(csv_path).pipe(cleanup).write_parquet(pq_path)
        Path(csv_path).unlink()

    # Put back the SSL setting that we had before
    ssl._create_default_https_context = original
    
    # Now that process is back to normal, return parquet file
    return pl.read_parquet(pq_path)


def fetch_datasette(url, name, data_home=None, return_X_y=False, input_cols=None, target_col=None, force=False, cleanup=lambda d: d):
    data_home = BENCHY_DIR if not data_home else data_home
    df = download_and_open_datasette(url + ".csv?_size=max", name, data_home, force, cleanup=cleanup)
    if return_X_y:
        return df.select(*input_cols), df[target_col]
    return df


def fetch_births_nchs(return_X_y=False, data_home=None, force=False):
    def cleanup(dataf):
        expr = pl.concat_str(
            pl.col("year").cast(pl.String),
            pl.lit("-"),
            pl.col("month").cast(pl.String),
            pl.lit("-"),
            pl.col("date_of_month").cast(pl.String),
        )
        return (dataf.with_columns(date=expr.cast(pl.Date)).select("date", "births"))

    return fetch_datasette(
        url="""https://fivethirtyeight.datasettes.com/fivethirtyeight/births~2FUS_births_1994-2003_CDC_NCHS""", 
        name="births_nchs", 
        return_X_y=return_X_y,
        data_home=data_home,
        force=force,
        cleanup=cleanup)


def fetch_births_ssa(return_X_y=False, data_home=None, force=False):
    def cleanup(dataf):
        expr = pl.concat_str(
            pl.col("year").cast(pl.String),
            pl.lit("-"),
            pl.col("month").cast(pl.String),
            pl.lit("-"),
            pl.col("date_of_month").cast(pl.String),
        )
        return (dataf.with_columns(date=expr.cast(pl.Date)).select("date", "births"))

    return fetch_datasette(
        url="""https://fivethirtyeight.datasettes.com/fivethirtyeight/births~2FUS_births_2000-2014_SSA""", 
        name="births_ssa", 
        return_X_y=return_X_y,
        data_home=data_home,
        force=force,
        cleanup=cleanup)

if __name__ == "__main__":
    print(fetch_births_nchs(force=True))
    print(fetch_births_ssa(force=True))
