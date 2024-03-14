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
    if force or (not pq_path.exists()):
        urllib.request.urlretrieve(url, csv_path)
        pl.read_csv(csv_path).pipe(cleanup).write_parquet(pq_path)
        Path(csv_path).unlink()

    # Put back the SSL setting that we had before
    ssl._create_default_https_context = original
    
    # Now that process is back to normal, return parquet file
    return pl.read_parquet(pq_path)


def fetch_datasette(url, name, data_home=None, return_X_y=False, target_col=None, force=False, cleanup=lambda d: d):
    data_home = BENCHY_DIR if not data_home else data_home
    df = download_and_open_datasette(url + ".csv?_dl=on&_stream=on&_size=max", name, data_home, force, cleanup=cleanup)
    if return_X_y:
        return df.drop(target_col), df[target_col]
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
        target_col="births",
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
        target_col="births",
        cleanup=cleanup)


def fetch_births_multistate(return_X_y=False, data_home=None, force=False):
    def cleanup(dataf):
        expr = pl.concat_str(
            pl.col("year").cast(pl.String),
            pl.lit("-"),
            pl.col("month").cast(pl.String),
            pl.lit("-"),
            pl.col("day").cast(pl.String),
        )
        return (dataf.with_columns(date=expr.cast(pl.Date), state=pl.col("state").cast(pl.Categorical)).select("state", "date", "births"))

    return fetch_datasette(
        url="""https://calmcode-datasette.fly.dev/calmcode/birthdays""", 
        name="births_multistate", 
        return_X_y=return_X_y,
        data_home=data_home,
        force=force,
        target_col="births",
        cleanup=cleanup)


def fetch_bigmac(return_X_y=False, data_home=None, force=False):
    def cleanup(dataf):
        return (dataf.with_columns(date=pl.col("date").cast(pl.Date), country=pl.col("name").cast(pl.Categorical)).select("date", "country", "dollar_price"))

    return fetch_datasette(
        url="""https://calmcode-datasette.fly.dev/calmcode/bigmac""", 
        name="bigmac", 
        return_X_y=return_X_y,
        data_home=data_home,
        force=force,
        target_col="dollar_price",
        cleanup=cleanup)


def fetch_smoking(return_X_y=False, data_home=None, force=False):
    def cleanup(dataf):
        return (dataf.with_columns(outcome=pl.col("outcome").cast(pl.Categorical), smoker=pl.col("smoker").cast(pl.Categorical)).select("outcome", "smoker", "age"))

    return fetch_datasette(
        url="""https://calmcode-datasette.fly.dev/calmcode/smoking""", 
        name="smoking",
        return_X_y=return_X_y,
        data_home=data_home,
        force=force,
        target_col="outcome",
        cleanup=cleanup)

if __name__ == "__main__":
    print(fetch_smoking())