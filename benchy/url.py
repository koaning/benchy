import ssl
import polars as pl
from pathlib import Path
import urllib.request


BENCHY_DIR = Path.home() / ".benchy"


def download_and_open_url(
    url: str, name: str, data_home: Path, force: bool = False, cleanup=lambda d: d
):
    # This is a temporary workaround for mac.
    # https://stackoverflow.com/a/28052583
    original = ssl._create_default_https_context
    ssl._create_default_https_context = ssl._create_unverified_context

    # Normal behavior from here
    Path(data_home).mkdir(parents=True, exist_ok=True)
    pq_path = data_home / f"{name}.parquet"
    if force or (not pq_path.exists()):
        if url.endswith(".csv"):
            csv_path = data_home / f"{name}.csv"
            urllib.request.urlretrieve(url, csv_path)
            pl.read_csv(csv_path).pipe(cleanup).write_parquet(pq_path)
            Path(csv_path).unlink()
        elif url.endswith(".parquet"):
            tmp_path = pq_path.parent / f"temp-{pq_path.parts[-1]}"
            urllib.request.urlretrieve(url, tmp_path)
            pl.read_parquet(tmp_path).pipe(cleanup).write_parquet(pq_path)
            Path(tmp_path).unlink()
        else:
            raise ValueError("url must end with .csv or .parquet")

    # Put back the SSL setting that we had before
    ssl._create_default_https_context = original
    # Now that process is back to normal, return parquet file
    return pl.read_parquet(pq_path)


def fetch_url(
    url,
    name,
    data_home=None,
    return_X_y=False,
    target_col=None,
    force=False,
    cleanup=lambda d: d,
):
    data_home = BENCHY_DIR if not data_home else data_home
    df = download_and_open_url(url, name, data_home, force, cleanup=cleanup)
    if return_X_y:
        return df.drop(target_col), df[target_col]
    return df
