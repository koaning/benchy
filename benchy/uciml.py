import polars as pl
from pathlib import Path
import ssl
from ucimlrepo import fetch_ucirepo


BENCHY_DIR = Path.home() / ".benchy"


def download_and_open_uci(
    identifier, name: str, data_home: Path, force: bool = False, cleanup=lambda d: d
):
    # This is a temporary workaround for mac.
    # https://stackoverflow.com/a/28052583
    original = ssl._create_default_https_context
    ssl._create_default_https_context = ssl._create_unverified_context

    # Normal behavior from here
    Path(data_home).mkdir(parents=True, exist_ok=True)
    pq_path = data_home / f"{name}.parquet"
    if force or (not pq_path.exists()):
        if isinstance(identifier, int):
            dataset = fetch_ucirepo(id=identifier)
        if isinstance(identifier, str):
            dataset = fetch_ucirepo(name=identifier)
        (
            pl.from_pandas(dataset.data.original)
            .select(pl.all().map_alias(lambda colName: colName.capitalize()))
            .pipe(cleanup)
            .write_parquet(pq_path)
        )

    # Put back the SSL setting that we had before
    ssl._create_default_https_context = original

    # Now that process is back to normal, return parquet file
    return pl.read_parquet(pq_path)


def fetch_uci(
    identifier,
    name,
    data_home=None,
    return_X_y=False,
    target_col=None,
    force=False,
    cleanup=lambda d: d,
):
    data_home = BENCHY_DIR if not data_home else data_home
    df = download_and_open_uci(identifier, name, data_home, force, cleanup=cleanup)
    if return_X_y:
        return df.drop(target_col), df[target_col]
    return df


def fetch_dry_bean(return_X_y=False, data_home=None, force=False):
    return fetch_uci(
        602,
        name="dry_beans",
        return_X_y=return_X_y,
        data_home=data_home,
        force=force,
        target_col="class",
    )


if __name__ == "__main__":
    print(fetch_dry_bean())
