from .url import fetch_url

METADATA = {
    "s3e26": {"target_col": "Status", "task": "classification"},
    "s3e25": {"target_col": "Hardness", "task": "regression"},
    "s3e24": {"target_col": "smoking", "task": "classification"},
    "s3e23": {"target_col": "defects", "task": "classification"},
    "s3e22": {"target_col": "outcome", "task": "classification"},
    "s3e20": {"target_col": "emission", "task": "regression"},
    "s3e19": {"target_col": "num_sold", "task": "regression"},
    "s3e17": {"target_col": "Machine failure", "task": "classification"},
    "s3e16": {"target_col": "Age", "task": "regression"},
    "s3e14": {"target_col": "yield", "task": "regression"},
    "s3e13": {"target_col": "prognosis", "task": "classification"},
    "s3e12": {"target_col": "target", "task": "classification"},
    "s3e11": {"target_col": "cost", "task": "regression"},
    "s3e10": {"target_col": "Class", "task": "classification"},
    "s3e9": {"target_col": "Strength", "task": "regression"},
    "s3e8": {"target_col": "price", "task": "regression"},
    "s3e7": {"target_col": "booking_status", "task": "classification"},
    "s3e6": {"target_col": "price", "task": "regression"},
    "s3e5": {"target_col": "quality", "task": "regression"},
    "s3e4": {"target_col": "Class", "task": "classification"},
    "s3e3": {"target_col": "Attrition", "task": "classification"},
    "s3e2": {"target_col": "stroke", "task": "classification"},
    "s3e1": {"target_col": "MedHouseVal", "task": "regression"},
    "s2e9": {"target_col": "num_sold", "task": "regression"},
}


def fetch_playground_series(
    season,
    episode,
    return_X_y=False,
    data_home=None,
    force=False,
    cleanup=lambda d: d.drop("id"),
):
    url = f"https://github.com/koaning/benchy/raw/main/datasets/playground-series-s{season}e{episode}.parquet"
    identifier = f"s{season}e{episode}"

    # Confirm we have the metadata if the user needs X, y pairs.
    target_col = None
    if return_X_y:
        try:
            target_col = METADATA[identifier]["target_col"]
        except KeyError:
            raise KeyError(
                f"Can't find {identifier} in our lookup. It's possible that the metadata still needs to be added manually to the project."
            )

    return fetch_url(
        url,
        name=f"kaggle-playground-{identifier}",
        target_col=target_col,
        return_X_y=return_X_y,
        data_home=data_home,
        force=force,
        cleanup=cleanup,
    )


fetch_playground_series(3, 26)
