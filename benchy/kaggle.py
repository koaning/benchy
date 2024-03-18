from .url import fetch_url

METADATA = {
    "s3e26": {
        "target_col": "Status",
        "task": "classification"
    },
    "s3e25": {
        "target_col": "Hardness",
        "task": "regression"
    },
    "s3e24": {
        "target_col": "smoking",
        "task": "classification"
    },
    "s3e23": {
        "target_col": "defects",
        "task": "classification"
    },
    "s3e22": {
        "target_col": "outcome",
        "task": "classification"
    }
}

def fetch_playground_series(season, episode, return_X_y=False, data_home=None, force=False, cleanup=lambda d: d.drop("id")):
    url = f"https://github.com/koaning/benchy/raw/main/datasets/playground-series-s{season}e{episode}.parquet"
    identifier = f"s{season}e{episode}"
    
    # Confirm we have the metadata if the user needs X, y pairs.
    target_col = None
    if return_X_y:
        try:
            target_col = METADATA[identifier]['target_col']
        except KeyError:
            raise KeyError(f"Can't find {identifier} in our lookup. It's possible that the metadata still needs to be added manually to the project.")
    
    return fetch_url(
        url, 
        name=f"kaggle-playground-{identifier}", 
        target_col=target_col, 
        return_X_y=return_X_y, 
        data_home=data_home, 
        force=force, 
        cleanup=cleanup
    )

fetch_playground_series(3, 26)