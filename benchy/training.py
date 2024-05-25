import polars as pl
from sklearn.model_selection import KFold
from skrub import TableVectorizer, SelectCols
import time

from benchy.kaggle import METADATA, fetch_playground_series
from benchy.estimators import ColumnDropper
from lightgbm import LGBMClassifier, LGBMRegressor
from xgboost import XGBClassifier, XGBRegressor
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import make_pipeline, make_union
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.ensemble import (
    HistGradientBoostingClassifier,
    HistGradientBoostingRegressor,
)
from sklearn.metrics import (
    classification_report,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from sklearn.preprocessing import LabelEncoder, FunctionTransformer, SplineTransformer
from sklearn.impute import SimpleImputer


def calc_scores(task, y_true, y_pred):
    if task == "classification":
        return classification_report(y_true, y_pred, output_dict=True)
    if task == "regression":
        return {
            "mae": mean_absolute_error(y_true, y_pred),
            "mse": mean_squared_error(y_true, y_pred),
            "r2": r2_score(y_true, y_pred),
        }


def get_datasets(task="classification"):
    for name, descr in METADATA.items():
        if descr["task"] == task:
            season = int(name.split("e")[0].replace("s", ""))
            episode = int(name.split("e")[1])
            yield name, fetch_playground_series(season, episode, return_X_y=True)


def get_dataset(dataname):
    for name, descr in METADATA.items():
        if name == dataname:
            season = int(name.split("e")[0].replace("s", ""))
            episode = int(name.split("e")[1])
            return fetch_playground_series(season, episode, return_X_y=True)


def datetime_feats(dataf):
    return dataf.with_columns(
        day_of_year=pl.col("date").str.to_datetime().dt.ordinal_day()
    ).select("day_of_year")


class ConditionalDateFeaturizer(BaseEstimator, TransformerMixin):
    def __init__(self) -> None:
        self.spline_tfm = SplineTransformer(n_knots=12, extrapolation="periodic")
    
    def fit(self, X, y):
        if 'date' in X.columns:
            self.pipeline_ = make_union(
                make_pipeline(
                    SelectCols('date'),
                    FunctionTransformer(datetime_feats),
                    SplineTransformer(n_knots=12, extrapolation="periodic")
                ),
                TableVectorizer()
            )
        else:
            self.pipeline_ = TableVectorizer()
        return self.pipeline_.fit(X, y)
    
    def transform(self, X, y=None):
        return self.pipeline_.transform(X)
    
def get_featurizers(task="classification", mod_name="", elaborate=False):
    if "ridge" not in mod_name and task == 'classification':
        yield "tablevec", TableVectorizer()
        yield "tablevec-noids", make_pipeline(ColumnDropper(substring='id'), TableVectorizer())
    elif task == 'regression':
        yield "tablevec-ts-impute", make_pipeline(
            ConditionalDateFeaturizer(),
            SimpleImputer()
        )
        yield "tablevec-impute", make_pipeline(TableVectorizer(), SimpleImputer())
        yield "tablevec-noids-impute", make_pipeline(ColumnDropper(substring='id'), TableVectorizer(), SimpleImputer())


def get_estimators(task="classification", elaborate=False):
    if task == "classification":
        yield "lr", LogisticRegression()
        yield "hgbt", HistGradientBoostingClassifier()
        yield "xgboost", XGBClassifier()
        yield "lgbm", LGBMClassifier()
    if task == "regression":
        yield "ridge", Ridge()
        yield "hgbt", HistGradientBoostingRegressor()
        yield "xgboost", XGBRegressor()
        yield "lgbm", LGBMRegressor()


def make_task_hash(
    task, dataname, random_seed, cv_id, tfm_name, n_splits, mod_name, **kwargs
):
    return f"{task}-{dataname}-{random_seed}-{cv_id}-{n_splits}-{tfm_name}-{mod_name}"


def task_generator(task, cache, n_seeds=1, n_splits=5, dry_run=False):
    for dataname, _ in get_datasets(task):
        for random_seed in range(n_seeds):
            for cv_id in range(n_splits):
                for mod_name, mod in get_estimators(task, elaborate=False):
                    for tfm_name, tfm in get_featurizers(
                        task, mod_name, elaborate=False
                    ):
                        train_hash = make_task_hash(
                                task,
                                dataname,
                                random_seed,
                                cv_id,
                                n_splits,
                                tfm_name,
                                mod_name,
                            )
                        if train_hash not in cache:
                            if dry_run:
                                yield 1
                            else:
                                yield {
                                    "task": task,
                                    "dataname": dataname,
                                    "random_seed": random_seed,
                                    "cv_id": cv_id,
                                    "n_splits": n_splits,
                                    "tfm_name": tfm_name,
                                    "mod_name": mod_name,
                                    "featurizer": tfm,
                                    "model": mod,
                                }


def train(
    task, dataname, random_seed, n_splits, cv_id, tfm_name, mod_name, featurizer, model
):
    # This is somewhat expensive to the memory, but kind of nice for parallelism
    X, y = get_dataset(dataname)
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_seed)
    for cv_idx, (train_idx, test_idx) in enumerate(kf.split(X)):
        if cv_idx == cv_id:
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]

    # XGBoost doesnt handle this nicely internally, so we gotta do this manually
    if task == "classification":
        lab_enc = LabelEncoder()
        y_train = lab_enc.fit_transform(y_train)
        y_test = lab_enc.transform(y_test)

    # Sometimes featurization can take embarassingly long, inspect.
    tic = time.time()
    feat_pipe = make_pipeline(
        featurizer, FunctionTransformer()
    )
    feat_pipe.fit(X_train, y_train)
    feat_time = time.time() - tic

    # Train the model
    tic = time.time()
    X_feat = feat_pipe.transform(X_train)
    model.fit(X_feat, y_train)
    train_time = time.time() - tic

    # Keep predictions around for reporting.
    pred_train = model.predict(X_feat)

    # Gather predictions for test set
    tic = time.time()
    X_feat_test = feat_pipe.transform(X_test)
    pred_test = model.predict(X_feat_test)
    infer_time = time.time() - tic

    return {
        "task": task, 
        "timestamp": int(time.time()),
        "dataname": dataname,
        "n_splits": n_splits,
        "cv_id": cv_id,
        "random_seed": random_seed,
        "estimator_name": f"{mod_name}-{tfm_name}",
        "mod_name": mod_name,
        "tfm_name": tfm_name,
        "feat_time": feat_time,
        "train_time": train_time,
        "infer_time": infer_time,
        "scores_train": calc_scores(task, y_train, pred_train),
        "scores_test": calc_scores(task, y_test, pred_test),
    }
