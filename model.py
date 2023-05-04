"""
Trains a model on the UCI ML Auto MPG dataset.
The dataset can be downloaded from:
https://archive.ics.uci.edu/ml/machine-learning-databases/auto-mpg/
"""

import shutil
import warnings
from datetime import timedelta

import mlflow
import mlflow.xgboost
import pandas as pd
import structlog
from prefect import flow, task
from prefect.tasks import task_input_hash
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBRegressor

warnings.filterwarnings("ignore")

logger = structlog.get_logger()


@task(log_prints=True, cache_key_fn=task_input_hash, cache_expiration=timedelta(days=1))
def read_data(data_path: str) -> pd.DataFrame:
    """
    Read data from a csv file

    Args:
        data_path (str): path to the data file

    Returns:
        pd.DataFrame: data
    """
    try:
        data = pd.read_csv(data_path)
    except FileNotFoundError:
        logger.error("File not found")
        return None
    return data


@task(log_prints=True)
def preprocess(data: pd.DataFrame) -> pd.DataFrame:
    """
    Preprocess the data

    Args:
        data (pd.DataFrame): input data

    Returns:
        pd.DataFrame: preprocessed data
    """
    logger.info("Preprocessing data ...")
    cleaned_df = data.copy()
    cleaned_df = cleaned_df.drop(
        ["car_ID", "symboling", "stroke", "compressionratio", "peakrpm"], axis=1
    )
    cleaned_df["CarBrand"] = cleaned_df["CarName"].str.split(" ", n=1, expand=True)[
        0]
    cleaned_df = cleaned_df.drop("CarName", axis=1)
    cleaned_df["CarBrand"] = cleaned_df["CarBrand"].replace(
        "alfa-romero", "alfa-romeo")
    cleaned_df["CarBrand"] = cleaned_df["CarBrand"].replace("maxda", "mazda")
    cleaned_df["CarBrand"] = cleaned_df["CarBrand"].replace("Nissan", "nissan")
    cleaned_df["CarBrand"] = cleaned_df["CarBrand"].replace(
        "porcshce", "porsche")
    cleaned_df["CarBrand"] = cleaned_df["CarBrand"].replace(
        "toyouta", "toyota")
    cleaned_df["CarBrand"] = cleaned_df["CarBrand"].replace(
        ["vokswagen", "vw"], "volkswagen"
    )
    return cleaned_df


@task(log_prints=True)
def create_classifier() -> Pipeline:
    """
    Create a classifier pipeline

    Returns:
        sklearn.pipeline.Pipeline: classifier pipeline
    """
    logger.info("Creating classifier ...")
    # Feature engineering
    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_transformer = OneHotEncoder(handle_unknown="ignore")
    preprocessor = ColumnTransformer(
        transformers=[
            (
                "cat",
                categorical_transformer,
                make_column_selector(dtype_include="object"),
            ),
            ("num", numeric_transformer, make_column_selector(dtype_exclude="object")),
        ]
    )
    clf = Pipeline(
        steps=[("preprocessor", preprocessor), ("classifier", XGBRegressor())]
    )
    return clf


@task(log_prints=True)
def split_data(data: pd.DataFrame, target: str = "price") -> tuple:
    """
    Split data into train and test sets

    Args:
        data (pd.DataFrame): input data
        target (str): target column name

    Returns:
        tuple: train and test sets
    """
    logger.info("Splitting data ...")
    X = data.drop(target, axis=1)
    y = data[target]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=1234
    )
    return X_train, X_test, y_train, y_test


def fetch_logged_data(run_id: str) -> tuple:
    """
    Fetch the data tracked by MLflow during the run with the specified ID.

    Args:
        run_id (str): ID of the run to fetch

    Returns:
        tuple: params, metrics, tags, artifacts
    """
    logger.info("Fetching logged data ...")
    client = mlflow.tracking.MlflowClient()
    data = client.get_run(run_id).data
    tags = {k: v for k, v in data.tags.items() if not k.startswith("mlflow.")}
    artifacts = [f.path for f in client.list_artifacts(run_id, "model")]
    return data.params, data.metrics, tags, artifacts


@task(log_prints=True)
def hyperparameters_optimization(clf, X_train, y_train):
    """
    Hyperparameters optimization

    Args:
        clf (sklearn.pipeline.Pipeline): classifier pipeline
        X_train (pd.DataFrame): train data
        y_train (pd.DataFrame): train target

    Returns:
        sklearn.pipeline.Pipeline: optimized classifier pipeline
    """
    logger.info("Hyperparameters optimization ...")
    # gridsearsh
    param_grid = {
        # "preprocessor__num__imputer__strategy": ["mean", "median"],
        "classifier__max_depth": [6, 7, 8],
        "classifier__learning_rate": [0.01, 0.015],
        "classifier__min_child_weight": [1, 2, 3],
        "classifier__subsample": [0.8, 0.9],
        "classifier__colsample_bytree": [0.7, 0.8],
        "classifier__n_estimators": [500],  # 600, 700],
        # "classifier__reg_alpha" : [0, 0.05],
        # "classifier__reg_lambda" : [0, 0.05],
        "classifier__objective": ["reg:squarederror"],
    }
    with mlflow.start_run(run_name="run") as run:

        grid_search = GridSearchCV(clf, param_grid, cv=5, verbose=4, n_jobs=-1)

        grid_search.fit(X_train, y_train)

        params, metrics, tags, artifacts = fetch_logged_data(run.info.run_id)
        logger.info("params: %s", params)
        logger.info("metrics: %s", metrics)
        logger.info("tags: %s", tags)
        logger.info("artifacts: %s", artifacts)
        logger.info("Best score: %f", grid_search.best_score_)
        logger.info("Best parameters: %s", grid_search.best_params_)
        logger.info("Best estimator: %s", grid_search.best_estimator_)
        logger.info("Run ID: %s", run.info.run_id)

        model_path = "mlruns/0/{}/artifacts/model".format(run.info.run_id)
        logger.info("Saving model to %s", model_path)

        shutil.copytree(model_path, "./model", dirs_exist_ok=True)
    return grid_search.best_estimator_


@task(log_prints=True)
def train_model(best_estimator, X_train, y_train, X_test, y_test):
    """
    Train a model with the best hyperparameters

    Args:
        best_estimator (sklearn.pipeline.Pipeline): classifier pipeline
        X_train (pd.DataFrame): train data
        y_train (pd.DataFrame): train target
        X_test (pd.DataFrame): test data
        y_test (pd.DataFrame): test target

    Returns:
        sklearn.pipeline.Pipeline: trained classifier pipeline
    """
    logger.info("Training model ...")
    with mlflow.start_run(run_name="run") as run:
        best_estimator.fit(X_train, y_train)
        y_pred = best_estimator.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        logger.info("mse: %s", mse)
        params, metrics, tags, artifacts = fetch_logged_data(run.info.run_id)
        logger.info("params: %s", params)
        logger.info("metrics: %s", metrics)
        logger.info("tags: %s", tags)
        logger.info("artifacts: %s", artifacts)
        logger.info("Run ID: %s", run.info.run_id)
        model_path = "mlruns/0/{}/artifacts/model".format(run.info.run_id)
        logger.info("Saving model to %s", model_path)
        shutil.copytree(model_path, "./model", dirs_exist_ok=True)


@flow(name="Traininig flow")
def main():
    """
    Main function
    """
    mlflow.sklearn.autolog(silent=True)
    data_path = "data/CarPrice_Assignment.csv"
    data = read_data(data_path)
    cleaned_data = preprocess(data)
    X_train, X_test, y_train, y_test = split_data(cleaned_data)
    clf = create_classifier()
    best_estimator = hyperparameters_optimization(clf, X_train, y_train)
    train_model(best_estimator, X_train, y_train, X_test, y_test)


if __name__ == "__main__":
    main()
