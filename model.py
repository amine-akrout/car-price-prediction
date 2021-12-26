import pandas as pd
import numpy as np
import json
from pprint import pprint
import shutil

from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error


from xgboost import XGBRegressor

import mlflow
import mlflow.xgboost

df = pd.read_csv('./data/CarPrice_Assignment.csv')
df.head()

df.info()

df = df.drop(['car_ID','symboling', 'stroke','compressionratio','peakrpm'], axis=1)

def preprocess_car_name(df):
    df['CarBrand'] = df['CarName'].str.split(' ', n=1, expand=True)[0]
    df = df.drop('CarName', axis=1)
    df['CarBrand'] = df['CarBrand'].replace('alfa-romero', 'alfa-romeo')
    df['CarBrand'] = df['CarBrand'].replace('maxda', 'mazda')
    df['CarBrand'] = df['CarBrand'].replace('Nissan', 'nissan')
    df['CarBrand'] = df['CarBrand'].replace('porcshce', 'porsche')
    df['CarBrand'] = df['CarBrand'].replace('toyouta', 'toyota')
    df['CarBrand'] = df['CarBrand'].replace(['vokswagen', 'vw'], 'volkswagen')
    return df
df = preprocess_car_name(df)

# Feature engineering
numeric_transformer = Pipeline(
    steps=[("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]
)
categorical_transformer = OneHotEncoder(handle_unknown="ignore")

preprocessor = ColumnTransformer(
    transformers=[
        ("cat", categorical_transformer, make_column_selector(dtype_include="object")),
        ("num", numeric_transformer, make_column_selector(dtype_exclude="object")),
    ]
)

clf = Pipeline(
    steps=[("preprocessor", preprocessor),
           ("classifier", XGBRegressor())]
)

# Split Data
price = df['price']
dataset = df.drop('price', axis=1)
X_train, X_test, y_train, y_test = train_test_split(dataset, price, test_size=0.2, random_state=1234)

mlflow.sklearn.autolog()

def fetch_logged_data(run_id):
    client = mlflow.tracking.MlflowClient()
    data = client.get_run(run_id).data
    tags = {k: v for k, v in data.tags.items() if not k.startswith("mlflow.")}
    artifacts = [f.path for f in client.list_artifacts(run_id, "model")]
    return data.params, data.metrics, tags, artifacts

# gridsearsh
param_grid = {
    #"preprocessor__num__imputer__strategy": ["mean", "median"],
    "classifier__max_depth": [6, 7, 8],
    "classifier__learning_rate": [0.01, 0.015],
    "classifier__min_child_weight": [1,2,3],
    "classifier__subsample": [0.8, 0.9],
    "classifier__colsample_bytree": [0.7, 0.8],
    "classifier__n_estimators" : [500], # 600, 700],
    #"classifier__reg_alpha" : [0, 0.05],
    #"classifier__reg_lambda" : [0, 0.05],
    "classifier__objective" : ['reg:squarederror']
}
with mlflow.start_run(run_name="run") as run:

    grid_search = GridSearchCV(clf, 
                                param_grid,
                                cv=5,
                                verbose=4,
                                n_jobs=-1)
    grid_search

    grid_search.fit(X_train, y_train)

    params, metrics, tags, artifacts = fetch_logged_data(run.info.run_id)
    pprint(params)
    pprint(metrics)
    pprint(tags)
    pprint(artifacts)
    pprint(run.info.run_id)

    model_path = 'mlruns/0/{}/artifacts/model'.format(run.info.run_id)

    shutil.copytree(model_path, './model', dirs_exist_ok=True)

    