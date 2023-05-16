"""
Module to test the training module
"""
# pylint: disable=W0621, E0401, C0413
import os
import sys

# Add the parent directory of the training directory to the system path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import pytest
from sklearn.pipeline import Pipeline
from xgboost import XGBRegressor

from training.get_data import download_data
from training.model_training import create_classifier, preprocess, split_data


@pytest.fixture(scope="session")
def data():
    """
    Fixture to load the data
    """
    data = pd.read_csv("./data/CarPrice_Assignment.csv")
    return data


# test preprocess function
def test_preprocess(data):
    """
    Test the preprocess function
    """
    cleaned_data = preprocess.fn(data)
    # Call the preprocess function on the input data

    # Check that the output has the expected columns
    assert "CarBrand" in cleaned_data.columns

    # Check that some brand names have been corrected
    assert "alfa-romeo" in cleaned_data["CarBrand"].unique()
    assert "mazda" in cleaned_data["CarBrand"].unique()
    assert "nissan" in cleaned_data["CarBrand"].unique()
    assert "porsche" in cleaned_data["CarBrand"].unique()
    assert "toyota" in cleaned_data["CarBrand"].unique()
    assert "volkswagen" in cleaned_data["CarBrand"].unique()
    assert "volvo" in cleaned_data["CarBrand"].unique()


def test_split_data(data):
    """
    Test the split_data function
    """
    # Call the split_data function on the input data
    cleaned_data = preprocess.fn(data)
    train, test, _, _ = split_data.fn(cleaned_data)
    # Check that the output objects are dataframes
    assert isinstance(train, pd.DataFrame)
    assert isinstance(test, pd.DataFrame)

    # Check that the output dataframes have the expected number of rows
    assert train.shape[0] >= 100
    assert test.shape[0] >= 35


def test_create_classifier():
    """
    Test the create_classifier function
    """
    # Call the create_classifier function
    model = create_classifier.fn()

    # Check that the output is a pipeline
    assert isinstance(model, Pipeline)

    # Check that the pipeline has the expected number of steps
    assert len(model.steps) == 2

    # Check that the model is a RandomForestRegressor
    assert isinstance(model.steps[1][1], XGBRegressor)
