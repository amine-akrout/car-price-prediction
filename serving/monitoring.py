"""
Monitoring module for car prediction app
"""

from datetime import datetime, timedelta

import pandas as pd
from evidently.metric_preset import DataDriftPreset
from evidently.report import Report
from pymongo import MongoClient

client = MongoClient("mongodb://mongo:27017")
db = client["car_prediction"]
collection = db["predictions"]


def preprocess(data):
    """Preprocess the data for dashboard"""
    cleaned_df = data.copy()
    cleaned_df = cleaned_df.drop(
        ["car_ID", "symboling", "stroke", "compressionratio", "peakrpm", "price"],
        axis=1,
    )
    cleaned_df["CarBrand"] = cleaned_df["CarName"].str.split(" ", n=1, expand=True)[0]
    cleaned_df = cleaned_df.drop("CarName", axis=1)
    cleaned_df["CarBrand"] = cleaned_df["CarBrand"].replace("alfa-romero", "alfa-romeo")
    cleaned_df["CarBrand"] = cleaned_df["CarBrand"].replace("maxda", "mazda")
    cleaned_df["CarBrand"] = cleaned_df["CarBrand"].replace("Nissan", "nissan")
    cleaned_df["CarBrand"] = cleaned_df["CarBrand"].replace("porcshce", "porsche")
    cleaned_df["CarBrand"] = cleaned_df["CarBrand"].replace("toyouta", "toyota")
    cleaned_df["CarBrand"] = cleaned_df["CarBrand"].replace(
        ["vokswagen", "vw"], "volkswagen"
    )
    return cleaned_df


def get_last_30_days_data():
    """Get data from MongoDB for the last 30 days"""
    # calculate date range
    end_date = datetime.now()
    start_date = end_date - timedelta(days=30)

    # query MongoDB for data within date range
    cursor = collection.find({"created_at": {"$gte": start_date, "$lt": end_date}})

    # convert MongoDB cursor to pandas dataframe
    current_df = pd.DataFrame(list(cursor))
    return current_df


def get_refence_data():
    """Get reference data from csv"""
    reference_df = pd.read_csv("ref_data.csv")
    # remove some cols from df
    return preprocess(reference_df)


def generate_dashboard():
    """Generate dashboard"""
    dasboard_name = "drift.html"
    data_drift_dashboard = Report(metrics=[DataDriftPreset()])

    reference_data = get_refence_data()
    current_data = get_last_30_days_data()

    data_drift_dashboard.run(
        reference_data=reference_data,
        current_data=current_data,
        column_mapping=None,
    )
    data_drift_dashboard.save_html("templates/" + dasboard_name)
    print(f"Dashboard saved to {dasboard_name}")
    return dasboard_name
