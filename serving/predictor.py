"""
This module contains the functions for making predictions and saving them to the database.
"""

import sqlite3
from datetime import datetime

import pandas as pd
from db import DB_FILE
from mlflow.pyfunc import load_model
from model_loader import LOADED_MODEL


def make_prediction(input_data):
    data = [[getattr(input_data, field) for field in input_data.dict()]]
    preds_df = pd.DataFrame(data, columns=input_data.dict().keys())
    preds = LOADED_MODEL.predict(preds_df)
    predicted_price = round(float(preds[0]), 2)
    return predicted_price


def save_prediction_to_db(input_data, predicted_price):
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    cursor.execute(
        """
        INSERT INTO predictions (
            CarBrand, fueltype, aspiration, doornumber, carbody, drivewheel, enginelocation, 
            wheelbase, carlength, carwidth, carheight, curbweight, enginetype, cylindernumber, 
            enginesize, fuelsystem, boreratio, horsepower, citympg, highwaympg, predicted_price, created_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            *input_data.dict().values(),
            predicted_price,
            datetime.utcnow().isoformat(),
        ),
    )
    conn.commit()
    conn.close()
