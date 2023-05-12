"""
This is a simple Flask web app that predicts the price of a car based on the user inputs.
"""
from __future__ import print_function

import os
import sys
from datetime import datetime

import pandas as pd
from flask import Flask, render_template, request
from mlflow.pyfunc import load_model
from pymongo import MongoClient

MODEL_PATH = "./model"
# Load model as a PyFuncModel.
LOADED_MODEL = load_model(MODEL_PATH)

client = MongoClient("mongodb://mongo:27017")
db = client["car_prediction"]
collection = db["predictions"]


app = Flask(__name__)


@app.route("/")
def entry_page():
    """Displays the input webpage"""
    # Nicepage template of the webpage
    return render_template("index.html")


@app.route("/predict", methods=["GET", "POST"])
def render_message():
    """Generates the predicted price and renders it in the html"""
    try:
        # Get data input
        CarBrand = request.form["CarBrand"]
        fueltype = request.form["fueltype"]
        aspiration = request.form["aspiration"]
        doornumber = request.form["doornumber"]
        carbody = request.form["carbody"]
        drivewheel = request.form["drivewheel"]
        enginelocation = request.form["enginelocation"]
        wheelbase = float(request.form["wheelbase"])
        carlength = float(request.form["carlength"])
        carwidth = float(request.form["carwidth"])
        carheight = float(request.form["carheight"])
        curbweight = int(request.form["curbweight"])
        enginetype = request.form["enginetype"]
        cylindernumber = request.form["cylindernumber"]
        enginesize = int(request.form["enginesize"])
        fuelsystem = request.form["fuelsystem"]
        boreratio = float(request.form["boreratio"])
        horsepower = int(request.form["horsepower"])
        citympg = int(request.form["citympg"])
        highwaympg = int(request.form["highwaympg"])
        data = [
            [
                CarBrand,
                fueltype,
                aspiration,
                doornumber,
                carbody,
                drivewheel,
                enginelocation,
                wheelbase,
                carlength,
                carwidth,
                carheight,
                curbweight,
                enginetype,
                cylindernumber,
                enginesize,
                fuelsystem,
                boreratio,
                horsepower,
                citympg,
                highwaympg,
            ]
        ]
        preds_df = pd.DataFrame(
            data,
            columns=[
                "CarBrand",
                "fueltype",
                "aspiration",
                "doornumber",
                "carbody",
                "drivewheel",
                "enginelocation",
                "wheelbase",
                "carlength",
                "carwidth",
                "carheight",
                "curbweight",
                "enginetype",
                "cylindernumber",
                "enginesize",
                "fuelsystem",
                "boreratio",
                "horsepower",
                "citympg",
                "highwaympg",
            ],
        )

        preds = LOADED_MODEL.predict(pd.DataFrame(preds_df))

        print("Python module executed successfully")
        message = "Estimated price : {} ".format(round(preds[0], 2))
        print(message, file=sys.stderr)

        # Save the data and prediction to MongoDB
        prediction_data = {
            "CarBrand": CarBrand,
            "fueltype": fueltype,
            "aspiration": aspiration,
            "doornumber": doornumber,
            "carbody": carbody,
            "drivewheel": drivewheel,
            "enginelocation": enginelocation,
            "wheelbase": float(wheelbase),
            "carlength": float(carlength),
            "carwidth": float(carwidth),
            "carheight": float(carheight),
            "curbweight": int(curbweight),
            "enginetype": enginetype,
            "cylindernumber": cylindernumber,
            "enginesize": int(enginesize),
            "fuelsystem": fuelsystem,
            "boreratio": float(boreratio),
            "horsepower": int(horsepower),
            "citympg": int(citympg),
            "highwaympg": int(highwaympg),
            "predicted_price": float(preds[0]),
            "created_at": datetime.utcnow(),
        }
        collection.insert_one(prediction_data)

    except Exception as error:
        # Store error to pass to the web page
        message = (
            "Error encountered. Try with other values. ErrorClass: {}, Argument: {} and"
            " Traceback details are: {}".format(
                error.__class__, error.args, error.__doc__
            )
        )

    # Return the model results to the web page
    return render_template("index.html", message=message)


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=int(os.environ.get("PORT", 8080)))
