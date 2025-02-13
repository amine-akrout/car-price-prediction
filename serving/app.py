"""
This is a simple FastAPI web app that predicts the price of a car based on user inputs.
"""

import os
import warnings
from contextlib import asynccontextmanager
from datetime import datetime
from typing import List, Optional

import pandas as pd
import structlog
import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI, Form, HTTPException, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from google.cloud import storage
from mlflow.pyfunc import load_model
from monitoring import generate_dashboard
from motor.motor_asyncio import AsyncIOMotorClient
from pydantic import BaseModel, validator
from utils import load_model_from_gcs

logger = structlog.get_logger()
warnings.filterwarnings("ignore")


load_dotenv(".env")


# Define Pydantic models for data validation
class CarPredictionInput(BaseModel):
    CarBrand: str
    fueltype: str
    aspiration: str
    doornumber: str
    carbody: str
    drivewheel: str
    enginelocation: str
    wheelbase: float
    carlength: float
    carwidth: float
    carheight: float
    curbweight: int
    enginetype: str
    cylindernumber: str
    enginesize: int
    fuelsystem: str
    boreratio: float
    horsepower: int
    citympg: int
    highwaympg: int

    # Add validators if needed
    @validator(
        "wheelbase",
        "carlength",
        "carwidth",
        "carheight",
        "curbweight",
        "enginesize",
        "boreratio",
        "horsepower",
        "citympg",
        "highwaympg",
    )
    def validate_positive_numbers(cls, value):
        if value <= 0:
            raise ValueError("Value must be positive")
        return value


# Global variable to store the loaded model
LOADED_MODEL = None


# Lifespan event to load the model on startup
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan event to load the model on startup."""
    global LOADED_MODEL
    MODEL_DIR = "./model"
    BUCKET_NAME = os.environ.get("GCS_BUCKET")
    MODEL_BLOB_PREFIX = "models/"

    # Download the model files from GCS
    load_model_from_gcs(BUCKET_NAME, MODEL_BLOB_PREFIX, MODEL_DIR)

    # Load the model
    LOADED_MODEL = load_model(MODEL_DIR)
    logger.info("Model loaded successfully.")

    yield  # FastAPI will keep running until shutdown

    # Cleanup on shutdown (if needed)
    LOADED_MODEL = None
    logger.info("Model unloaded.")


# Initialize FastAPI app with lifespan event
app = FastAPI(lifespan=lifespan)

# Mount static files and templates
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

# MongoDB setup
MONGO_URI = "mongodb://mongo:27017"
client = AsyncIOMotorClient(MONGO_URI)
db = client["car_prediction"]
collection = db["predictions"]


@app.get("/", response_class=HTMLResponse)
async def entry_page(request: Request):
    """Displays the input webpage."""
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/predict", response_class=HTMLResponse)
async def render_message(
    request: Request,
    CarBrand: str = Form(...),
    fueltype: str = Form(...),
    aspiration: str = Form(...),
    doornumber: str = Form(...),
    carbody: str = Form(...),
    drivewheel: str = Form(...),
    enginelocation: str = Form(...),
    wheelbase: float = Form(...),
    carlength: float = Form(...),
    carwidth: float = Form(...),
    carheight: float = Form(...),
    curbweight: int = Form(...),
    enginetype: str = Form(...),
    cylindernumber: str = Form(...),
    enginesize: int = Form(...),
    fuelsystem: str = Form(...),
    boreratio: float = Form(...),
    horsepower: int = Form(...),
    citympg: int = Form(...),
    highwaympg: int = Form(...),
):
    """Generates the predicted price and renders it in the HTML."""
    try:
        # Validate input data using Pydantic
        input_data = CarPredictionInput(
            CarBrand=CarBrand,
            fueltype=fueltype,
            aspiration=aspiration,
            doornumber=doornumber,
            carbody=carbody,
            drivewheel=drivewheel,
            enginelocation=enginelocation,
            wheelbase=wheelbase,
            carlength=carlength,
            carwidth=carwidth,
            carheight=carheight,
            curbweight=curbweight,
            enginetype=enginetype,
            cylindernumber=cylindernumber,
            enginesize=enginesize,
            fuelsystem=fuelsystem,
            boreratio=boreratio,
            horsepower=horsepower,
            citympg=citympg,
            highwaympg=highwaympg,
        )

        # Prepare data for prediction
        data = [
            [
                input_data.CarBrand,
                input_data.fueltype,
                input_data.aspiration,
                input_data.doornumber,
                input_data.carbody,
                input_data.drivewheel,
                input_data.enginelocation,
                input_data.wheelbase,
                input_data.carlength,
                input_data.carwidth,
                input_data.carheight,
                input_data.curbweight,
                input_data.enginetype,
                input_data.cylindernumber,
                input_data.enginesize,
                input_data.fuelsystem,
                input_data.boreratio,
                input_data.horsepower,
                input_data.citympg,
                input_data.highwaympg,
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

        # Make prediction
        preds = LOADED_MODEL.predict(preds_df)
        message = f"Estimated price: {round(preds[0], 2)}"

        # Save the data and prediction to MongoDB
        prediction_data = {
            **input_data.dict(),
            "predicted_price": float(preds[0]),
            "created_at": datetime.utcnow(),
        }
        logger.info(f"Saving prediction data to MongoDB: {prediction_data}")
        await collection.insert_one(prediction_data)
        logger.info("Prediction data saved successfully.")

    except Exception as error:
        message = f"Error encountered. Try with other values. Error: {str(error)}"
        logger.error(message)
        return templates.TemplateResponse(
            "index.html", {"request": request, "message": "Please enter valid values."}
        )

    return templates.TemplateResponse(
        "index.html", {"request": request, "message": message}
    )


@app.get("/dashboard", response_class=HTMLResponse)
async def dashboard(request: Request):
    """Generates and displays the monitoring dashboard."""
    dashboard_path = generate_dashboard()
    return templates.TemplateResponse(dashboard_path, {"request": request})


# Add health check endpoint
@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "ok"}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
