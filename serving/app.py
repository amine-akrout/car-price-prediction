import os
import sqlite3
import warnings
from contextlib import asynccontextmanager
from datetime import datetime, timedelta
from typing import List

import pandas as pd
import structlog
import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI, Form, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from mlflow.pyfunc import load_model
from monitoring import generate_dashboard
from pydantic import BaseModel, validator
from starlette.exceptions import HTTPException as StarletteHTTPException
from utils import load_model_from_gcs

logger = structlog.get_logger()
warnings.filterwarnings("ignore")

load_dotenv(".env")

DB_FILE = "predictions.db"


def init_db():
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            CarBrand TEXT,
            fueltype TEXT,
            aspiration TEXT,
            doornumber TEXT,
            carbody TEXT,
            drivewheel TEXT,
            enginelocation TEXT,
            wheelbase REAL,
            carlength REAL,
            carwidth REAL,
            carheight REAL,
            curbweight INTEGER,
            enginetype TEXT,
            cylindernumber TEXT,
            enginesize INTEGER,
            fuelsystem TEXT,
            boreratio REAL,
            horsepower INTEGER,
            citympg INTEGER,
            highwaympg INTEGER,
            predicted_price REAL,
            created_at TEXT
        )
        """
    )
    conn.commit()
    conn.close()


init_db()


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


LOADED_MODEL = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global LOADED_MODEL
    MODEL_DIR = "./model"
    BUCKET_NAME = os.environ.get("GCS_BUCKET")
    MODEL_BLOB_PREFIX = "models/"
    load_model_from_gcs(BUCKET_NAME, MODEL_BLOB_PREFIX, MODEL_DIR)
    LOADED_MODEL = load_model(MODEL_DIR)
    logger.info("Model loaded successfully.")
    yield
    LOADED_MODEL = None
    logger.info("Model unloaded.")


app = FastAPI(lifespan=lifespan)
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    """Handles validation errors and returns the index page with an error message."""
    message = "Error: Please enter valid values."
    return templates.TemplateResponse(
        "index.html", {"request": request, "message": message}
    )


@app.exception_handler(StarletteHTTPException)
async def http_exception_handler(request: Request, exc: StarletteHTTPException):
    """Handles general HTTP exceptions."""
    message = "An error occurred. Please try again."
    return templates.TemplateResponse(
        "index.html", {"request": request, "message": message}
    )


@app.get("/", response_class=HTMLResponse)
async def entry_page(request: Request):
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
        print(input_data)
        data = [[getattr(input_data, field) for field in input_data.dict()]]
        preds_df = pd.DataFrame(data, columns=input_data.dict().keys())
        preds = LOADED_MODEL.predict(preds_df)
        predicted_price = round(float(preds[0]), 2)

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
        message = f"Estimated price: {predicted_price}"
    except Exception as error:
        message = "Error: Please enter valid values."
        logger.error(f"Error encountered: {str(error)}")

    return templates.TemplateResponse(
        "index.html", {"request": request, "message": message}
    )


@app.get("/dashboard", response_class=HTMLResponse)
async def dashboard(request: Request):
    dashboard_path = generate_dashboard()
    return templates.TemplateResponse(dashboard_path, {"request": request})


@app.get("/health")
async def health_check():
    return {"status": "ok"}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
