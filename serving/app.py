# app.py
import os
import warnings
from contextlib import asynccontextmanager

import structlog
import uvicorn
from db import init_db
from dotenv import load_dotenv
from fastapi import FastAPI, Form, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from model_loader import load_model_from_gcs_and_initialize, unload_model
from models import CarPredictionInput
from monitoring import generate_dashboard
from predictor import make_prediction, save_prediction_to_db
from pydantic import BaseModel
from starlette.exceptions import HTTPException as StarletteHTTPException

logger = structlog.get_logger()
warnings.filterwarnings("ignore")
load_dotenv(".env")


init_db()


# Move model loading/unloading to a separate context manager function
@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Context manager to load and unload the model when the app starts and stops.
    """
    load_model_from_gcs_and_initialize()
    yield
    unload_model()


app = FastAPI(lifespan=lifespan)
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):

    message = "Error: Please enter valid values."
    return templates.TemplateResponse(
        "index.html", {"request": request, "message": message}
    )


@app.exception_handler(StarletteHTTPException)
async def http_exception_handler(request: Request, exc: StarletteHTTPException):
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
        predicted_price = make_prediction(input_data)
        save_prediction_to_db(input_data, predicted_price)
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
