"""
Pydantic models for serving module.
"""

from pydantic import BaseModel, validator


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
