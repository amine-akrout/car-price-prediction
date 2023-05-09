"""
This module contains functions to get data from Kaggle
and save it to data folder in the root directory
"""


import json
import os
import zipfile

import structlog

logger = structlog.get_logger()


def download_data():
    """
    Get data from Kaggle and save it to data folder in the root directory
    """
    logger.info("Getting data ...")
    with open("kaggle.json") as file:
        kaggle_api_key = json.load(file)
    os.environ["KAGGLE_USERNAME"] = kaggle_api_key["username"]
    os.environ["KAGGLE_KEY"] = kaggle_api_key["key"]
    os.system(
        "kaggle datasets download -d shaistashaikh/carprice-assignment"
    )
    with zipfile.ZipFile("carprice-assignment.zip", "r") as zip_ref:
        zip_ref.extractall("./data")
    os.remove("carprice-assignment.zip")


