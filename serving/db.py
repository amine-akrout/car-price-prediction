"""
This module initializes a SQLite database for storing car price predictions.
"""

import sqlite3

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
