import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def raw_races():
    rows = []
    for year, rounds in ((2023, 6), (2024, 3)):
        for round_number in range(1, rounds + 1):
            for driver in range(1, 23):
                finish = driver
                if round_number % 2 == 0 and driver in (1, 2):
                    finish = 3 - driver
                rows.append(
                    {
                        "Year": year,
                        "Round": round_number,
                        "GPName": f"Race {round_number}",
                        "CircuitKey": f"circuit-{round_number % 3}",
                        "DriverId": f"driver-{driver:02}",
                        "Abbreviation": f"D{driver:02}",
                        "DriverNumber": str(driver),
                        "TeamId": f"team-{(driver - 1) // 2}",
                        "TeamName": f"Team {(driver - 1) // 2}",
                        "QualiPos": driver,
                        "QualiNorm": (driver - 1) * 0.001,
                        "QualiSegment": 3 if driver <= 10 else 1,
                        "QualiMissing": 0,
                        "FinishPos": finish,
                        "DNF": float(driver >= 21),
                        "Points": max(0, 11 - finish),
                        "FieldSize": 22,
                        "ForecastTempC": np.nan,
                        "ForecastPrecipMM": np.nan,
                        "ForecastWindKph": np.nan,
                        "ForecastHumidity": np.nan,
                        "ForecastWet": np.nan,
                        "WeatherMissing": 1,
                        "WeatherSource": "synthetic missing",
                    }
                )
    return pd.DataFrame(rows)
