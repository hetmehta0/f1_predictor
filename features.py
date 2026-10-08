"""History features computed strictly before each race, without global imputation."""

from __future__ import annotations

import numpy as np
import pandas as pd

if __package__:
    from .weather import WEATHER_FEATURES, missing_weather
else:
    from weather import WEATHER_FEATURES, missing_weather

HISTORICAL_FEATURE_COLUMNS = [
    "DriverRecentFinish",
    "DriverRecentQuali",
    "DriverRecentGain",
    "DriverAvgFinish_Circuit",
    "DriverAvgFinish_Season",
    "TeamRecentFinish",
    "DriverDNFRate",
    "TeamDNFRate",
    "DriverSeasonPointsPerRace",
    "DriverHistoryCount",
    "CircuitHistoryCount",
    "DriverForecastWetGain",
]
FEATURE_COLUMNS = [
    "QualiNorm",
    "QualiPos",
    "QualiPositionFraction",
    "QualiSegment",
    "QualiMissing",
    "FieldSize",
    *HISTORICAL_FEATURE_COLUMNS,
    "IsFirstSeasonOfEra",
    *WEATHER_FEATURES,
    "WetFormInteraction",
]
ERA_START_YEARS = {2014, 2017, 2022, 2026}


def before_race(frame: pd.DataFrame, year: int, round_number: int) -> pd.DataFrame:
    return frame.loc[
        (frame["Year"] < year) | ((frame["Year"] == year) & (frame["Round"] < round_number))
    ].copy()


def _recent(series: pd.Series, limit=5) -> float:
    values = series.tail(limit).dropna().to_numpy(dtype=float)
    if not len(values):
        return np.nan
    weights = 0.75 ** np.arange(len(values) - 1, -1, -1)
    return float(np.average(values, weights=weights))


def _finish_scale(frame: pd.DataFrame, column: str) -> pd.Series:
    """Express history on a 20-car reference scale while keeping actual labels intact."""
    return 1 + 19 * (frame[column] - 1) / (frame["FieldSize"] - 1).clip(lower=1)


def _apply_historical_features_for_race(race_df, history_df) -> pd.DataFrame:
    if race_df.empty or len(race_df[["Year", "Round"]].drop_duplicates()) != 1:
        raise ValueError("Expected exactly one nonempty race.")
    result = race_df.copy()
    year, round_number = int(result["Year"].iloc[0]), int(result["Round"].iloc[0])
    # Defend this boundary even when callers accidentally pass the complete dataset.
    history = before_race(history_df, year, round_number).sort_values(["Year", "Round", "DriverId"])
    history = history.loc[history["Year"] >= year - 3].copy()
    for key, value in missing_weather("not supplied").items():
        if key not in result:
            result[key] = value
    result["QualiPositionFraction"] = (result["QualiPos"] - 1) / (result["FieldSize"] - 1).clip(
        lower=1
    )
    result["IsFirstSeasonOfEra"] = int(year in ERA_START_YEARS)
    for feature in HISTORICAL_FEATURE_COLUMNS:
        result[feature] = np.nan
    result[["DriverHistoryCount", "CircuitHistoryCount"]] = 0
    if not history.empty:
        history["_Finish"] = _finish_scale(history, "FinishPos")
        history["_Quali"] = _finish_scale(history, "QualiPos")
        history["_Gain"] = history["_Quali"] - history["_Finish"]
        for index, row in result.iterrows():
            driver = history.loc[history["DriverId"] == row["DriverId"]]
            team = history.loc[history["TeamId"] == row["TeamId"]]
            circuit = driver.loc[driver["CircuitKey"] == row["CircuitKey"]].tail(3)
            season = driver.loc[driver["Year"] == year]
            recent = _recent(driver["_Finish"])
            circuit_finish = circuit["_Finish"].mean()
            # Sparse circuit histories shrink toward general recent form.
            if len(circuit) and pd.notna(recent):
                circuit_finish = (len(circuit) * circuit_finish + 3 * recent) / (len(circuit) + 3)
            team_by_race = team.groupby(["Year", "Round"])["_Finish"].mean()
            wet = driver.loc[driver.get("ForecastWet", pd.Series(np.nan, index=driver.index)) == 1]
            wet_gain = float(wet["_Gain"].tail(10).sum() / (min(len(wet), 10) + 3))
            values = {
                "DriverRecentFinish": recent,
                "DriverRecentQuali": _recent(driver["_Quali"]),
                "DriverRecentGain": _recent(driver["_Gain"]),
                "DriverAvgFinish_Circuit": circuit_finish,
                "DriverAvgFinish_Season": season["_Finish"].mean(),
                "TeamRecentFinish": _recent(team_by_race),
                "DriverDNFRate": (driver["DNF"].tail(10).sum() + 0.24) / (min(len(driver), 10) + 2),
                "TeamDNFRate": (team["DNF"].tail(20).sum() + 0.48) / (min(len(team), 20) + 4),
                "DriverSeasonPointsPerRace": season["Points"].mean(),
                "DriverHistoryCount": len(driver),
                "CircuitHistoryCount": len(circuit),
                "DriverForecastWetGain": wet_gain,
            }
            for feature, value in values.items():
                result.loc[index, feature] = value
    result["WetFormInteraction"] = result["ForecastWet"] * result["DriverForecastWetGain"]
    return result


def add_historical_features(master_df: pd.DataFrame) -> pd.DataFrame:
    if master_df.empty:
        raise ValueError("Cannot create features for an empty dataset.")
    sorted_df = master_df.sort_values(["Year", "Round", "DriverId"]).reset_index(drop=True)
    races = []
    for _, race in sorted_df.groupby(["Year", "Round"], sort=True):
        races.append(_apply_historical_features_for_race(race, sorted_df))
    # Missing values stay missing until each individual training fold fits its imputer.
    return pd.concat(races, ignore_index=True)
