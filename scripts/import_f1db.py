"""Convert an F1DB checkout into the predictor's CSV schema for reproducible backtests.

Data: https://github.com/f1db/f1db, CC BY 4.0. This converts the ordered result
records to classification positions (including distinct DNF/DNS/DSQ places).
No final grid, pit-stop, fastest-lap or race-lap data becomes a predictor.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from f1_data import _build_qualifying_frame, validate_master  # noqa: E402
from weather import WeatherClient  # noqa: E402


def read_yaml(path):
    # BaseLoader avoids YAML 1.1 silently coercing times into sexagesimal integers.
    return yaml.load(path.read_text(), Loader=yaml.BaseLoader)


def import_checkout(source, seasons, *, weather=False, cache_dir="f1_cache"):
    root = Path(source) / "src" / "data"
    forecasts = WeatherClient(Path(cache_dir) / "weather", enabled=weather)
    frames = []
    for year in seasons:
        for folder in sorted((root / "seasons" / str(year) / "races").glob("*")):
            if not (folder / "race-results.yml").exists():
                continue
            meta = read_yaml(folder / "race.yml")
            race_results = read_yaml(folder / "race-results.yml")
            qualifying = read_yaml(folder / "qualifying-results.yml")
            circuit = read_yaml(root / "circuits" / (meta["circuitId"] + ".yml"))
            qualifiers = []
            for index, result in enumerate(qualifying, 1):
                driver = read_yaml(root / "drivers" / (result["driverId"] + ".yml"))
                position = result["position"]
                qualifiers.append(
                    {
                        "DriverNumber": result["driverNumber"],
                        "DriverId": result["driverId"],
                        "Abbreviation": driver["abbreviation"],
                        "TeamId": result["constructorId"],
                        "TeamName": result["constructorId"],
                        "Position": int(position) if position.isdigit() else index,
                        **{name.upper(): result.get(name, "") for name in ("q1", "q2", "q3")},
                    }
                )
            qframe = _build_qualifying_frame(SimpleNamespace(results=pd.DataFrame(qualifiers)))
            # Retain the legacy min(Q1,Q2,Q3) gap solely for the baseline comparison.
            from f1_data import _timedelta_to_seconds

            times = (
                pd.DataFrame(qualifiers)[["Q1", "Q2", "Q3"]].map(_timedelta_to_seconds).min(axis=1)
            )
            best = times.min()
            qframe["LegacyQualiNorm"] = ((times - best) / best).fillna(
                ((times - best) / best).max() + 0.005
            )
            rows = []
            for index, result in enumerate(race_results, 1):
                driver = read_yaml(root / "drivers" / (result["driverId"] + ".yml"))
                retired = bool(result.get("reasonRetired")) or result["position"] in {
                    "DNF",
                    "DNS",
                    "DSQ",
                    "NC",
                    "DNQ",
                }
                rows.append(
                    {
                        "DriverId": result["driverId"],
                        "DriverNumber": result["driverNumber"],
                        "Abbreviation": driver["abbreviation"],
                        "TeamId": result["constructorId"],
                        "TeamName": result["constructorId"],
                        "FinishPos": index,
                        "DNF": int(retired),
                        "Points": float(result.get("points") or 0),
                    }
                )
            frame = pd.DataFrame(rows).merge(
                qframe.drop(
                    columns=["DriverNumber", "Abbreviation", "TeamId", "TeamName", "FieldSize"]
                ),
                on="DriverId",
                how="left",
                validate="one_to_one",
            )
            frame["QualiPos"] = frame["QualiPos"].fillna(len(frame))
            frame["QualiMissing"] = frame["QualiMissing"].fillna(1)
            frame["QualiSegment"] = frame["QualiSegment"].fillna(0)
            frame["Year"] = year
            frame["Round"] = int(meta["round"])
            frame["GPName"] = meta["grandPrixId"]
            frame["CircuitKey"] = meta["circuitId"]
            frame["RaceStart"] = (
                pd.Timestamp(meta["date"] + "T" + meta["time"] + "Z")
                if meta.get("time")
                else pd.NaT
            )
            frame["Latitude"] = float(circuit["latitude"])
            frame["Longitude"] = float(circuit["longitude"])
            weather_data = forecasts.for_race(
                frame["Latitude"].iloc[0], frame["Longitude"].iloc[0], frame["RaceStart"].iloc[0]
            )
            for key, value in weather_data.items():
                frame[key] = value
            frames.append(frame)
    return validate_master(pd.concat(frames, ignore_index=True))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--seasons", type=int, nargs="+", default=[2014, 2022, 2023, 2024, 2025])
    parser.add_argument("--weather", action="store_true")
    parser.add_argument("--output", type=Path, default=Path("outputs/f1db_race_data.csv"))
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    frame = import_checkout(args.source, args.seasons, weather=args.weather)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.output, index=False)
    print(frame.groupby("Year").agg(races=("Round", "nunique"), drivers=("DriverId", "size")))


if __name__ == "__main__":
    main()
