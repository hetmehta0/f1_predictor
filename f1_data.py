"""Cached Jolpica/Ergast results, preserving classification and stable driver IDs."""

from __future__ import annotations

import hashlib
import json
import logging
import re
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

if __package__:
    from .weather import WeatherClient, utc
else:
    from weather import WeatherClient, utc

LOGGER = logging.getLogger(__name__)
FINISHED = re.compile(r"^\+\s*\d+\s+Laps?$", re.IGNORECASE)
BASE_URL = "https://api.jolpi.ca/ergast/f1"


def is_dnf(status) -> bool:
    if status is None or pd.isna(status) or not str(status).strip():
        raise ValueError("Missing race status cannot establish a DNF.")
    value = str(status).strip()
    return value.lower() != "finished" and not bool(FINISHED.fullmatch(value))


def _timedelta_to_seconds(value) -> float:
    if value is None or pd.isna(value):
        return np.nan
    if hasattr(value, "total_seconds"):
        return float(value.total_seconds())
    try:
        # Jolpica uses m:ss.sss; FastF1 exposes Timedelta objects.
        minutes, seconds = str(value).rsplit(":", 1)
        return float(minutes) * 60 + float(seconds)
    except (ValueError, TypeError):
        return np.nan


def _build_qualifying_frame(quali_session) -> pd.DataFrame:
    results = quali_session.results.copy().reset_index(drop=True)
    if results.empty:
        raise ValueError("Qualifying results are not available yet.")
    n = len(results)
    output = results.reindex(
        columns=["DriverNumber", "DriverId", "Abbreviation", "TeamName", "TeamId"]
    ).copy()
    output["DriverNumber"] = output["DriverNumber"].astype(str)
    output["DriverId"] = output["DriverId"].replace("", np.nan).fillna(output["Abbreviation"])
    output["TeamId"] = output["TeamId"].replace("", np.nan).fillna(output["TeamName"])
    output["QualiPos"] = (
        pd.to_numeric(results.get("Position"), errors="coerce").where(lambda x: x > 0).fillna(n)
    )
    output["QualiNorm"] = np.nan
    output["LapTimeSeconds"] = np.nan
    output["QualiSegment"] = 0
    # Compare each driver's last completed segment against that segment's best.
    # Comparing a dry Q1 time directly with a wet Q3 time is misleading.
    for segment, name in enumerate(("Q1", "Q2", "Q3"), 1):
        times = results.get(name, pd.Series(np.nan, index=results.index)).map(_timedelta_to_seconds)
        times = times.where(times > 0)
        fastest = times.min()
        valid = times.notna()
        if pd.notna(fastest):
            output.loc[valid, "QualiNorm"] = (times[valid] - fastest) / fastest
            output.loc[valid, "LapTimeSeconds"] = times[valid]
            output.loc[valid, "QualiSegment"] = segment
    output["QualiMissing"] = output["QualiNorm"].isna().astype(int)
    output["FieldSize"] = n
    return output


def qualifying_frame(entry: dict) -> pd.DataFrame:
    rows = []
    for result in entry.get("QualifyingResults", []):
        driver, team = result["Driver"], result["Constructor"]
        rows.append(
            {
                "DriverNumber": result.get("number", driver.get("permanentNumber", "")),
                "DriverId": driver["driverId"],
                "Abbreviation": driver.get("code", driver["driverId"]),
                "TeamId": team["constructorId"],
                "TeamName": team["name"],
                "Position": result.get("position"),
                **{name: result.get(name) for name in ("Q1", "Q2", "Q3")},
            }
        )
    return _build_qualifying_frame(SimpleNamespace(results=pd.DataFrame(rows)))


def race_metadata(entry: dict) -> dict:
    location = entry["Circuit"]["Location"]
    # A missing start time is unknown, not midnight disguised as an exact time.
    start = utc(entry["date"] + "T" + entry["time"]) if entry.get("time") else pd.NaT
    return {
        "Year": int(entry["season"]),
        "Round": int(entry["round"]),
        "GPName": entry["raceName"],
        "CircuitKey": entry["Circuit"]["circuitId"],
        "RaceStart": start,
        "Latitude": pd.to_numeric(location.get("lat"), errors="coerce"),
        "Longitude": pd.to_numeric(location.get("long"), errors="coerce"),
    }


class F1DataClient:
    def __init__(self, cache_dir: Path | str = "f1_cache"):
        self.cache_dir = Path(cache_dir) / "jolpica"
        self.session = requests.Session()
        self.session.mount(
            "https://",
            HTTPAdapter(
                max_retries=Retry(
                    total=2,
                    backoff_factor=0.5,
                    status_forcelist=[500, 502, 503, 504],
                    respect_retry_after_header=False,
                )
            ),
        )

    def races(self, year: int, kind: str, round_number: int | None = None) -> list[dict]:
        if kind not in {"results", "qualifying"}:
            raise ValueError("Expected results or qualifying.")
        endpoint = (
            f"{year}/" + (f"{round_number}/" if round_number is not None else "") + f"{kind}.json"
        )
        result_key = "Results" if kind == "results" else "QualifyingResults"
        races = {}
        offset = 0
        while True:
            params = {"limit": 100, "offset": offset}
            key = hashlib.sha256(
                json.dumps([endpoint, params], sort_keys=True).encode()
            ).hexdigest()
            path = self.cache_dir / f"{key}.json"
            now = pd.Timestamp.now(tz="UTC")
            payload = None
            try:
                cached = json.loads(path.read_text())
                if now - utc(cached["fetched_at"]) < pd.Timedelta(hours=6):
                    payload = cached["payload"]
            except (OSError, ValueError, KeyError, TypeError):
                pass
            if payload is None:
                response = self.session.get(
                    f"{BASE_URL}/{endpoint}", params=params, timeout=(5, 20)
                )
                response.raise_for_status()
                payload = response.json()
                self.cache_dir.mkdir(parents=True, exist_ok=True)
                path.write_text(json.dumps({"fetched_at": now.isoformat(), "payload": payload}))
            data = payload["MRData"]
            page_count = 0
            for race in data["RaceTable"]["Races"]:
                identity = (race["season"], race["round"])
                rows = race[result_key]
                page_count += len(rows)
                if identity not in races:
                    races[identity] = {**race, result_key: list(rows)}
                else:
                    races[identity][result_key].extend(rows)
            offset += page_count
            if offset >= int(data["total"]):
                break
            if not page_count:
                raise RuntimeError(f"Incomplete pagination for {endpoint} at offset {offset}.")
        return list(races.values())


def build_race_frame(race: dict, qualifying: dict) -> pd.DataFrame:
    qualifiers = qualifying_frame(qualifying)
    rows = []
    for result in race["Results"]:
        driver, team = result["Driver"], result["Constructor"]
        rows.append(
            {
                "DriverId": driver["driverId"],
                "DriverNumber": str(result.get("number", driver.get("permanentNumber", ""))),
                "Abbreviation": driver.get("code", driver["driverId"]),
                "TeamName": team["name"],
                "TeamId": team["constructorId"],
                "FinishPos": float(result["position"]),
                "Points": float(result.get("points", 0)),
                "Status": result["status"],
                "DNF": float(is_dnf(result["status"])),
                "ActualGridPos": float(result.get("grid", 0)),
            }
        )
    frame = pd.DataFrame(rows)
    if frame.empty:
        raise ValueError("Race has no classification.")
    fields = ["DriverId", "QualiPos", "QualiNorm", "LapTimeSeconds", "QualiSegment", "QualiMissing"]
    frame = frame.merge(qualifiers[fields], on="DriverId", how="left", validate="one_to_one")
    frame["FieldSize"] = len(frame)
    frame["QualiPos"] = frame["QualiPos"].fillna(len(frame))
    frame["QualiMissing"] = frame["QualiMissing"].fillna(1)
    frame["QualiSegment"] = frame["QualiSegment"].fillna(0)
    for key, value in race_metadata(race).items():
        frame[key] = value
    return frame


def validate_master(frame: pd.DataFrame) -> pd.DataFrame:
    required = {
        "Year",
        "Round",
        "DriverId",
        "Abbreviation",
        "TeamId",
        "TeamName",
        "CircuitKey",
        "GPName",
        "QualiPos",
        "QualiNorm",
        "FinishPos",
        "DNF",
        "Points",
    }
    if missing := required - set(frame.columns):
        raise ValueError(f"Dataset is missing columns: {sorted(missing)}")
    if frame.empty:
        raise ValueError("No completed races available.")
    frame = frame.copy()
    keys = ["Year", "Round", "DriverId"]
    if (
        frame[keys].isna().any().any()
        or frame.duplicated(keys).any()
        or frame["DriverId"].astype(str).str.strip().eq("").any()
    ):
        raise ValueError("Missing or duplicate race/driver identities.")
    for name in ("Year", "Round", "QualiPos", "FinishPos", "DNF", "Points"):
        frame[name] = pd.to_numeric(frame[name], errors="raise")
        if not np.isfinite(frame[name]).all():
            raise ValueError(f"Non-finite {name}.")
    for name in ("Year", "Round", "QualiPos"):
        if (frame[name] < 1).any() or (frame[name] % 1 != 0).any():
            raise ValueError(f"{name} must contain positive integers.")
    if not frame["DNF"].isin([0, 1]).all():
        raise ValueError("DNF must contain explicit 0/1 statuses.")
    for (year, round_number), race in frame.groupby(["Year", "Round"]):
        positions = sorted(race["FinishPos"].tolist())
        if len(race) < 2 or positions != list(range(1, len(race) + 1)):
            raise ValueError(
                f"Incomplete/duplicate classification for {year} round {round_number}."
            )
    frame["FieldSize"] = frame.groupby(["Year", "Round"])["DriverId"].transform("size")
    frame["QualiMissing"] = frame.get("QualiMissing", frame["QualiNorm"].isna().astype(int))
    frame["QualiSegment"] = frame.get("QualiSegment", 0)
    return frame.sort_values(["Year", "Round", "DriverId"]).reset_index(drop=True)


def build_master_dataframe(seasons, cache_dir="f1_cache", weather=True) -> pd.DataFrame:
    client = F1DataClient(cache_dir)
    forecasts = WeatherClient(Path(cache_dir) / "weather", enabled=weather)
    frames = []
    for year in sorted(set(seasons)):
        LOGGER.info("Loading %s results and qualifying", year)
        races = client.races(year, "results")
        if not races:
            LOGGER.warning("No completed results for %s", year)
            continue
        qualifiers = {int(q["round"]): q for q in client.races(year, "qualifying")}
        for race in races:
            round_number = int(race["round"])
            if round_number not in qualifiers:
                raise RuntimeError(f"Missing qualifying for completed race {year}/{round_number}.")
            frame = build_race_frame(race, qualifiers[round_number])
            meta = frame.iloc[0]
            if pd.notna(meta["RaceStart"]) and utc(meta["RaceStart"]) > pd.Timestamp.now(tz="UTC"):
                continue
            for key, value in forecasts.for_race(
                meta["Latitude"], meta["Longitude"], meta["RaceStart"]
            ).items():
                frame[key] = value
            frames.append(frame)
    if not frames:
        raise RuntimeError(
            "No completed race data was collected. Check seasons and network access."
        )
    return validate_master(pd.concat(frames, ignore_index=True))


def load_prediction_frame(year, round_number, cache_dir="f1_cache", weather=True) -> pd.DataFrame:
    entries = F1DataClient(cache_dir).races(year, "qualifying", round_number)
    if not entries:
        raise RuntimeError(f"Qualifying for {year} round {round_number} is not published yet.")
    entry = entries[0]
    frame = qualifying_frame(entry)
    meta = race_metadata(entry)
    for key, value in meta.items():
        frame[key] = value
    forecast = WeatherClient(Path(cache_dir) / "weather", enabled=weather).for_race(
        meta["Latitude"], meta["Longitude"], meta["RaceStart"]
    )
    for key, value in forecast.items():
        frame[key] = value
    return frame
