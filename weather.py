"""Race-window forecasts. Observed race weather is never a predictor.

Historical inputs use Open-Meteo's fixed-lead previous-runs archive, not
reanalysis or the historical API's stitched latest forecasts.
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
from pathlib import Path

import numpy as np
import pandas as pd
import requests

LOGGER = logging.getLogger(__name__)
VARIABLES = {
    "temperature_2m": "ForecastTempC",
    "precipitation": "ForecastPrecipMM",
    "wind_speed_10m": "ForecastWindKph",
    "relative_humidity_2m": "ForecastHumidity",
}
WEATHER_FEATURES = [*VARIABLES.values(), "ForecastWet", "WeatherMissing"]
ARCHIVE_START = pd.Timestamp("2024-01-01", tz="UTC")


def utc(value) -> pd.Timestamp:
    timestamp = pd.Timestamp(value)
    return timestamp.tz_localize("UTC") if timestamp.tzinfo is None else timestamp.tz_convert("UTC")


def missing_weather(reason: str) -> dict:
    return {
        **dict.fromkeys(VARIABLES.values(), np.nan),
        "ForecastWet": np.nan,
        "WeatherMissing": 1,
        "WeatherSource": reason,
        "WeatherForecastBefore": "",
    }


class WeatherClient:
    """Cached, bounded requests; an outage is missing data, never invented sunshine."""

    def __init__(self, cache_dir: Path | str = "f1_cache/weather", enabled: bool = True):
        self.cache_dir = Path(cache_dir)
        self.enabled = enabled
        self.session = requests.Session()
        self.failures = 0

    def for_race(self, latitude, longitude, race_start, *, now=None, cutoff=None) -> dict:
        if not self.enabled:
            return missing_weather("disabled")
        if pd.isna(race_start) or pd.isna(latitude) or pd.isna(longitude):
            return missing_weather("missing circuit coordinates or race start")
        start = utc(race_start)
        now = utc(now) if now is not None else pd.Timestamp.now(tz="UTC")
        cutoff = utc(cutoff) if cutoff is not None else start - pd.Timedelta(hours=3)
        cutoff = min(cutoff, now)
        if cutoff >= start:
            raise ValueError("Weather cutoff must precede the race start.")
        # Include every hourly bucket intersecting a three-hour race window.
        first_hour = start.floor("h")
        end = start + pd.Timedelta(hours=3)
        last_hour = end.ceil("h") - pd.Timedelta(hours=1)
        historical = now > cutoff
        suffix = ""
        params = {
            "latitude": float(latitude),
            "longitude": float(longitude),
            "start_date": first_hour.date().isoformat(),
            "end_date": last_hour.date().isoformat(),
            "timezone": "UTC",
            "wind_speed_unit": "kmh",
            "temperature_unit": "celsius",
            "precipitation_unit": "mm",
            "models": "ecmwf_ifs025",
        }
        if historical:
            if first_hour < ARCHIVE_START:
                return missing_weather("forecast archive unavailable before 2024")
            # A six-hour margin also keeps model-publication lag before cutoff.
            lead_days = max(1, math.ceil((last_hour - cutoff).total_seconds() / 86400 + 6 / 24))
            if lead_days > 7:
                return missing_weather("cutoff exceeds archived forecast lead times")
            suffix = f"_previous_day{lead_days}"
            url = "https://previous-runs-api.open-meteo.com/v1/forecast"
            issued_before = last_hour - pd.Timedelta(days=lead_days)
            source = f"open-meteo previous-day{lead_days} forecast"
        else:
            if start - now > pd.Timedelta(days=10):
                return missing_weather("outside forecast horizon")
            url = "https://api.open-meteo.com/v1/forecast"
            issued_before = now
            source = "open-meteo live forecast"
        params["hourly"] = ",".join(name + suffix for name in VARIABLES)
        key = hashlib.sha256(json.dumps([url, params], sort_keys=True).encode()).hexdigest()
        path = self.cache_dir / f"{key}.json"
        payload = None
        try:
            cached = json.loads(path.read_text())
            fetched = utc(cached["fetched_at"])
            if historical or (fetched <= cutoff and now - fetched < pd.Timedelta(hours=1)):
                payload = cached["payload"]
                if not historical:
                    issued_before = fetched
        except (OSError, ValueError, KeyError, TypeError):
            pass
        if payload is None:
            if self.failures >= 3:
                return missing_weather("weather service unavailable for this run")
            try:
                response = self.session.get(url, params=params, timeout=(5, 15))
                response.raise_for_status()
                payload = response.json()
                # Validate before caching (including complete hourly coverage).
                self._aggregate(payload, suffix, first_hour, last_hour)
                self.cache_dir.mkdir(parents=True, exist_ok=True)
                path.write_text(json.dumps({"fetched_at": now.isoformat(), "payload": payload}))
                self.failures = 0
            except (requests.RequestException, ValueError, KeyError, TypeError, OSError) as exc:
                self.failures += 1
                LOGGER.warning("Weather unavailable for %s: %s", start, exc)
                return missing_weather("weather request failed")
        try:
            features = self._aggregate(payload, suffix, first_hour, last_hour)
        except (ValueError, KeyError, TypeError) as exc:
            LOGGER.warning("Invalid cached weather for %s: %s", start, exc)
            return missing_weather("incomplete weather response")
        return {
            **features,
            "WeatherMissing": 0,
            "WeatherSource": source,
            # This is a conservative bound, not an invented model issue timestamp.
            "WeatherForecastBefore": issued_before.isoformat(),
        }

    @staticmethod
    def _aggregate(payload, suffix, first_hour, last_hour) -> dict:
        hourly = pd.DataFrame(payload["hourly"])
        hourly.index = pd.to_datetime(hourly.pop("time"), utc=True)
        expected = pd.date_range(first_hour, last_hour, freq="h")
        values = hourly.reindex(expected)[[name + suffix for name in VARIABLES]]
        values = values.apply(pd.to_numeric, errors="coerce")
        if values.empty or not np.isfinite(values.to_numpy()).all():
            raise ValueError("Missing or non-finite forecast hours/variables.")
        features = {}
        for variable, feature in VARIABLES.items():
            series = values[variable + suffix]
            features[feature] = float(
                series.sum() if variable == "precipitation" else series.mean()
            )
        features["ForecastWet"] = float(features["ForecastPrecipMM"] >= 0.2)
        return features
