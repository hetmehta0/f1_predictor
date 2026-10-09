from unittest.mock import Mock

import numpy as np
import pandas as pd
import pytest
import requests

from weather import WeatherClient


def payload(suffix="_previous_day1"):
    hours = pd.date_range("2025-03-16", periods=24, freq="h")
    return {
        "hourly": {
            "time": [hour.isoformat() for hour in hours],
            "temperature_2m" + suffix: list(range(24)),
            "precipitation" + suffix: [0.1] * 24,
            "wind_speed_10m" + suffix: [10] * 24,
            "relative_humidity_2m" + suffix: [70] * 24,
        }
    }


def test_historical_weather_uses_pre_race_forecast_and_race_window(tmp_path):
    client = WeatherClient(tmp_path)
    client.session.get = Mock(return_value=Mock(json=Mock(return_value=payload())))
    result = client.for_race(-37, 145, "2025-03-16T04:00:00Z", now="2026-10-08")
    assert result["ForecastTempC"] == 5
    assert result["ForecastPrecipMM"] == pytest.approx(0.3)
    assert result["ForecastWet"] == 1
    assert result["WeatherMissing"] == 0
    assert pd.Timestamp(result["WeatherForecastBefore"]) < pd.Timestamp("2025-03-16T01:00:00Z")
    request = client.session.get.call_args
    assert "previous-runs-api" in request.args[0]
    assert "_previous_day1" in request.kwargs["params"]["hourly"]
    again = client.for_race(-37, 145, "2025-03-16T04:00:00Z", now="2026-10-08")
    assert result == again
    assert client.session.get.call_count == 1


def test_live_weather_path(tmp_path):
    client = WeatherClient(tmp_path)
    client.session.get = Mock(return_value=Mock(json=Mock(return_value=payload(""))))
    result = client.for_race(-37, 145, "2025-03-16T04:00:00Z", now="2025-03-15T22:00:00Z")
    assert result["WeatherSource"] == "open-meteo live forecast"
    assert client.session.get.call_args.args[0] == "https://api.open-meteo.com/v1/forecast"


def test_old_race_has_missing_forecast_not_realized_weather(tmp_path):
    client = WeatherClient(tmp_path)
    client.session.get = Mock()
    result = client.for_race(-37, 145, "2022-03-16T04:00:00Z", now="2026-10-08")
    assert result["WeatherMissing"] == 1
    assert np.isnan(result["ForecastWet"])
    client.session.get.assert_not_called()


def test_partial_response_not_presented_as_full_weather(tmp_path):
    client = WeatherClient(tmp_path)
    bad = payload()
    bad["hourly"]["precipitation_previous_day1"][5] = None
    client.session.get = Mock(return_value=Mock(json=Mock(return_value=bad)))
    result = client.for_race(-37, 145, "2025-03-16T04:00:00Z", now="2026-10-08")
    assert result["WeatherMissing"] == 1
    assert np.isnan(result["ForecastPrecipMM"])


def test_outage_stops_repeated_requests(tmp_path):
    client = WeatherClient(tmp_path)
    client.session.get = Mock(side_effect=requests.Timeout("unavailable"))
    for _ in range(5):
        result = client.for_race(-37, 145, "2025-03-16T04:00:00Z", now="2026-10-08")
        assert result["WeatherMissing"] == 1
    assert client.session.get.call_count == 3


def test_post_race_cutoff_is_rejected(tmp_path):
    with pytest.raises(ValueError, match="precede"):
        WeatherClient(tmp_path).for_race(
            1, 2, "2025-03-16T04:00Z", now="2026-10-08", cutoff="2025-03-16T05:00Z"
        )
