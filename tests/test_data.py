from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pandas as pd
import pytest

from f1_data import F1DataClient, _build_qualifying_frame, build_race_frame, is_dnf, validate_master
from run_pipeline import parse_args


@pytest.mark.parametrize("status", ["Finished", "+1 Lap", "+ 1 Lap", "+2 Laps"])
def test_finish_status_is_not_dnf(status):
    assert not is_dnf(status)


@pytest.mark.parametrize("status", ["Engine", "Accident", "Disqualified", "Did not start"])
def test_nonfinishing_status(status):
    assert is_dnf(status)


def test_unknown_status_is_not_invented():
    with pytest.raises(ValueError, match="Missing"):
        is_dnf(None)


def qualifying_results(n=22):
    return pd.DataFrame(
        {
            "DriverNumber": range(1, n + 1),
            "DriverId": [f"d{i}" for i in range(n)],
            "Abbreviation": [f"D{i}" for i in range(n)],
            "TeamName": "Team",
            "TeamId": "team",
            "Position": range(1, n + 1),
            "Q1": [pd.Timedelta(seconds=70 + i) for i in range(n)],
        }
    )


def test_qualifying_keeps_22_cars():
    frame = _build_qualifying_frame(SimpleNamespace(results=qualifying_results()))
    assert frame["QualiPos"].tolist() == list(range(1, 23))
    assert (frame["FieldSize"] == 22).all()


def test_qualifying_compares_the_same_segment():
    data = qualifying_results(3)
    data["Q3"] = [pd.Timedelta(seconds=100), pd.Timedelta(seconds=102), pd.NaT]
    frame = _build_qualifying_frame(SimpleNamespace(results=data))
    assert frame.loc[0, "LapTimeSeconds"] == 100
    assert frame.loc[1, "QualiNorm"] == pytest.approx(0.02)
    assert frame.loc[2, "QualiNorm"] == pytest.approx(2 / 70)


def test_missing_quali_lap_is_explicit():
    data = qualifying_results(2)
    data.loc[1, "Q1"] = pd.NaT
    frame = _build_qualifying_frame(SimpleNamespace(results=data))
    assert frame.loc[1, "QualiMissing"] == 1
    assert np.isnan(frame.loc[1, "QualiNorm"])


def test_classification_preserves_distinct_retirements():
    meta = {
        "season": "2026",
        "round": "1",
        "raceName": "Test",
        "date": "2026-03-01",
        "time": "05:00:00Z",
        "Circuit": {"circuitId": "test", "Location": {"lat": "1", "long": "2"}},
    }
    drivers = [
        {
            "Driver": {"driverId": f"d{i}", "code": f"D{i}"},
            "Constructor": {"constructorId": "team", "name": "Team"},
            "number": str(i),
            "position": str(i),
            "Q1": "1:20.000",
            "points": "0",
            "status": "Finished" if i < 20 else "Engine",
        }
        for i in range(1, 23)
    ]
    frame = build_race_frame({**meta, "Results": drivers}, {**meta, "QualifyingResults": drivers})
    assert frame["FinishPos"].tolist() == list(range(1, 23))
    assert frame["DNF"].sum() == 3
    validate_master(frame)


def test_dataset_rejects_duplicate_or_incomplete_results(raw_races):
    with pytest.raises(ValueError, match="classification"):
        validate_master(raw_races.drop(index=[0]))
    with pytest.raises(ValueError, match="duplicate"):
        validate_master(pd.concat([raw_races, raw_races.head(1)]))


def test_pagination_reassembles_a_race_split_across_pages(tmp_path):
    client = F1DataClient(tmp_path)
    pages = [
        {
            "MRData": {
                "total": "3",
                "RaceTable": {
                    "Races": [{"season": "2025", "round": "1", "Results": [{"id": 1}, {"id": 2}]}]
                },
            }
        },
        {
            "MRData": {
                "total": "3",
                "RaceTable": {"Races": [{"season": "2025", "round": "1", "Results": [{"id": 3}]}]},
            }
        },
    ]
    client.session.get = Mock(side_effect=[Mock(json=Mock(return_value=page)) for page in pages])
    races = client.races(2025, "results")
    assert [r["id"] for r in races[0]["Results"]] == [1, 2, 3]
    assert client.session.get.call_args_list[1].kwargs["params"]["offset"] == 2


def test_bad_prediction_arguments_fail_before_loading():
    with pytest.raises(SystemExit):
        parse_args(["--predict-year", "2026"])
    with pytest.raises(SystemExit):
        parse_args(["--top-n", "0"])
