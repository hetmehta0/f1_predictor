import numpy as np
import pandas as pd
import pytest

from features import FEATURE_COLUMNS, add_historical_features
from modeling import chronological_splits, fit_model, metrics_for_predictions, rank_predictions
from pipeline import _predict_race_with_artifacts, train_full_pipeline


def test_future_and_same_race_results_cannot_change_features(raw_races):
    original = add_historical_features(raw_races)
    changed = raw_races.copy()
    changed.loc[changed["Year"] == 2024, ["FinishPos", "DNF", "Points"]] = [22, 1, 99]
    updated = add_historical_features(changed)
    cutoff = (original["Year"] < 2024) | ((original["Year"] == 2024) & (original["Round"] == 1))
    pd.testing.assert_frame_equal(
        original.loc[cutoff, FEATURE_COLUMNS], updated.loc[cutoff, FEATURE_COLUMNS]
    )
    assert original.query("Year == 2023 and Round == 1")["DriverRecentFinish"].isna().all()


def test_current_season_and_separate_dnf_history(raw_races):
    featured = add_historical_features(raw_races)
    driver = featured.query("Year == 2024 and Round == 2 and DriverId == 'driver-21'").iloc[0]
    assert driver["DriverDNFRate"] > 0.5
    assert driver["DriverHistoryCount"] == 7
    assert np.isfinite(driver["DriverAvgFinish_Season"])


def test_chronological_validation_has_no_future_or_split_races(raw_races):
    for train, valid in chronological_splits(raw_races):
        train_events = set(map(tuple, raw_races.iloc[train][["Year", "Round"]].to_numpy()))
        valid_events = set(map(tuple, raw_races.iloc[valid][["Year", "Round"]].to_numpy()))
        assert max(train_events) < min(valid_events)
        assert train_events.isdisjoint(valid_events)
        assert len(train) % 22 == len(valid) % 22 == 0


def test_imputation_is_fitted_on_training_only(raw_races):
    features = add_historical_features(raw_races)
    train = features.query("Year == 2023 and Round == 1")
    model = fit_model(train)
    future = features.query("Year == 2024").copy()
    future["DriverRecentFinish"] = 999
    assert model.feature_medians["DriverRecentFinish"] == 0
    assert model.transform(train)["DriverRecentFinish"].eq(0).all()


def test_order_and_membership_are_distinct_metrics(raw_races):
    frame = raw_races.query("Year == 2023 and Round == 1").copy()
    scores = list(range(10, 0, -1)) + list(range(11, 23))
    metrics, _ = metrics_for_predictions(frame, scores)
    assert metrics["Top-10 membership (%)"] == 100
    assert metrics["Top-10 exact positions (%)"] == 0
    assert metrics["Winner accuracy (%)"] == 0


def test_score_ties_do_not_reuse_actual_result_order(raw_races):
    race = raw_races.query("Year == 2023 and Round == 1")
    a = rank_predictions(race, np.zeros(len(race)))
    b = rank_predictions(race.sample(frac=1, random_state=7), np.zeros(len(race)))
    assert a["DriverId"].tolist() == b["DriverId"].tolist()


def test_pairwise_ranker_returns_consistent_full_field(raw_races):
    features = add_historical_features(raw_races)
    train = features.query("Year == 2023")
    test = features.query("Year == 2024 and Round == 1")
    model = fit_model(train, "pairwise")
    scores = model.predict(test)
    assert np.isfinite(scores).all()
    assert scores.min() >= 1 and scores.max() <= 22
    assert scores.sum() == pytest.approx(22 * 23 / 2)
    shuffled = test.sample(frac=1, random_state=4)
    score_map = dict(zip(test["DriverId"], scores, strict=True))
    assert model.predict(shuffled) == pytest.approx([score_map[d] for d in shuffled["DriverId"]])


def test_historical_prediction_refits_without_target_or_future_results(raw_races, tmp_path):
    artifacts, _, metrics = train_full_pipeline(
        seasons=[2023, 2024],
        master_df=raw_races,
        cache_dir=tmp_path,
        test_year=2024,
        model="forest",
        weather=False,
    )
    first = _predict_race_with_artifacts(2024, 2, artifacts, top_n=22)
    future = (artifacts.master_df["Year"] == 2024) & (artifacts.master_df["Round"] >= 2)
    artifacts.master_df.loc[future, ["FinishPos", "Points", "DNF"]] = [22, 100, 1]
    second = _predict_race_with_artifacts(2024, 2, artifacts, top_n=22)
    pd.testing.assert_frame_equal(first, second)
    assert first.attrs["trained_through"] == (2024, 1)
    assert len(first) == 22
    assert len(metrics) == 2
    assert set(artifacts.evaluation_predictions["ModelTrainedThrough"]) == {
        "2023-6",
        "2024-1",
        "2024-2",
    }
