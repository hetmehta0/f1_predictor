"""End-to-end Formula 1 race finish prediction pipeline."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import fastf1
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import GridSearchCV, GroupKFold

CACHE_DIR = Path("./f1_cache")
SEASONS: tuple[int, int, int] = (2014, 2022, 2026)
ERA_START_YEARS = set(SEASONS)
FEATURE_COLUMNS = [
    "QualiNorm",
    "QualiPos",
    "DriverAvgFinish_Circuit",
    "DriverAvgFinish_Season",
    "TeamDNFRate",
    "DriverChampPos",
    "IsFirstSeasonOfEra",
]
HISTORICAL_FEATURE_COLUMNS = [
    "DriverAvgFinish_Circuit",
    "DriverAvgFinish_Season",
    "TeamDNFRate",
    "DriverChampPos",
]
FINISHED_LAP_PATTERN = re.compile(r"^\+\d+\sLap(?:s)?$")


@dataclass
class PipelineArtifacts:
    """Container for trained model state and reusable metadata."""

    model: RandomForestRegressor
    feature_columns: list[str]
    feature_medians: dict[str, float]
    master_df: pd.DataFrame


TRAINED_ARTIFACTS: PipelineArtifacts | None = None


def enable_cache(cache_dir: Path | str = CACHE_DIR) -> Path:
    """Enable local FastF1 cache in the requested directory."""
    cache_path = Path(cache_dir)
    cache_path.mkdir(parents=True, exist_ok=True)
    fastf1.Cache.enable_cache(str(cache_path))
    return cache_path


def _timedelta_to_seconds(value: Any) -> float:
    """Convert timedelta-like values to seconds."""
    if value is None or pd.isna(value):
        return float("nan")
    if isinstance(value, pd.Timedelta):
        return float(value.total_seconds())
    if hasattr(value, "total_seconds"):
        return float(value.total_seconds())
    return float("nan")


def is_dnf(status: Any) -> bool:
    """Return True when race status indicates retirement/non-finish."""
    if status is None or pd.isna(status):
        return True
    status_text = str(status).strip()
    if not status_text:
        return True
    finished = status_text == "Finished" or bool(FINISHED_LAP_PATTERN.match(status_text))
    return not finished


def _round_numbers_for_season(year: int) -> list[int]:
    """Fetch all non-testing round numbers for a season."""
    schedule = fastf1.get_event_schedule(year, include_testing=False)
    round_numbers = pd.to_numeric(schedule["RoundNumber"], errors="coerce").dropna().astype(int).tolist()
    return sorted({round_number for round_number in round_numbers if round_number > 0})


def _build_qualifying_frame(quali_session: Any) -> pd.DataFrame:
    """Build qualifying features for all drivers in a qualifying session."""
    quali_results = quali_session.results.copy()
    if quali_results.empty:
        raise ValueError("Qualifying results are empty.")

    quali_results["DriverNumber"] = quali_results["DriverNumber"].astype(str)
    lap_columns = [column for column in ("Q1", "Q2", "Q3") if column in quali_results.columns]

    if lap_columns:
        lap_times_seconds = quali_results[lap_columns].apply(lambda column: column.map(_timedelta_to_seconds))
        best_quali_lap_seconds = lap_times_seconds.min(axis=1, skipna=True)
    else:
        best_quali_lap_seconds = pd.Series(np.nan, index=quali_results.index, dtype=float)

    pole_lap_seconds = best_quali_lap_seconds.min(skipna=True)
    if pd.isna(pole_lap_seconds) or pole_lap_seconds <= 0:
        raw_norm = pd.Series(np.nan, index=quali_results.index, dtype=float)
    else:
        raw_norm = (best_quali_lap_seconds - pole_lap_seconds) / pole_lap_seconds

    worst_norm = raw_norm.max(skipna=True)
    if pd.isna(worst_norm):
        worst_norm = 0.05
    imputed_quali_norm = raw_norm.fillna(float(worst_norm) + 0.005)

    lap_time_seconds = best_quali_lap_seconds.copy()
    if pd.notna(pole_lap_seconds) and pole_lap_seconds > 0:
        lap_time_seconds = lap_time_seconds.fillna(pole_lap_seconds * (1.0 + imputed_quali_norm))

    quali_pos = pd.to_numeric(quali_results.get("Position"), errors="coerce").fillna(20).clip(1, 20).astype(int)

    return pd.DataFrame(
        {
            "DriverNumber": quali_results["DriverNumber"],
            "Abbreviation": quali_results.get("Abbreviation"),
            "TeamName": quali_results.get("TeamName"),
            "LapTimeSeconds": lap_time_seconds.astype(float),
            "QualiNorm": imputed_quali_norm.astype(float),
            "QualiPos": quali_pos.astype(int),
        }
    )


def _load_round_dataframe(year: int, round_number: int) -> pd.DataFrame | None:
    """Load race + qualifying data and return one row per race driver."""
    try:
        quali_session = fastf1.get_session(year, round_number, "Q")
        race_session = fastf1.get_session(year, round_number, "R")
        quali_session.load()
        race_session.load()
    except Exception as exc:  # noqa: BLE001
        print(f"Warning: skipping {year} round {round_number} because session load failed: {exc}")
        return None

    race_results = race_session.results.copy()
    if race_results.empty:
        print(f"Warning: skipping {year} round {round_number} because race results are empty.")
        return None

    race_results["DriverNumber"] = race_results["DriverNumber"].astype(str)
    dnf_mask = race_results.get("Status", pd.Series(index=race_results.index, dtype=object)).map(is_dnf)
    raw_finish_pos = pd.to_numeric(race_results.get("Position"), errors="coerce").fillna(20)
    finish_pos = pd.Series(np.where(dnf_mask, 20, raw_finish_pos), index=race_results.index).clip(1, 20).astype(int)
    points = pd.to_numeric(race_results.get("Points"), errors="coerce").fillna(0.0).astype(float)

    race_df = pd.DataFrame(
        {
            "DriverNumber": race_results["DriverNumber"],
            "Abbreviation": race_results.get("Abbreviation"),
            "TeamName": race_results.get("TeamName"),
            "FinishPos": finish_pos,
            "Points": points,
        }
    )

    try:
        quali_df = _build_qualifying_frame(quali_session)
    except Exception as exc:  # noqa: BLE001
        print(f"Warning: skipping {year} round {round_number} because qualifying parse failed: {exc}")
        return None

    merged = race_df.merge(quali_df, on="DriverNumber", how="left", suffixes=("", "_Q"))
    merged["Abbreviation"] = merged["Abbreviation"].fillna(merged.get("Abbreviation_Q"))
    merged["TeamName"] = merged["TeamName"].fillna(merged.get("TeamName_Q"))
    merged = merged.drop(columns=[column for column in ("Abbreviation_Q", "TeamName_Q") if column in merged.columns])

    fallback_norm = float(quali_df["QualiNorm"].max(skipna=True)) + 0.005 if not quali_df.empty else 0.055
    merged["QualiNorm"] = pd.to_numeric(merged["QualiNorm"], errors="coerce").fillna(fallback_norm).astype(float)
    merged["QualiPos"] = pd.to_numeric(merged["QualiPos"], errors="coerce").fillna(20).clip(1, 20).astype(int)

    pole_lap_seconds = quali_df["LapTimeSeconds"].min(skipna=True) if not quali_df.empty else np.nan
    merged["LapTimeSeconds"] = pd.to_numeric(merged["LapTimeSeconds"], errors="coerce")
    if pd.notna(pole_lap_seconds) and pole_lap_seconds > 0:
        merged["LapTimeSeconds"] = merged["LapTimeSeconds"].fillna(pole_lap_seconds * (1.0 + merged["QualiNorm"]))

    event = race_session.event
    gp_name = str(event.get("EventName", f"{year} Round {round_number}"))
    circuit_key = str(event.get("Location", gp_name))

    merged["Year"] = year
    merged["Round"] = round_number
    merged["GPName"] = gp_name
    merged["CircuitKey"] = circuit_key
    merged["RegEra"] = year

    columns = [
        "Year",
        "Round",
        "GPName",
        "CircuitKey",
        "RegEra",
        "DriverNumber",
        "Abbreviation",
        "TeamName",
        "LapTimeSeconds",
        "QualiNorm",
        "QualiPos",
        "FinishPos",
        "Points",
    ]
    return merged[columns].reset_index(drop=True)


def build_master_dataframe(seasons: Sequence[int] = SEASONS) -> pd.DataFrame:
    """Build the complete race-driver dataset across configured seasons."""
    all_rows: list[pd.DataFrame] = []
    for year in seasons:
        for round_number in _round_numbers_for_season(year):
            round_df = _load_round_dataframe(year, round_number)
            if round_df is not None:
                all_rows.append(round_df)

    if not all_rows:
        raise RuntimeError("No race data was collected from FastF1.")

    master_df = pd.concat(all_rows, ignore_index=True)
    return master_df.sort_values(["Year", "Round", "FinishPos"]).reset_index(drop=True)


def _apply_historical_features_for_race(race_df: pd.DataFrame, history_df: pd.DataFrame) -> pd.DataFrame:
    """Add history-based features for one race using only past race rows."""
    race_feature_df = race_df.copy()
    year = int(race_feature_df["Year"].iloc[0])

    if history_df.empty:
        race_feature_df["DriverAvgFinish_Circuit"] = np.nan
        race_feature_df["DriverAvgFinish_Season"] = np.nan
        race_feature_df["TeamDNFRate"] = np.nan
        race_feature_df["DriverChampPos"] = np.nan
        race_feature_df["IsFirstSeasonOfEra"] = int(year in ERA_START_YEARS)
        return race_feature_df

    driver_circuit_avg = history_df.groupby(["Abbreviation", "CircuitKey"])["FinishPos"].mean()
    season_history = history_df[history_df["Year"] == year]
    driver_season_avg = season_history.groupby("Abbreviation")["FinishPos"].mean()
    team_dnf_rate = (
        history_df.assign(_DNF=(history_df["FinishPos"] == 20).astype(float))
        .groupby("TeamName")["_DNF"]
        .mean()
    )

    if season_history.empty:
        champ_pos_map: dict[str, float] = {}
    else:
        cumulative_points = season_history.groupby("Abbreviation")["Points"].sum()
        champ_pos_map = cumulative_points.rank(method="min", ascending=False).to_dict()

    race_feature_df["DriverAvgFinish_Circuit"] = [
        driver_circuit_avg.get((driver, circuit), np.nan)
        for driver, circuit in zip(race_feature_df["Abbreviation"], race_feature_df["CircuitKey"])
    ]
    race_feature_df["DriverAvgFinish_Season"] = race_feature_df["Abbreviation"].map(driver_season_avg)
    race_feature_df["TeamDNFRate"] = race_feature_df["TeamName"].map(team_dnf_rate)
    race_feature_df["DriverChampPos"] = race_feature_df["Abbreviation"].map(champ_pos_map)
    race_feature_df["IsFirstSeasonOfEra"] = int(year in ERA_START_YEARS)
    return race_feature_df


def _fill_feature_nulls_with_session_median(
    df: pd.DataFrame,
    feature_columns: Sequence[str],
    fallback_medians: dict[str, float] | None = None,
) -> pd.DataFrame:
    """Fill missing values race-by-race using session median, then fallbacks."""
    filled_df = df.copy()
    for feature in feature_columns:
        filled_df[feature] = filled_df.groupby(["Year", "Round"])[feature].transform(
            lambda series: series.fillna(series.median())
        )
        if fallback_medians and feature in fallback_medians:
            filled_df[feature] = filled_df[feature].fillna(float(fallback_medians[feature]))
        filled_df[feature] = filled_df[feature].fillna(filled_df[feature].median())
    return filled_df


def add_historical_features(master_df: pd.DataFrame) -> pd.DataFrame:
    """Add all required no-leakage historical features to the master dataframe."""
    sorted_df = master_df.sort_values(["Year", "Round", "Abbreviation"]).reset_index(drop=True)
    history_df = sorted_df.iloc[0:0].copy()
    featured_races: list[pd.DataFrame] = []

    for (_, _), race_df in sorted_df.groupby(["Year", "Round"], sort=True):
        race_with_features = _apply_historical_features_for_race(race_df, history_df)
        featured_races.append(race_with_features)
        history_df = pd.concat([history_df, race_df], ignore_index=True)

    featured_df = pd.concat(featured_races, ignore_index=True)
    featured_df = _fill_feature_nulls_with_session_median(featured_df, HISTORICAL_FEATURE_COLUMNS)
    return featured_df


def _prepare_train_test_frames(
    featured_df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, float]]:
    """Split train/test using the temporal split and apply median feature fills."""
    train_df = featured_df[featured_df["Year"].isin([2014, 2022])].copy()
    test_df = featured_df[featured_df["Year"] == 2026].copy()

    if train_df.empty:
        raise RuntimeError("Training set is empty after temporal split.")
    if test_df.empty:
        raise RuntimeError("Test set (2026) is empty after temporal split.")

    medians = train_df[FEATURE_COLUMNS].median().to_dict()
    for frame in (train_df, test_df):
        for feature in FEATURE_COLUMNS:
            frame[feature] = pd.to_numeric(frame[feature], errors="coerce").fillna(float(medians[feature]))

    return train_df, test_df, {feature: float(value) for feature, value in medians.items()}


def train_random_forest(train_df: pd.DataFrame) -> RandomForestRegressor:
    """Tune and train RandomForestRegressor on training races only."""
    x_train = train_df[FEATURE_COLUMNS]
    y_train = train_df["FinishPos"]
    groups = train_df["Year"].astype(str) + "-" + train_df["Round"].astype(str)

    param_grid = {
        "n_estimators": [100, 200, 500],
        "max_depth": [None, 10, 20],
    }
    base_model = RandomForestRegressor(random_state=42, n_jobs=-1)

    unique_groups = groups.nunique()
    if unique_groups >= 2:
        cv = GroupKFold(n_splits=min(5, unique_groups))
        grid = GridSearchCV(
            estimator=base_model,
            param_grid=param_grid,
            scoring="neg_mean_absolute_error",
            cv=cv,
            n_jobs=-1,
        )
        grid.fit(x_train, y_train, groups=groups)
        model = grid.best_estimator_
    else:
        model = RandomForestRegressor(
            n_estimators=200,
            max_depth=None,
            random_state=42,
            n_jobs=-1,
        )
        model.fit(x_train, y_train)

    return model


def feature_importance_table(model: RandomForestRegressor) -> pd.DataFrame:
    """Return feature importances sorted descending."""
    importance_df = pd.DataFrame(
        {
            "Feature": FEATURE_COLUMNS,
            "Importance": model.feature_importances_,
        }
    )
    return importance_df.sort_values("Importance", ascending=False).reset_index(drop=True)


def evaluate_model(model: RandomForestRegressor, test_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Evaluate model on 2026 races using MAE, mean race Spearman, and top-3 winner accuracy."""
    evaluation_df = test_df.copy()
    evaluation_df["PredictedFinishPos"] = model.predict(evaluation_df[FEATURE_COLUMNS])
    mae = mean_absolute_error(evaluation_df["FinishPos"], evaluation_df["PredictedFinishPos"])

    race_spearman_values: list[float] = []
    winner_top3_hits: list[int] = []

    for (_, _, _), race_df in evaluation_df.groupby(["Year", "Round", "GPName"], sort=True):
        corr, _ = spearmanr(race_df["FinishPos"], race_df["PredictedFinishPos"])
        if not np.isnan(corr):
            race_spearman_values.append(float(corr))

        winner_rows = race_df[race_df["FinishPos"] == 1]
        if winner_rows.empty:
            winner_top3_hits.append(0)
            continue
        winner = winner_rows.iloc[0]["Abbreviation"]
        predicted_top3 = race_df.nsmallest(3, "PredictedFinishPos")["Abbreviation"].tolist()
        winner_top3_hits.append(int(winner in predicted_top3))

    mean_spearman = float(np.mean(race_spearman_values)) if race_spearman_values else float("nan")
    top3_accuracy = float(np.mean(winner_top3_hits) * 100.0) if winner_top3_hits else float("nan")

    results_table = pd.DataFrame(
        {
            "Metric": [
                "MAE (Finish Position)",
                "Mean Spearman Rank Correlation (per race)",
                "Top-3 Accuracy for Actual Winner (%)",
            ],
            "Value": [mae, mean_spearman, top3_accuracy],
        }
    )
    return results_table, evaluation_df


def train_full_pipeline(
    seasons: Sequence[int] = SEASONS,
    cache_dir: Path | str = CACHE_DIR,
) -> tuple[PipelineArtifacts, pd.DataFrame, pd.DataFrame]:
    """Run full data->feature->train->evaluate pipeline and store trained artifacts."""
    enable_cache(cache_dir)
    master_df = build_master_dataframe(seasons=seasons)
    featured_df = add_historical_features(master_df)
    train_df, test_df, medians = _prepare_train_test_frames(featured_df)
    model = train_random_forest(train_df)
    importances = feature_importance_table(model)
    results_table, _ = evaluate_model(model, test_df)

    artifacts = PipelineArtifacts(
        model=model,
        feature_columns=list(FEATURE_COLUMNS),
        feature_medians=medians,
        master_df=featured_df,
    )

    global TRAINED_ARTIFACTS
    TRAINED_ARTIFACTS = artifacts
    return artifacts, importances, results_table


def _load_prediction_quali_frame(year: int, round_number: int) -> pd.DataFrame:
    """Load qualifying data for a target race and construct base prediction rows."""
    try:
        quali_session = fastf1.get_session(year, round_number, "Q")
        quali_session.load()
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(f"Unable to load qualifying data for {year} round {round_number}: {exc}") from exc

    quali_df = _build_qualifying_frame(quali_session)
    event = quali_session.event
    gp_name = str(event.get("EventName", f"{year} Round {round_number}"))
    circuit_key = str(event.get("Location", gp_name))

    prediction_df = quali_df.copy()
    prediction_df["Year"] = year
    prediction_df["Round"] = round_number
    prediction_df["GPName"] = gp_name
    prediction_df["CircuitKey"] = circuit_key
    prediction_df["RegEra"] = year
    return prediction_df[
        [
            "Year",
            "Round",
            "GPName",
            "CircuitKey",
            "RegEra",
            "DriverNumber",
            "Abbreviation",
            "TeamName",
            "LapTimeSeconds",
            "QualiNorm",
            "QualiPos",
        ]
    ]


def _predict_race_with_artifacts(year: int, round_number: int, artifacts: PipelineArtifacts) -> pd.DataFrame:
    """Predict top-10 finishing order for a race using provided artifacts."""
    race_df = _load_prediction_quali_frame(year, round_number)
    history_df = artifacts.master_df[
        (artifacts.master_df["Year"] < year)
        | ((artifacts.master_df["Year"] == year) & (artifacts.master_df["Round"] < round_number))
    ].copy()

    race_with_features = _apply_historical_features_for_race(race_df, history_df)
    race_with_features = _fill_feature_nulls_with_session_median(
        race_with_features,
        HISTORICAL_FEATURE_COLUMNS,
        fallback_medians=artifacts.feature_medians,
    )

    for feature in FEATURE_COLUMNS:
        race_with_features[feature] = pd.to_numeric(race_with_features[feature], errors="coerce").fillna(
            artifacts.feature_medians[feature]
        )

    race_with_features["PredictedFinishPos"] = artifacts.model.predict(race_with_features[FEATURE_COLUMNS])
    ranked_df = race_with_features.sort_values("PredictedFinishPos", ascending=True).reset_index(drop=True)
    ranked_df["Rank"] = np.arange(1, len(ranked_df) + 1)
    top_10 = ranked_df.loc[:9, ["Rank", "Abbreviation", "TeamName", "PredictedFinishPos"]].copy()
    top_10 = top_10.rename(
        columns={
            "Abbreviation": "Driver",
            "TeamName": "Team",
            "PredictedFinishPos": "PredictedFinishPosition",
        }
    )
    return top_10


def predict_race(year: int, round_number: int) -> pd.DataFrame:
    """Predict the top-10 for a race round after `train_full_pipeline` has been run."""
    if TRAINED_ARTIFACTS is None:
        raise RuntimeError("No trained model in memory. Run train_full_pipeline() before calling predict_race().")
    return _predict_race_with_artifacts(year=year, round_number=round_number, artifacts=TRAINED_ARTIFACTS)

