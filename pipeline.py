"""F1 data -> past-only features -> ranking -> chronological evaluation."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import fastf1
import pandas as pd

if __package__:
    from .f1_data import (
        _build_qualifying_frame,  # noqa: F401 - compatibility with original module
        is_dnf,  # noqa: F401
        load_prediction_frame,
        validate_master,
    )
    from .f1_data import (
        build_master_dataframe as _collect,
    )
    from .features import (
        FEATURE_COLUMNS,
        HISTORICAL_FEATURE_COLUMNS,  # noqa: F401
        _apply_historical_features_for_race,
        add_historical_features,
        before_race,
    )
    from .modeling import (
        FittedModel,
        fit_model,
        metrics_for_predictions,
        rank_predictions,
        select_model,
    )
    from .weather import missing_weather
else:
    from f1_data import (
        _build_qualifying_frame,  # noqa: F401 - compatibility with original module
        is_dnf,  # noqa: F401
        load_prediction_frame,
        validate_master,
    )
    from f1_data import (
        build_master_dataframe as _collect,
    )
    from features import (
        FEATURE_COLUMNS,
        HISTORICAL_FEATURE_COLUMNS,  # noqa: F401
        _apply_historical_features_for_race,
        add_historical_features,
        before_race,
    )
    from modeling import (
        FittedModel,
        fit_model,
        metrics_for_predictions,
        rank_predictions,
        select_model,
    )
    from weather import missing_weather

CACHE_DIR = Path("f1_cache")
SEASONS = tuple(range(max(2022, pd.Timestamp.now().year - 4), pd.Timestamp.now().year + 1))


@dataclass
class PipelineArtifacts:
    model: FittedModel
    feature_columns: list[str]
    feature_medians: dict[str, float]
    master_df: pd.DataFrame
    validation_results: pd.DataFrame
    evaluation_predictions: pd.DataFrame
    selected_method: str
    selection_through: tuple[int, int]
    cache_dir: Path
    weather: bool
    requested_method: str = "auto"


TRAINED_ARTIFACTS: PipelineArtifacts | None = None


def enable_cache(cache_dir: Path | str = CACHE_DIR) -> Path:
    path = Path(cache_dir)
    path.mkdir(parents=True, exist_ok=True)
    fastf1.set_log_level("WARNING")
    fastf1.Cache.enable_cache(str(path))
    return path


def build_master_dataframe(seasons=SEASONS, cache_dir=CACHE_DIR, weather=True):
    return _collect(seasons, cache_dir=cache_dir, weather=weather)


def train_random_forest(train_df: pd.DataFrame) -> FittedModel:
    """Compatibility helper; training-fold imputation now belongs to the model."""
    return fit_model(train_df, "forest")


def feature_importance_table(model: FittedModel) -> pd.DataFrame:
    if model.method != "forest":
        return pd.DataFrame(columns=["Feature", "Importance"])
    return (
        pd.DataFrame(
            {"Feature": FEATURE_COLUMNS, "Importance": model.estimator.feature_importances_}
        )
        .sort_values("Importance", ascending=False)
        .reset_index(drop=True)
    )


def evaluate_model(model: FittedModel, test_df: pd.DataFrame):
    metrics, predictions = metrics_for_predictions(test_df, model.predict(test_df))
    return pd.DataFrame({"Metric": metrics.keys(), "Value": metrics.values()}), predictions


def walk_forward_evaluate(featured: pd.DataFrame, test_year: int, method: str):
    test = featured.loc[featured["Year"] == test_year]
    if test.empty:
        raise ValueError(f"No completed races in evaluation year {test_year}.")
    predictions = []
    for (year, round_number), race in test.groupby(["Year", "Round"], sort=True):
        history = before_race(featured, year, round_number)
        model = fit_model(history, method)
        result = rank_predictions(race, model.predict(race))
        result["ModelTrainedThrough"] = f"{model.trained_through[0]}-{model.trained_through[1]}"
        result["Model"] = method
        predictions.append(result)
    evaluation = pd.concat(predictions, ignore_index=True)
    model_metrics, evaluation = metrics_for_predictions(
        evaluation, evaluation["PredictedFinishPos"]
    )
    baseline, _ = metrics_for_predictions(test, test["QualiPos"])
    return pd.DataFrame(
        [{"Model": method, **model_metrics}, {"Model": "qualifying baseline", **baseline}]
    ), evaluation


def train_full_pipeline(
    seasons=SEASONS,
    cache_dir=CACHE_DIR,
    *,
    master_df: pd.DataFrame | None = None,
    test_year: int | None = None,
    model: str = "auto",
    weather: bool = True,
) -> tuple[PipelineArtifacts, pd.DataFrame, pd.DataFrame]:
    cache_path = enable_cache(cache_dir)
    raw = (
        build_master_dataframe(seasons, cache_path, weather)
        if master_df is None
        else validate_master(master_df)
    )
    raw = raw.loc[raw["Year"].isin(seasons)].copy()
    if raw.empty:
        raise ValueError("No races match the selected seasons.")
    if test_year is not None and test_year not in raw["Year"].unique():
        raise ValueError(f"No completed races in evaluation year {test_year}.")
    if not weather:
        raw = raw.assign(**missing_weather("disabled"))
    featured = add_historical_features(raw)
    test_year = int(featured["Year"].max()) if test_year is None else test_year
    initial_train = featured.loc[featured["Year"] < test_year].copy()
    if initial_train.empty:
        raise ValueError(
            "Need completed races before the evaluation year. Include earlier --seasons."
        )
    selected, cv_results = select_model(initial_train, model)
    results, predictions = walk_forward_evaluate(featured, test_year, selected)
    final_model = fit_model(featured, selected)
    artifacts = PipelineArtifacts(
        final_model,
        list(FEATURE_COLUMNS),
        final_model.feature_medians,
        featured,
        cv_results,
        predictions,
        selected,
        max(zip(initial_train["Year"], initial_train["Round"], strict=True)),
        cache_path,
        weather,
        model,
    )
    global TRAINED_ARTIFACTS
    TRAINED_ARTIFACTS = artifacts
    return artifacts, feature_importance_table(final_model), results


def _load_prediction_quali_frame(year, round_number, cache_dir=CACHE_DIR, weather=True):
    return load_prediction_frame(year, round_number, cache_dir, weather)


def _predict_race_with_artifacts(
    year: int, round_number: int, artifacts: PipelineArtifacts, top_n=10
):
    if top_n < 1:
        raise ValueError("top_n must be positive.")
    history = before_race(artifacts.master_df, year, round_number)
    if history.empty:
        raise ValueError("No training races before the requested race.")
    existing = artifacts.master_df.loc[
        (artifacts.master_df["Year"] == year) & (artifacts.master_df["Round"] == round_number)
    ]
    if existing.empty:
        race = _load_prediction_quali_frame(
            year, round_number, artifacts.cache_dir, artifacts.weather
        )
    else:
        race = existing.drop(
            columns=["FinishPos", "Points", "DNF", "Status", "ActualGridPos"], errors="ignore"
        ).copy()
    method = artifacts.selected_method
    if (year, round_number) <= artifacts.selection_through:
        method, _ = select_model(history, artifacts.requested_method)
    model = (
        artifacts.model
        if (year, round_number) > artifacts.model.trained_through
        else fit_model(history, method)
    )
    prepared = _apply_historical_features_for_race(race, history)
    ranked = rank_predictions(prepared, model.predict(prepared)).head(top_n)
    output = (
        ranked[["PredictedRank", "Abbreviation", "TeamName", "PredictedFinishPos"]]
        .rename(
            columns={
                "PredictedRank": "Rank",
                "Abbreviation": "Driver",
                "TeamName": "Team",
                "PredictedFinishPos": "RankingScore",
            }
        )
        .reset_index(drop=True)
    )
    output.attrs.update(
        {
            "model": model.method,
            "trained_through": model.trained_through,
            "weather_source": prepared["WeatherSource"].iloc[0],
            "weather_missing": bool(prepared["WeatherMissing"].iloc[0]),
        }
    )
    return output


def predict_race(year: int, round_number: int, top_n=10):
    if TRAINED_ARTIFACTS is None:
        raise RuntimeError("Run train_full_pipeline() before predict_race().")
    return _predict_race_with_artifacts(year, round_number, TRAINED_ARTIFACTS, top_n)
