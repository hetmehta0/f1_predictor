"""Ranking-aware model selection and whole-race, forward-only validation."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestRegressor
from threadpoolctl import threadpool_limits

if __package__:
    from .features import FEATURE_COLUMNS
else:
    from features import FEATURE_COLUMNS
if __package__:
    from .weather import WEATHER_FEATURES
else:
    from weather import WEATHER_FEATURES


def chronological_splits(frame: pd.DataFrame, n_splits=3):
    """Expanding windows, never splitting drivers from one race across folds."""
    events = sorted(set(zip(frame["Year"], frame["Round"], strict=True)))
    if len(events) < 4:
        raise ValueError("Need at least four races for chronological model selection.")
    chunks = np.array_split(np.arange(len(events)), min(n_splits + 1, len(events)))
    row_events = list(zip(frame["Year"], frame["Round"], strict=True))
    for index in range(1, len(chunks)):
        train_events = {events[i] for chunk in chunks[:index] for i in chunk}
        valid_events = {events[i] for i in chunks[index]}
        yield (
            np.array([i for i, event in enumerate(row_events) if event in train_events]),
            np.array([i for i, event in enumerate(row_events) if event in valid_events]),
        )


def rank_predictions(frame: pd.DataFrame, scores) -> pd.DataFrame:
    result = frame.copy()
    result["PredictedFinishPos"] = np.asarray(scores)
    if not np.isfinite(result["PredictedFinishPos"]).all():
        raise ValueError("Model returned a non-finite ranking score.")
    # Input result order must never break ties: it usually contains the true answer.
    result = result.sort_values(
        ["Year", "Round", "PredictedFinishPos", "QualiPos", "DriverId"], kind="stable"
    )
    result["PredictedRank"] = result.groupby(["Year", "Round"]).cumcount() + 1
    return result


def metrics_for_predictions(frame: pd.DataFrame, scores) -> tuple[dict, pd.DataFrame]:
    ranked = rank_predictions(frame, scores)
    per_race = []
    for _, race in ranked.groupby(["Year", "Round"], sort=True):
        actual = race.sort_values("FinishPos")["DriverId"].tolist()
        predicted = race.sort_values("PredictedRank")["DriverId"].tolist()
        k = min(10, len(race))
        actual_top = set(actual[:k])
        per_race.append(
            {
                "Rank MAE": float((race["PredictedRank"] - race["FinishPos"]).abs().mean()),
                "Spearman": float(spearmanr(race["FinishPos"], race["PredictedRank"]).statistic),
                "Top-10 membership (%)": 100 * len(actual_top & set(predicted[:k])) / k,
                "Top-10 exact positions (%)": 100
                * sum(a == p for a, p in zip(actual[:k], predicted[:k], strict=True))
                / k,
                "Winner accuracy (%)": 100 * float(actual[0] == predicted[0]),
                "Winner in predicted top 3 (%)": 100 * float(actual[0] in predicted[:3]),
                "Exact podium (%)": 100 * float(actual[:3] == predicted[:3]),
            }
        )
    if not per_race:
        raise ValueError("Cannot evaluate an empty dataset.")
    summary = pd.DataFrame(per_race).mean().to_dict()
    summary["Races"] = len(per_race)
    return summary, ranked


def _pair_rows(values: np.ndarray, indices: np.ndarray):
    left, right = np.where(~np.eye(len(indices), dtype=bool))
    a, b = values[indices[left]], values[indices[right]]
    context_indices = [FEATURE_COLUMNS.index(name) for name in WEATHER_FEATURES]
    # Weather is shared by a whole race and cancels out in a difference vector.
    # Include it as context so the ranker can learn weather x driver interactions.
    context = (a[:, context_indices] + b[:, context_indices]) / 2
    pair_features = np.column_stack([a - b, context, (a[:, 1] + b[:, 1]) / 2])
    return pair_features, left, right


@dataclass
class FittedModel:
    method: str
    estimator: object
    feature_medians: dict[str, float]
    trained_through: tuple[int, int]

    def transform(self, frame: pd.DataFrame) -> pd.DataFrame:
        return (
            frame[FEATURE_COLUMNS]
            .apply(pd.to_numeric, errors="coerce")
            .replace([np.inf, -np.inf], np.nan)
            .fillna(self.feature_medians)
        )

    def predict(self, frame: pd.DataFrame) -> np.ndarray:
        x = self.transform(frame)
        if self.method == "qualifying":
            return frame["QualiPos"].to_numpy(dtype=float)
        if self.method == "forest":
            return self.estimator.predict(x)
        result = np.empty(len(frame), dtype=float)
        values = x.to_numpy()
        for indices in frame.reset_index(drop=True).groupby(["Year", "Round"]).indices.values():
            pair_x, left, right = _pair_rows(values, indices)
            with threadpool_limits(limits=2):
                p = self.estimator.predict_proba(pair_x)[:, 1]
            matrix = np.zeros((len(indices), len(indices)))
            matrix[left, right] = p
            # Enforce P(A beats B) = 1 - P(B beats A), then expected rank.
            symmetric = (matrix + 1 - matrix.T) / 2
            np.fill_diagonal(symmetric, 0)
            result[indices] = 1 + symmetric.sum(axis=0)
        return result


def fit_model(frame: pd.DataFrame, method="forest", *, min_samples_leaf=3) -> FittedModel:
    if frame.empty:
        raise ValueError("Training data is empty.")
    x = (
        frame[FEATURE_COLUMNS]
        .apply(pd.to_numeric, errors="coerce")
        .replace([np.inf, -np.inf], np.nan)
    )
    medians = x.median().fillna(0).to_dict()
    x = x.fillna(medians)
    events = sorted(set(zip(frame["Year"], frame["Round"], strict=True)))
    event_index = {event: i for i, event in enumerate(events)}
    # Recent races have more influence, including completed races in the new season.
    weights = np.array(
        [
            0.5 ** ((len(events) - 1 - event_index[event]) / 24)
            for event in zip(frame["Year"], frame["Round"], strict=True)
        ]
    )
    if method == "qualifying":
        estimator = None
    elif method == "forest":
        estimator = RandomForestRegressor(
            n_estimators=160,
            min_samples_leaf=min_samples_leaf,
            max_features=0.8,
            random_state=42,
            n_jobs=2,
        )
        estimator.fit(x, frame["FinishPos"], sample_weight=weights)
    elif method == "pairwise":
        pair_x, pair_y, pair_weights = [], [], []
        target = frame["FinishPos"].to_numpy()
        for indices in frame.reset_index(drop=True).groupby(["Year", "Round"]).indices.values():
            values, left, right = _pair_rows(x.to_numpy(), indices)
            pair_x.append(values)
            pair_y.append((target[indices[left]] < target[indices[right]]).astype(int))
            pair_weights.append(np.repeat(weights[indices[0]], len(left)))
        estimator = HistGradientBoostingClassifier(
            max_iter=100,
            max_leaf_nodes=15,
            min_samples_leaf=30,
            l2_regularization=2,
            early_stopping=False,
            random_state=42,
        )
        with threadpool_limits(limits=2):
            estimator.fit(
                np.vstack(pair_x),
                np.concatenate(pair_y),
                sample_weight=np.concatenate(pair_weights),
            )
    else:
        raise ValueError(f"Unknown model: {method}")
    return FittedModel(method, estimator, medians, events[-1])


def select_model(train_df: pd.DataFrame, requested="auto") -> tuple[str, pd.DataFrame]:
    methods = ["qualifying", "forest", "pairwise"] if requested == "auto" else [requested]
    rows = []
    for method in methods:
        predictions = []
        for train, valid in chronological_splits(train_df):
            model = fit_model(train_df.iloc[train], method)
            frame = train_df.iloc[valid].copy()
            frame["_score"] = model.predict(frame)
            predictions.append(frame)
        combined = pd.concat(predictions, ignore_index=True)
        metrics, _ = metrics_for_predictions(combined, combined.pop("_score"))
        rows.append({"Model": method, **metrics})
    report = pd.DataFrame(rows)
    # The product outputs a top ten: selecting the right drivers is insufficient.
    # Prefer exact top-ten slots; full-field order breaks validation ties.
    report = report.sort_values(
        ["Top-10 exact positions (%)", "Spearman", "Rank MAE"], ascending=[False, False, True]
    )
    return str(report.iloc[0]["Model"]), report.reset_index(drop=True)
