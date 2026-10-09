"""Run the trusted original pipeline against the frozen 2026 benchmark snapshot.

Export the baseline first:
  git show f354770:pipeline.py > /tmp/f1_predictor_baseline.py

Only load a baseline Python file that you trust. It is executable Python code.
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from modeling import metrics_for_predictions  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-csv", type=Path, required=True)
    parser.add_argument("--baseline-file", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/legacy"))
    args = parser.parse_args()
    spec = importlib.util.spec_from_file_location("legacy_baseline", args.baseline_file)
    legacy = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = legacy
    spec.loader.exec_module(legacy)

    # Bound CPU use; estimator definitions, tuning candidates and seeds are unchanged.
    def forest(*a, **kw):
        kw["n_jobs"] = 1
        return RandomForestRegressor(*a, **kw)

    def grid(*a, **kw):
        kw["n_jobs"] = 2
        return GridSearchCV(*a, **kw)

    legacy.RandomForestRegressor = forest
    legacy.GridSearchCV = grid
    raw = pd.read_csv(args.data_csv)
    raw = raw.loc[raw["Year"].isin([2014, 2022, 2026])].copy()
    if "LegacyQualiNorm" not in raw:
        raise ValueError("Snapshot must include LegacyQualiNorm from min(Q1,Q2,Q3).")
    truth = raw[["Year", "Round", "DriverId", "FinishPos"]].rename(
        columns={"FinishPos": "ActualFinish"}
    )
    raw["QualiNorm"] = raw["LegacyQualiNorm"]
    raw["QualiPos"] = raw["QualiPos"].clip(1, 20)
    raw["FinishPos"] = np.where(raw["DNF"] == 1, 20, raw["FinishPos"]).clip(1, 20)
    train, test, _ = legacy._prepare_train_test_frames(legacy.add_historical_features(raw))
    model = legacy.train_random_forest(train)
    test = test.merge(truth, on=["Year", "Round", "DriverId"], validate="one_to_one")
    scores = model.predict(test[legacy.FEATURE_COLUMNS])
    test["FinishPos"] = test["ActualFinish"]
    metrics, predictions = metrics_for_predictions(test, scores)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(args.output_dir / "predictions.csv", index=False)
    pd.DataFrame([{"Model": "original forest (2014 + 2022)", **metrics}]).to_csv(
        args.output_dir / "metrics.csv", index=False
    )
    print(model.get_params())
    print(metrics)


if __name__ == "__main__":
    main()
