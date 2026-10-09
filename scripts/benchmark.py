"""Reproduce model selection and weather ablations on a frozen CSV snapshot."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from f1_data import validate_master  # noqa: E402
from features import add_historical_features  # noqa: E402
from modeling import select_model  # noqa: E402
from pipeline import walk_forward_evaluate  # noqa: E402
from weather import missing_weather  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-csv", type=Path, required=True)
    parser.add_argument("--test-year", type=int, default=2026)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/benchmark"))
    args = parser.parse_args()
    raw = validate_master(pd.read_csv(args.data_csv))
    raw = raw.loc[(raw["Year"] >= 2022) & (raw["Year"] <= args.test_year)].copy()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    reports = []
    for weather in (False, True):
        frame = raw if weather else raw.assign(**missing_weather("disabled"))
        features = add_historical_features(frame)
        selected, cv = select_model(features.loc[features["Year"] < args.test_year])
        cv.to_csv(args.output_dir / f"selection_weather_{weather}.csv", index=False)
        for method in ("qualifying", "forest", "pairwise"):
            metrics, predictions = walk_forward_evaluate(features, args.test_year, method)
            report = metrics.iloc[[0]].assign(
                Weather=weather, SelectedBeforeHoldout=method == selected
            )
            reports.append(report)
            predictions.to_csv(
                args.output_dir / f"predictions_{method}_weather_{weather}.csv", index=False
            )
            print(report.to_string(index=False), flush=True)
    pd.concat(reports, ignore_index=True).to_csv(args.output_dir / "comparison.csv", index=False)


if __name__ == "__main__":
    main()
