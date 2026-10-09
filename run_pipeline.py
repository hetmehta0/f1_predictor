"""uv run python run_pipeline.py --help"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import pandas as pd

from pipeline import SEASONS, _predict_race_with_artifacts, train_full_pipeline


def _format_table(df, float_digits=4):
    return df.to_string(index=False, float_format=lambda value: f"{value:.{float_digits}f}")


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Train and backtest F1 finishing order after qualifying."
    )
    parser.add_argument("--seasons", type=int, nargs="+", default=list(SEASONS))
    parser.add_argument(
        "--test-year", type=int, help="Walk-forward holdout year (default: latest in data)."
    )
    parser.add_argument(
        "--model", choices=["auto", "forest", "pairwise", "qualifying"], default="auto"
    )
    parser.add_argument(
        "--data-csv", type=Path, help="Use an exported raw dataset instead of network requests."
    )
    parser.add_argument("--cache-dir", type=Path, default=Path("f1_cache"))
    parser.add_argument(
        "--no-weather", action="store_true", help="Disable weather for a controlled ablation."
    )
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"))
    parser.add_argument("--predict-year", type=int)
    parser.add_argument("--predict-round", type=int)
    parser.add_argument("--top-n", type=int, default=10)
    args = parser.parse_args(argv)
    if (args.predict_year is None) != (args.predict_round is None):
        parser.error("Provide both --predict-year and --predict-round together.")
    if args.top_n < 1 or (args.predict_round is not None and args.predict_round < 1):
        parser.error("--top-n and --predict-round must be positive.")
    if args.test_year is not None and args.data_csv is None and args.test_year not in args.seasons:
        parser.error("--test-year must be included in --seasons.")
    return args


def main(argv=None):
    args = parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    raw = pd.read_csv(args.data_csv) if args.data_csv else None
    artifacts, importances, metrics = train_full_pipeline(
        seasons=args.seasons,
        cache_dir=args.cache_dir,
        master_df=raw,
        test_year=args.test_year,
        model=args.model,
        weather=not args.no_weather,
    )
    print("\nChronological model selection (before the holdout year):")
    print(_format_table(artifacts.validation_results))
    print("\nWalk-forward evaluation (refit using only earlier races):")
    print(_format_table(metrics))
    print(f"\nSelected model: {artifacts.selected_method}")
    races = artifacts.master_df.drop_duplicates(["Year", "Round"])
    print(f"Weather coverage: {int((races['WeatherMissing'] == 0).sum())}/{len(races)} races")
    if not importances.empty:
        print("\nFeature importances (training tree impurity; not causal effects):")
        print(_format_table(importances))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    from features import HISTORICAL_FEATURE_COLUMNS

    artifacts.master_df.drop(
        columns=[
            *HISTORICAL_FEATURE_COLUMNS,
            "WetFormInteraction",
            "IsFirstSeasonOfEra",
            "QualiPositionFraction",
        ],
        errors="ignore",
    ).to_csv(args.output_dir / "race_data.csv", index=False)
    metrics.to_csv(args.output_dir / "metrics.csv", index=False)
    artifacts.validation_results.to_csv(args.output_dir / "model_selection.csv", index=False)
    artifacts.evaluation_predictions.to_csv(
        args.output_dir / "backtest_predictions.csv", index=False
    )
    if args.predict_year is not None:
        prediction = _predict_race_with_artifacts(
            args.predict_year, args.predict_round, artifacts, args.top_n
        )
        print(f"\nPrediction for {args.predict_year} round {args.predict_round}:")
        print(_format_table(prediction, 3))
        print(
            f"Model: {prediction.attrs['model']}; trained through {prediction.attrs['trained_through']}"
        )
        print(f"Weather: {prediction.attrs['weather_source']}")
        print(
            "RankingScore is an ordering score, not a calibrated probability or guaranteed finish."
        )
        prediction.to_csv(
            args.output_dir / f"prediction_{args.predict_year}_{args.predict_round}.csv",
            index=False,
        )
    print(f"\nReports saved in {args.output_dir}")


if __name__ == "__main__":
    main()
