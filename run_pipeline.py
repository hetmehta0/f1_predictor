"""CLI entrypoint for the F1 race finish predictor pipeline."""

from __future__ import annotations

import argparse

import pandas as pd

from pipeline import _predict_race_with_artifacts, train_full_pipeline


def _format_table(df: pd.DataFrame, float_digits: int = 4) -> str:
    """Format a dataframe for readable CLI output."""
    return df.to_string(index=False, float_format=lambda x: f"{x:.{float_digits}f}")


def parse_args() -> argparse.Namespace:
    """Parse command-line options."""
    parser = argparse.ArgumentParser(description="Train and evaluate F1 race finish predictor.")
    parser.add_argument(
        "--predict-year",
        type=int,
        default=None,
        help="Optional: year to run post-training race prediction for.",
    )
    parser.add_argument(
        "--predict-round",
        type=int,
        default=None,
        help="Optional: round number to run post-training race prediction for.",
    )
    return parser.parse_args()


def main() -> None:
    """Run full pipeline, print metrics/importances, and optionally predict one race."""
    args = parse_args()
    artifacts, importances, results = train_full_pipeline()

    print("\nFeature Importances (descending):")
    print(_format_table(importances, float_digits=6))

    print("\n2026 Evaluation Results:")
    print(_format_table(results, float_digits=4))

    if (args.predict_year is None) ^ (args.predict_round is None):
        raise ValueError("Provide both --predict-year and --predict-round together.")

    if args.predict_year is not None and args.predict_round is not None:
        top10 = _predict_race_with_artifacts(args.predict_year, args.predict_round, artifacts)
        print(f"\nPredicted Top-10 for {args.predict_year} Round {args.predict_round}:")
        print(_format_table(top10, float_digits=3))


if __name__ == "__main__":
    main()

