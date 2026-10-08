# F1 finishing-order predictor

Predict the top ten **after qualifying**, compare learned models with qualifying order, and evaluate races chronologically. Python 3.12 and `uv` manage the environment. Default seasons are the current season plus the previous four (2022–2026 in 2026).

## Install and run

Install [uv](https://docs.astral.sh/uv/getting-started/installation/) if needed (`brew install uv` on macOS), then:

```bash
uv sync --locked
uv run python run_pipeline.py
```

`uv` uses `.python-version`, creates `.venv`, and installs the versions in `uv.lock`. The original numerical-library versions are retained. `requirements.txt` is an exported compatibility file; make dependency changes with `uv add`, then regenerate it with `uv export --no-dev --no-hashes -o requirements.txt`.

Predict a round whose qualifying results have been published:

```bash
uv run python run_pipeline.py --predict-year 2026 --predict-round 16
```

For a completed round this is a **historical prediction**: training excludes the target race and every later race. Qualifying data must exist; the code does not fabricate a future starting field. Use `--top-n 22` to display the full current field.

## What changed and why

The original predictor often identified the contenders while misordering them:

- Every DNF was changed to P20, and legitimate P21/P22 positions were clipped to P20. This corrupted the target and the reliability feature, which inferred DNFs from `FinishPos == 20`.
- Training used only 2014 and 2022. No completed 2026 race updated the fitted model. The era flag was constant for all configured seasons, so it could not teach the model to adapt.
- Taking `min(Q1, Q2, Q3)` compared times from different conditions. Gaps now compare a driver's final completed qualifying segment with that segment's fastest time; qualifying position and segment are separate features.
- `GroupKFold` kept races together but could train on later races to validate earlier ones. Global median filling also exposed future feature distributions. Splits now move forward in time, and each training fold fits its own imputer.
- A position regressor predicts an average finishing position, not an exact ordered list. The old winner-in-top-three metric did not measure top-ten ordering.

Results retain actual classification positions, including distinct retirements. Stable driver and constructor IDs identify history. New inputs include recent driver/team performance, qualifying-to-finish gains, current-season race form, circuit history shrunk toward general form, and explicit driver/team DNF rates. Recent training races receive more weight. Circuit and form histories use only earlier races, with a three-year lookback.

The former `DriverChampPos` feature was only a ranking of Grand Prix points: it omitted sprint points and official tie-breaks. It is replaced with accurately named **Grand Prix points per race**, not presented as championship position.

## Models and honest evaluation

`--model auto` compares:

1. **Qualifying order**, the baseline.
2. A **random forest** predicting classification position.
3. A **pairwise ranker** learning which driver finishes ahead of another. Shared race weather is included separately because it cancels out in driver-to-driver feature differences.

Selection uses expanding chronological folds **before the holdout year**, prioritising exact top-ten positions, then full-field Spearman correlation. Within the holdout year, the selected model is refitted before each race using only earlier races. Model choice is not tuned on holdout results. The final in-memory model uses all completed races for future predictions; historical requests refit and, where necessary, reselect using earlier races only.

The simplest baseline remains eligible. Adding features is not proof of improvement. The checked-in 2026 snapshot selected **qualifying order**; see [VALIDATION.md](VALIDATION.md) for the full comparison, including learned models that do worse and weather ablations.

Reported metrics distinguish:

- **Top-10 membership:** whether the correct drivers were picked, irrespective of order.
- **Top-10 exact positions:** whether each predicted P1–P10 driver finished in that slot.
- Winner, exact podium, full-field rank MAE, and Spearman correlation.

Ties are resolved by qualifying position and stable driver ID, never by the input's actual finishing order. `RankingScore` is not a calibrated probability, confidence interval, or guaranteed finishing position.

## Weather

Open-Meteo supplies race-window air temperature, precipitation, wind speed, and humidity using circuit coordinates and UTC race start times from Jolpica. No API key is needed for the public non-commercial endpoints. A precipitation threshold supplies a forecast-wet flag; it is **not** a rain probability.

- Future-race requests use a current forecast when it is available before the cutoff.
- Historical races use the **Previous Runs API**, selecting fixed-lead forecasts that precede a race-start-minus-three-hours cutoff, with an additional publication margin. They do not use observed race weather or a stitched historical/reanalysis series.
- Forecast features are aggregated over hourly buckets intersecting a three-hour race window. Air temperature is not track temperature.
- Missing coordinates, missing start times, unavailable forecast history, incomplete API responses, and outages remain explicit missing values. They are never silently called dry weather. Live forecasts expire from cache after an hour. Repeated weather failures stop further network attempts for that client run.
- Historical coverage is unavailable before 2024 for the selected model. Training reports coverage. `DriverForecastWetGain` measures performance in **previous races forecast to be wet**, with shrinkage for sparse samples; it is not a proven driver wet-weather skill rating.

Weather is integrated, but it did **not** improve exact ordering on the checked-in 2026 comparison. Use `--no-weather` for an ablation. The automatic baseline can win even when weather features are present.

## Useful commands

```bash
# Reproduce the frozen benchmark without any race/weather API requests
uv run python run_pipeline.py --data-csv validation/race_snapshot.csv --seasons 2022 2023 2024 2025 2026 --test-year 2026

# Inspect a learned model explicitly, without weather
uv run python run_pipeline.py --data-csv validation/race_snapshot.csv --seasons 2022 2023 2024 2025 2026 --test-year 2026 --model pairwise --no-weather

# Compare every candidate with and without weather
uv run python scripts/benchmark.py --data-csv validation/race_snapshot.csv

# Regression checks (offline)
uv run pytest -q
uv run ruff check .
uv run ruff format --check .
```

`outputs/` contains `race_data.csv`, `model_selection.csv`, `metrics.csv`, `backtest_predictions.csv`, and an optional prediction CSV. Race data requests use cached, paginated Jolpica results and qualifying records, which avoids downloading telemetry. Transient server/network failures get bounded retries; invalid or incomplete classification fails explicitly. First-time collection still depends on the providers being reachable. `--data-csv` enables reproducible offline runs. Cached historical records are refreshed after six hours so corrections are not permanently ignored.

Programmatic usage from the repository directory:

```python
from pipeline import train_full_pipeline, predict_race

artifacts, importances, metrics = train_full_pipeline()
top10 = predict_race(2026, 16)
```

Forest impurity importances are available; the ranker does not pretend to expose the same importance measure. They are not causal explanations.

## Limits

This is an after-qualifying research model. It does not yet model verified grid penalties, FP2 long-run pace, tyre compounds/degradation, pit-stop strategy, safety cars, or unpredictable failures. Final race grid positions are retained only as metadata where available, never fed into training when live prediction only knows qualifying order. Historical qualifying/results records may contain later official corrections; these providers do not offer a complete as-of publication archive. Historical field membership follows the provider's recorded classification and may omit a DNS. Avoid interpreting a retrospective test as a perfectly timestamped live replay.

The 2026 evaluation has only 16 races. Repeatedly optimising against this snapshot would turn it into training data. Freeze a future evaluation window before further tuning.

## Sources and layout

- Race/qualifying records: [Jolpica F1](https://github.com/jolpica/jolpica-f1), the Ergast-compatible data source also used by FastF1.
- Forecasts: [Open-Meteo](https://open-meteo.com/), [Previous Runs API](https://open-meteo.com/en/docs/previous-runs-api), [licence](https://open-meteo.com/en/licence).
- Optional offline importer: [F1DB](https://github.com/f1db/f1db), CC BY 4.0. `scripts/import_f1db.py` converts an existing checkout; it is a development/validation helper.

`f1_data.py` handles results; `features.py` builds prior-history inputs; `weather.py` handles forecasts; `modeling.py` trains and ranks; `pipeline.py` orchestrates training, validation, and historical-safe prediction. `tests/` exercises temporal boundaries, ranking, weather, pagination, and 22-car classification. The original `train_full_pipeline` / `predict_race` entry points are retained.
