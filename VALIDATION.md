# Validation: 8 October 2026

The original pipeline could identify strong drivers while ordering them poorly. On the retrieved **16 completed 2026 races (352 driver-race rows, 22 per race)**, it recovered 76.25% of top-ten members but only 13.125% of exact P1–P10 slots. These measure different things.

## Actual results

All rows below use the same 2026 classification truth. Exact positions and membership are percentages averaged across races; lower rank MAE and higher Spearman are better.

| Method | Weather | Top-10 members | Exact top-10 slots | Winner | Rank MAE | Spearman |
|---|---|---:|---:|---:|---:|---:|
| Original forest, trained on 2014 + 2022 | No | 76.25% | 13.125% | 12.50% | 3.881 | 0.6234 |
| Auto-selected qualifying baseline | Inputs available; unused by baseline | 75.625% | 17.50% | 68.75% | 3.432 | 0.6706 |
| Updated forest | No | 73.125% | 19.375% | 56.25% | 3.665 | 0.6491 |
| Pairwise ranker | No | 75.00% | 19.375% | 62.50% | 3.511 | 0.6683 |
| Updated forest | Yes | 72.50% | 16.25% | 68.75% | 3.688 | 0.6458 |
| Pairwise ranker | Yes | 76.25% | 15.00% | 56.25% | 3.528 | 0.6649 |

**Weather did not improve exact ordering in this test.** The two learned models without weather achieved more exact slots on the holdout, but neither was retroactively selected using those outcomes. Pre-2026 validation favoured qualifying order: exact-slot accuracy was 21.739% versus 19.275% for the weather-enabled ranker and 17.681% for the weather-enabled forest. Those values come from 69 forward-validation races. Auto therefore retains the qualifying baseline.

This is an honest fallback, not evidence that weather or historical form has no value. It means the tested implementations and this sample have not established enough benefit to replace the baseline automatically. In particular, the default output should not be described as a weather-driven ML improvement when its selected model is qualifying order.

## Protocol

- Original source: repository commit `f354770296f8f27ad728c1f21029742ecf4b258d`.
- Original model was reconstructed with the original features, min(Q1,Q2,Q3) gap, P20 cap/DNF transformation, training seasons 2014 + 2022, GroupKFold tuning, and seed 42. Selected hyperparameters were 200 trees, depth 10. Parallelism was bounded to avoid excessive CPU use; candidates/objective were unchanged.
- Original imputation/validation shortcomings were preserved for this comparison. It is not a claim that the original validation was leakage-free.
- The original target distortion was **not** applied to evaluation truth: all methods were scored against the real distinct classification order. Reported rank MAE is calculated from the sorted prediction's ranks, not unranked regression scores.
- Updated models use 2022–2025 for initial model selection. Each 2026 race is predicted after refitting on all strictly earlier races, including completed 2026 races. No target or later race outcome enters that prediction's history, training, or imputation.
- Candidate definitions were fixed for the 2026 comparison. Weather/no-weather ablations are reported together; no winning holdout variant was silently promoted to the default.
- The qualifying baseline has no fitted weather/form effect. Deterministic tie-breaking uses qualifying position and driver ID.
- The frozen snapshot contains 2014 and 2022–2026: 127 races and 2,597 driver-race rows. The default modern training/evaluation subset contains 108 races and 2,190 rows. `LegacyQualiNorm` exists exclusively to reproduce the old model and is excluded from the new feature list.

## Weather provenance and coverage

The snapshot uses Open-Meteo `ecmwf_ifs025` fixed-lead `_previous_day1` forecasts across each race window. `WeatherForecastBefore` is a conservative issuance-time bound, not an asserted exact publication timestamp. Forecasts precede the historical three-hour pre-race cutoff. No realized race weather is substituted.

| Season | Forecasts available | Races |
|---|---:|---:|
| 2014 | 0 | 19 |
| 2022 | 0 | 22 |
| 2023 | 0 | 22 |
| 2024 | 22 | 24 |
| 2025 | 24 | 24 |
| 2026 | 16 | 16 |

Two 2024 forecast requests timed out and remain marked missing. Older seasons predate the selected forecast archive. Modern-subset coverage is 62/108 races. Availability flags and train-fitted imputation keep the pipeline usable without fabricating weather.

## Reproduction

The snapshot and summaries are committed under `validation/`; generated working reports go to ignored `outputs/`.

```bash
uv sync --locked
uv run python run_pipeline.py --data-csv validation/race_snapshot.csv --seasons 2022 2023 2024 2025 2026 --test-year 2026
uv run python scripts/benchmark.py --data-csv validation/race_snapshot.csv

git show f354770:pipeline.py > /tmp/f1_predictor_baseline.py
uv run python scripts/compare_legacy.py --data-csv validation/race_snapshot.csv --baseline-file /tmp/f1_predictor_baseline.py
```

The normal pipeline and benchmark commands work offline after dependencies are installed. The legacy script executes the supplied trusted baseline source. `validation/comparison_2026.csv` includes all six rows above; `model_selection.csv` records the selection decision; `backtest_predictions.csv` records selected-model predictions and their training cutoff per race.

An additional development check used F1DB snapshot `e360484b42234b719eb13b2f55590c57225d0512` with a 2025 holdout. Its no-weather pairwise model achieved Spearman 0.6686 versus qualifying's 0.6547, but exact top-ten slots were 19.17% versus 26.67%. Weather raised its exact-slot score to 21.25% but reduced Spearman to 0.6633. That check prompted explicit reporting of both membership and exact ordering. The committed 2026 comparison uses Jolpica throughout; sources are not mixed within its training/evaluation dataset.

## Checks and limitations

- Automated offline regressions cover temporal splits, training-only imputation, future-outcome invariance, historical refitting, distinct DNFs, 22-car grids, segment-local qualifying gaps, deterministic ties, pairwise symmetry, pagination, weather cutoff/cache/failure paths, and CLI validation.
- Installed the locked environment, checked dependency compatibility, ran lint/format checks and the full test suite, and exercised the real weather and race APIs plus end-to-end CLI training/prediction.
- Race APIs initially timed out; bounded retries later obtained every requested season. This validates successful retrieval here, not future service uptime.
- Results/qualifying corrections are not timestamp-versioned. The benchmark is chronological with respect to outcomes and forecasts, but is not a complete as-published replay. Provider classifications can omit DNS entrants. Retrospective final-grid penalties are intentionally excluded as predictors.
- Sixteen races are too few to claim robust future superiority. The exact-slot improvements are descriptive, not significance tests or calibrated certainty. Safety cars, contact, failures, penalties, pit-stop execution and strategy remain important unmodelled factors.

## Data attribution

The committed CSV is transformed from [Jolpica F1](https://github.com/jolpica/jolpica-f1) Ergast-compatible results/qualifying records and [Open-Meteo](https://open-meteo.com/) forecasts, retrieved 8 October 2026. Transformations include per-driver joins, qualifying-gap calculations, forecast aggregation, and explicit missingness. The combined benchmark follows Jolpica’s CC BY-NC-SA 4.0 data terms (see `validation/DATA_LICENSE.md`); Open-Meteo data is attributed under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) with [provider licence details](https://open-meteo.com/en/licence). The optional F1DB development check used [F1DB](https://github.com/f1db/f1db), also CC BY 4.0. No source endorses these predictions.
