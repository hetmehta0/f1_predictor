# F1 Race Finish Predictor

This project trains a **Random Forest Regressor** to predict Formula 1 finishing position (1–20) from qualifying performance plus historical context, then ranks drivers by predicted finish to produce a top-10.

## What the pipeline does

1. Pulls FastF1 data for seasons **2014, 2022, 2026**.
2. Loads both **Qualifying (`Q`)** and **Race (`R`)** sessions for each round.
3. Builds one master dataframe where each row is one driver in one race.
4. Creates no-leakage historical features (circuit average, season rolling average, team DNF rate, championship position entering the round).
5. Trains and tunes `RandomForestRegressor` using temporal split:
   - **Train:** 2014 + 2022
   - **Test:** 2026
6. Prints:
   - Feature importances (descending)
   - Evaluation metrics table:
     - MAE on finish position
     - Mean per-race Spearman rank correlation
     - Top-3 winner accuracy

## Repository structure

```text
f1_predictor/
├── __init__.py
├── pipeline.py
├── run_pipeline.py
├── requirements.txt
├── README.md
└── .gitignore
```

## Setup

```bash
cd f1_predictor
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

## Run end-to-end training + evaluation

```bash
python run_pipeline.py
```

FastF1 cache is enabled at:

```text
./f1_cache
```

Rounds that fail to load are skipped with a warning.

## Predict a specific race (top-10)

After training in the same process, you can request a race prediction:

```bash
python run_pipeline.py --predict-year 2026 --predict-round 8
```

This prints a formatted top-10 table with:
- Rank
- Driver abbreviation
- Team
- Predicted finish position

## Programmatic usage

```python
from pipeline import predict_race, train_full_pipeline

artifacts, importances, results = train_full_pipeline()
top10 = predict_race(2026, 8)

print(importances)
print(results)
print(top10)
```

