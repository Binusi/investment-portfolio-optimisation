import pandas as pd
from src.invest_port_opt.models.ml import train_baseline_model

X = pd.read_parquet("data/processed/X.parquet")
y = pd.read_parquet("data/processed/y.parquet").squeeze("columns")

model, scores = train_baseline_model(X, y)
print("TimeSeriesSplit R^2 scores:", scores, "Mean:", scores.mean())

# Save with joblib
import joblib, pathlib
pathlib.Path("models").mkdir(exist_ok=True)
joblib.dump(model, "models/baseline_ridge.joblib")
print("Saved models/baseline_ridge.joblib")
