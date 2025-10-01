from pathlib import Path
import pandas as pd
from src.invest_port_opt.features.technical import make_technical_features
from src.invest_port_opt.features.targets import make_regression_target

# Read the saved tables (date is a normal column)
adj = pd.read_parquet("data/raw/prices_adj_close.parquet")

# Restore DatetimeIndex
if "date" not in adj.columns:
    raise ValueError("Expected a 'date' column in prices_adj_close.parquet")

adj["date"] = pd.to_datetime(adj["date"])
adj = adj.set_index("date").sort_index()

X = make_technical_features(adj)
y = make_regression_target(adj, horizon_days=21)

Path("data/processed").mkdir(parents=True, exist_ok=True)
X.to_parquet("data/processed/X.parquet", index=True)
y.to_frame(name=y.name or "y").to_parquet("data/processed/y.parquet", index=True)
print("Features & target saved.")