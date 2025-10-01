import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.model_selection import TimeSeriesSplit, cross_val_score

def train_baseline_model(X: pd.DataFrame, y: pd.Series):
    # Align & drop NA rows for both X and y
    df = X.join(y, how="inner").dropna()
    X_aligned = df.drop(columns=[y.name])
    y_aligned = df[y.name]

    # Simple, robust baseline: standardize features + Ridge regression
    model = Pipeline([
        ("scaler", StandardScaler(with_mean=True, with_std=True)),
        ("ridge", Ridge(alpha=1.0, random_state=42))
    ])

    # 5-fold expanding window CV for time series
    tscv = TimeSeriesSplit(n_splits=5)
    scores = cross_val_score(model, X_aligned, y_aligned, cv=tscv, scoring="r2")
    model.fit(X_aligned, y_aligned)
    return model, scores