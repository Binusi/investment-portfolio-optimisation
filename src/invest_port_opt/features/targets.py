import pandas as pd

def make_regression_target(adj_close: pd.DataFrame, horizon_days: int = 21) -> pd.Series:
    future = adj_close.pct_change(horizon_days).shift(-horizon_days)
    # We'll later train one model per ticker or stack them with a multiindex
    # For a simple baseline, return the SPY target:
    return future["SPY"].rename("y_future_ret")