import pandas as pd

def _rsi(series: pd.Series, length: int = 14) -> pd.Series:
    """
    Plain-Pandas RSI implementation (Wilder's smoothing approximation).
    Works on a single price series. Returns a Series aligned to the input index.
    """
    delta = series.diff()

    # Gains (positive deltas) and losses (negative deltas)
    up = delta.clip(lower=0)
    down = -delta.clip(upper=0)

    # Use exponential weighted mean as a simple Wilder approximation
    # (alpha ~ 1/length)
    roll_up = up.ewm(alpha=1/length, adjust=False).mean()
    roll_down = down.ewm(alpha=1/length, adjust=False).mean()

    rs = roll_up / roll_down
    rsi = 100 - (100 / (1 + rs))
    return rsi

def make_technical_features(adj_close: pd.DataFrame) -> pd.DataFrame:
    """
    Build simple, robust features for all tickers in a wide adj_close DataFrame
    (columns are tickers). No external TA libraries required.
    """
    # Daily returns
    rets = adj_close.pct_change().add_suffix("_ret1")

    # Volatility (20-day rolling std of returns)
    vol20 = adj_close.pct_change().rolling(20).std().add_suffix("_vol20")

    # Simple moving averages
    sma20 = adj_close.rolling(20).mean().add_suffix("_sma20")
    sma50 = adj_close.rolling(50).mean().add_suffix("_sma50")

    # 20-day momentum (percentage change over 20 sessions)
    mom20 = adj_close.pct_change(20).add_suffix("_mom20")

    # RSI(14) per ticker
    rsi_frames = []
    for col in adj_close.columns:
        rsi_col = _rsi(adj_close[col], length=14).rename(f"{col}_rsi14")
        rsi_frames.append(rsi_col)
    rsi14 = pd.concat(rsi_frames, axis=1)

    # Combine
    feats = pd.concat([rets, vol20, sma20, sma50, mom20, rsi14], axis=1)

    # Drop initial NaNs from rolling windows
    feats = feats.dropna()
    return feats