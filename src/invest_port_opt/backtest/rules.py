import pandas as pd

def sma_crossover_signals(price: pd.Series, fast: int = 20, slow: int = 50) -> pd.DataFrame:
    sma_fast = price.rolling(fast).mean()
    sma_slow = price.rolling(slow).mean()
    entries  = (sma_fast > sma_slow) & (sma_fast.shift(1) <= sma_slow.shift(1))
    exits    = (sma_fast < sma_slow) & (sma_fast.shift(1) >= sma_slow.shift(1))
    return pd.DataFrame({"entries": entries, "exits": exits})