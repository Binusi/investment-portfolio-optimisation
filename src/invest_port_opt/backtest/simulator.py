import pandas as pd
import vectorbt as vbt

def run_vectorbt(price: pd.Series, signals: pd.DataFrame, fees_bps: float = 5.0):
    # fees_bps: 5 bps = 0.05% per trade for a simple cost model
    pf = vbt.Portfolio.from_signals(
        close=price,
        entries=signals["entries"],
        exits=signals["exits"],
        fees=fees_bps / 10000.0
    )
    return pf  # pf.stats(), pf.total_return(), pf.sharpe_ratio(), etc.
