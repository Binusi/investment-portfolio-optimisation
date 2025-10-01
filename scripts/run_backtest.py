import pandas as pd
from src.invest_port_opt.backtest.rules import sma_crossover_signals
from src.invest_port_opt.backtest.simulator import run_vectorbt

adj = pd.read_parquet("data/raw/prices_adj_close.parquet")
price = adj["SPY"].dropna()
signals = sma_crossover_signals(price, 20, 50)
pf = run_vectorbt(price, signals, fees_bps=5.0)
print(pf.stats())