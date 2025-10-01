import pandas as pd
from pypfopt.expected_returns import mean_historical_return
from pypfopt.risk_models import CovarianceShrinkage
from pypfopt.efficient_frontier import EfficientFrontier

def optimize_mean_var(adj_close: pd.DataFrame, rebalance: str = "M") -> pd.DataFrame:
    # Compute monthly weights with shrinkage covariance
    weights_list = []
    dates = []
    for dt, window in adj_close.resample(rebalance):
        if window.shape[0] < 252:  # need ~1y of data
            continue
        mu = mean_historical_return(window)  # simple baseline
        S  = CovarianceShrinkage(window).ledoit_wolf()
        ef = EfficientFrontier(mu, S)
        w = ef.max_sharpe()
        weights_list.append(pd.Series(ef.clean_weights()))
        dates.append(dt)
    W = pd.DataFrame(weights_list, index=dates).fillna(0.0)
    return W