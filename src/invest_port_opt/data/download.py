from pathlib import Path
import pandas as pd
import yfinance as yf

DATA_RAW = Path("data/raw")

def download_prices(tickers: list[str], start: str = "2010-01-01", interval: str = "1d") -> pd.DataFrame:
    """
    Downloads OHLCV with yfinance and returns a DataFrame of adjusted close
    (or close if auto_adjust=True). Also saves adj_close and volume to Parquet.
    """
    DATA_RAW.mkdir(parents=True, exist_ok=True)

    # auto_adjust=True => "Close" is already adjusted and "Adj Close" is removed
    # Disable threads to avoid pandas/numpy type conversion issues in yfinance's multi downloader.
    try:
        df = yf.download(
            tickers,
            start=start,
            interval=interval,
            auto_adjust=True,
            progress=False,
            group_by="column",   # MultiIndex with ('Close','Volume',...) x tickers
            threads=False,       # <-- important
            repair=True,         # <-- yfinance tries to fix bad rows
            timeout=30
        )
    except Exception as e:
        # Fallback: download each ticker sequentially and concat
        frames = []
        for t in (tickers if isinstance(tickers, list) else [tickers]):
            dft = yf.download(
                t,
                start=start,
                interval=interval,
                auto_adjust=True,
                progress=False,
                group_by="column",
                threads=False,
                repair=True,
                timeout=30
            )
            if dft is None or dft.empty:
                continue
            # Normalize to MultiIndex like the multi-download would give
            dft.columns = pd.MultiIndex.from_product([dft.columns, [t]])
            frames.append(dft)
        if not frames:
            raise ValueError("No data downloaded for any ticker") from e
        df = pd.concat(frames, axis=1).sort_index()

    if df is None or df.empty:
        raise ValueError(
            "yfinance returned an empty DataFrame. "
            "Check your tickers, start date, and internet connectivity."
        )

    # Figure out which price field exists
    price_field = None
    if isinstance(df.columns, pd.MultiIndex):
        # Level 0 is field name (e.g., 'Close', 'Volume'); level 1 is ticker
        level0 = set(df.columns.get_level_values(0))
        if "Adj Close" in level0:
            price_field = "Adj Close"
        elif "Close" in level0:
            price_field = "Close"
        else:
            raise KeyError(f"No 'Adj Close' or 'Close' in columns: {sorted(level0)}")

        if "Volume" not in level0:
            raise KeyError(f"No 'Volume' in columns: {sorted(level0)}")

        adj = df[price_field].copy()   # wide df: columns=tickers
        vol = df["Volume"].copy()      # wide df: columns=tickers

    else:
        # Single-ticker, columns are simple Index like ['Open','High','Low','Close','Volume']
        cols = set(df.columns)
        if "Adj Close" in cols:
            price_field = "Adj Close"
        elif "Close" in cols:
            price_field = "Close"
        else:
            raise KeyError(f"No 'Adj Close' or 'Close' in columns: {sorted(cols)}")

        if "Volume" not in cols:
            raise KeyError(f"No 'Volume' in columns: {sorted(cols)}")

        # Normalize to a 2D wide frame with one column named by the ticker
        tkr = tickers[0] if isinstance(tickers, list) and len(tickers) == 1 else "SINGLE"
        adj = df[[price_field]].rename(columns={price_field: tkr})
        vol = df[["Volume"]].rename(columns={"Volume": tkr})

    # --- SAVE with 'date' as a normal column (no pandas index in parquet metadata) ---
    adj_out = adj.copy().reset_index().rename(columns={adj.index.name or "index": "date"})
    vol_out = vol.copy().reset_index().rename(columns={vol.index.name or "index": "date"})

    # Optional, but safe if sources vary:
    # adj_out["date"] = pd.to_datetime(adj_out["date"])
    # vol_out["date"] = pd.to_datetime(vol_out["date"])

    adj_out.to_parquet(DATA_RAW / "prices_adj_close.parquet", index=False)
    vol_out.to_parquet(DATA_RAW / "prices_volume.parquet", index=False)

    return adj  # unchanged return for callers

    return adj
