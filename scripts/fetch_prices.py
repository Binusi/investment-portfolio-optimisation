import yaml
from src.invest_port_opt.data.download import download_prices  # adjust path to your package

with open("configs/universe.yaml") as f:
    cfg = yaml.safe_load(f)

tickers = cfg["tickers"]
print("Tickers:", tickers, "| Start:", cfg.get("start", "2010-01-01"))

adj = download_prices(tickers, start=cfg.get("start","2010-01-01"))
print("Downloaded adj shape:", adj.shape)
print("Adj columns (tickers):", list(adj.columns)[:10])
print("OK.")
