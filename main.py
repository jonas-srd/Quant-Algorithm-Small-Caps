from StockPredictor import StockPredictor
from Backtester import Backtester
import pdb

small_caps = [
    "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA",
    "TSLA", "META", "JPM", "V",
    "MA", "UNH", "HD", "PEP", "KO",
    "LLY", "ABBV", "XOM", "CVX", "BAC",
    "WMT", "PG", "AVGO", "ADBE", "COST",
    "ORCL", "TMO", "NKE", "INTC", "DIS"

]


predictor = StockPredictor(tickers=small_caps)
predictor.run(force_train=True)


# ✅ Pass only the trained model to Backtester
backtester = Backtester(model=predictor.model)

# ✅ Select stocks and run backtest
tickers = filtered_tickers = ["KOPN", "EYPT", "INBX", "AVXL", "ADMA", "BCLI", "ACET", "KRUS", "BGSF",
"LUNA", "ONCT", "VTSI", "SLGL", "KZIA", "DMRC", "FNCH", "XAIR", "TCON",
"SABS", "LCTX", "CLRB", "FENC", "MNOV", "SNES", "ACHV", "RHE"
]

start_date = "2023-01-01"
end_date = "2024-01-01"

backtester.run_backtest(tickers, start_date, end_date)

