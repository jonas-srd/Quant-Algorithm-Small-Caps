from StockPredictor import StockPredictor
from Backtester import Backtester
from TFTRL import StockPredictor_TFT
from datetime import datetime, timedelta
import random
import pdb




small_caps = [
    "GME", "AMC", "PLTR"
]



# ✅ Modell ausführen
# ✅ Initialize Stock Predictor and Train/Load Model
#predictor = StockPredictor(tickers=small_caps)
#predictor.run(force_train=False)

predictor = StockPredictor_TFT(tickers=small_caps)
predictor.run(force_train=False)
pdb.set_trace()
# ✅ Pass only the trained model to Backtester
backtester = Backtester(model=predictor.model)

# ✅ Select stocks and run backtest
tickers = filtered_tickers = ["KOPN", "EYPT", "INBX", "AVXL", "ADMA", "BCLI", "ACET", "KRUS", "BGSF",
"LUNA", "ONCT", "VTSI", "SLGL", "KZIA", "DMRC", "FNCH", "XAIR", "TCON",
"SABS", "LCTX", "CLRB", "FENC", "TFFP", "MNOV", "SNES", "ACHV", "RHE"
]

start_date = "2023-01-01"
end_date = "2024-01-01"

backtester.run_backtest(tickers, start_date, end_date)

