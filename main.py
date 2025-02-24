from StockPredictor import StockPredictor
from Backtester import Backtester
from datetime import datetime, timedelta
import random
import pdb

small_caps = [
    "GME", "AMC", "PLTR", "SOFI", "MARA", "RIOT", "NOK", "SNDL",
    "FUBO", "FIZZ", "CARA", "CLNE", "CRON", "NNDM", "SKLZ", "OSTK",
    "MULN", "BB", "CLOV", "DKNG", "ACB", "PINS", "CSSE", "WKHS",
    "IINN", "VYNE", "INND", "ZOM", "VYGR", "ONVO", "DGLY", "KOS",
    "BNTX", "BYSI", "GSAT", "SCSC", "DCBO", "MTC", "DLO", "LITE",
    "QLYS", "BOX", "SHAK", "POWI", "PI", "POWL", "CIEN", "CRDO",
    "AHCO", "AG", "SMR", "OXLC", "HIMS", "LRE", "MCM", "MDF", "NEA",
    "NXT", "OLE", "PRS", "RIO", "CASH", "DOM", "GAM", "NTH", "ORY",
    "SOL", "UBS", "TRG", "TCRT", "NAYA", "OLB", "SBET", "AQB",
    "SHPH", "ENVB", "AZTR", "PALI", "UOKA", "YCBD", "VTAK", "PCSA",
    "ATXI"
]



# ✅ Modell ausführen
# ✅ Initialize Stock Predictor and Train/Load Model
predictor = StockPredictor(tickers=small_caps)
predictor.run(force_train=False)

# ✅ Pass only the trained model to Backtester
backtester = Backtester(model=predictor.model)

# ✅ Select stocks and run backtest
tickers = filtered_tickers = [
    "GME", "AMC", "PLTR", "SOFI", "MARA", "RIOT", "NOK", "SNDL",
    "FUBO", "FIZZ", "CARA", "CLNE", "CRON", "NNDM", "SKLZ",
    "MULN", "BB", "CLOV", "DKNG", "ACB", "PINS", "CSSE", "WKHS",
    "IINN", "VYNE", "INND", "ZOM", "VYGR", "ONVO", "DGLY", "KOS",
    "BNTX", "BYSI", "GSAT", "SCSC", "DCBO",  "DLO", "LITE",
    "QLYS", "SHAK", "POWI", "PI", "POWL", "CIEN", "CRDO",
    "AHCO", "AG", "SMR", "OXLC", "HIMS", "LRE", "NXT",
     "RIO", "CASH", "GAM", "SOL", "UBS", "TCRT", "NAYA",
    "OLB", "SBET", "AQB", "SHPH", "ENVB", "AZTR", "PALI", "UOKA",
    "YCBD", "VTAK",  "ATXI"
]

start_date = "2023-01-01"
end_date = "2024-01-01"

backtester.run_backtest(tickers, start_date, end_date)

