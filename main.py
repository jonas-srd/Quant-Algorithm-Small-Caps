from StockPredictor import StockPredictor
from Backtester import Backtester


small_caps = [
    "GME", "AMC", "PLTR", "SOFI", "MARA", "RIOT", "NOK", "SNDL",
    "FUBO", "FIZZ", "CARA", "CLNE", "CRON", "NNDM", "SKLZ", "OSTK",
    "SPCE", "MULN", "BB", "CLOV", "DKNG", "ACB", "PINS", "CSSE",
    "WKHS", "IINN", "VYNE", "HCMC", "INND", "ZOM", "VYGR", "ONVO",
    "DGLY", "KOS", "BNTX", "BYSI", "GSAT", "SCSC", "DCBO", "MTC", "DLO"
]



# ✅ Modell ausführen
predictor = StockPredictor(tickers=small_caps)
predictor.run()

# ✅ Backtest starten
if "BB" in predictor.data:
    backtester = Backtester(model=predictor.model)
    backtester.backtest(predictor.data["BB"])