from StockPredictor import StockPredictor
from Backtester import Backtester


small_caps = [
    "GME", "AMC", "PLTR", "SOFI", "MARA", "RIOT", "NOK", "SNDL",
    "FUBO", "FIZZ", "CARA", "CLNE", "CRON", "NNDM", "SKLZ", "OSTK",
     "MULN", "BB", "CLOV", "DKNG", "ACB", "PINS", "CSSE",
    "WKHS", "IINN", "VYNE", "INND", "ZOM", "VYGR", "ONVO",
    "DGLY", "KOS", "BNTX", "BYSI", "GSAT", "SCSC", "DCBO", "MTC", "DLO"
]




# ✅ Modell ausführen
predictor = StockPredictor(tickers=small_caps)
predictor.run(force_train = False)

# ✅ Backtest mit verschiedenen Aktien durchführen
for stock in ["PLTR"]:
    if stock in predictor.data:
        print(f"🔍 Backtest für {stock} startet...")
        backtester = Backtester(model=predictor.model)
        backtester.backtest(predictor.data[stock])