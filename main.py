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


stock = "OLB"
start_date = "2023-01-01"

if not start_date:
    end_date = datetime.today() - timedelta(days=365 * 5)
    random_days = random.randint(0, 365 * 4)
    start_date = (end_date + timedelta(days=random_days)).strftime('%Y-%m-%d')

end_date = (datetime.strptime(start_date, '%Y-%m-%d') + timedelta(days=365)).strftime('%Y-%m-%d')
df = backtester.fetch_stock_data(stock, start_date, end_date)
backtester.backtest(df)
backtester.print_prediction_accuracy(df)

