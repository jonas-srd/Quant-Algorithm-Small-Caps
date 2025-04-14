import os
from dotenv import load_dotenv
load_dotenv()

from stockpredictor import StockPredictor, Backtester, PaperTrader
from datetime import datetime, timedelta

# Define Training Time-window
start_date = "2016-01-09"
end_date = datetime.now() - timedelta(hours=6)

small_caps = [
    "ACLS", "AEIS", "ALRM", "AMBA", "ATKR", "AVT", "BC", "BHE",
    "CNK", "CR", "CVLT", "DORM", "ENSG", "ESNT", "FCFS", "FELE", "FNB", "FORM",
    "GMS", "HUBG", "INSM", "IOSP", "KAI", "KAR", "LANC", "MMS", "MTX", "NPO",
    "OSUR", "PATK", "PBH", "PDFS", "PRGS", "PRIM", "PTCT", "RDNT", "RELY",
    "RMBS", "RMR", "SATS", "SBH", "SCL", "SFNC", "SGRY", "SHOO", "SLAB",
    "SLGN", "SMTC", "SPTN", "SPWH", "SR", "SRCL", "SSRM", "STAG", "STKL",
    "STRA", "SUM", "SYNA", "TCMD", "TCX", "TDC", "THRM", "TMHC", "TPIC",
    "TREX", "TROX", "TRUP", "TTGT", "UFPI", "UMBF", "VICR", "VRNT", "WERN",
    "WINA", "WNC", "XPER", "ZWS", "ACIU", "ADUS", "ALKS", "AMN", "ANGO",
    "ASGN", "ASIX", "AVNS", "AZTA", "BCPC", "BEAT", "BELFA", "BLBD", "BLKB",
    "BRKR", "BWXT", "CBNK", "CCRN", "CEVA", "CHDN", "CHRS", "CLFD", "CMTL",
    "CNXN", "CORT", "CSGS", "CTRE", "CUTR", "CWST", "DAKT", "DHIL", "DLHC",
    "DYNT", "EHTH", "ELMD", "EMKR", "ENSG", "ENTG", "ENVA", "ERII", "ESGR",
    "EYPT", "FGEN", "FLGT", "FOLD", "FORR", "FWRD", "GENC", "GERN", "GLDD",
    "GMED", "GNSS", "GOLF", "GRBK", "GRFS", "HAFC", "HCSG", "HSTM", "ICUI",
    "IDXX", "INDB", "INSG", "INVA", "IPAR", "IRMD", "ITRI", "JACK", "JYNT",
    "KOPN", "KTOS", "LECO", "LIVN", "LPSN", "LUNA", "LWAY"
]

predictor = StockPredictor(
    tickers=small_caps,
    start_date=start_date,
    end_date=end_date
)
predictor.run(force_train=False)

mode = "paper"

if mode == "backtest":
    backtester = Backtester(model=predictor.model)
    tickers = ["ANDE", "CVCO", "HLIT", "MMS", "LANC", "AIR", "OSUR", "RDNT", "BXC", "PATK"]
    backtester.run_backtest(tickers, start_date, end_date)
    backtester.plot_equity_curve(start_date, end_date)
    backtester.plot_rolling_roi()
    backtester.plot_return_distribution()
    backtester.plot_accuracy_per_stock()
    backtester.plot_trade_count_per_stock()

elif mode == "paper":
    trader = PaperTrader(
        predictor,
        alpaca_key=os.getenv("ALPACA_API_KEY"),
        alpaca_secret=os.getenv("ALPACA_SECRET_KEY"),
        email_address=os.getenv("EMAIL_ADDRESS"),
        email_password=os.getenv("EMAIL_PASSWORD")
    )
    trader.run_live_loop(interval_minutes=60)