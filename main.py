from stockpredictor import StockPredictor, Backtester
import pdb

small_caps = [
    "ACLS", "AEIS", "ALRM", "AMBA", "ATKR", "AVT", "BC", "BHE", "BJ",
    "CNK", "CR", "CVLT", "DORM", "ENSG", "ESNT", "FCFS", "FELE", "FNB", "FORM",
    "GMS", "HUBG", "INSM", "IOSP", "KAI", "KAR", "LANC", "MMS", "MTX", "NPO",
    "OSUR", "PATK", "PBH", "PDFS", "PRGS", "PRIM", "PRVA", "PSN", "PTCT",
    "RDNT", "RELY", "REPL", "RMBS", "RMR", "SATS", "SBH", "SCL", "SFNC", "SGRY",
    "SHOO",  "SLAB", "SLGN", "SMTC", "SPTN", "SPWH", "SR", "SRCL", "SSRM",
    "STAG", "STKL", "STRA", "SUM",  "SYNA", "TCMD", "TCX", "TDC", "THRM",
    "TMHC", "TPIC", "TREX", "TROX", "TRUP", "TTGT", "UFPI",  "UMBF", "VICR",
    "VRNT", "WERN", "WGS", "WINA",  "WNC", "XPER", "ZWS", "ACIU",
    "ADUS", "ALKS", "AMN", "ANGO",  "APLS", "ARLO", "ARVN", "ASGN", "ASIX",
    "AVNS", "AZTA", "BCPC", "BEAT", "BELFA", "BLBD", "BLKB", "BRKR", "BSBK",
    "BWXT", "CBNK", "CCRN", "CDMO", "CEVA", "CHDN", "CHRS", "CLFD", "CMTL", "CNXN",
    "CODX", "CORT", "CRCT", "CSGS", "CTRE", "CUTR", "CWST", "DAKT", "DCGO", "DHIL",
    "DLHC", "DNLI", "DOYU", "DYNT", "EHTH", "ELMD", "EMBC", "EMKR", "ENSG",
    "ENTG", "ENVA",  "ERII", "ESGR", "EYPT", "FGEN", "FLGT", "FOLD", "FORR",
     "FULC", "FWRD",  "GENC", "GERN", "GLDD", "GMED", "GNSS", "GOLF",
    "GRBK", "GRFS",  "HAFC", "HCSG",  "HSTM", "ICUI", "IDXX",
    "INDB", "INSG", "INVA", "IPAR", "IRMD", "ITRI", "JACK", "JYNT", "KIDS", "KNSA",
    "KOPN",  "KTOS",  "LECO", "LIVN", "LPRO", "LPSN", "LUNA", "LWAY"

]

start_date = "2010-01-01"
end_date = "2025-01-01"

predictor = StockPredictor(tickers=small_caps, start_date=start_date, end_date=end_date)
predictor.run(force_train=False)


# ✅ Pass only the trained model to Backtester
backtester = Backtester(model=predictor.model)

# ✅ Select stocks and run backtest
tickers = filtered_tickers = [
     "CLW", "CRUS", "CUBI", "CVCO", "CYTK",
      "ESEA", "FARO", "FOSL", "GCO", "HLIT",  "IIIN"

]

start_date = "2018-01-01"
end_date = "2025-04-07"

backtester.run_backtest(tickers, start_date, end_date)
backtester.plot_equity_curve(start_date, end_date)
backtester.plot_rolling_roi()
backtester.plot_return_distribution()
backtester.plot_accuracy_per_stock()
backtester.plot_trade_count_per_stock()


