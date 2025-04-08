from stockpredictor import StockPredictor, Backtester
import pdb

small_caps = [
    "ACLS", "AEIS", "ALRM", "AMBA", "ATKR", "AVT", "BC", "BHE", "BJ", "CIR",
    "CNK", "CR", "CVLT", "CVLT", "DORM", "ENSG", "ESNT", "FCFS", "FELE", "FNB",
    "FORM", "GMS", "HUBG", "INSM", "IOSP", "KAI", "KAR", "LANC", "MMS"

]

start_date = "2010-01-01"
end_date = "2025-01-01"

predictor = StockPredictor(tickers=small_caps, start_date=start_date, end_date=end_date)
predictor.run(force_train=True)


# ✅ Pass only the trained model to Backtester
backtester = Backtester(model=predictor.model)

# ✅ Select stocks and run backtest
tickers = filtered_tickers = [
    "BB", "PEP","TSLA","aapl","msft","amzn","googl","meta","nvda","brk-b","jpm","jnj","xom"

]

start_date = "2023-01-01"
end_date = "2024-01-01"

backtester.run_backtest(tickers, start_date, end_date)

backtester.plot_equity_curve()
backtester.plot_rolling_roi()
backtester.plot_return_distribution()
backtester.plot_accuracy_per_stock()


