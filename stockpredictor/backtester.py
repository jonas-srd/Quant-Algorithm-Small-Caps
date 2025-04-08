import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from collections import defaultdict
from .predictor import StockPredictor
import pdb
import yfinance as yf



class Backtester:
    def __init__(self, model, future_prediction_days=5, trading_fee=0.005, risk_per_trade=0.1, stop_loss=0.01,
                 take_profit=0.04, lookback_period=10, std_dev_factor=2, min_prob_threshold=0.9):
        self.model = model
        self.future_prediction_days = future_prediction_days
        self.trading_fee = trading_fee
        self.risk_per_trade = risk_per_trade
        self.stop_loss = stop_loss
        self.take_profit = take_profit
        self.lookback_period = lookback_period
        self.std_dev_factor = std_dev_factor
        self.min_prob_threshold = min_prob_threshold
        self.trade_log = []
        self.correct_predictions = 0

        # Create StockPredictor instance to reuse its fetch methods
        self.stock_predictor = StockPredictor(
            tickers=[],
            lookback_period=self.lookback_period,
            std_dev_factor=self.std_dev_factor,
            future_prediction_days=self.future_prediction_days
        )

    def fetch_stock_data(self, tickers, start_date, end_date):
        self.stock_predictor.tickers = tickers


        # Hole Makrodaten einmal
        macro_df = self.stock_predictor.fetch_macro_data(start_date=start_date, end_date=end_date)

        macro_df.index = pd.to_datetime(macro_df.index)

        stock_data = {}
        for ticker in tickers:
            df = self.stock_predictor.fetch_stock_data(ticker)
            if df is not None:
                df = df.copy()

                # 🛠 Fix für MultiIndex-Spalten
                if isinstance(df.columns, pd.MultiIndex):
                    df.columns = df.columns.get_level_values(0)

                # 🛠 Sicherstellen, dass Index korrekt ist
                df.index.name = "Date"
                df = df.reset_index().set_index("Date")

                # 🧠 Merge der Makrodaten wie in prepare_data()
                df = df.join(macro_df, how="left")

                df.dropna(inplace=True)

                # Optional: MultiIndex (Ticker, Date), wie in deinem bisherigen Code
                df["Ticker"] = ticker
                df = df.set_index("Ticker", append=True).reorder_levels(["Ticker", "Date"])

                stock_data[ticker] = df

        return stock_data

    def predict_best_stock(self, stock_data, index):
        predictions = {}

        for ticker, df in stock_data.items():
            if df is None or len(df) <= index:
                continue

            features = df.iloc[index][[
                "Close", "Moving_Avg", "Upper_Band", "Lower_Band",
                "MACD", "MACD_Signal", "RSI", "ATR",
                "Bollinger_Width", "ROC", "ADX",
                "CPI", "Zinsen", "Arbeitslosenquote"
            ]].values.reshape(1, -1)

            probs = self.model.predict_proba(features)[0]
            predictions[ticker] = probs


        if not predictions:
            return None, None, None

        best_ticker, (p_down, p_up) = max(predictions.items(), key=lambda item: max(item[1]))
        trade_type = "Long" if p_up >= p_down else "Short"
        trade_prob = max(p_up, p_down)

        if trade_prob < self.min_prob_threshold:
            return None, None, None

        return best_ticker, trade_type, trade_prob

    def run_backtest(self, tickers, start_date, end_date):
        stock_data = self.fetch_stock_data(tickers, start_date, end_date)

        if not stock_data:
            print("⚠ No valid stock data found. Exiting backtest.")
            return

        initial_balance = 10_000
        balance = initial_balance
        trade_count = 0
        open_trade = None

        print(f"\n🔍 Running backtest with initial capital: ${initial_balance}")

        tickers = list(stock_data.keys())
        max_length = min(len(df) for df in stock_data.values())

        if max_length == 0:
            print("⚠ No valid data for backtest. Exiting.")
            return

        i = 0
        while i < max_length - self.future_prediction_days:
            if open_trade is None:

                best_stock, trade_type, trade_prob = self.predict_best_stock(stock_data, i)

                if best_stock:
                    trade_stock = best_stock
                    trade_entry_price = stock_data[trade_stock].iloc[i]["Close"].item()
                    open_trade = trade_type
                    trade_count += 1
                else:
                    i += 1
                    continue
            else:
                future_prices = stock_data[trade_stock].iloc[i + 1: i + 1 + self.future_prediction_days]["Close"]

                if future_prices.empty:
                    open_trade = None
                    i += 1
                    continue

                max_price = future_prices.max().item()
                min_price = future_prices.min().item()

                if open_trade == "Long":
                    if max_price >= trade_entry_price * (1 + self.take_profit):
                        exit_price = trade_entry_price * (1 + self.take_profit)
                        trade_outcome = "Long Profit"
                    elif min_price <= trade_entry_price * (1 - self.stop_loss):
                        exit_price = trade_entry_price * (1 - self.stop_loss)
                        trade_outcome = "Long Stop Loss"
                    else:
                        i += 1
                        continue

                elif open_trade == "Short":
                    if min_price <= trade_entry_price * (1 - self.take_profit):
                        exit_price = trade_entry_price * (1 + self.take_profit)
                        trade_outcome = "Short Profit"
                    elif max_price >= trade_entry_price * (1 + self.stop_loss):
                        exit_price = trade_entry_price * (1 - self.stop_loss)
                        trade_outcome = "Short Stop Loss"
                    else:
                        i += 1
                        continue

                trade_amount = balance * self.risk_per_trade
                trade_fee = trade_amount * self.trading_fee
                profit_loss = (exit_price - trade_entry_price) / trade_entry_price
                trade_result = trade_amount * profit_loss
                balance += trade_result - trade_fee

                actual_movement = int((exit_price > trade_entry_price) if open_trade == "Long" else (exit_price < trade_entry_price))
                predicted_movement = 1 if open_trade == "Long" else 0

                if actual_movement == predicted_movement:
                    self.correct_predictions += 1

                trade_start_date = stock_data[trade_stock].index[i][1].strftime("%Y-%m-%d")
                trade_end_date = stock_data[trade_stock].index[i + self.future_prediction_days][1].strftime("%Y-%m-%d")

                self.trade_log.append({
                    "Trade Start Date": trade_start_date,
                    "Trade End Date": trade_end_date,
                    "Day": i,
                    "Stock": trade_stock,
                    "Trade Type": trade_outcome,
                    "Trade Price": trade_entry_price,
                    "Exit Price": exit_price,
                    "Profit/Loss": trade_result,
                    "Fees": trade_fee,
                    "Balance": balance
                })

                open_trade = None

            i += 1

        final_balance = balance
        roi = ((final_balance - initial_balance) / initial_balance) * 100
        prediction_accuracy = (self.correct_predictions / trade_count) * 100 if trade_count > 0 else 0

        print(f"🏁 End capital: ${final_balance:,.2f}")
        print(f"📊 Number of trades: {trade_count}")
        print(f"📈 Total ROI: {roi:.2f}%")
        print(f"🎯 Prediction Accuracy: {prediction_accuracy:.2f}%")

    def plot_equity_curve(self, start_date, end_date):
        trade_df = pd.DataFrame(self.trade_log)
        if trade_df.empty:
            print("⚠ Kein Trade-Log vorhanden. Keine Equity Curve anzeigbar.")
            return

        trade_df["Trade End Date"] = pd.to_datetime(trade_df["Trade End Date"])
        trade_df.sort_values("Trade End Date", inplace=True)
        trade_df.set_index("Trade End Date", inplace=True)

        # Initialwert
        initial_balance = trade_df["Balance"].iloc[0]
        trade_df["Equity %"] = (trade_df["Balance"] / initial_balance - 1) * 100

        # Hole S&P 500 Daten (SPY als Proxy)
        spy = yf.download("^GSPC", start=start_date, end=end_date, interval="1d")
        spy = spy.loc[spy.index >= trade_df.index[0]]  # auf den Backtest-Zeitraum beschränken
        spy["SPY %"] = (spy["Close"] / spy["Close"].iloc[0] - 1) * 100

        # Plot
        plt.figure(figsize=(12, 6))
        plt.plot(trade_df.index, trade_df["Equity %"], label="Quant Strategy", marker='o')
        plt.plot(spy.index, spy["SPY %"], label="S&P 500", linestyle='--')
        plt.title("Prozentuale Equity Curve vs. S&P 500")
        plt.xlabel("Datum")
        plt.ylabel("Veränderung [%]")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    def plot_rolling_roi(self, window=5):
        trade_df = pd.DataFrame(self.trade_log)
        rolling_roi = trade_df["Profit/Loss"].rolling(window=window).sum()

        plt.figure(figsize=(10, 5))
        plt.plot(rolling_roi, label=f"Rolling ROI ({window} Trades)")
        plt.axhline(0, color='red', linestyle='--')
        plt.title("Rolling ROI")
        plt.xlabel("Trade #")
        plt.ylabel(f"Cumulative P/L over {window} trades")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    def plot_return_distribution(self):
        trade_df = pd.DataFrame(self.trade_log)
        plt.figure(figsize=(8, 5))
        plt.hist(trade_df["Profit/Loss"], bins=20, edgecolor="black")
        plt.title("Distribution of Trade Returns")
        plt.xlabel("Profit / Loss")
        plt.ylabel("Number of Trades")
        plt.grid(True)
        plt.tight_layout()
        plt.show()


    def plot_accuracy_per_stock(self):
        stock_hits = defaultdict(lambda: {"correct": 0, "total": 0})

        for trade in self.trade_log:
            stock = trade["Stock"]
            correct = 1 if trade["Profit/Loss"] > 0 else 0
            stock_hits[stock]["correct"] += correct
            stock_hits[stock]["total"] += 1

        stock_accuracy = {s: v["correct"] / v["total"] for s, v in stock_hits.items() if v["total"] > 0}

        plt.figure(figsize=(10, 5))
        plt.bar(stock_accuracy.keys(), stock_accuracy.values())
        plt.xticks(rotation=90)
        plt.title("Accuracy by Stock")
        plt.ylabel("Success Rate")
        plt.tight_layout()
        plt.grid(True)
        plt.show()


    def plot_trade_count_per_stock(self):
        trade_df = pd.DataFrame(self.trade_log)

        if trade_df.empty:
            print("⚠ Kein Trade-Log vorhanden.")
            return

        trade_counts = trade_df["Stock"].value_counts()

        plt.figure(figsize=(10, 5))
        plt.bar(trade_counts.index, trade_counts.values)
        plt.xticks(rotation=90)
        plt.title("Anzahl der Trades pro Aktie")
        plt.xlabel("Aktie")
        plt.ylabel("Trade-Anzahl")
        plt.grid(True)
        plt.tight_layout()
        plt.show()


