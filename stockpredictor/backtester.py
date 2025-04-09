import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from collections import defaultdict
from .predictor import StockPredictor
import pdb
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame

class Backtester:
    def __init__(self, model, future_prediction_hours=24, trading_fee=0.005, risk_per_trade=0.1, stop_loss=0.01,
                 take_profit=0.04, lookback_period=10, std_dev_factor=2, min_prob_threshold=0.9):
        self.model = model
        self.future_prediction_hours = future_prediction_hours
        self.trading_fee = trading_fee
        self.risk_per_trade = risk_per_trade
        self.stop_loss = stop_loss
        self.take_profit = take_profit
        self.lookback_period = lookback_period
        self.std_dev_factor = std_dev_factor
        self.min_prob_threshold = min_prob_threshold
        self.trade_log = []
        self.correct_predictions = 0

        self.alpaca_client = StockHistoricalDataClient(
            api_key="PKHZI2GPSTTI8XH7FVA8",
            secret_key="fZhOrdjXLDhhggvMaBCBcAQ5wVvJ5OopIIZkZKve"
        )

        self.stock_predictor = StockPredictor(
            tickers=[],
            lookback_period=self.lookback_period,
            std_dev_factor=self.std_dev_factor,
            future_prediction_days=1
        )

    def fetch_stock_data(self, tickers, start_date, end_date):
        self.stock_predictor.tickers = tickers

        macro_df = self.stock_predictor.fetch_macro_data(start_date=start_date, end_date=end_date)
        macro_df.index = pd.to_datetime(macro_df.index).tz_localize(None)
        macro_df.index.name = "timestamp"

        stock_data = {}
        for ticker in tickers:
            request_params = StockBarsRequest(
                symbol_or_symbols=ticker,
                timeframe=TimeFrame.Hour,
                start=start_date,
                end=end_date
            )
            try:
                bars = self.alpaca_client.get_stock_bars(request_params).df

                if bars.empty:
                    print(f"⚠ No data for {ticker}")
                    continue

                df = bars.reset_index()
                df = df[df["symbol"] == ticker].drop(columns="symbol").copy()
                df.set_index("timestamp", inplace=True)
                df.index = df.index.tz_convert(None)

                df["Moving_Avg"] = df["close"].rolling(window=self.lookback_period).mean()
                df["Std_Dev"] = df["close"].rolling(window=self.lookback_period).std()
                df["Upper_Band"] = df["Moving_Avg"] + (self.std_dev_factor * df["Std_Dev"])
                df["Lower_Band"] = df["Moving_Avg"] - (self.std_dev_factor * df["Std_Dev"])
                df["ATR"] = df["high"].rolling(window=14).max() - df["low"].rolling(window=14).min()
                df["MACD"] = df["close"].ewm(span=12, adjust=False).mean() - df["close"].ewm(span=26, adjust=False).mean()
                df["MACD_Signal"] = df["MACD"].ewm(span=9, adjust=False).mean()
                df["Bollinger_Width"] = (df["Upper_Band"] - df["Lower_Band"]) / df["Moving_Avg"]
                df["ROC"] = ((df["close"] - df["close"].shift(10)) / df["close"].shift(10)) * 100
                df["ADX"] = df["ATR"].rolling(window=14).mean()

                delta = df["close"].diff()
                gain = delta.where(delta > 0, 0)
                loss = -delta.where(delta < 0, 0)
                avg_gain = gain.rolling(window=14, min_periods=1).mean()
                avg_loss = loss.rolling(window=14, min_periods=1).mean()
                rs = avg_gain / avg_loss
                df["RSI"] = 100 - (100 / (1 + rs))

                df.rename(columns={
                    "close": "Close",
                    "high": "High",
                    "low": "Low",
                    "open": "Open",
                    "volume": "Volume"
                }, inplace=True)

                df.replace([np.inf, -np.inf], np.nan, inplace=True)
                df.dropna(inplace=True)

                df["Ticker"] = ticker
                df = df.set_index("Ticker", append=True).reorder_levels(["Ticker", "timestamp"])
                df = df.join(macro_df, how="left")
                df.dropna(inplace=True)

                stock_data[ticker] = df

            except Exception as e:
                print(f"❌ Failed to load data for {ticker}: {e}")
        return stock_data

    def run_backtest(self, tickers, start_date, end_date):
        stock_data = self.fetch_stock_data(tickers, start_date, end_date)

        if not stock_data:
            print("⚠ No valid stock data found. Exiting backtest.")
            return

        initial_balance = 10_000
        balance = initial_balance
        trade_count = 0
        self.trade_log = []
        self.correct_predictions = 0
        equity_curve = []

        print(f"\n🔍 Running backtest with initial capital: ${initial_balance}")

        tickers = list(stock_data.keys())
        max_length = min(len(df) for df in stock_data.values())

        if max_length == 0:
            print("⚠ No valid data for backtest. Exiting.")
            return

        open_trades = []
        i = 0
        while i < max_length - self.future_prediction_hours:
            current_date = None
            available_balance = balance * (1 - self.risk_per_trade * len(open_trades))

            for ticker in tickers:
                df = stock_data[ticker]
                if len(df) <= i:
                    continue

                if current_date is None:
                    current_date = df.index[i][1]

                features = df.iloc[i][[
                    "Close", "Moving_Avg", "Upper_Band", "Lower_Band",
                    "MACD", "MACD_Signal", "RSI", "ATR",
                    "Bollinger_Width", "ROC", "ADX",
                    "CPI", "Zinsen", "Arbeitslosenquote"
                ]].values.reshape(1, -1)

                probs = self.model.predict_proba(features)[0]
                p_down, p_up = probs
                prob = max(p_down, p_up)

                if prob >= self.min_prob_threshold:
                    trade_type = "Long" if p_up >= p_down else "Short"
                    trade_entry_price = df.iloc[i]["Close"].item()
                    trade_amount = balance * self.risk_per_trade

                    if available_balance >= trade_amount:
                        open_trades.append({
                            "ticker": ticker,
                            "type": trade_type,
                            "entry_price": trade_entry_price,
                            "entry_day": i,
                            "trade_amount": trade_amount,
                            "open": True,
                            "prob": prob,
                            "entry_date": df.index[i][1].strftime("%Y-%m-%d %H:%M:%S"),
                            "max_duration": 240  # e.g., 24 hours
                        })
                        trade_count += 1

            new_open_trades = []
            for trade in open_trades:
                if not trade["open"]:
                    continue

                df = stock_data[trade["ticker"]]
                entry_day = trade["entry_day"]
                trade_age = i - entry_day

                if entry_day + self.future_prediction_hours >= len(df):
                    continue

                future_prices = df.iloc[entry_day + 1: entry_day + 1 + self.future_prediction_hours]["Close"]
                max_price = future_prices.max()
                min_price = future_prices.min()

                exit_price = None
                trade_outcome = None

                if trade["type"] == "Long":
                    if max_price >= trade["entry_price"] * (1 + self.take_profit):
                        exit_price = trade["entry_price"] * (1 + self.take_profit)
                        trade_outcome = "Long Profit"
                    elif min_price <= trade["entry_price"] * (1 - self.stop_loss):
                        exit_price = trade["entry_price"] * (1 - self.stop_loss)
                        trade_outcome = "Long Stop Loss"
                elif trade["type"] == "Short":
                    if min_price <= trade["entry_price"] * (1 - self.take_profit):
                        exit_price = trade["entry_price"] * (1 + self.take_profit)
                        trade_outcome = "Short Profit"
                    elif max_price >= trade["entry_price"] * (1 + self.stop_loss):
                        exit_price = trade["entry_price"] * (1 - self.stop_loss)
                        trade_outcome = "Short Stop Loss"

                if exit_price is None and trade_age >= trade["max_duration"]:
                    exit_price = df.iloc[i]["Close"]
                    trade_outcome = f"{trade['type']} Timeout"

                    trade_fee = trade["trade_amount"] * self.trading_fee
                    profit_loss = (exit_price - trade["entry_price"]) / trade["entry_price"]
                    if trade["type"] == "Short":
                        profit_loss = -profit_loss  # Gewinn bei fallendem Kurs
                    trade_result = trade["trade_amount"] * profit_loss
                    balance += trade_result - trade_fee
                    trade["open"] = False

                    actual_movement = int((exit_price > trade["entry_price"]) if trade["type"] == "Long" else (
                                exit_price < trade["entry_price"]))
                    predicted_movement = 1 if trade["type"] == "Long" else 0

                    if actual_movement == predicted_movement:
                        self.correct_predictions += 1

                    trade_start_date = trade["entry_date"]
                    trade_end_date = df.index[i][1].strftime("%Y-%m-%d %H:%M:%S")

                    self.trade_log.append({
                        "Trade Start Date": trade_start_date,
                        "Trade End Date": trade_end_date,
                        "Hour": entry_day,
                        "Stock": trade["ticker"],
                        "Trade Type": trade_outcome,
                        "Trade Price": trade["entry_price"],
                        "Exit Price": exit_price,
                        "Profit/Loss": trade_result,
                        "Fees": trade_fee,
                        "Balance": balance,
                        "Prediction Prob": trade["prob"]
                    })
                    continue

                if exit_price is not None:
                    trade_fee = trade["trade_amount"] * self.trading_fee
                    profit_loss = (exit_price - trade["entry_price"]) / trade["entry_price"]
                    trade_result = trade["trade_amount"] * profit_loss
                    balance += trade_result - trade_fee
                    trade["open"] = False

                    actual_movement = int((exit_price > trade["entry_price"]) if trade["type"] == "Long" else (exit_price < trade["entry_price"]))
                    predicted_movement = 1 if trade["type"] == "Long" else 0

                    if actual_movement == predicted_movement:
                        self.correct_predictions += 1

                    trade_start_date = trade["entry_date"]
                    trade_end_date = df.index[i][1].strftime("%Y-%m-%d %H:%M:%S")

                    self.trade_log.append({
                        "Trade Start Date": trade_start_date,
                        "Trade End Date": trade_end_date,
                        "Hour": entry_day,
                        "Stock": trade["ticker"],
                        "Trade Type": trade_outcome,
                        "Trade Price": trade["entry_price"],
                        "Exit Price": exit_price,
                        "Profit/Loss": trade_result,
                        "Fees": trade_fee,
                        "Balance": balance,
                        "Prediction Prob": trade["prob"]
                    })
                else:
                    new_open_trades.append(trade)

            open_trades = new_open_trades

            if current_date:
                equity_curve.append({"Date": current_date.strftime("%Y-%m-%d %H:%M:%S"), "Equity": balance})

            i += 1

        final_balance = balance
        roi = ((final_balance - initial_balance) / initial_balance) * 100
        prediction_accuracy = (self.correct_predictions / trade_count) * 100 if trade_count > 0 else 0

        print(f"🏁 End capital: ${final_balance:,.2f}")
        print(f"📊 Number of trades: {trade_count}")
        print(f"📈 Total ROI: {roi:.2f}%")
        print(f"🎯 Prediction Accuracy: {prediction_accuracy:.2f}%")

        log_df = pd.DataFrame(self.trade_log)
        log_df.to_csv("trade_log.csv", index=False)
        print("📝 Trade log saved as 'trade_log.csv'")

        equity_df = pd.DataFrame(equity_curve)
        equity_df.to_csv("equity_curve.csv", index=False)
        print("📈 Equity curve saved as 'equity_curve.csv'")

    def plot_equity_curve(self, start_date, end_date):
        # 📥 Load Equity Curve
        equity_df = pd.read_csv("equity_curve.csv")
        equity_df["Date"] = pd.to_datetime(equity_df["Date"])
        equity_df.set_index("Date", inplace=True)


        request_params = StockBarsRequest(
            symbol_or_symbols="SPY",
            timeframe=TimeFrame.Hour,
            start=start_date,
            end=end_date
        )

        try:
            bars = self.alpaca_client.get_stock_bars(request_params).df

            if isinstance(bars.index, pd.MultiIndex):
                bars = bars.reset_index(level="symbol", drop=True)

            # Zeitzone entfernen
            bars.index = bars.index.tz_convert(None)

            bars.rename(columns={"close": "Close"}, inplace=True)
            bars = bars[["Close"]]
            bars = bars.loc[bars.index.isin(equity_df.index)]
            bars["SPY Equity"] = bars["Close"] / bars["Close"].iloc[0] * equity_df["Equity"].iloc[0]

            # 📊 Calculate percentage change
            equity_df["Equity %"] = (equity_df["Equity"] / equity_df["Equity"].iloc[0] - 1) * 100
            bars["SPY %"] = (bars["SPY Equity"] / bars["SPY Equity"].iloc[0] - 1) * 100

            # 🖼️ Plot
            plt.figure(figsize=(12, 6))
            plt.plot(equity_df.index, equity_df["Equity %"], label="XGBoost Share Predictor", linewidth=2)
            plt.plot(bars.index, bars["SPY %"], label="S&P 500", linestyle='--', linewidth=2)
            plt.title("Percentage Change in Portfolio Value vs. S&P 500")
            plt.xlabel("Time")
            plt.ylabel("Change [%]")
            plt.legend()
            plt.grid(True)
            plt.tight_layout()
            plt.show()

        except Exception as e:
            print(f"❌ Failed to load SPY data: {e}")

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


