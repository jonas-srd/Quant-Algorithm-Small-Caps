import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from .predictor import StockPredictor
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame


class Backtester:
    def __init__(
        self,
        model,
        future_prediction_hours=24,
        trading_fee=0.005,
        risk_per_trade=0.2,
        stop_loss=0.01,
        take_profit=0.04,
        lookback_period=10,
        std_dev_factor=2,
        min_prob_threshold=0.7
    ):
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
            api_key=os.getenv("ALPACA_API_KEY"),
            secret_key=os.getenv("ALPACA_SECRET_KEY")
        )

        self.stock_predictor = StockPredictor(
            tickers=[],
            lookback_period=self.lookback_period,
            std_dev_factor=self.std_dev_factor,
            future_prediction_days=1
        )

    def fetch_stock_data(self, tickers, start_date, end_date):
        self.stock_predictor.tickers = tickers
        macro_df = self.stock_predictor.fetch_macro_data(start_date, end_date)
        macro_df.index = pd.to_datetime(macro_df.index).tz_localize(None)
        macro_df.index.name = "timestamp"

        stock_data = {}
        for ticker in tickers:
            request = StockBarsRequest(
                symbol_or_symbols=ticker,
                timeframe=TimeFrame.Hour,
                start=start_date,
                end=end_date
            )
            try:
                bars = self.alpaca_client.get_stock_bars(request).df
                if bars.empty:
                    continue

                df = bars.reset_index()
                df = df[df["symbol"] == ticker].drop(columns="symbol")
                df.set_index("timestamp", inplace=True)
                df.index = df.index.tz_convert(None)

                # Feature engineering
                df["Moving_Avg"] = df["close"].rolling(window=self.lookback_period).mean()
                df["Std_Dev"] = df["close"].rolling(window=self.lookback_period).std()
                df["Upper_Band"] = df["Moving_Avg"] + (self.std_dev_factor * df["Std_Dev"])
                df["Lower_Band"] = df["Moving_Avg"] - (self.std_dev_factor * df["Std_Dev"])
                df["ATR"] = df["high"].rolling(window=14).max() - df["low"].rolling(window=14).min()
                df["MACD"] = df["close"].ewm(span=12).mean() - df["close"].ewm(span=26).mean()
                df["MACD_Signal"] = df["MACD"].ewm(span=9).mean()
                df["Bollinger_Width"] = (df["Upper_Band"] - df["Lower_Band"]) / df["Moving_Avg"]
                df["ROC"] = ((df["close"] - df["close"].shift(10)) / df["close"].shift(10)) * 100
                df["ADX"] = df["ATR"].rolling(window=14).mean()

                delta = df["close"].diff()
                gain = delta.where(delta > 0, 0)
                loss = -delta.where(delta < 0, 0)
                avg_gain = gain.rolling(window=14).mean()
                avg_loss = loss.rolling(window=14).mean()
                rs = avg_gain / avg_loss
                df["RSI"] = 100 - (100 / (1 + rs))

                df.rename(columns={
                    "close": "Close", "high": "High", "low": "Low",
                    "open": "Open", "volume": "Volume"
                }, inplace=True)

                df.replace([np.inf, -np.inf], np.nan, inplace=True)
                df.dropna(inplace=True)
                df["Ticker"] = ticker
                df = df.set_index("Ticker", append=True).reorder_levels(["Ticker", "timestamp"])
                df = df.join(macro_df, how="left").dropna()

                stock_data[ticker] = df
            except Exception:
                continue
        return stock_data

    def run_backtest(self, tickers, start_date, end_date):
        stock_data = self.fetch_stock_data(tickers, start_date, end_date)
        if not stock_data:
            return

        self.balance = 10_000
        initial_balance = self.balance
        self.trade_log = []
        self.correct_predictions = 0
        trade_count = 0
        equity_curve = []

        tickers = list(stock_data.keys())
        max_length = min(len(df) for df in stock_data.values())
        if max_length == 0:
            return

        open_trades = []
        for i in range(max_length - self.future_prediction_hours):
            current_date = None
            available_balance = self.balance * (1 - self.risk_per_trade * len(open_trades))

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
                    "CPI", "Zinsen", "Arbeitslosenquote", "VIX", "Oil_Price"
                ]].values.reshape(1, -1)

                p_down, p_up = self.model.predict_proba(features)[0]
                prob = max(p_down, p_up)

                if prob >= self.min_prob_threshold:
                    trade_type = "Long" if p_up >= p_down else "Short"
                    trade_entry_price = df.iloc[i]["Close"]
                    trade_amount = self.balance * self.risk_per_trade

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
                            "max_duration": 240
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

                tp = trade["entry_price"] * (1 + self.take_profit)
                sl = trade["entry_price"] * (1 - self.stop_loss)
                exit_price = None
                outcome = None

                for price in future_prices:
                    if trade["type"] == "Long":
                        if price >= tp:
                            exit_price, outcome = tp, "Long Profit"
                            break
                        elif price <= sl:
                            exit_price, outcome = sl, "Long Stop Loss"
                            break
                    else:  # Short
                        tp_s = trade["entry_price"] * (1 - self.take_profit)
                        sl_s = trade["entry_price"] * (1 + self.stop_loss)
                        if price <= tp_s:
                            exit_price, outcome = tp_s, "Short Profit"
                            break
                        elif price >= sl_s:
                            exit_price, outcome = sl_s, "Short Stop Loss"
                            break

                if exit_price is None and trade_age >= trade["max_duration"]:
                    exit_price = df.iloc[entry_day + 1 + self.future_prediction_hours]["Close"]
                    outcome = f"{trade['type']} Timeout"
                    self.close_trade(trade, exit_price, df.index[i][1], outcome)
                elif exit_price:
                    self.close_trade(trade, exit_price, df.index[i][1], outcome)
                else:
                    new_open_trades.append(trade)

            open_trades = new_open_trades
            if current_date:
                equity_curve.append({"Date": current_date.strftime("%Y-%m-%d %H:%M:%S"), "Equity": self.balance})

        final_balance = self.balance
        roi = ((final_balance - initial_balance) / initial_balance) * 100
        prediction_accuracy = (self.correct_predictions / trade_count) * 100 if trade_count > 0 else 0
        pd.DataFrame(self.trade_log).to_csv("data/trade_log.csv", index=False)
        pd.DataFrame(equity_curve).to_csv("data/equity_curve.csv", index=False)
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

    def close_trade(self, trade, exit_price, current_date, outcome):
        fee = trade["trade_amount"] * self.trading_fee
        pl = (exit_price - trade["entry_price"]) / trade["entry_price"]
        if trade["type"] == "Short":
            pl = -pl

        result = trade["trade_amount"] * pl
        self.balance += result - fee
        trade["open"] = False

        actual = int((exit_price > trade["entry_price"]) if trade["type"] == "Long" else (exit_price < trade["entry_price"]))
        predicted = 1 if trade["type"] == "Long" else 0
        if actual == predicted:
            self.correct_predictions += 1

        self.trade_log.append({
            "Trade Start Date": trade["entry_date"],
            "Trade End Date": current_date.strftime("%Y-%m-%d %H:%M:%S"),
            "Hour": trade["entry_day"],
            "Stock": trade["ticker"],
            "Trade Type": outcome,
            "Trade Price": trade["entry_price"],
            "Exit Price": exit_price,
            "Profit/Loss": result,
            "Fees": fee,
            "Balance": self.balance,
            "Prediction Prob": trade["prob"]
        })

    def plot_results(self):
        try:
            equity_df = pd.read_csv("data/equity_curve.csv")
            trade_df = pd.read_csv("data/trade_log.csv")

            # Equity curve
            plt.figure(figsize=(12, 6))
            plt.plot(pd.to_datetime(equity_df["Date"]), equity_df["Equity"], label="Equity")
            plt.xlabel("Date")
            plt.ylabel("Equity")
            plt.title("Equity Curve")
            plt.grid(True)
            plt.legend()
            plt.tight_layout()
            plt.savefig("plots/equity_curve.png")
            plt.show()

            # Drawdown
            equity_df["Peak"] = equity_df["Equity"].cummax()
            equity_df["Drawdown"] = equity_df["Equity"] / equity_df["Peak"] - 1
            plt.figure(figsize=(12, 4))
            plt.fill_between(pd.to_datetime(equity_df["Date"]), equity_df["Drawdown"], color="red")
            plt.title("Drawdown Curve")
            plt.ylabel("Drawdown")
            plt.xlabel("Date")
            plt.grid(True)
            plt.tight_layout()
            plt.savefig("plots/drawdown.png")
            plt.show()

            # Rolling Sharpe Ratio
            returns = equity_df["Equity"].pct_change().fillna(0)
            rolling_sharpe = returns.rolling(window=20).mean() / returns.rolling(window=20).std()
            plt.figure(figsize=(12, 4))
            plt.plot(pd.to_datetime(equity_df["Date"]), rolling_sharpe, label="Rolling Sharpe Ratio")
            plt.axhline(0, color='black', linestyle='--', linewidth=1)
            plt.title("Rolling Sharpe Ratio (20 steps)")
            plt.xlabel("Date")
            plt.ylabel("Sharpe Ratio")
            plt.grid(True)
            plt.tight_layout()
            plt.savefig("plots/rolling_sharpe.png")
            plt.show()

            # Profit/Loss distribution
            plt.figure(figsize=(10, 5))
            plt.hist(trade_df["Profit/Loss"], bins=30, color="skyblue", edgecolor="black")
            plt.title("Distribution of Profit and Loss")
            plt.xlabel("Profit/Loss")
            plt.ylabel("Number of Trades")
            plt.grid(True)
            plt.tight_layout()
            plt.savefig("plots/pl_distribution.png")
            plt.show()

            # Trades over time
            trade_df["Trade Start Date"] = pd.to_datetime(trade_df["Trade Start Date"])
            trades_per_day = trade_df.groupby(trade_df["Trade Start Date"].dt.date).size()
            plt.figure(figsize=(10, 4))
            trades_per_day.plot(kind="bar")
            plt.title("Trades per Day")
            plt.xlabel("Date")
            plt.ylabel("Number of Trades")
            plt.tight_layout()
            plt.xticks(rotation=45)
            plt.savefig("plots/trades_per_day.png")
            plt.show()

            # Win rate per ticker
            trade_df["Win"] = trade_df["Profit/Loss"] > 0
            win_rate = trade_df.groupby("Stock")["Win"].mean()
            plt.figure(figsize=(10, 4))
            win_rate.plot(kind="bar", color="green")
            plt.title("Win Rate per Ticker")
            plt.xlabel("Ticker")
            plt.ylabel("Win Rate")
            plt.ylim(0, 1)
            plt.tight_layout()
            plt.savefig("plots/winrate_per_ticker.png")
            plt.show()

        except FileNotFoundError:
            print("CSV-Dateien für Plotting nicht gefunden. Bitte zuerst Backtest ausführen.")
