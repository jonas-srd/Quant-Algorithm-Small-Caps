import numpy as np
import pandas as pd
import yfinance as yf
from StockPredictor import StockPredictor
import pdb


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
        self.min_prob_threshold = min_prob_threshold  # Minimum probability for a trade
        self.trade_log = []
        self.correct_predictions = 0


    def fetch_stock_data(self, tickers, start_date, end_date):
        """Fetches stock data for multiple tickers and returns a dictionary of dataframes."""
        data = {}
        for ticker in tickers:
            print(f"📊 Fetching data for {ticker} from {start_date} to {end_date}...")
            stock_data = yf.download(ticker, start=start_date, end=end_date)

            if stock_data.empty:
                print(f"⚠️ No data found for {ticker}. Skipping.")
                continue

            # Compute technical indicators
            stock_data['Moving_Avg'] = stock_data['Close'].rolling(window=self.lookback_period).mean()
            stock_data['Std_Dev'] = stock_data['Close'].rolling(window=self.lookback_period).std()
            stock_data['Upper_Band'] = stock_data['Moving_Avg'] + (self.std_dev_factor * stock_data['Std_Dev'])
            stock_data['Lower_Band'] = stock_data['Moving_Avg'] - (self.std_dev_factor * stock_data['Std_Dev'])
            stock_data['ATR'] = stock_data['High'].rolling(window=14).max() - stock_data['Low'].rolling(window=14).min()
            stock_data['MACD'] = stock_data['Close'].ewm(span=12, adjust=False).mean() - stock_data['Close'].ewm(span=26, adjust=False).mean()
            stock_data['MACD_Signal'] = stock_data['MACD'].ewm(span=9, adjust=False).mean()
            stock_data['Bollinger_Width'] = (stock_data['Upper_Band'] - stock_data['Lower_Band']) / stock_data['Moving_Avg']
            stock_data['ROC'] = ((stock_data['Close'] - stock_data['Close'].shift(10)) / stock_data['Close'].shift(10)) * 100
            stock_data['ADX'] = stock_data['ATR'].rolling(window=14).mean()

            delta = stock_data["Close"].diff()
            gain = delta.where(delta > 0, 0)
            loss = -delta.where(delta < 0, 0)
            avg_gain = gain.rolling(window=14, min_periods=1).mean()
            avg_loss = loss.rolling(window=14, min_periods=1).mean()
            rs = avg_gain / avg_loss
            stock_data["RSI"] = 100 - (100 / (1 + rs))

            stock_data.replace([np.inf, -np.inf], np.nan, inplace=True)
            stock_data.dropna(inplace=True)

            data[ticker] = stock_data

        return data

    def predict_best_stock(self, stock_data, index):
        """Selects the single best stock (long or short) with the highest absolute probability."""
        predictions = {}

        for ticker, df in stock_data.items():
            if df is None or len(df) <= index:
                continue

            features = df.iloc[index][["Close", "Moving_Avg", "Upper_Band", "Lower_Band",
                                       "MACD", "MACD_Signal", "RSI", "ATR",
                                       "Bollinger_Width", "ROC", "ADX"]].values.reshape(1, -1)

            probs = self.model.predict_proba(features)[0]  # Get [P(fall), P(rise)]
            predictions[ticker] = probs

        if not predictions:
            return None, None, None

        # Select stock with the highest absolute probability (long or short)
        best_ticker, (p_down, p_up) = max(predictions.items(), key=lambda item: max(item[1]))

        # Determine trade type based on highest probability
        trade_type = "Long" if p_up >= p_down else "Short"
        trade_prob = max(p_up, p_down)

        # Apply probability threshold
        if trade_prob < self.min_prob_threshold:
            return None, None, None

        return best_ticker, trade_type, trade_prob

    def run_backtest(self, tickers, start_date, end_date):
        """Runs the backtest dynamically selecting the best stock and waits until a trade is closed before entering another."""
        stock_data = self.fetch_stock_data(tickers, start_date, end_date)

        if not stock_data:
            print("⚠ No valid stock data found. Exiting backtest.")
            return

        initial_balance = 10_000
        balance = initial_balance
        trade_count = 0
        open_trade = None
        trade_entry_price = None
        trade_stock = None
        trade_type = None

        print(f"\n🔍 Running backtest with initial capital: ${initial_balance}")

        max_length = min((len(df) for df in stock_data.values()), default=0)

        if max_length == 0:
            print("⚠ No valid data for backtest. Exiting.")
            return

        i = 0
        while i < max_length - self.future_prediction_days:
            if open_trade is None:
                best_stock, trade_type, trade_prob = self.predict_best_stock(stock_data, i)

                if best_stock:
                    trade_stock = best_stock
                    trade_entry_price = stock_data[best_stock].iloc[i]["Close"].item()
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

                # Determine actual market movement based on trade type
                if open_trade == "Long":
                    actual_movement = 1 if exit_price > trade_entry_price else 0
                elif open_trade == "Short":
                    actual_movement = 1 if exit_price < trade_entry_price else 0  # ✅ Short profits when price drops
                else:
                    actual_movement = None

                # Predicted movement is based on whether the model suggested Long or Short
                predicted_movement = 1 if open_trade == "Long" else 0

                # Only count if there's a valid actual movement to compare
                if actual_movement is not None and actual_movement == predicted_movement:
                    self.correct_predictions += 1
                trade_start_date = stock_data[trade_stock].index[i].strftime("%Y-%m-%d")  # Date of trade entry
                trade_end_date = stock_data[trade_stock].index[i + self.future_prediction_days].strftime(
                    "%Y-%m-%d")  # Exit date

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

        trade_df = pd.DataFrame(self.trade_log)
        trade_df.to_csv("backtest_results.csv", index=False)
        print("✅ Backtest results saved as 'backtest_results.csv'")




