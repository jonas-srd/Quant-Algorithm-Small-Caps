import numpy as np
import pandas as pd
import yfinance as yf
from StockPredictor import StockPredictor


class Backtester:
    def __init__(self, model, future_prediction_days=5, trading_fee=0.005, risk_per_trade=0.1, stop_loss=0.01,
                 take_profit=0.04, lookback_period=10, std_dev_factor=2):
        self.model = model
        self.future_prediction_days = future_prediction_days
        self.trading_fee = trading_fee
        self.risk_per_trade = risk_per_trade
        self.stop_loss = stop_loss
        self.take_profit = take_profit
        self.lookback_period = lookback_period
        self.std_dev_factor = std_dev_factor
        self.trade_log = []

    def fetch_stock_data(self, ticker, start_date, end_date):
        print(f"Fetching data for {ticker} from {start_date} to {end_date}...")
        stock_data = yf.download(ticker, start=start_date, end=end_date)

        if stock_data.empty:
            print(f"⚠️ No data found for {ticker}. Check the ticker symbol and date range.")
            return None

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

        return stock_data.dropna()

    def backtest(self, df):
        if df is None or df.empty:
            print("⚠️ No valid data to backtest.")
            return

        initial_balance = 10_000
        balance = initial_balance
        position = 0.0
        short_position = 0.0
        trade_count = 0
        daily_returns = []

        print(f"\n🔍 Backtest started with initial capital: ${initial_balance}")

        for i in range(len(df) - self.future_prediction_days):
            features = np.array(df.iloc[i][["Close", "Moving_Avg", "Upper_Band", "Lower_Band",
                                            "MACD", "MACD_Signal", "ATR",
                                            "Bollinger_Width", "ROC", "ADX", "RSI"]].values).reshape(1, -1)

            if np.any(np.isnan(features)) or np.any(np.isinf(features)):
                continue

            prediction = self.model.predict(features)[0]

            try:
                trade_price = df.iloc[i]["Close"].item()
                max_price = df.iloc[i+1:i+self.future_prediction_days+1]["Close"].max().item()
                min_price = df.iloc[i+1:i+self.future_prediction_days+1]["Close"].min().item()
            except ValueError:
                continue

            if position == 0 and short_position == 0:
                if prediction == 1:
                    invest_amount = balance * self.risk_per_trade
                    balance -= invest_amount
                    position = invest_amount / trade_price
                    long_entry_price = trade_price
                    trade_count += 1
                    self.trade_log.append(
                        {"Day": i, "Trade Type": "Long Open", "Trade Price": trade_price, "Investment": invest_amount,
                         "Balance": balance})
                elif prediction == 0:
                    invest_amount = balance * self.risk_per_trade
                    balance -= invest_amount
                    short_position = invest_amount / trade_price
                    short_entry_price = trade_price
                    trade_count += 1
                    self.trade_log.append(
                        {"Day": i, "Trade Type": "Short Open", "Trade Price": trade_price, "Investment": invest_amount,
                         "Balance": balance})

            if position > 0.0:
                if (max_price / long_entry_price - 1 >= self.take_profit) or (
                        1 - min_price / long_entry_price >= self.stop_loss):
                    exit_price = long_entry_price * (1 + self.take_profit) if (
                                max_price / long_entry_price - 1 >= self.take_profit) else long_entry_price * (
                                1 - self.stop_loss)
                    balance += position * exit_price - (position * exit_price * self.trading_fee)
                    position = 0.0
                    self.trade_log.append(
                        {"Day": i, "Trade Type": "Long Close", "Trade Price": exit_price, "Investment": 0.0,
                         "Balance": balance})

            if short_position > 0.0:
                if (short_entry_price / min_price - 1 >= self.take_profit) or (
                        1 - short_entry_price / max_price >= self.stop_loss):
                    exit_price = short_entry_price * (1 - self.take_profit) if (
                                short_entry_price / min_price - 1 >= self.take_profit) else short_entry_price * (
                                1 + self.stop_loss)
                    balance += short_position * exit_price - (short_position * exit_price * self.trading_fee)
                    short_position = 0.0
                    self.trade_log.append(
                        {"Day": i, "Trade Type": "Short Close", "Trade Price": exit_price, "Investment": 0.0,
                         "Balance": balance})

            daily_return = (balance - initial_balance) / initial_balance
            daily_returns.append(daily_return)

        final_balance = balance
        roi = ((final_balance - initial_balance) / initial_balance) * 100

        print(f"\n🔹 Backtest complete!")
        print(f"📊 Number of trades: {trade_count}")
        print(f"💰 Start capital: ${initial_balance:,.2f}")
        print(f"🏁 End capital: ${final_balance:,.2f}")
        print(f"📈 Total ROI: {roi:.2f}%")

        trade_df = pd.DataFrame(self.trade_log)
        trade_df.to_csv("backtest_results.csv", index=False)
        print("✅ Backtest results saved as 'backtest_results.csv'")
