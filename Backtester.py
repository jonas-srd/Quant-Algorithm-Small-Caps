import numpy as np
import pandas as pd
from StockPredictor import StockPredictor
import pdb

class Backtester:
    def __init__(self, model, future_prediction_days=5, trading_fee=0.005, risk_per_trade=0.1, stop_loss=0.01, take_profit=0.04):
        self.model = model
        self.future_prediction_days = future_prediction_days
        self.trading_fee = trading_fee
        self.risk_per_trade = risk_per_trade
        self.stop_loss = stop_loss
        self.take_profit = take_profit
        self.trade_log = []

    def backtest(self, df):
        initial_balance = 10_000
        balance = initial_balance
        position = 0.0
        short_position = 0.0
        trade_count = 0
        daily_returns = []

        print(f"🔍 Backtest gestartet mit Startkapital: ${initial_balance}")

        for i in range(len(df) - self.future_prediction_days):
            features = np.array(df.iloc[i][["Close", "Moving_Avg", "Upper_Band", "Lower_Band",
                                            "MACD", "MACD_Signal", "RSI", "ATR",
                                            "Bollinger_Width", "ROC", "ADX"]].values).reshape(1, -1)

            if np.any(np.isnan(features)) or np.any(np.isinf(features)):
                continue

            prediction = self.model.predict(features)[0]

            try:
                trade_price = df.iloc[i]["Close"].item()
                max_price = df.iloc[i+1:i+self.future_prediction_days+1]["Close"].max().item()
                min_price = df.iloc[i+1:i+self.future_prediction_days+1]["Close"].min().item()
            except ValueError:
                continue

            # **Trade-Öffnung nur wenn keine offene Position existiert**
            if position == 0 and short_position == 0:
                if prediction == 1:  # Long
                    invest_amount = balance * self.risk_per_trade
                    balance -= invest_amount
                    position = invest_amount / trade_price
                    long_entry_price = trade_price  # Store entry price
                    trade_count += 1
                    self.trade_log.append({
                        "Day": i,
                        "Trade Type": "Long Open",
                        "Trade Price": trade_price,
                        "Investment": invest_amount,
                        "Balance": balance
                    })

                elif prediction == 0:  # Short
                    invest_amount = balance * self.risk_per_trade
                    balance -= invest_amount
                    short_position = invest_amount / trade_price
                    short_entry_price = trade_price  # Store entry price
                    trade_count += 1
                    self.trade_log.append({
                        "Day": i,
                        "Trade Type": "Short Open",
                        "Trade Price": trade_price,
                        "Investment": invest_amount,
                        "Balance": balance
                    })

            # **Position schließen (Take Profit oder Stop Loss)**
            if position > 0.0:
                if (max_price / long_entry_price - 1 >= self.take_profit) or \
                        (1 - min_price / long_entry_price >= self.stop_loss):
                    exit_price = (1 + self.take_profit) * long_entry_price if (max_price / long_entry_price - 1 >= self.take_profit) else (1 - self.stop_loss)*long_entry_price
                    sell_amount = position * exit_price
                    fee = sell_amount * self.trading_fee
                    balance += sell_amount - fee
                    position = 0.0  # Close position

                    self.trade_log.append({
                        "Day": i,
                        "Trade Type": "Long Close",
                        "Trade Price": exit_price,  # Corrected to actual exit price
                        "Investment": 0.0,
                        "Balance": balance
                    })

            if short_position > 0.0:
                if (short_entry_price / min_price - 1 >= self.take_profit) or \
                        (1 - short_entry_price / max_price >= self.stop_loss):
                    exit_price = (1 + self.take_profit) * short_entry_price if (short_entry_price / min_price - 1 >= self.take_profit) else (1 - self.stop_loss)*short_entry_price
                    profit_loss = exit_price * short_position
                    fee = abs(profit_loss) * self.trading_fee  # Ensure fees are subtracted correctly
                    balance += profit_loss - fee
                    short_position = 0.0  # Close position

                    self.trade_log.append({
                        "Day": i,
                        "Trade Type": "Short Close",
                        "Trade Price": exit_price,  # Corrected to actual exit price
                        "Investment": 0.0,
                        "Balance": balance
                    })

            daily_return = (balance - initial_balance) / initial_balance
            daily_returns.append(daily_return)

        final_balance = balance
        roi = ((final_balance - initial_balance) / initial_balance) * 100

        print(f"\n🔹 Backtest abgeschlossen!")
        print(f"📊 Anzahl der Trades: {trade_count}")
        print(f"💰 Startkapital: ${initial_balance:,.2f}")
        print(f"🏁 Endkapital: ${final_balance:,.2f}")
        print(f"📈 Gesamtrendite: {roi:.2f}%")

        # Speichern der Trades in CSV-Datei
        trade_df = pd.DataFrame(self.trade_log)
        trade_df.to_csv("backtest_results.csv", index=False)

        print("✅ Backtest-Ergebnisse gespeichert als 'backtest_results.csv'")
