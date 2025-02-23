import numpy as np
from StockPredictor import StockPredictor

class Backtester:
    def __init__(self, model, future_prediction_days=5, trading_fee=0.00, risk_per_trade=0.1, stop_loss=0.01, take_profit=0.04):
        self.model = model
        self.future_prediction_days = future_prediction_days
        self.trading_fee = trading_fee
        self.risk_per_trade = risk_per_trade
        self.stop_loss = stop_loss
        self.take_profit = take_profit

    def backtest(self, df):
        initial_balance = 10_000
        balance = initial_balance
        position = 0.0
        short_position = 0.0
        trade_count = 0

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

            if prediction == 1:  # Long
                invest_amount = balance * self.risk_per_trade
                balance -= invest_amount
                position = invest_amount / trade_price
                trade_count += 1
            elif prediction == 0:  # Short
                invest_amount = balance * self.risk_per_trade
                balance += invest_amount
                short_position = invest_amount / trade_price
                trade_count += 1

            # Gewinn/Verlust berechnen
            if position > 0.0:
                if ((max_price - trade_price) / trade_price >= self.take_profit or
                    (trade_price - min_price) / trade_price >= self.stop_loss):
                    sell_amount = position * trade_price
                    fee = sell_amount * self.trading_fee
                    balance += sell_amount - fee
                    position = 0.0

            if short_position > 0.0:
                if ((trade_price - min_price) / trade_price >= self.take_profit or
                    (max_price - trade_price) / trade_price >= self.stop_loss):
                    buyback_amount = short_position * trade_price
                    fee = buyback_amount * self.trading_fee
                    balance -= buyback_amount + fee
                    short_position = 0.0
            #print(balance)

        final_balance = balance
        roi = ((final_balance - initial_balance) / initial_balance) * 100

        print(f"🔹 Backtest abgeschlossen!")
        print(f"📊 Anzahl der Trades: {trade_count}")
        print(f"💰 Startkapital: ${initial_balance:,.2f}")
        print(f"🏁 Endkapital: ${final_balance:,.2f}")
        print(f"📈 Gesamtrendite: {roi:.2f}%")



