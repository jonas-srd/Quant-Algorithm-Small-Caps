import numpy as np
import pandas as pd
from StockPredictor import StockPredictor

class Backtester:
    def __init__(self, model, future_prediction_days=5, trading_fee=0.00, risk_per_trade=0.1, stop_loss=0.01, take_profit=0.04):
        self.model = model
        self.future_prediction_days = future_prediction_days
        self.trading_fee = trading_fee
        self.risk_per_trade = risk_per_trade
        self.stop_loss = stop_loss
        self.take_profit = take_profit
        self.trade_log = []

    def calculate_risk_metrics(self, returns, risk_free_rate=0.02):
        returns = np.array(returns)
        mean_return = np.mean(returns)
        std_dev = np.std(returns)
        downside_std = np.std(returns[returns < 0]) if len(returns[returns < 0]) > 0 else 1
        max_drawdown = np.max(np.maximum.accumulate(returns) - returns) if len(returns) > 0 else 0
        annualized_return = np.mean(returns) * 252

        sharpe_ratio = (annualized_return - risk_free_rate) / (std_dev * np.sqrt(252)) if std_dev > 0 else np.nan
        sortino_ratio = (annualized_return - risk_free_rate) / (downside_std * np.sqrt(252)) if downside_std > 0 else np.nan
        calmar_ratio = annualized_return / max_drawdown if max_drawdown > 0 else np.nan

        return {
            "Sharpe Ratio": sharpe_ratio,
            "Sortino Ratio": sortino_ratio,
            "Max Drawdown": max_drawdown,
            "Calmar Ratio": calmar_ratio,
            "Volatilität": std_dev * np.sqrt(252)
        }

    def backtest(self, df):
        initial_balance = 10_000
        balance = initial_balance
        position = 0.0
        short_position = 0.0
        last_buy_day = -3  # Damit direkt ein Kauf am ersten Tag möglich ist
        trade_interval = 3  # Nur alle 3 Tage kaufen
        trade_count = 0
        daily_returns = []

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

            # Kauf (nur alle 3 Tage erlaubt)
            if i - last_buy_day >= trade_interval:
                if prediction == 1:  # Long
                    invest_amount = balance * self.risk_per_trade
                    balance -= invest_amount
                    position = invest_amount / trade_price
                    last_buy_day = i  # Letzten Kauf-Tag speichern
                    trade_count += 1
                elif prediction == 0:  # Short
                    invest_amount = balance * self.risk_per_trade
                    balance += invest_amount
                    short_position = invest_amount / trade_price
                    last_buy_day = i  # Letzten Kauf-Tag speichern
                    trade_count += 1

            # Verkauf (jederzeit erlaubt)
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

            daily_return = (balance - initial_balance) / initial_balance
            daily_returns.append(daily_return)

            self.trade_log.append({
                "Day": i,
                "Balance": balance,
                "Trade Type": "Long" if prediction == 1 else "Short",
                "Trade Price": trade_price
            })

        final_balance = balance
        roi = ((final_balance - initial_balance) / initial_balance) * 100
        risk_metrics = self.calculate_risk_metrics(daily_returns)

        print(f"🔹 Backtest abgeschlossen!")
        print(f"📊 Anzahl der Trades: {trade_count}")
        print(f"💰 Startkapital: ${initial_balance:,.2f}")
        print(f"🏁 Endkapital: ${final_balance:,.2f}")
        print(f"📈 Gesamtrendite: {roi:.2f}%")
        print(f"📉 Risikokennzahlen:")
        for metric, value in risk_metrics.items():
            print(f"   {metric}: {value:.4f}")

        pd.DataFrame(self.trade_log).to_csv("backtest_results.csv", index=False)
        print("✅ Backtest-Ergebnisse gespeichert als 'backtest_results.csv'")
