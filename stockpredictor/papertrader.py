import time
import pandas as pd
import numpy as np
import alpaca_trade_api as tradeapi
from datetime import datetime
import matplotlib.pyplot as plt
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame


class PaperTrader:
    def __init__(self, predictor, alpaca_key, alpaca_secret, paper_url="https://paper-api.alpaca.markets"):
        self.predictor = predictor
        self.api = tradeapi.REST(
            alpaca_key,
            alpaca_secret,
            base_url=paper_url
        )
        self.cash = 10_000.0
        self.risk_per_trade = 0.2
        self.stop_loss = 0.01
        self.take_profit = 0.04
        self.min_prob_threshold = 0.7
        self.positions = {}
        self.trade_log = []
        self.last_macro_update_day = None
        self.update_macro_if_needed()

    def update_macro_if_needed(self):
        today = datetime.today().date()
        if self.last_macro_update_day != today:
            print("🔄 Überprüfe Makrodaten...")
            self.predictor.auto_update_macro()
            self.last_macro_update_day = today

    def get_latest_features(self, df):
        latest = df.iloc[-1][[
            "Close", "Moving_Avg", "Upper_Band", "Lower_Band", "MACD", "MACD_Signal", "RSI", "ATR",
            "Bollinger_Width", "ROC", "ADX", "CPI", "Zinsen", "Arbeitslosenquote", "VIX", "Oil_Price"
        ]]
        if np.any(np.isnan(latest)) or np.any(np.isinf(latest)):
            return None, None
        return latest.values.reshape(1, -1), latest["Close"]

    def run_live_loop(self, interval_minutes=60, total_cycles=7):
        """
        Live-Trading Loop.
        Führt alle 'interval_minutes' einen Trade-Zyklus aus und stoppt nach 'total_cycles' Durchläufen.
        """
        print(f"🚀 Starte Paper-Trading mit 10.000 $ Kapital")
        cycle = 0

        while cycle < total_cycles:
            print(f"\n🕒 Zyklus {cycle + 1} von {total_cycles}")
            self.update_macro_if_needed()

            # Schritt 1: Nur 20% der Ticker zufällig wählen
            selected_tickers = np.random.choice(
                self.predictor.tickers,
                size=max(1, len(self.predictor.tickers) // 5),
                replace=False
            )

            # Schritt 2: Pro Ticker Wahrscheinlichkeiten berechnen
            ticker_candidates = []

            for ticker in selected_tickers:
                print(f"\n🔍 {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} – Analysiere {ticker}")
                df = self.predictor.fetch_stock_data(ticker)
                if df is None or df.empty:
                    continue

                try:
                    macro_df = pd.read_csv("macro_data_hourly.csv", index_col="timestamp", parse_dates=True)
                    df = df.join(macro_df, how="left")
                    df.dropna(inplace=True)
                except Exception as e:
                    print(f"❌ Makrodatenfehler: {e}")
                    continue

                features, price = self.get_latest_features(df)
                if features is None:
                    continue

                probs = self.predictor.model.predict_proba(features)[0]
                p_down, p_up = probs
                prob = max(p_down, p_up)

                if prob >= self.min_prob_threshold:
                    ticker_candidates.append({
                        "ticker": ticker,
                        "price": price,
                        "prob": prob,
                        "p_up": p_up,
                        "p_down": p_down,
                        "features": features,
                        "df": df
                    })

            # Schritt 3: Sortiere nach Wahrscheinlichkeit und wähle die Top 6
            top_candidates = sorted(ticker_candidates, key=lambda x: x["prob"], reverse=True)[:6]

            if not top_candidates:
                print("❗ Kein Trade-Kandidat mit ausreichender Wahrscheinlichkeit.")
                cycle += 1
                time.sleep(interval_minutes * 60)
                continue

            for candidate in top_candidates:
                ticker = candidate["ticker"]
                price = candidate["price"]
                features = candidate["features"]
                df = candidate["df"]
                p_up = candidate["p_up"]
                p_down = candidate["p_down"]
                prob = candidate["prob"]

                trade_type = "Long" if p_up >= p_down else "Short"
                entry_price = price
                position = self.positions.get(ticker)

                if position is None:
                    trade_amount = self.cash * self.risk_per_trade
                    qty = int(trade_amount // entry_price)
                    if qty == 0:
                        print(f"⚠ Nicht genug Kapital für Trade mit {ticker}")
                        continue

                    self.positions[ticker] = {
                        "type": trade_type,
                        "entry_price": entry_price,
                        "entry_time": datetime.now(),
                        "qty": qty,
                        "amount": qty * entry_price,
                        "prob": prob
                    }
                    self.cash -= qty * entry_price
                    print(f"✅ {trade_type} eröffnet: {qty} x {ticker} @ {entry_price:.2f} $ | Cash: {self.cash:.2f} $")
                else:
                    current_price = entry_price
                    outcome = None
                    if position["type"] == "Long":
                        if current_price >= position["entry_price"] * (1 + self.take_profit):
                            outcome = "TP"
                        elif current_price <= position["entry_price"] * (1 - self.stop_loss):
                            outcome = "SL"
                    elif position["type"] == "Short":
                        if current_price <= position["entry_price"] * (1 - self.take_profit):
                            outcome = "TP"
                        elif current_price >= position["entry_price"] * (1 + self.stop_loss):
                            outcome = "SL"

                    if outcome:
                        pnl = (current_price - position["entry_price"]) * position["qty"]
                        if position["type"] == "Short":
                            pnl = -pnl
                        self.cash += position["qty"] * current_price + pnl
                        print(
                            f"💰 {outcome} – {ticker} geschlossen @ {current_price:.2f} $ | PnL: {pnl:.2f} $ | Cash: {self.cash:.2f} $")
                        self.trade_log.append({
                            "Ticker": ticker,
                            "Type": position["type"],
                            "Entry": position["entry_price"],
                            "Exit": current_price,
                            "Qty": position["qty"],
                            "Outcome": outcome,
                            "PnL": pnl,
                            "Prob": position["prob"],
                            "Open Time": position["entry_time"],
                            "Close Time": datetime.now()
                        })
                        del self.positions[ticker]

            print(
                f"\n📊 Zyklus {cycle + 1} abgeschlossen – Portfolio-Wert: {self.cash:.2f} $ | Offene Positionen: {len(self.positions)}")
            cycle += 1
            time.sleep(interval_minutes * 60)

        # ✅ Nach letztem Zyklus: Log & Plot
        print("\n📁 Trading-Zyklus abgeschlossen. Exportiere Log & visualisiere Performance...")
        self.export_trade_log()
        self.visualize_performance()

    def export_trade_log(self, filename="paper_trade_log.csv"):
        if not self.trade_log:
            print("⚠ Kein Trade-Log zum Speichern vorhanden.")
            return
        df = pd.DataFrame(self.trade_log)
        df.to_csv(filename, index=False)
        print(f"📝 Trade-Log gespeichert als: {filename}")

    def visualize_performance(self):
        if not self.trade_log:
            print("⚠ Kein Trade-Log vorhanden – kein Plot möglich.")
            return

        df = pd.DataFrame(self.trade_log)
        df["Time"] = pd.to_datetime(df["Close Time"])
        df = df.sort_values("Time")
        df["Cumulative PnL"] = df["PnL"].cumsum()

        plt.figure(figsize=(10, 5))
        plt.plot(df["Time"], df["Cumulative PnL"], marker='o', linestyle='-', label="Trader Performance")
        plt.axhline(0, color="red", linestyle="--", label="Break-even")

        # Vergleich mit SPY
        start_time = df["Time"].iloc[0].strftime("%Y-%m-%d")
        end_time = df["Time"].iloc[-1].strftime("%Y-%m-%d")

        try:
            client = StockHistoricalDataClient(
                api_key="PKHZI2GPSTTI8XH7FVA8",
                secret_key="fZhOrdjXLDhhggvMaBCBcAQ5wVvJ5OopIIZkZKve"
            )
            request = StockBarsRequest(
                symbol_or_symbols="SPY",
                timeframe=TimeFrame.Hour,
                start=start_time,
                end=end_time
            )
            bars = client.get_stock_bars(request).df
            bars = bars.reset_index(level="symbol", drop=True)
            bars.index = bars.index.tz_convert(None)
            bars = bars[~bars.index.duplicated(keep='first')]
            bars = bars[["close"]].rename(columns={"close": "SPY"})
            bars = bars.loc[bars.index.isin(df["Time"])]

            bars["SPY Performance"] = (bars["SPY"] / bars["SPY"].iloc[0] - 1) * df["Cumulative PnL"].iloc[-1]
            plt.plot(bars.index, bars["SPY Performance"], linestyle='--', label="S&P 500")

        except Exception as e:
            print(f"⚠ SPY-Daten konnten nicht geladen werden: {e}")

        plt.title("Kumulierte Performance vs. S&P 500")
        plt.xlabel("Zeit")
        plt.ylabel("Kumulierter Gewinn / Verlust ($)")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()
