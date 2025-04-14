import time
import pandas as pd
import numpy as np
import alpaca_trade_api as tradeapi
from datetime import datetime
import yagmail


class PaperTrader:
    def __init__(self, predictor, alpaca_key, alpaca_secret, email_address, email_password, paper_url="https://paper-api.alpaca.markets"):
        self.predictor = predictor
        self.api = tradeapi.REST(alpaca_key, alpaca_secret, base_url=paper_url)
        self.cash = 10_000.0
        self.risk_per_trade = 0.2
        self.stop_loss = 0.01
        self.take_profit = 0.04
        self.min_prob_threshold = 0.7
        self.positions = {}
        self.trade_log = []
        self.last_macro_update_day = None
        self.email_address = email_address
        self.email_password = email_password
        self.selected_tickers = None
        self.update_macro_if_needed()

    def update_macro_if_needed(self):
        today = datetime.today().date()
        if self.last_macro_update_day != today:
            print("🔄 Aktualisiere Makrodaten...")
            self.predictor.auto_update_macro()
            self.last_macro_update_day = today

    def fetch_stock_data_cached(self, ticker):
        try:
            df = self.predictor.fetch_stock_data(ticker)
            if df is not None and not df.empty:
                print(f"📥 Daten für {ticker} geladen bis {df.index.get_level_values('timestamp').max()}.")
                return df
            else:
                print(f"⚠ Keine Daten für {ticker} verfügbar.")
        except Exception as e:
            print(f"❌ Fehler beim Laden der Daten für {ticker}: {e}")
        return None

    def run_live_loop(self, interval_minutes=60):
        print(f"🚀 Starte Paper-Trading Loop mit 10.000 $ Kapital")
        cycle = 0
        while True:
            print(f"\n🕒 Zyklus {cycle + 1}")
            self.update_macro_if_needed()

            if self.selected_tickers is None:
                self.selected_tickers = np.random.choice(
                    self.predictor.tickers,
                    size=min(20, len(self.predictor.tickers)),
                    replace=False
                )
                print(f"🎯 Zufällig ausgewählte Aktien: {self.selected_tickers}")

            candidates = []
            for ticker in self.selected_tickers:
                df = self.fetch_stock_data_cached(ticker)
                if df is None or df.empty:
                    continue

                try:
                    macro_df = pd.read_csv("macro_data_hourly.csv", index_col="timestamp", parse_dates=True)
                    df = df.join(macro_df, how="left").dropna()
                except Exception as e:
                    print(f"❌ Makrodatenfehler: {e}")
                    continue

                features = df.iloc[-1][[
                    "Close", "Moving_Avg", "Upper_Band", "Lower_Band", "MACD", "MACD_Signal", "RSI", "ATR",
                    "Bollinger_Width", "ROC", "ADX", "CPI", "Zinsen", "Arbeitslosenquote", "VIX", "Oil_Price"
                ]].values.reshape(1, -1)

                if np.any(np.isnan(features)) or np.any(np.isinf(features)):
                    continue

                p_down, p_up = self.predictor.model.predict_proba(features)[0]
                prob = max(p_down, p_up)
                if prob >= self.min_prob_threshold:
                    candidates.append((ticker, prob, p_up, p_down, df))

            top_candidates = sorted(candidates, key=lambda x: x[1], reverse=True)[:6]

            for ticker, prob, p_up, p_down, df in top_candidates:
                price = df.iloc[-1]["Close"]
                trade_type = "Long" if p_up >= p_down else "Short"
                position = self.positions.get(ticker)

                if position is None:
                    trade_amount = self.cash * self.risk_per_trade
                    qty = int(trade_amount // price)
                    if qty == 0:
                        print(f"⚠ Nicht genug Kapital für {ticker}")
                        continue

                    self.positions[ticker] = {
                        "type": trade_type,
                        "entry_price": price,
                        "entry_time": datetime.now(),
                        "qty": qty,
                        "amount": qty * price,
                        "prob": prob
                    }
                    self.cash -= qty * price
                    print(f"✅ {trade_type} eröffnet: {qty} x {ticker} @ {price:.2f} $ | Cash: {self.cash:.2f} $")

                else:
                    current_price = price
                    exit_trade = False
                    if trade_type == "Long":
                        if current_price >= position["entry_price"] * (1 + self.take_profit):
                            outcome = "TP"
                            exit_trade = True
                        elif current_price <= position["entry_price"] * (1 - self.stop_loss):
                            outcome = "SL"
                            exit_trade = True
                    else:
                        if current_price <= position["entry_price"] * (1 - self.take_profit):
                            outcome = "TP"
                            exit_trade = True
                        elif current_price >= position["entry_price"] * (1 + self.stop_loss):
                            outcome = "SL"
                            exit_trade = True

                    if exit_trade:
                        pnl = (current_price - position["entry_price"]) * position["qty"]
                        if trade_type == "Short":
                            pnl = -pnl
                        self.cash += position["qty"] * current_price + pnl
                        print(f"💰 {outcome} – {ticker} geschlossen @ {current_price:.2f} $ | PnL: {pnl:.2f} $ | Cash: {self.cash:.2f} $")
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

            if datetime.now().hour == 20:
                self.send_daily_report()

            cycle += 1
            time.sleep(interval_minutes * 60)

    def export_trade_log(self, filename="paper_trade_log.csv"):
        if not self.trade_log:
            print("⚠ Kein Trade-Log zum Speichern vorhanden.")
            return
        pd.DataFrame(self.trade_log).to_csv(filename, index=False)
        print(f"📝 Trade-Log gespeichert: {filename}")

    def send_daily_report(self):
        if not self.trade_log:
            print("⚠ Kein Trade-Log zum Versenden vorhanden.")
            return

        df = pd.DataFrame(self.trade_log)
        df["Time"] = pd.to_datetime(df["Close Time"])
        df.sort_values("Time", inplace=True)
        df["Cumulative PnL"] = df["PnL"].cumsum()

        last_day = df["Time"].dt.date.max()
        summary = df[df["Time"].dt.date == last_day]

        message = f"Daily Report – {last_day}\n"
        message += f"Trades: {len(summary)}\n"
        message += f"Tages-PnL: {summary['PnL'].sum():.2f} $\n"
        message += f"Gesamt-PnL: {df['Cumulative PnL'].iloc[-1]:.2f} $\n"

        filename = f"daily_report_{last_day}.csv"
        summary.to_csv(filename, index=False)

        yag = yagmail.SMTP(self.email_address, self.email_password)
        yag.send(to=self.email_address, subject=f"Trading Report – {last_day}", contents=message, attachments=filename)
        print(f"📧 Report gesendet an {self.email_address}")
