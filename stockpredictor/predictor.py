import numpy as np
import pandas as pd
import requests
import os
import joblib
import yfinance as yf
import xgboost as xgb

from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame


class StockPredictor:
    def __init__(self, tickers, lookback_period=10, std_dev_factor=2, future_prediction_days=2, start_date=None, end_date=None):
        self.tickers = tickers
        self.lookback_period = lookback_period
        self.std_dev_factor = std_dev_factor
        self.future_prediction_days = future_prediction_days
        self.start_date = start_date
        self.end_date = end_date
        self.data = {}
        self.model = None
        self.fred_api_key = os.getenv("FRED_API_KEY", "")
        self.alpaca_client = StockHistoricalDataClient(
            api_key=os.getenv("ALPACA_API_KEY"),
            secret_key=os.getenv("ALPACA_SECRET_KEY")
        )

    def fetch_macro_data(self, start_date=None, end_date=None):
        start_date = start_date or self.start_date
        end_date = end_date or self.end_date
        fred_series = {"CPI": "CPIAUCSL", "Zinsen": "FEDFUNDS", "Arbeitslosenquote": "UNRATE"}
        macro_data = {}

        for key, series_id in fred_series.items():
            url = f"https://api.stlouisfed.org/fred/series/observations"
            params = {
                "series_id": series_id,
                "api_key": self.fred_api_key,
                "file_type": "json",
                "frequency": "m",
                "observation_start": start_date
            }
            if end_date:
                params["observation_end"] = end_date

            response = requests.get(url, params=params)
            if response.status_code == 200:
                data = response.json()["observations"]
                df = pd.DataFrame(data)
                df["date"] = pd.to_datetime(df["date"])
                df["value"] = pd.to_numeric(df["value"], errors="coerce")
                macro_data[key] = df.set_index("date")["value"]

        try:
            vix = yf.download("^VIX", start=start_date, end=end_date, interval="1d")["Close"]
            vix.name = "VIX"
            macro_data["VIX"] = vix
        except Exception:
            pass

        try:
            oil = yf.download("CL=F", start=start_date, end=end_date, interval="1d")["Close"]
            oil.name = "Oil_Price"
            macro_data["Oil_Price"] = oil
        except Exception:
            pass

        macro_df = pd.concat(macro_data.values(), axis=1)
        macro_df.columns = macro_data.keys()
        macro_df.ffill(inplace=True)
        macro_df.index = pd.to_datetime(macro_df.index)
        macro_df = macro_df[~macro_df.index.duplicated(keep='first')]
        macro_df = macro_df.resample("h").ffill()
        macro_df.index.name = "timestamp"
        macro_df.to_csv("data/macro_data_hourly.csv")
        return macro_df

    def fetch_stock_data(self, ticker):
        try:
            request = StockBarsRequest(
                symbol_or_symbols=ticker,
                timeframe=TimeFrame.Hour,
                start=self.start_date,
                end=self.end_date
            )
            bars = self.alpaca_client.get_stock_bars(request).df
            if bars.empty:
                return None

            df = bars.reset_index()
            df = df[df["symbol"] == ticker].drop(columns="symbol").copy()
            df.set_index("timestamp", inplace=True)
            df.index = df.index.tz_convert(None)

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
            return df

        except Exception as e:
            print(f"Error fetching {ticker}: {e}")
            return None

    def prepare_data(self):
        X, y, self.data = [], [], {}
        macro_df = self.fetch_macro_data(start_date=self.start_date, end_date=self.end_date)

        for ticker in self.tickers:
            df = self.fetch_stock_data(ticker)
            if df is not None:
                df = df.join(macro_df, how="left")
                df.dropna(inplace=True)
                self.data[ticker] = df

                for i in range(self.lookback_period, len(df) - self.future_prediction_days):
                    features = df.iloc[i][[
                        "Close", "Moving_Avg", "Upper_Band", "Lower_Band",
                        "MACD", "MACD_Signal", "RSI", "ATR",
                        "Bollinger_Width", "ROC", "ADX",
                        "CPI", "Zinsen", "Arbeitslosenquote", "VIX", "Oil_Price"
                    ]].values

                    if not np.any(np.isnan(features)) and not np.any(np.isinf(features)):
                        X.append(features)
                        y.append(int(df["Close"].iloc[i + self.future_prediction_days] > df["Close"].iloc[i]))

        return np.array(X), np.array(y)

    def train_model(self):
        X, y = self.prepare_data()
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        self.X_test, self.y_test = X_test, y_test
        grid = GridSearchCV(xgb.XGBClassifier(), {
            "n_estimators": [100], "max_depth": [6], "learning_rate": [0.1]
        }, cv=5, scoring="accuracy")

        grid.fit(X_train, y_train)
        self.model = grid.best_estimator_
        print(f"Best Params: {grid.best_params_}")
        print(f"Train Accuracy: {grid.best_score_:.2f}")

        joblib.dump((self.model, self.data, self.X_test, self.y_test), "data/trained_model.pkl")

    def load_model(self, force_train=False):
        if force_train or not os.path.exists("data/trained_model.pkl"):
            self.train_model()
        else:
            self.model, self.data, self.X_test, self.y_test = joblib.load("data/trained_model.pkl")

    def evaluate_model(self):
        y_pred = self.model.predict(self.X_test)
        print("Accuracy:", accuracy_score(self.y_test, y_pred))
        print("F1 Score:", f1_score(self.y_test, y_pred))
        print(confusion_matrix(self.y_test, y_pred))
        print(classification_report(self.y_test, y_pred))

    def run(self, force_train=False):
        self.load_model(force_train=force_train)
        self.evaluate_model()

    def predict(self, X):
        if self.model is None:
            raise ValueError("Model not trained")
        return self.model.predict(X)

    def predict_proba(self, X):
        if self.model is None:
            raise ValueError("Model not trained")
        return self.model.predict_proba(X)