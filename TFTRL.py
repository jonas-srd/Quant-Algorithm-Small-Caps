import torch
torch.set_num_threads(1)
import yfinance as yf
import numpy as np
import pandas as pd
import torch.optim as optim
import joblib
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import ccxt
import requests
import seaborn as sns
import matplotlib.pyplot as plt
import lightning.pytorch as pl
import pdb
import feedparser
import random

from bs4 import BeautifulSoup
from pytorch_forecasting import TimeSeriesDataSet
from pytorch_forecasting.models.temporal_fusion_transformer import TemporalFusionTransformer
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

from transformers import pipeline
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler


# 📌 Klasse für Datenverarbeitung & Feature-Engineering
class DataFetcher:
    def __init__(self, tickers, lookback_period=20):
        self.tickers = tickers
        self.lookback_period = lookback_period
        self.data = {}
        self.finbert = pipeline("sentiment-analysis", model="ProsusAI/finbert")
        self.vader = SentimentIntensityAnalyzer()
        self.fred_api_key = "d0176aa190f9a4db6dbf2ba6de6efc82"
        self.NEWS_ARCHIVE_FILE = "yahoo_news_archive.csv"

    def fetch_stock_data(self, ticker):
        """Fetch historical stock data and compute technical indicators."""
        df = yf.download(ticker, period="10y", interval="1d")

        if df.empty:
            print(f"⚠ Skipping {ticker}: No data available.")
            return None

        # 📌 Fix MultiIndex Issue (Rename & Flatten)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)  # Take first level values (e.g., "Close")

        if ticker in df.columns:  # Case when single ticker data is fetched
            df.rename(columns={ticker: "Close"}, inplace=True)

        if "Close" not in df.columns:
            print(f"⚠ Warning: 'Close' column missing. Available columns: {df.columns}")
            return None

        # 📌 Technical Indicators
        df["SMA_14"] = df["Close"].rolling(window=14).mean()
        df["RSI"] = 100 - (100 / (1 + df["Close"].diff().rolling(window=14).mean() / df["Close"].diff().abs().rolling(
            window=14).mean()))
        df["MACD"] = df["Close"].ewm(span=12, adjust=False).mean() - df["Close"].ewm(span=26, adjust=False).mean()
        df["ATR"] = df["High"].rolling(window=14).max() - df["Low"].rolling(window=14).min()
        df["VWAP"] = (df["Volume"] * (df["High"] + df["Low"] + df["Close"]) / 3).rolling(window=14).sum() / df[
            "Volume"].rolling(window=14).sum()

        # 📌 Ensure EPS is a proper Series and does not become a DataFrame
        df["EPS"] = pd.Series(np.random.uniform(1, 10, len(df)), index=df.index, dtype=float)

        # 📌 Ensure Close is a Series and not a DataFrame
        df["Close"] = df["Close"].astype(float)

        # 📌 Compute KGV safely
        df["KGV"] = df["Close"] / df["EPS"]
        df["KGV"] = df["KGV"].astype(float)  # Ensure KGV stays as float


        df.dropna(inplace=True)  # Drop NaN values
        df.reset_index(inplace=True)  # Reset index after dropping NaNs

        return df


    def fetch_macro_data(self, start_date="2010-01-01"):
        """Holt historische Makrodaten (CPI, Zinsen, Arbeitslosenquote) von der FRED API."""
        fred_series = {
            "CPI": "CPIAUCSL",  # Verbraucherpreisindex (inflationsbereinigt)
            "Zinsen": "FEDFUNDS",  # Federal Funds Rate
            "Arbeitslosenquote": "UNRATE"  # US Arbeitslosenquote
        }

        macro_data = {}

        for key, series_id in fred_series.items():
            url = f"https://api.stlouisfed.org/fred/series/observations"
            params = {
                "series_id": series_id,
                "api_key": self.fred_api_key,
                "file_type": "json",
                "frequency": "m",  # Monatliche Daten
                "observation_start": start_date
            }

            response = requests.get(url, params=params)
            if response.status_code == 200:
                data = response.json()["observations"]
                df = pd.DataFrame(data)
                df["date"] = pd.to_datetime(df["date"])  # Konvertiere Datum in datetime-Format
                df["value"] = df["value"].astype(float)  # Konvertiere Werte in Float
                macro_data[key] = df.set_index("date")["value"]
            else:
                print(f"⚠ Fehler beim Abrufen von {key}: {response.status_code}")
                macro_data[key] = None

        # Kombiniere alle Daten in ein DataFrame
        macro_df = pd.concat(macro_data.values(), axis=1)
        macro_df.columns = macro_data.keys()  # Setze Spaltennamen (CPI, Zinsen, Arbeitslosenquote)

        return macro_df

    def prepare_data(self):
        """Bereitet Daten für TFT vor"""
        for ticker in self.tickers:
            df = self.fetch_stock_data(ticker)
            if df is not None:
                macro_data = self.fetch_macro_data()
                df["Macro"] = macro_data.mean().values[0] if isinstance(macro_data, pd.DataFrame) else macro_data
                df["ticker"] = ticker
                self.data[ticker] = df

        full_df = pd.concat(self.data.values())

        # 📌 Fix: Stelle sicher, dass der Index ein Datum ist
        if "Date" in full_df.columns:
            full_df["Date"] = pd.to_datetime(full_df["Date"])
            full_df = full_df.set_index("Date")  # Setze den Index auf das Datum

        # 📌 Fix: Entferne doppelte Datumswerte
        full_df = full_df[~full_df.index.duplicated(keep="first")]

        # 📌 Fix: Resample auf tägliche Daten
        full_df = full_df.resample("D").ffill()  # Fehlende Tage mit letzten Werten füllen

        # 📌 Fix: Zeitindex für TFT-Modell berechnen
        full_df = full_df.reset_index()
        full_df["time_idx"] = (full_df["Date"] - full_df["Date"].min()).dt.days

        return full_df


# 📌 TFT-Modell Klasse
class StockPredictor_TFT:
    def __init__(self, tickers, lookback_period=20, future_prediction_days=5):
        self.tickers = tickers
        self.lookback_period = lookback_period
        self.future_prediction_days = future_prediction_days
        self.data = {}
        self.model = None

    def prepare_data(self):
        """Bereitet Daten für TFT vor"""
        data_fetcher = DataFetcher(self.tickers)  # ✅ Erstelle ein DataFetcher-Objekt
        df = data_fetcher.prepare_data()  # ✅ Nutze die Methode aus `DataFetcher`
        return df

    def train_tft(self):
        """Trainiert das TFT-Modell mit PyTorch Lightning Trainer"""
        df = self.prepare_data()

        dataset = TimeSeriesDataSet(
            df,
            time_idx="time_idx",
            target="Close",
            group_ids=["ticker"],
            max_encoder_length=30,
            max_prediction_length=self.future_prediction_days,
            time_varying_known_reals=["time_idx"],
            time_varying_unknown_reals=["Close", "SMA_14", "RSI", "MACD", "ATR", "VWAP"]
        )

        train_dataloader = dataset.to_dataloader(
            train=True,
            batch_size=64,
            num_workers=0,  # Fix für MacOS multiprocessing Problem
            persistent_workers=False  # Verhindert Deadlocks
        )

        self.model = TemporalFusionTransformer.from_dataset(
            dataset,
            learning_rate=0.01,
            hidden_size=16,
            attention_head_size=2,
            dropout=0.1
        )

        trainer = pl.Trainer(max_epochs=10, accelerator="mps" if torch.backends.mps.is_available() else "cpu")
        trainer.fit(self.model, train_dataloader)

        # 📌 **Modell mit `torch.save()` statt `joblib` speichern**
        torch.save(self.model.state_dict(), "tft_model.pth")
        print("✅ TFT Modell gespeichert als 'tft_model.pth'")

    def evaluate_model(self):

        # Stelle sicher, dass das Modell geladen wurde
        if self.model is None:
            print("⚠ Kein Modell gefunden. Lade gespeichertes Modell...")
            self.load_model()

        df = self.prepare_data()

        # 📌 Erstelle den TimeSeriesDataSet für Vorhersagen
        dataset = TimeSeriesDataSet(
            df,
            time_idx="time_idx",
            target="Close",
            group_ids=["ticker"],
            max_encoder_length=30,
            max_prediction_length=self.future_prediction_days,
            time_varying_known_reals=["time_idx"],
            time_varying_unknown_reals=["Close", "SMA_14", "RSI", "MACD", "ATR", "VWAP"]
        )

        # Erstelle DataLoader für das Modell
        test_dataloader = dataset.to_dataloader(train=False, batch_size=64, num_workers=0)

        # 📌 Nutze das bereits trainierte Modell für Vorhersagen
        predictions = self.model.predict(test_dataloader)

        # 📌 Falls das Modell mehrere Quantile zurückgibt, wähle das mittlere Quantil (50%)
        if predictions.ndim == 3 and predictions.shape[-1] > 1:
            print(
                f"⚠ Mehrdimensionale Vorhersagen gefunden: {predictions.shape}. Nutze das mittlere Quantil (Index 2).")
            predictions = predictions[:, :, 3]  # Nimm das 50%-Quantil als zentrale Vorhersage

        # 📌 Falls Tensor, auf CPU verschieben und in NumPy-Array umwandeln
        if isinstance(predictions, torch.Tensor):
            predictions = predictions.cpu().numpy()

        # 📌 Trimme `predictions` auf die Länge von `df`
        predictions = predictions[:len(df)]

        # **🔧 KORREKTUR: Wähle nur das mittlere Quantil (`50%`-Quantil, Index 2)**
        y_pred = (predictions[:, 2] > df["Close"].values[:len(predictions)]).astype(int)

        # 📌 Bereite `y_true` vor (Ist der Preis in 5 Tagen gestiegen?)
        y_true = (df["Close"].shift(-self.future_prediction_days) > df["Close"]).astype(int).dropna().values

        # 📌 Falls die Längen nicht übereinstimmen, trimme beide
        min_length = min(len(y_true), len(y_pred))
        y_true = y_true[:min_length]
        y_pred = y_pred[:min_length]

        # **Prüfe nochmals, ob `y_true` und `y_pred` jetzt binär sind**
        print(f"y_true unique values: {np.unique(y_true)}")
        print(f"y_pred unique values: {np.unique(y_pred)}")

        # **Test-Accuracy berechnen**
        print(f"✅ Test-Accuracy: {accuracy_score(y_true, y_pred) * 100:.2f}%")
        print("📊 Classification Report:")
        print(classification_report(y_true, y_pred))

        # 📌 Confusion Matrix Visualisierung
        cm = confusion_matrix(y_true, y_pred)
        plt.figure(figsize=(6, 5))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=["Down", "Up"], yticklabels=["Down", "Up"])
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        plt.title("Confusion Matrix")
        plt.show()

    def load_model(self, force_train):
        """Lädt das gespeicherte Modell oder trainiert ein neues, falls nötig."""
        if force_train or not os.path.exists("tft_model.pth"):
            print("🔄 Training eines neuen Modells wird gestartet...")
            self.train_tft()
        else:
            print("✅ Lade gespeichertes Modell...")

            df = self.prepare_data()

            dataset = TimeSeriesDataSet(
                df,
                time_idx="time_idx",
                target="Close",
                group_ids=["ticker"],
                max_encoder_length=30,
                max_prediction_length=self.future_prediction_days,
                time_varying_known_reals=["time_idx"],
                time_varying_unknown_reals=["Close", "SMA_14", "RSI", "MACD", "ATR", "VWAP"],
                allow_missing_timesteps=True  # Lücken im Zeitindex erlauben
            )

            self.model = TemporalFusionTransformer.from_dataset(dataset)

            try:
                self.model.load_state_dict(
                    torch.load("tft_model.pth",
                               map_location=torch.device("mps" if torch.backends.mps.is_available() else "cpu")),
                    strict=False  # Ignoriere Layer-Mismatch
                )
                print("✅ Modell erfolgreich geladen!")
            except RuntimeError as e:
                print(f"⚠ Fehler beim Laden des Modells: {e}")
                print("🚨 Starte das Training eines neuen Modells...")
                self.train_tft()

    def run(self, force_train):
        """Startet das Modell"""
        self.load_model(force_train=force_train)
        self.evaluate_model()




