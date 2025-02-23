import yfinance as yf
import numpy as np
import xgboost as xgb
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
import pdb
import joblib
import os


class StockPredictor:
    def __init__(self, tickers, lookback_period=20, std_dev_factor=2, future_prediction_days=5):
        self.tickers = tickers
        self.lookback_period = lookback_period
        self.std_dev_factor = std_dev_factor
        self.future_prediction_days = future_prediction_days
        self.data = {}
        self.model = None

    def fetch_stock_data(self, ticker):
        try:
            df = yf.download(ticker, period="1y", interval="1d")

            if df.empty:
                print(f"⚠ Skipping {ticker}: No data available.")
                return None

            # Berechnung technischer Indikatoren
            df["Moving_Avg"] = df["Close"].rolling(window=self.lookback_period).mean()
            df["Std_Dev"] = df["Close"].rolling(window=self.lookback_period).std()
            df["Upper_Band"] = df["Moving_Avg"] + (self.std_dev_factor * df["Std_Dev"])
            df["Lower_Band"] = df["Moving_Avg"] - (self.std_dev_factor * df["Std_Dev"])
            df["ATR"] = df["High"].rolling(window=14).max() - df["Low"].rolling(window=14).min()
            df["MACD"] = df["Close"].ewm(span=12, adjust=False).mean() - df["Close"].ewm(span=26, adjust=False).mean()
            df["MACD_Signal"] = df["MACD"].ewm(span=9, adjust=False).mean()
            df["Bollinger_Width"] = (df["Upper_Band"] - df["Lower_Band"]) / df["Moving_Avg"]
            df["ROC"] = ((df["Close"] - df["Close"].shift(10)) / df["Close"].shift(10)) * 100
            df["ADX"] = df["ATR"].rolling(window=14).mean()

            # RSI berechnen
            delta = df["Close"].diff()
            gain = delta.where(delta > 0, 0)
            loss = -delta.where(delta < 0, 0)
            avg_gain = gain.rolling(window=14, min_periods=1).mean()
            avg_loss = loss.rolling(window=14, min_periods=1).mean()
            rs = avg_gain / avg_loss
            df["RSI"] = 100 - (100 / (1 + rs))

            df.replace([np.inf, -np.inf], np.nan, inplace=True)
            df.dropna(inplace=True)
            return df
        except Exception as e:
            print(f"❌ Error fetching data for {ticker}: {e}")
            return None

    def prepare_data(self):
        self.data = {ticker: self.fetch_stock_data(ticker) for ticker in self.tickers if
                     self.fetch_stock_data(ticker) is not None}
        X, y = [], []
        for ticker, df in self.data.items():
            for i in range(self.lookback_period, len(df) - self.future_prediction_days):
                features = df.iloc[i][["Close", "Moving_Avg", "Upper_Band", "Lower_Band",
                                       "MACD", "MACD_Signal", "RSI", "ATR",
                                       "Bollinger_Width", "ROC", "ADX"]].values
                if np.any(np.isnan(features)) or np.any(np.isinf(features)):
                    continue
                X.append(features)
                y.append(1 if df["Close"].iloc[i + self.future_prediction_days].item() > df["Close"].iloc[i].item() else 0)
        return np.array(X), np.array(y)

    def train_model(self):
        X, y = self.prepare_data()
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        self.X_test = X_test  # Store for later evaluation
        self.y_test = y_test  # Store for later evaluation
        param_grid = {
            "n_estimators": [1000],
            "max_depth": [9],
            "learning_rate": [0.1],
            "subsample": [1.0],
            "colsample_bytree": [0.75],
            "gamma": [0],
            "min_child_weight": [1],
            "reg_lambda": [1],
            "reg_alpha": [0]
        }

        grid_search = GridSearchCV(xgb.XGBClassifier(random_state=42), param_grid, cv=5, scoring="accuracy", n_jobs=-1,
                                   verbose=2)
        print("🔍 Starte Grid Search...")
        grid_search.fit(X_train, y_train)
        self.model = grid_search.best_estimator_
        print(f"🎯 Beste Hyperparameter: {grid_search.best_params_}")

        # Cross-Validation Score Berechnung
        cv_scores = cross_val_score(self.model, X_train, y_train, cv=5, scoring="accuracy")
        print(f"✅ Cross-Validation Scores: {cv_scores}")
        print(f"✅ Durchschnittliche Cross-Validation-Genauigkeit: {np.mean(cv_scores) * 100:.2f}%")

        # ✅ Modell speichern
        joblib.dump(self.model, "trained_model.pkl")
        print("✅ Modell gespeichert als 'trained_model.pkl'")

    def evaluate_model(self):
        y_pred = self.model.predict(self.X_test)  # Use stored test set
        print(f"✅ Test-Accuracy: {accuracy_score(self.y_test, y_pred) * 100:.2f}%")
        print(f"F1 Score: {f1_score(self.y_test, y_pred):.4f}")
        print("Classification Report:")
        print(classification_report(self.y_test, y_pred))

        # Confusion Matrix Visualization
        cm = confusion_matrix(self.y_test, y_pred)
        plt.figure(figsize=(6,5))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=["Down", "Up"], yticklabels=["Down", "Up"])
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        plt.title("Confusion Matrix")
        plt.show()

        # Feature Importance Plot
        feature_importance = self.model.feature_importances_
        plt.figure(figsize=(8,6))
        plt.barh(["Close", "Moving_Avg", "Upper_Band", "Lower_Band", "MACD", "MACD_Signal", "RSI", "ATR", "Bollinger_Width", "ROC", "ADX"], feature_importance)
        plt.xlabel("Feature Importance")
        plt.ylabel("Features")
        plt.title("Feature Importance Plot")
        plt.show()

    def load_model(self, force_train=False):
        if force_train or not os.path.exists("trained_model.pkl"):
            print("🔄 Training eines neuen Modells wird gestartet...")
            self.train_model()
        else:
            print("✅ Lade gespeichertes Modell...")
            self.model = joblib.load("trained_model.pkl")
            print("✅ Modell geladen!")

    def run(self, force_train=False):
        self.load_model(force_train=force_train)
        self.evaluate_model()
