import os
import pymongo
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVC
import joblib

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

from imblearn.over_sampling import SMOTE

from value_engine import ValueEngine


################################################################################
# Główna klasa AITraining (dla SVC w Pipeline)
################################################################################
class AITraining(ValueEngine):
    def __init__(self):
        super().__init__()
        self.model_path = "trained_model_svc.pkl"  # finalny pipeline (scaler + SVC)
        self.final_model = None

    def load_data(self):
        """
        Wczytuje dane z kolekcji MongoDB (self.database_parameters_coll).
        """
        data = pd.DataFrame(list(self.database_parameters_coll.find()))
        # Usuwamy niewykorzystywane pola
        for col in ["_id", "processed", "edited", "datetime", "home_name", "away_name"]:
            if col in data.columns:
                data.drop(columns=[col], errors="ignore", inplace=True)
        return data

    def preprocess_data(self, data):
        """
        Przygotowanie danych (bez ręcznego skalowania!):
         1. Usuwamy wiersze bez winner.
         2. Przekształcamy etykiety: winner=0 => 2, następnie: 1->0, 2->1.
         3. Tworzymy 'diff_dominance_percentage' = (dominance_percentage_home - dominance_percentage_away).
         4. (Opcjonalnie) Rozbijamy pola typu trends_home, diff_walecznosc.
         5. Wybieramy 11 cech.
         6. Usuwamy outliery (z-score > 3).
         7. Bez skalowania (zrobi to pipeline).
         8. Dzielimy dane na X i y.
        """
        # 1. Usuwamy wiersze bez wartości w kolumnie 'winner'
        data = data[data["winner"].notnull()]
        
        # 2. Przekształcamy etykiety: zamieniamy 0 -> 2, potem -1 => (1->0, 2->1)
        data["winner"] = data["winner"].replace(0, 2)
        data["winner"] = data["winner"] - 1

        # Rozbicie trends_home (opcjonalnie)
        if "trends_home" in data.columns:
            data["trends_home_100_60"] = data["trends_home"].apply(lambda x: x["home"]["100-60"] if x else 0)
            data["trends_home_60_25"] = data["trends_home"].apply(lambda x: x["home"]["60-25"] if x else 0)
            data["trends_home_25_0"] = data["trends_home"].apply(lambda x: x["home"]["25-0"] if x else 0)
            data["trends_away_100_60"] = data["trends_home"].apply(lambda x: x["away"]["100-60"] if x else 0)
            data["trends_away_60_25"] = data["trends_home"].apply(lambda x: x["away"]["60-25"] if x else 0)
            data["trends_away_25_0"] = data["trends_home"].apply(lambda x: x["away"]["25-0"] if x else 0)
            data.drop(columns=["trends_home"], errors="ignore", inplace=True)

        # Rozbicie diff_walecznosc (opcjonalnie)
        if "diff_walecznosc" in data.columns:
            data["diff_walecznosc_3"] = data["diff_walecznosc"].apply(lambda x: x["3+"] if x else 0)
            data["diff_walecznosc_4"] = data["diff_walecznosc"].apply(lambda x: x["4+"] if x else 0)
            data["diff_walecznosc_5"] = data["diff_walecznosc"].apply(lambda x: x["5+"] if x else 0)
            data["diff_walecznosc_6"] = data["diff_walecznosc"].apply(lambda x: x["6+"] if x else 0)
            data.drop(columns=["diff_walecznosc"], errors="ignore", inplace=True)

        # Wypełnianie braków
        data.fillna(0, inplace=True)

        # 3. Tworzymy 'diff_dominance_percentage'
        if "dominance_percentage_home" in data.columns and "dominance_percentage_away" in data.columns:
            data["diff_dominance_percentage"] = data["dominance_percentage_home"] - data["dominance_percentage_away"]
            data.drop(columns=["dominance_percentage_home", "dominance_percentage_away"], inplace=True, errors="ignore")
        else:
            data["diff_dominance_percentage"] = 0

        # 4. Wybieramy 11 cech
        selected_features = [
            'elo_diff_scaled',
            'diff_last_tournaments',
            'diff_zmeczenie',
            'h2h_home_scaled',
            'diff_h2h',
            'diff_players_vs_opponents_50',
            'diff_dominance_percentage',
            'diff_closest_trends',
            'diff_saved',
            'diff_wasted',
            'diff_walecznosc_3'
        ]
        features_and_target = selected_features + ["winner"]
        existing_cols = [c for c in features_and_target if c in data.columns]
        data = data[existing_cols]

        # 5. Usuwamy outliery (z-score > 3)
        numeric_cols = [c for c in existing_cols if c != "winner"]
        z_scores = np.abs((data[numeric_cols] - data[numeric_cols].mean()) / data[numeric_cols].std(ddof=0))
        mask = (z_scores < 3).all(axis=1)
        data = data[mask]

        # 6. Nie skalujemy cech tutaj (zrobi to pipeline!)

        # 7. Podział na X i y
        X = data.drop(columns=["winner"])
        y = data["winner"]
        return X, y

    def evaluate_with_threshold_both(self, X, y, pipeline):
        """
        Ocena coverage vs. accuracy dla obu klas (0 i 1), 
        z użyciem pipeline (skalowanie + SVC).
        """
        thresholds = np.arange(0.40, 0.81, 0.01)
        # Predykcja prawdopodobieństw (pipeline sam skaluje w środku)
        proba = pipeline.predict_proba(X)
        p1 = proba[:, 1]  # p(klasa=1)
        p0 = proba[:, 0]  # p(klasa=0)

        print("\nOcena dla klasy 1 (p1 >= threshold):")
        for t in thresholds:
            mask = (p1 >= t)
            coverage = mask.mean()
            if coverage == 0:
                print(f" Threshold={t:.2f} => 0% coverage")
                continue
            acc_local = accuracy_score(y[mask], np.ones(np.sum(mask), dtype=int))
            print(f" Threshold={t:.2f} => coverage={coverage*100:.2f}%, accuracy={acc_local*100:.2f}%")

        print("\nOcena dla klasy 0 (p0 >= threshold):")
        for t in thresholds:
            mask = (p0 >= t)
            coverage = mask.mean()
            if coverage == 0:
                print(f" Threshold={t:.2f} => 0% coverage")
                continue
            acc_local = accuracy_score(y[mask], np.zeros(np.sum(mask), dtype=int))
            print(f" Threshold={t:.2f} => coverage={coverage*100:.2f}%, accuracy={acc_local*100:.2f}%")

    def run_training(self):
        # 1. Wczytanie danych
        data = self.load_data()

        # 2. Preprocessing danych (bez skalowania)
        X, y = self.preprocess_data(data)

        # 3. Podział na zbiór treningowy i testowy
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # 4. Definiujemy pipeline: scaler + SVC(probability=True)
        pipeline = Pipeline([
            ("scaler", MinMaxScaler()),
            ("svc", SVC(probability=True, random_state=42, kernel='rbf', C=1.0))
        ])

        # 5. Trenujemy pipeline
        pipeline.fit(X_train, y_train)

        # 6. Ocena na zbiorze treningowym
        y_pred_train = pipeline.predict(X_train)
        train_acc = accuracy_score(y_train, y_pred_train)
        print(f"\nTrain Accuracy: {train_acc*100:.2f}%")

        # Wyświetlenie parametrów SVC:
        print("\nWybrane parametry SVC:")
        print(pipeline.named_steps['svc'].get_params())

        # 7. Ocena na zbiorze testowym (próg=0.5)
        y_pred_test = pipeline.predict(X_test)
        test_acc = accuracy_score(y_test, y_pred_test)
        print(f"\nTest Accuracy (próg=0.5): {test_acc*100:.2f}%")

        # 8. Ocena coverage vs. accuracy dla obu klas
        self.evaluate_with_threshold_both(X_test, y_test, pipeline)

        # 9. Zapisujemy cały pipeline (scaler + SVC) do pliku
        joblib.dump(pipeline, self.model_path)
        print(f"\nFinal pipeline (scaler+SVC) saved to {self.model_path}")


################################################################################
# Uruchomienie
################################################################################
if __name__ == "__main__":
    trainer = AITraining()
    trainer.run_training()
