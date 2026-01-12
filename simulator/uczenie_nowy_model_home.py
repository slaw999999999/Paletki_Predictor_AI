import os
import pymongo
import pandas as pd
import numpy as np
import sys
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import GradientBoostingClassifier
import joblib

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

from imblearn.over_sampling import SMOTE

# Dodajemy folder real-time do ścieżki (zakładamy, że katalogi ai_training i real-time są rodzeństwem)
current_dir = os.path.dirname(__file__)
parent_dir = os.path.abspath(os.path.join(current_dir, ".."))
#real_time_dir = os.path.join(parent_dir, "real-time")
#sys.path.append(real_time_dir)

from value_engine import ValueEngine


################################################################################
# Główna klasa AITraining (dla Gradient Boosting w Pipeline)
################################################################################
class AITraining(ValueEngine):
    def __init__(self):
        super().__init__()
        self.model_path = "nowy_model_home.pkl"  # finalny pipeline (scaler + model)
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
        Przygotowanie danych bez ręcznego skalowania (bo to zrobi Pipeline):
         1. Usuwamy wiersze bez winner.
         2. Przekształcamy etykiety: winner=0 => 2, następnie: 1->0, 2->1.
         3. Tworzymy 'diff_dominance_percentage' = (dominance_percentage_home - dominance_percentage_away).
         4. (Opcjonalnie) Rozbijamy pola typu trends_home, diff_walecznosc.
         5. Wybieramy 11 cech.
         6. Usuwamy outliery (z-score > 3).
         7. (Bez skalowania!) - Pipeline zrobi to za nas.
         8. Dzielimy dane na X i y.
        """
        # 1. Wiersze bez wartości w kolumnie 'winner'
        data = data[data["winner"].notnull()]

        # 2. Przekształcamy etykiety: zamieniamy 0 na 2, następnie odejmujemy 1 (1->0, 2->1)
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

        data['diff_wasted'] = data['diff_wasted'].apply(lambda x: x[0] - x[1])

        z_scores = np.abs((data[numeric_cols] - data[numeric_cols].mean()) / data[numeric_cols].std(ddof=0))
        mask = (z_scores < 3).all(axis=1)
        data = data[mask]

        # 6. Nie skalujemy tutaj (zrobimy to w pipeline!)
        # 7. Podział na cechy (X) i etykiety (y)
        X = data.drop(columns=["winner"])
        y = data["winner"]
        return X, y

    def evaluate_with_threshold_both(self, X, y, model_pipeline):
        """
        Ocena coverage vs. accuracy dla obu klas, 
        korzystając z modelu w pipeline (skalowanie + GB).
        """
        thresholds = np.arange(0.40, 0.81, 0.01)
        proba = model_pipeline.predict_proba(X)
        p1 = proba[:, 1]  # p(klasa=1)
        p0 = proba[:, 0]  # p(klasa=0)

        print("\nOcena dla klasy 1:")
        for t in thresholds:
            mask = (p1 >= t)
            coverage = mask.mean()
            if coverage == 0:
                print(f" Threshold={t:.2f} => 0% coverage")
                continue
            acc_local = accuracy_score(y[mask], np.ones(np.sum(mask), dtype=int))
            print(f" Threshold={t:.2f} => coverage={coverage*100:.2f}%, accuracy={acc_local*100:.2f}%")

        print("\nOcena dla klasy 0:")
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

        #############################################################
        # Faza 1: Wstępne trenowanie na pełnym zbiorze treningowym
        #############################################################
        prelim_pipeline = Pipeline([
            ("scaler", MinMaxScaler()),
            ("gb", GradientBoostingClassifier(
                random_state=42,
                n_estimators=200,
                learning_rate=0.01,
                subsample=0.75,
                max_features='sqrt',
                loss='log_loss',
                min_samples_leaf=2,
                min_samples_split=2,
                max_depth=7
            ))
        ])

        prelim_pipeline.fit(X_train, y_train)

        # Predykcje oraz prawdopodobieństwa na zbiorze treningowym
        probs_train = prelim_pipeline.predict_proba(X_train)
        # Dla każdej próbki bierzemy maksymalne prawdopodobieństwo (czyli pewność modelu)
        p_max = np.max(probs_train, axis=1)
        pred_train = prelim_pipeline.predict(X_train)

        # 4. Wybór threshold: szukamy najniższego threshold, dla którego accuracy >= 80%
        t_opt = None
        for t in np.arange(0.40, 1.0, 0.01):
            mask = (p_max >= t)
            if np.sum(mask) == 0:
                continue
            acc = accuracy_score(y_train[mask], pred_train[mask])
            if acc >= 0.85:
                t_opt = t
                print(f"Wybrano threshold t_opt={t_opt:.2f} z coverage={(mask.mean()*100):.2f}% i accuracy={acc*100:.2f}%")
                break
        if t_opt is None:
            t_opt = 0.5
            print("Nie znaleziono threshold z accuracy >= 85%. Użyto domyślnego t_opt=0.5")

        # 5. Wybieramy podzbiór treningowy ("perełki") na podstawie t_opt
        mask_opt = p_max >= t_opt
        X_train_optimal = X_train[mask_opt]
        y_train_optimal = y_train[mask_opt]
        print(f"Wybrano {np.sum(mask_opt)} optymalnych próbek treningowych spośród {len(X_train)}.")

        #############################################################
        # Faza 2: Trenowanie finalnego modelu na optymalnym zbiorze
        #############################################################
        final_pipeline = Pipeline([
            ("scaler", MinMaxScaler()),
            ("gb", GradientBoostingClassifier(
                random_state=42,
                n_estimators=200,
                learning_rate=0.01,
                subsample=0.75,
                max_features='sqrt',
                loss='log_loss',
                min_samples_leaf=2,
                min_samples_split=2,
                max_depth=7
            ))
        ])

        final_pipeline.fit(X_train_optimal, y_train_optimal)

        # Ocena na zbiorze treningowym (optymalnym)
        y_pred_train_final = final_pipeline.predict(X_train_optimal)
        train_acc_final = accuracy_score(y_train_optimal, y_pred_train_final)
        print(f"\nTrain Accuracy (optymalne dane): {train_acc_final*100:.2f}%")

        # Wyświetlenie wybranych parametrów modelu Gradient Boosting
        print("\nWybrane parametry Gradient Boosting:")
        print(final_pipeline.named_steps['gb'].get_params())

        # Ocena na zbiorze testowym (próg=0.5)
        y_pred_test = final_pipeline.predict(X_test)
        test_acc = accuracy_score(y_test, y_pred_test)
        print(f"\nTest Accuracy (próg=0.5): {test_acc*100:.2f}%")

        # Ocena coverage vs. accuracy dla obu klas na zbiorze testowym
        self.evaluate_with_threshold_both(X_test, y_test, final_pipeline)

        # Zapis finalnego pipeline do pliku
        joblib.dump(final_pipeline, self.model_path)
        print(f"\nFinal pipeline (scaler+GB) saved to {self.model_path}")


################################################################################
# Uruchomienie
################################################################################
if __name__ == "__main__":
    trainer = AITraining()
    trainer.run_training()
