import os
import pymongo
import pandas as pd
import numpy as np
import sys

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
import joblib

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

from imblearn.over_sampling import SMOTE

# Dodajemy folder real-time do ścieżki (zakładamy, że katalogi ai_training i real-time są rodzeństwem)
current_dir = os.path.dirname(__file__)
parent_dir = os.path.abspath(os.path.join(current_dir, ".."))
real_time_dir = os.path.join(parent_dir, "real-time")
sys.path.append(real_time_dir)

from value_engine import ValueEngine


################################################################################
# Główna klasa AITraining
################################################################################
class AITraining(ValueEngine):
    def __init__(self):
        super().__init__()
        self.model_path = "trained_model_random_forest.pkl"  # finalny model
        self.final_model = None

    

    def load_data(self):
        """
        Wczytaj dane z kolekcji MongoDB (self.database_parameters_coll).
        """
        data = pd.DataFrame(list(self.database_parameters_coll.find()))
        # Drop niewykorzystywanych pól
        for col in ["_id", "processed", "edited", "datetime", "home_name", "away_name"]:
            if col in data.columns:
                data.drop(columns=[col], errors="ignore", inplace=True)
        return data

    def preprocess_data(self, data):
        """
        1. Usuwamy wiersze bez winner.
        2. winner=0 => 2 => finalnie {1->0, 2->1}.
        3. Tworzymy 'diff_dominance_percentage' = (dominance_home - dominance_away).
        4. (Dodatkowo) Generujemy heatmapę korelacji oraz wykres feature importance dla wszystkich dostępnych cech.
        5. Wybieramy 11 cech.
        6. Usuwamy outliery (z-score>3).
        7. Skalujemy MinMaxScaler.
        8. Dzielimy na X,y.
        """
        # 1. Wiersze bez wartości w kolumnie 'winner'
        data = data[data["winner"].notnull()]
        # 2. winner=0 => 2, następnie: 1->0, 2->1
        data["winner"] = data["winner"].replace(0, 2)
        data["winner"] = data["winner"] - 1  # teraz etykiety: {0,1}

        # Rozbicie trends_home (opcjonalnie, jeśli występuje)
        if "trends_home" in data.columns:
            data["trends_home_100_60"] = data["trends_home"].apply(lambda x: x["home"]["100-60"] if x else 0)
            data["trends_home_60_25"] = data["trends_home"].apply(lambda x: x["home"]["60-25"] if x else 0)
            data["trends_home_25_0"] = data["trends_home"].apply(lambda x: x["home"]["25-0"] if x else 0)
            data["trends_away_100_60"] = data["trends_home"].apply(lambda x: x["away"]["100-60"] if x else 0)
            data["trends_away_60_25"] = data["trends_home"].apply(lambda x: x["away"]["60-25"] if x else 0)
            data["trends_away_25_0"] = data["trends_home"].apply(lambda x: x["away"]["25-0"] if x else 0)
            data.drop(columns=["trends_home"], errors="ignore", inplace=True)

        # Rozbicie diff_walecznosc
        if "diff_walecznosc" in data.columns:
            data["diff_walecznosc_3"] = data["diff_walecznosc"].apply(lambda x: x["3+"] if x else 0)
            data["diff_walecznosc_4"] = data["diff_walecznosc"].apply(lambda x: x["4+"] if x else 0)
            data["diff_walecznosc_5"] = data["diff_walecznosc"].apply(lambda x: x["5+"] if x else 0)
            data["diff_walecznosc_6"] = data["diff_walecznosc"].apply(lambda x: x["6+"] if x else 0)
            data.drop(columns=["diff_walecznosc"], errors="ignore", inplace=True)

        # Wypełnianie braków
        data.fillna(0, inplace=True)
        

        # Obliczamy ważność cech przy użyciu prostego modelu ExtraTreesClassifier
        if "winner" in data.columns:
            X_temp = data.drop(columns=["winner"])
            y_temp = data["winner"]
            temp_model = ExtraTreesClassifier(n_estimators=50, random_state=42)
            temp_model.fit(X_temp, y_temp)
            
        # ---------------------------------------------------------------

        # 3. Tworzymy 'diff_dominance_percentage'
        if "dominance_percentage_home" in data.columns and "dominance_percentage_away" in data.columns:
            data["diff_dominance_percentage"] = data["dominance_percentage_home"] - data["dominance_percentage_away"]
            data.drop(columns=["dominance_percentage_home", "dominance_percentage_away"], inplace=True, errors="ignore")
        else:
            data["diff_dominance_percentage"] = 0

        # 4. Wybieramy 14 cech
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

        # 5. Usuwamy outliery (z-score>3)
        numeric_cols = [c for c in existing_cols if c != "winner"]
        z_scores = np.abs((data[numeric_cols] - data[numeric_cols].mean()) / data[numeric_cols].std(ddof=0))
        mask = (z_scores < 3).all(axis=1)
        data = data[mask]

        # 6. Skalujemy cechy
        scaler = MinMaxScaler()
        data[numeric_cols] = scaler.fit_transform(data[numeric_cols])

        # 7. Podział na cechy (X) i etykiety (y)
        X = data.drop(columns=["winner"])
        y = data["winner"]
        return X, y

    ############################################################################
    # Grid Search dla Random Forest (lub inna metoda optymalizacji)
    ############################################################################
    def grid_search_random_forest(self, X_train, y_train):
        rf_params = {
            'n_estimators': [200],
            'max_depth': [8],
            'min_samples_leaf': [4],
            'min_samples_split': [2]
        }
        rf_grid = GridSearchCV(
            RandomForestClassifier(random_state=42),
            rf_params,
            cv=3,
            scoring='accuracy',
            n_jobs=-1
        )
        rf_grid.fit(X_train, y_train)
        best_rf = rf_grid.best_estimator_

        print("\n=== GridSearch RESULTS (Random Forest) ===")
        print(f"  Best Params: {rf_grid.best_params_}")
        print(f"  Accuracy on train: {accuracy_score(y_train, best_rf.predict(X_train)):.4f}")

        return best_rf

    ############################################################################
    # coverage vs accuracy dla różnych thresholdów
    ############################################################################
    def evaluate_with_threshold(self, X, y, model):
        p = model.predict_proba(X)[:, 1]
        thresholds = np.arange(0.40, 0.81, 0.01)

        print("\nCoverage vs. Accuracy (threshold w [0.40..0.80]):")
        for t in thresholds:
            mask = (p >= t)
            coverage = mask.mean()
            if coverage == 0:
                print(f" Threshold={t:.2f} => 0% coverage")
                continue
            y_pred_local = (p[mask] >= t).astype(int)
            acc_local = accuracy_score(y[mask], y_pred_local)
            print(f" Threshold={t:.2f} => coverage={coverage*100:.2f}%, accuracy={acc_local*100:.2f}%")

    ############################################################################
    # Główny workflow
    ############################################################################
    def run_training(self):
        # 1. Wczytanie danych
        data = self.load_data()

        # 2. Preprocessing danych
        X, y = self.preprocess_data(data)

        # 3. Podział na zbiór treningowy i testowy
        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=0.2,
            random_state=42
        )

        # 4. Grid Search dla Random Forest (lub inna metoda optymalizacji)
        self.final_model = self.grid_search_random_forest(X_train, y_train)

        # 5. Ocena accuracy na zbiorze testowym
        y_pred_test = self.final_model.predict(X_test)
        test_acc = accuracy_score(y_test, y_pred_test)
        print(f"\nTest Accuracy (próg=0.5): {test_acc*100:.2f}%")

        # 6. coverage vs accuracy
        self.evaluate_with_threshold(X_test, y_test, self.final_model)

        # 7. Zapis finalnego modelu do pliku
        joblib.dump(self.final_model, self.model_path)
        print(f"\nFinal model saved to {self.model_path}")


################################################################################
# Uruchomienie
################################################################################
if __name__ == "__main__":
    trainer = AITraining()
    trainer.run_training()
