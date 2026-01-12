import os
import sys
import pymongo
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split, ParameterGrid, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.impute import SimpleImputer
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import joblib

# Dodajemy folder real-time do ścieżki (zakładamy, że katalogi ai_training i real-time są rodzeństwem)
current_dir = os.path.dirname(__file__)
parent_dir = os.path.abspath(os.path.join(current_dir, ".."))
real_time_dir = os.path.join(parent_dir, "real-time")
sys.path.append(real_time_dir)

from value_engine import ValueEngine


def custom_grid_search(pipeline, param_grid, X, y, cv=5, scoring="accuracy"):
    """
    Przeszukuje siatkę hiperparametrów (ParameterGrid) dla podanego pipeline’u.
    Dla każdej kombinacji oblicza cross validation score (średnia i odchylenie standardowe)
    i wypisuje wynik w czasie rzeczywistym.
    Zwraca najlepsze parametry, najlepszy wynik oraz wytrenowany model.
    """
    best_score = -np.inf
    best_params = None
    best_estimator = None
    grid = list(ParameterGrid(param_grid))
    print(f"Rozpoczynam przeszukiwanie {len(grid)} kombinacji parametrów...")
    for i, params in enumerate(grid, 1):
        pipeline.set_params(**params)
        scores = cross_val_score(pipeline, X, y, cv=cv, scoring=scoring)
        mean_score = np.mean(scores)
        std_score = np.std(scores)
        print(f"[{i}/{len(grid)}] Parametry: {params} -> CV średnia: {mean_score:.4f} (std: {std_score:.4f})")
        if mean_score > best_score:
            best_score = mean_score
            best_params = params.copy()
            best_estimator = pipeline.fit(X, y)
    print("Najlepsze parametry:", best_params)
    print("Najlepszy CV wynik:", best_score)
    return best_params, best_score, best_estimator


class AITrainingBinary(ValueEngine):
    def __init__(self):
        super().__init__()
        # Finalny pipeline (scaler + model) zostanie zapisany do pliku – możesz zmieniać nazwę w zależności od modelu.
        self.model_output_dir = os.getcwd()

    def load_data(self):
        """
        Ładuje dane z kolekcji MongoDB (self.database_parameters_coll) 
        oraz usuwa zbędne pola (np. nazwy drużyn, daty).
        """
        print("Ładowanie danych z MongoDB...")
        data = pd.DataFrame(list(self.database_parameters_coll.find()))
        for col in ["_id", "processed", "edited", "datetime", "home_name", "away_name"]:
            if col in data.columns:
                data.drop(columns=[col], errors="ignore", inplace=True)
        print("Dane załadowane. Liczba rekordów:", len(data))
        return data

    def preprocess_data(self, data):
        """
        Przetwarza dane do treningu klasyfikatora binarnego.
        
        1. Usuwamy rekordy bez wartości w kolumnie 'winner'.
        2. Transformujemy etykiety: zamieniamy 0 na 2, a potem odejmujemy 1.
           Rezultat: 1 – wygrana gospodarzy (home), 0 – wygrana gości (away).
        3. Wypełniamy braki (fillna(0)).
        4. (Opcjonalnie) Rozbijamy pola, jeśli występują.
        5. Tworzymy cechę 'diff_dominance_percentage' (jeśli dostępna).
        6. Wybieramy zestaw 11 cech.
        7. Usuwamy outliery – usuwamy wiersze, w których dla którejkolwiek cechy z-score ≥ 3.
        8. Dzielimy dane na cechy (X) i target (y).
        """
        print("Przetwarzanie danych...")
        data = data[data["winner"].notnull()]
        # Transformacja: jeśli winner == 0 (np. oznaczające wygraną gości) to zmieniamy na 2, a potem odejmujemy 1.
        # Wynik: 1 oznacza wygraną gospodarzy, 0 – wygraną gości.
        data["winner"] = data["winner"].replace(0, 2)
        data["winner"] = data["winner"] - 1

        # Opcjonalne rozbicie pól (np. trends_home, diff_walecznosc) – przykładowo:
        if "trends_home" in data.columns:
            data["trends_home_100_60"] = data["trends_home"].apply(lambda x: x["home"]["100-60"] if x else 0)
            data["trends_home_60_25"] = data["trends_home"].apply(lambda x: x["home"]["60-25"] if x else 0)
            data["trends_home_25_0"] = data["trends_home"].apply(lambda x: x["home"]["25-0"] if x else 0)
            data["trends_away_100_60"] = data["trends_home"].apply(lambda x: x["away"]["100-60"] if x else 0)
            data["trends_away_60_25"] = data["trends_home"].apply(lambda x: x["away"]["60-25"] if x else 0)
            data["trends_away_25_0"] = data["trends_home"].apply(lambda x: x["away"]["25-0"] if x else 0)
            data.drop(columns=["trends_home"], errors="ignore", inplace=True)
        
        if "diff_walecznosc" in data.columns:
            data["diff_walecznosc_3"] = data["diff_walecznosc"].apply(lambda x: x["3+"] if x else 0)
            data["diff_walecznosc_4"] = data["diff_walecznosc"].apply(lambda x: x["4+"] if x else 0)
            data["diff_walecznosc_5"] = data["diff_walecznosc"].apply(lambda x: x["5+"] if x else 0)
            data["diff_walecznosc_6"] = data["diff_walecznosc"].apply(lambda x: x["6+"] if x else 0)
            data.drop(columns=["diff_walecznosc"], errors="ignore", inplace=True)
        
        # Wypełniamy braki
        data.fillna(0, inplace=True)
        
        # Tworzymy cechę 'diff_dominance_percentage'
        if "dominance_percentage_home" in data.columns and "dominance_percentage_away" in data.columns:
            data["diff_dominance_percentage"] = data["dominance_percentage_home"] - data["dominance_percentage_away"]
            data.drop(columns=["dominance_percentage_home", "dominance_percentage_away"], inplace=True, errors="ignore")
        else:
            data["diff_dominance_percentage"] = 0

        # Wybrane cechy – lista 11 cech (dostosuj ją do swoich danych)
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

        # Usuwamy outliery – dla każdej cechy liczymy z-score i usuwamy rekordy, gdzie dowolny z-score >= 3
        numeric_cols = [c for c in existing_cols if c != "winner"]
        z_scores = np.abs((data[numeric_cols] - data[numeric_cols].mean()) / data[numeric_cols].std(ddof=0))
        mask = (z_scores < 3).all(axis=1)
        data = data[mask]

        X = data.drop(columns=["winner"])
        y = data["winner"]
        print("Przetwarzanie zakończone.")
        return X, y

    def evaluate_with_threshold_both(self, X, y, model_pipeline):
        """
        Ocena coverage vs. accuracy.
        Dla różnych progów prawdopodobieństwa (od 0.40 do 0.80) oblicza coverage oraz dokładność.
        """
        thresholds = np.arange(0.40, 0.81, 0.01)
        proba = model_pipeline.predict_proba(X)
        p1 = proba[:, 1]  # prawdopodobieństwo klasy 1 (wygrana gospodarzy)
        p0 = proba[:, 0]  # prawdopodobieństwo klasy 0 (wygrana gości)

        print("\nOcena dla klasy 1 (wygrana gospodarzy):")
        for t in thresholds:
            mask = (p1 >= t)
            coverage = mask.mean()
            if coverage == 0:
                print(f" Threshold={t:.2f} => 0% coverage")
                continue
            acc_local = accuracy_score(y[mask], np.ones(np.sum(mask), dtype=int))
            print(f" Threshold={t:.2f} => coverage={coverage*100:.2f}%, accuracy={acc_local*100:.2f}%")

        print("\nOcena dla klasy 0 (wygrana gości):")
        for t in thresholds:
            mask = (p0 >= t)
            coverage = mask.mean()
            if coverage == 0:
                print(f" Threshold={t:.2f} => 0% coverage")
                continue
            acc_local = accuracy_score(y[mask], np.zeros(np.sum(mask), dtype=int))
            print(f" Threshold={t:.2f} => coverage={coverage*100:.2f}%, accuracy={acc_local*100:.2f}%")

    def run_training(self):
        print("Rozpoczynam trenowanie modeli binarnego klasyfikatora (HOME)...")
        data = self.load_data()
        X, y = self.preprocess_data(data)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        ###############################################
        # MODELE – GRID SEARCH
        ###############################################
        print("=== MODELE BINARY (HOME) ===")
        # Model 1: Gradient Boosting
        print("GridSearch dla HOME - Gradient Boosting (real-time)...")
        pipeline_gb = Pipeline([
            ("imputer", SimpleImputer(strategy="mean")),
            ("scaler", MinMaxScaler()),
            ("gb", GradientBoostingClassifier(random_state=42))
        ])
        param_grid_gb = {
            "gb__n_estimators": [90, 100, 110, 125, 160, 200],
            "gb__learning_rate": [0.01, 0.05],
            "gb__max_depth": [7, 8, 9, 10, 12],
            "gb__subsample": [ 0.75, 0.8, 0.85, 0.9],
            "gb__max_features": [None, "sqrt", "log2"],
            "gb__min_samples_split": [2,4, 5],
            "gb__min_samples_leaf": [2, 3],
            "gb__loss": ["log_loss", "exponential"]
        }
        best_gb_params, best_gb_score, best_gb_est = custom_grid_search(
            pipeline_gb, param_grid_gb, X_train, y_train, cv=5, scoring="accuracy")
        print("Test accuracy (HOME GB):", best_gb_est.score(X_test, y_test))
        joblib.dump(best_gb_est, os.path.join(self.model_output_dir, "best_home_gb.pkl"))

        # Model 2: Random Forest
        print("GridSearch dla HOME - Random Forest (real-time)...")
        pipeline_rf = Pipeline([
            ("imputer", SimpleImputer(strategy="mean")),
            ("scaler", MinMaxScaler()),
            ("rf", RandomForestClassifier(random_state=42))
        ])
        param_grid_rf = {
            "rf__n_estimators": [ 100, 150,  200, 250, 300],
            "rf__max_depth": [None, 10, 15, 20, 30,  40],
            "rf__min_samples_split": [2, 3, 5, 7, 10],
            "rf__min_samples_leaf": [1, 2, 3, 5],
            "rf__max_features": ["auto", "sqrt", "log2"],
            "rf__bootstrap": [True, False],
            "rf__class_weight": [None, "balanced", "balanced_subsample"]
        }
        best_rf_params, best_rf_score, best_rf_est = custom_grid_search(
            pipeline_rf, param_grid_rf, X_train, y_train, cv=5, scoring="accuracy")
        print("Test accuracy (HOME RF):", best_rf_est.score(X_test, y_test))
        joblib.dump(best_rf_est, os.path.join(self.model_output_dir, "best_home_rf.pkl"))

        # Model 3: SVC
        print("GridSearch dla HOME - SVC (real-time)...")
        pipeline_svc = Pipeline([
            ("imputer", SimpleImputer(strategy="mean")),
            ("scaler", MinMaxScaler()),
            ("svc", SVC(probability=True, random_state=42))
        ])
        param_grid_svc = {
            "svc__C": [0.01, 0.1, 1, 10, 100, 1000],
            "svc__kernel": ["linear", "rbf", "poly", "sigmoid"],
            "svc__gamma": ["scale", "auto", 0.001, 0.01, 0.1, 1],
            "svc__degree": [2, 3],
            "svc__coef0": [0.0, 0.5, 1.0],
            "svc__shrinking": [True, False],
            "svc__tol": [1e-4, 1e-5]
        }
        best_svc_params, best_svc_score, best_svc_est = custom_grid_search(
            pipeline_svc, param_grid_svc, X_train, y_train, cv=5, scoring="accuracy")
        print("Test accuracy (HOME SVC):", best_svc_est.score(X_test, y_test))
        joblib.dump(best_svc_est, os.path.join(self.model_output_dir, "best_home_svc.pkl"))

        print("Wszystkie modele zostały zapisane w katalogu:", self.model_output_dir)


if __name__ == "__main__":
    trainer = AITrainingBinary()
    trainer.run_training()
