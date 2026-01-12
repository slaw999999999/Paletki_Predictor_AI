import pymongo
import pandas as pd
from itertools import combinations
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import (
    VotingClassifier, GradientBoostingClassifier, StackingClassifier,
    AdaBoostClassifier, ExtraTreesClassifier, RandomForestClassifier
)
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import MinMaxScaler
import joblib
from value_engine import ValueEngine

class AITraining(ValueEngine):
    def __init__(self):
        super().__init__()
        self.model_path = "trained_model_ensemble.joblib"

    def load_data(self):
        data = pd.DataFrame(list(self.database_parameters_coll.find()))
        for col in ["_id", "processed", "edited", "datetime", "home_name", "away_name"]:
            if col in data.columns:
                data = data.drop(columns=[col], errors="ignore")
        return data

    def preprocess_data(self, data):
        if "winner" not in data.columns:
            raise KeyError("The 'winner' column is missing from the data. Ensure the input data includes this column.")

        missing_winner_count = data["winner"].isnull().sum()
        total_records = len(data)
        print(f"Missing 'winner' values: {missing_winner_count}/{total_records} "
              f"({(missing_winner_count / total_records) * 100:.2f}%)")

        data = data[data["winner"].notnull()]
        data["winner"] = data["winner"].replace(0, 2)

        if "trends_home" in data.columns:
            data["trends_home_100_60"] = data["trends_home"].apply(lambda x: x["home"]["100-60"] if x else 0)
            data["trends_home_60_25"] = data["trends_home"].apply(lambda x: x["home"]["60-25"] if x else 0)
            data["trends_home_25_0"] = data["trends_home"].apply(lambda x: x["home"]["25-0"] if x else 0)
            data["trends_away_100_60"] = data["trends_home"].apply(lambda x: x["away"]["100-60"] if x else 0)
            data["trends_away_60_25"] = data["trends_home"].apply(lambda x: x["away"]["60-25"] if x else 0)
            data["trends_away_25_0"] = data["trends_home"].apply(lambda x: x["away"]["25-0"] if x else 0)
            data = data.drop(columns=["trends_home"], errors="ignore")

        if "diff_walecznosc" in data.columns:
            data["diff_walecznosc_3"] = data["diff_walecznosc"].apply(lambda x: x["3+"] if x else 0)
            data["diff_walecznosc_4"] = data["diff_walecznosc"].apply(lambda x: x["4+"] if x else 0)
            data["diff_walecznosc_5"] = data["diff_walecznosc"].apply(lambda x: x["5+"] if x else 0)
            data["diff_walecznosc_6"] = data["diff_walecznosc"].apply(lambda x: x["6+"] if x else 0)
            data = data.drop(columns=["diff_walecznosc"], errors="ignore")

        data.fillna(0, inplace=True)
        numeric_columns = data.select_dtypes(include=["number"]).columns
        data[numeric_columns] = data[numeric_columns].astype(float)

        selected_features = [
            'elo_diff', 'elo_diff_scaled', 'experience', 'diff_last_tournaments',
            'diff_zmeczenie', 'h2h_home_scaled', 'h2h_away_scaled', 'diff_h2h',
            'diff_players_vs_opponents_50', 'diff_big_wins', 'diff_big_losses',
            'dominance_percentage_home', 'dominance_percentage_away',
            'diff_closest_trends', 'diff_saved', 'diff_wasted',
            'diff_walecznosc_3'
        ]

        features_and_target = selected_features + ["winner"]
        existing_cols = [col for col in features_and_target if col in data.columns]
        data = data[existing_cols]

        print("Columns used for training (including 'winner'):", data.columns.tolist())

        scaler = MinMaxScaler()
        data[existing_cols] = scaler.fit_transform(data[existing_cols])

        if "winner" not in data.columns:
            raise KeyError("Kolumna 'winner' zniknęła po normalizacji, sprawdź dane wejściowe.")
        features = data.drop(columns=["winner"])
        target = data["winner"]

        return features, target

    def perform_grid_search(self, model, param_grid, X_train, y_train):
        grid = GridSearchCV(model, param_grid, cv=3, n_jobs=-1, verbose=1)
        grid.fit(X_train, y_train)
        print(f"Najlepsze parametry dla {model.__class__.__name__}: {grid.best_params_}")
        return grid.best_estimator_

    def print_sorted_ensemble_results(self, ensemble_results):
        if not ensemble_results:
            print("Brak wyników ensemble do wyświetlenia.")
            return []
        sorted_results = sorted(ensemble_results, key=lambda x: x[2], reverse=True)
        print("\nWszystkie wyniki ensemble (posortowane od najwyższej do najniższej dokładności):")
        for name, _, acc in sorted_results:
            print(f"{name}: {acc*100:.2f}%")
        return sorted_results

    def train_model(self, features, target):
        X_train, X_test, y_train, y_test = train_test_split(
            features, target, test_size=0.2, random_state=42
        )

        models = {
            'XGBoost': XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42),
            'LogisticRegression': LogisticRegression(max_iter=1000, random_state=42),
            'GradientBoosting': GradientBoostingClassifier(random_state=42),
            'KNeighbors': KNeighborsClassifier(),
            'AdaBoost': AdaBoostClassifier(random_state=42),
            'RandomForest': RandomForestClassifier(random_state=42),
            'ExtraTrees': ExtraTreesClassifier(random_state=42)
        }

        param_grids = {
            'XGBoost': { 
                'max_depth': [5,6,7,8,9, 10,11,12,13,14, 15],
                'learning_rate': [0.01, 0.05, 0.1, 0.2],
                'n_estimators': [100, 120, 140,150, 175, 200,220, 250, 300],
                'subsample': [0.7, 0.8, 1.0],
                'colsample_bytree': [0.7, 0.8, 1.0]
            },
            'LogisticRegression': { 
                'C': [0.01, 0.1, 1, 10, 100],
                'solver': ['lbfgs', 'liblinear', 'sag', 'saga'],
                'penalty': ['l2']
            },
            'GradientBoosting': { 
                'n_estimators': [100, 120, 140,150, 175, 200,220, 250, 300],
                'learning_rate': [0.01,0.05, 0.1, 0.2],
                'max_depth': [3,4,5,6,7,8,9, 10,11,12,15],
                'subsample': [0.8, 1.0],
                'max_features': ['sqrt', 'log2', None]
            },
            'KNeighbors': { 
                'n_neighbors': [5, 6,7,8,10,12,15,17,18, 20,22,25, 30],
                'weights': ['uniform', 'distance'],
                'metric': ['euclidean', 'manhattan', 'minkowski']
            },
            'AdaBoost': { 
                'n_estimators': [50,80, 90,100,135, 150,170, 200],
                'learning_rate': [0.01,0.05, 0.1, 0.2, 1]
            },
            
            'RandomForest': {
                'n_estimators': [100,130,150,165,180,190, 200,225,240,270,300,400, 500],
                'max_depth': [None, 10,12,15,17,18, 20,22,25,30],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4],
                'bootstrap': [True, False],
                'max_features': ['sqrt', 'log2', None]
            },
            'ExtraTrees': {
                'n_estimators': [100,130,150,165,180,190, 200,225,240,270,300,400, 500],
                'max_depth': [None, 10,12,15,17,18, 20,22,25,30,40,50],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4],
                'bootstrap': [True, False],
                'max_features': ['sqrt', 'log2', None]
            }
        }

        optimized_models = {}
        for name, model in models.items():
            print(f"\nOptymalizacja hiperparametrów dla {name}...")
            best_model = self.perform_grid_search(model, param_grids[name], X_train, y_train)
            optimized_models[name] = best_model

        for name, model in optimized_models.items():
            print(f"\nTrening zoptymalizowanego modelu {name}...")
            model.fit(X_train, y_train)
            preds = model.predict(X_test)
            acc = accuracy_score(y_test, preds)
            print(f"{name} accuracy po optymalizacji: {acc * 100:.2f}%")

        ensemble_results = []

        for ensemble_size in [3, 4, 5, 6]:
            for combo in combinations(list(optimized_models.items()), ensemble_size):
                estimators = list(combo)

                # Hard Voting Ensemble
                ensemble_hard = VotingClassifier(estimators=estimators, voting='hard')
                ensemble_hard.fit(X_train, y_train)
                y_pred = ensemble_hard.predict(X_test)
                acc = accuracy_score(y_test, y_pred)
                print(f"Accuracy dla kombinacji {tuple(name for name,_ in estimators)}: {acc*100:.2f}%")
                ensemble_results.append((f"Hard Voting {tuple(name for name,_ in estimators)}", ensemble_hard, acc))

        self.print_sorted_ensemble_results(ensemble_results)

        best_ensemble = max(ensemble_results, key=lambda x: x[2], default=(None, None, 0))
        if best_ensemble[0]:
            print(f"\nNajlepszy ensemble to: {best_ensemble[0]} z dokładnością {best_ensemble[2] * 100:.2f}%")
            return best_ensemble[1]
        else:
            print("Nie znaleziono najlepszego ensemble.")
            return None

    def save_model(self, model):
        if model:
            joblib.dump(model, self.model_path)
            print(f"\nModel saved to {self.model_path}")

    def run_training(self):
        data = self.load_data()
        features, target = self.preprocess_data(data)
        best_model = self.train_model(features, target)
        self.save_model(best_model)

if __name__ == "__main__":
    trainer = AITraining()
    trainer.run_training()
