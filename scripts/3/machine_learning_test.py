from pymongo import MongoClient
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score

# Połączenie z bazą MongoDB
conn = MongoClient('')
db = conn['czech_liga_pro_test']
collection = db['matches']  # Zmień na swoją kolekcję

# Pobranie danych z MongoDB i konwersja do DataFrame
data = list(collection.find({}))
df = pd.DataFrame(data)

# Konwersja kluczowych kolumn na typy numeryczne
df['home_elo'] = pd.to_numeric(df['home_elo'], errors='coerce')
df['away_elo'] = pd.to_numeric(df['away_elo'], errors='coerce')
df['elo_diff'] = pd.to_numeric(df['elo_diff'], errors='coerce')
df['winner'] = pd.to_numeric(df['winner'], errors='coerce')  # 1 - home, 0 - away
df['delta_elo'] = pd.to_numeric(df['delta_elo'], errors='coerce')


def calculate_form(df, player_col, n):
    """
    Oblicza średnią zmianę ELO dla ostatnich n meczów rozegranych przez gracza, 
    uwzględniając mecze zarówno jako home, jak i away.

    Args:
    - df: DataFrame z danymi meczów.
    - player_col: Kolumna zawierająca nazwę gracza (np. 'home_name' lub 'away_name').
    - date_col: Kolumna z datą rozegrania meczu.
    - elo_change_col: Kolumna z wartościami zmian ELO (np. 'delta_elo').
    - n: Liczba ostatnich meczów do uwzględnienia.

    Returns:
    - form: Tablica wartości formy gracza dla każdego meczu.
    """
    form = np.zeros(len(df))  # Inicjalizacja wynikowej kolumny formy
    date_col = 'datetime'
    elo_change_col = 'delta_elo'

    # Iteracja po unikalnych graczach
    unique_players = df[player_col].unique()

    for player in unique_players:
        # Filtracja meczów z udziałem danego gracza jako home lub away
        player_matches = df[(df['home_name'] == player) | (df['away_name'] == player)]
        player_matches = player_matches.sort_values(by=date_col)  # Sortowanie chronologiczne

        for idx, row in player_matches.iterrows():

            # Filtrujemy mecze przed aktualnym meczem
            past_matches = player_matches[player_matches[date_col] < row[date_col]].tail(n)
            if player_col == "away_name":
                past_matches['delta_elo'] *= -1
            # Wyliczamy średnią zmianę ELO
            form_value = past_matches['delta_elo'].mean() if not past_matches.empty else 0

            # Znajdujemy indeks meczu w oryginalnym DataFrame i zapisujemy wynik
            match_idx = df.index[
                (df[date_col] == row[date_col]) & 
                ((df['home_name'] == row["home_name"]) & (df['away_name'] == row["away_name"])) & 
                (df['league_name'] == row["league_name"])
            ]
            if not match_idx.empty:
                form[match_idx[0]] = form_value

    return form

def calculate_form_multiple_ranges(df, player_col, ranges):
    """
    Oblicza średnią zmianę ELO dla ostatnich n meczów rozegranych przez gracza, 
    dla wielu różnych zakresów n, uwzględniając mecze zarówno jako home, jak i away.

    Args:
    - df: DataFrame z danymi meczów.
    - player_col: Kolumna zawierająca nazwę gracza (np. 'home_name' lub 'away_name').
    - ranges: Lista wartości n (np. [20, 50, 100]) określających zakres meczów do analizy.

    Returns:
    - forms: Słownik, gdzie kluczami są wartości n, a wartościami są tablice formy gracza dla każdego meczu.
    """
    forms = {n: np.zeros(len(df)) for n in ranges}  # Inicjalizacja wynikowych kolumn formy
    date_col = 'datetime'
    elo_change_col = 'delta_elo'

    # Iteracja po unikalnych graczach
    unique_players = df[player_col].unique()

    for player in unique_players:
        # Filtracja meczów z udziałem danego gracza jako home lub away
        player_matches = df[(df['home_name'] == player) | (df['away_name'] == player)]
        player_matches = player_matches.sort_values(by=date_col)  # Sortowanie chronologiczne

        # Przygotowanie "bufora" ostatnich meczów
        past_matches_buffer = []

        for idx, row in player_matches.iterrows():
            # Usuwamy z bufora mecze, które są po aktualnym meczu
            past_matches = [
                match for match in past_matches_buffer 
                if match[date_col] < row[date_col]
            ]

            # Dodajemy aktualny mecz do bufora
            delta = row[elo_change_col]
            if row['away_name'] == player:
                delta *= -1
            past_matches.append({
                date_col: row[date_col],
                elo_change_col: delta
            })

            # Ograniczamy bufor do maksymalnej długości (największego zakresu)
            max_n = max(ranges)
            if len(past_matches) > max_n:
                past_matches = past_matches[-max_n:]

            # Obliczamy formę dla każdego zakresu n
            for n in ranges:
                relevant_matches = past_matches[-n:]  # Pobieramy ostatnie n meczów
                form_value = (
                    np.mean([match[elo_change_col] for match in relevant_matches])
                    if relevant_matches else 0
                )

                # Znajdujemy indeks meczu w oryginalnym DataFrame i zapisujemy wynik
                match_idx = df.index[
                    (df[date_col] == row[date_col]) &
                    ((df['home_name'] == row["home_name"]) & (df['away_name'] == row["away_name"])) &
                    (df['league_name'] == row["league_name"])
                ]
                if not match_idx.empty:
                    forms[n][match_idx[0]] = form_value

    return forms

ranges = [20, 50, 100]  # Zakresy analizowane
forms = calculate_form_multiple_ranges(df, 'home_name', ranges)
# Przypisanie wyników do odpowiednich kolumn DataFrame
for n in ranges:
    df[f'home_form_{n}'] = forms[n]

forms = calculate_form_multiple_ranges(df, 'away_name', ranges)
for n in ranges:
    df[f'away_form_{n}'] = forms[n]
# # Modyfikacja delta_elo w zależności od kategorii meczu
# def adjust_delta_elo(row):
#     if row['match_category'] == 'final':
#         if row['winner'] == 1:  # Home wygrywa
#             return row['delta_elo'] + 0.8
#         else:  # Away wygrywa
#             return row['delta_elo'] - 0.6
#     if row['match_category'] == '3rd':
#         if row['winner'] == 1:  # Home wygrywa
#             return row['delta_elo'] + 0.4
#         else:  # Away wygrywa
#             return row['delta_elo'] - 0.6
#     return row['delta_elo']

# df['adjusted_delta_elo'] = df.apply(adjust_delta_elo, axis=1)

# Przygotowanie danych wejściowych
features = [
    'home_elo', 'away_elo', 'elo_diff', 
    'home_form_50', 'away_form_50', 
    'home_form_100', 'away_form_100',
]

X = df[features].fillna(0)
y = df['winner']

# Podział na zbiory
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Trenowanie modelu
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Ocena modelu
y_pred = model.predict(X_test)
y_pred_proba = model.predict_proba(X_test)[:, 1]

print(f"Accuracy: {accuracy_score(y_test, y_pred):.2f}")
print(f"ROC-AUC: {roc_auc_score(y_test, y_pred_proba):.2f}")

# Przewidywanie na podstawie nowych danych
new_match_data = {
    'home_elo': 720,
    'away_elo': 659,
    'elo_diff': 720 - 659,
    'home_form_50': 20.0,
    'away_form_50': -10.0,
    'home_form_100': 30.0,
    'away_form_100': -5.0,
}

new_match_df = pd.DataFrame([new_match_data])

# Predykcja
win_probability = model.predict_proba(new_match_df)[:, 1][0]
print(f"Szacowane prawdopodobieństwo wygranej gracza domowego: {win_probability * 100:.2f}%")