from pymongo import MongoClient
from datetime import datetime

# Połączenie z bazą danych MongoDB
conn = MongoClient('')
db = conn['czech_liga_pro_test']
matches_coll = db['matches']
players_coll = db['players']

player_name = "Dorko T"

def get_matches_history_of_player(player_name, datetime_ref):

    # Agregacja: filtrujemy mecze, grupujemy po dacie i nazwie ligi, sortujemy po dacie rosnąco
    pipeline = [
        # Filtrujemy mecze, gdzie gracz występuje jako gospodarz lub gość
        {
            "$match": {
                "$or": [
                    {"home_name": player_name},
                    {"away_name": player_name}
                ],
            "datetime": {
                "$lt": datetime_ref
            }
            }
        },
        # Grupa według daty (formatowane do daty bez godziny) i nazwy ligi
        {
            "$group": {
                "_id": {
                    "date": {"$dateToString": {"format": "%Y-%m-%d", "date": "$datetime"}},  # Grupa po dacie (tylko data, bez godziny)
                    "league_name": "$league_name"  # Grupa po nazwie ligi
                },
                "matches": {"$push": "$$ROOT"}  # Wszystkie mecze w tej grupie
            }
        },
        # Sortowanie po dacie rosnąco
        {
            "$sort": {
                "_id.date": 1  # 1 oznacza rosnąco
            }
        }
    ]

    # Uruchomienie agregacji
    result = matches_coll.aggregate(pipeline)

    return result

def get_actual_player_elo(player_name, match, actual_elo):

    delta_sum = 0

    if match["home_name"] == player_name:
        delta_sum += match["delta_elo"]
        if match["match_category"] == "final":
            if match["delta_elo"] > 0:
                delta_sum += 0.8
            else:
                delta_sum += 0.6
        if match["match_category"] == "3rd":
            if match["delta_elo"] > 0:
                delta_sum += 0.4
            
    else:
        delta_sum += match["delta_elo"] * -1
        if match["match_category"] == "final":
            if match["delta_elo"] > 0:
                delta_sum += 0.6
            else:
                delta_sum += 0.8
        if match["match_category"] == "3rd":
            if match["delta_elo"] < 0:
                delta_sum += 0.4

    return actual_elo + delta_sum

date_ref = datetime(2024, 12, 20, 23, 15, 15)

player_history = get_matches_history_of_player(player_name, date_ref)
history = []
# Wyświetlanie wyników
for group in player_history:
    if group['matches'][0]["home_name"] == player_name:
        actual_elo = group['matches'][0]["home_elo"]
    else:
        actual_elo = group['matches'][0]["away_elo"]
    for match in group['matches']:
        match["actual_elo"] = get_actual_player_elo(player_name, match, actual_elo)
        actual_elo = match["actual_elo"]
        history.append({'elo_after': round(actual_elo, 2)})


print(f' Gracz {player_name}, Na dzien {date_ref} aktualne ELO wynosi {history[-1]['elo_after']}')

import numpy as np

# Ustalamy ziarno dla powtarzalności wyników
#np.random.seed(43)

# Inicjalizujemy początkowe Elo
#elo_start = 400
#history = []

# # Generowanie historii 100 meczy
# elo_before = elo_start
# for i in range(1000):
#     if i < 250:
#         # Losowa zmiana Elo w zależności od wyniku meczu
#         result = np.random.choice([0, 1], p=[0.4, 0.6])
#     elif i >= 250 and i < 500:
#         # Losowa zmiana Elo w zależności od wyniku meczu
#         result = np.random.choice([0, 1], p=[0.6, 0.4])
#     elif i >= 500 and i < 750:
#         # Losowa zmiana Elo w zależności od wyniku meczu
#         result = np.random.choice([0, 1], p=[0.4, 0.6])
#     else:
#         # Losowa zmiana Elo w zależności od wyniku meczu
#         result = np.random.choice([0, 1], p=[0.6, 0.4])            

#     if result == 1:  # 1 - wygrana
#         change = np.random.normal(20, 5)  # Wzrost Elo przy wygranej
#         result = 1
#     else:  # 0 - przegrana
#         change = np.random.normal(-20, 5)  # Spadek Elo przy przegranej
#         result = 0

#     elo_after = elo_before + change
    
#     # Dodanie meczu do historii
#     history.append({'elo_before': round(elo_before, 2), 'elo_after': round(elo_after, 2), 'result': result})
    
#     # Ustawiamy nowe Elo jako Elo po meczu
#     elo_before = elo_after


import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import argrelextrema
from scipy.stats import linregress

# Przykładowa lista historii Elo (zastąp to rzeczywistymi danymi)
elo_values = [h['elo_after'] for h in history]

# Funkcja do wygładzania danych za pomocą średniej kroczącej
def moving_average(data, window_size):
    return np.convolve(data, np.ones(window_size)/window_size, mode='valid')

# Wygładzenie danych
window_size = 2
smoothed_elo = moving_average(elo_values, window_size)
smoothed_x = np.arange(len(smoothed_elo))  # Nowe indeksy dla wygładzonych danych

# Wykrywanie lokalnych minimów i maksimów
local_maxima = argrelextrema(smoothed_elo, np.greater)[0]
local_minima = argrelextrema(smoothed_elo, np.less)[0]
transition_points = np.sort(np.concatenate((local_maxima, local_minima)))

# Analiza trendu za pomocą regresji liniowej
slope, intercept, r_value, p_value, std_err = linregress(smoothed_x, smoothed_elo)
if slope > 0:
    overall_trend = "Wzrost"
elif slope < 0:
    overall_trend = "Spadek"
else:
    overall_trend = "Stabilność"

# Alternatywne podejście: porównanie średnich wartości z pierwszej i drugiej połowy
midpoint = len(smoothed_elo) // 2
mean_first_half = np.mean(smoothed_elo[:midpoint])
mean_second_half = np.mean(smoothed_elo[midpoint:])

if mean_second_half > mean_first_half:
    trend_by_means = "Wzrost"
elif mean_second_half < mean_first_half:
    trend_by_means = "Spadek"
else:
    trend_by_means = "Stabilność"

# Analiza trendu co 50 meczów
segment_size = 50
num_segments = len(smoothed_elo) // segment_size
segment_trends = []

for i in range(num_segments):
    start_idx = i * segment_size
    end_idx = start_idx + segment_size
    segment_x = smoothed_x[start_idx:end_idx]
    segment_y = smoothed_elo[start_idx:end_idx]
    if len(segment_x) > 1:  # Sprawdzenie, czy segment ma wystarczającą liczbę punktów
        segment_slope, _, _, _, _ = linregress(segment_x, segment_y)
        if segment_slope > 0:
            segment_trends.append("Wzrost")
        elif segment_slope < 0:
            segment_trends.append("Spadek")
        else:
            segment_trends.append("Stabilność")

# # Wyświetlenie trendów segmentów
# for i, trend in enumerate(segment_trends):
#     print(f"Segment {i+1}: {trend}")

# # Ocena tendencji formy gracza
# # Funkcja obliczająca średni czas między ekstremami
# def analyze_extrema_dynamics(local_maxima, local_minima):
#     extrema = np.sort(np.concatenate((local_maxima, local_minima)))
#     intervals = np.diff(extrema)
#     avg_interval = np.mean(intervals) if len(intervals) > 0 else None
#     return avg_interval

# avg_interval = analyze_extrema_dynamics(local_maxima, local_minima)
# print(f"Średni odstęp między ekstremami: {avg_interval} meczów")

# Szacowanie prawdopodobieństwa wygranej lub przegranej
#current_elo = smoothed_elo[-1]

# # Znajdowanie ostatniego ekstremum
# if len(local_maxima) > 0 and len(local_minima) > 0:
#     last_maximum = local_maxima[local_maxima < len(smoothed_elo)][-1] if any(local_maxima < len(smoothed_elo)) else None
#     last_minimum = local_minima[local_minima < len(smoothed_elo)][-1] if any(local_minima < len(smoothed_elo)) else None
#     if last_maximum is not None and (last_minimum is None or last_maximum > last_minimum):
#         last_extremum = last_maximum
#         extremum_type = "maximum"
#     elif last_minimum is not None:
#         last_extremum = last_minimum
#         extremum_type = "minimum"
#     else:
#         last_extremum = None
#         extremum_type = None
# elif len(local_maxima) > 0:
#     last_extremum = local_maxima[local_maxima < len(smoothed_elo)][-1] if any(local_maxima < len(smoothed_elo)) else None
#     extremum_type = "maximum"
# elif len(local_minima) > 0:
#     last_extremum = local_minima[local_minima < len(smoothed_elo)][-1] if any(local_minima < len(smoothed_elo)) else None
#     extremum_type = "minimum"
# else:
#     last_extremum = None
#     extremum_type = None

# if last_extremum is not None:
#     distance_from_last_extremum = len(smoothed_elo) - last_extremum

#     if extremum_type == "maximum":
#         print(f"Ostatnie ekstremum to maksimum, odległość: {distance_from_last_extremum} meczów")
#         win_probability = max(0, 1 - (distance_from_last_extremum / avg_interval))  # Im dalej od maksimum, tym mniejsze szanse
#     elif extremum_type == "minimum":
#         print(f"Ostatnie ekstremum to minimum, odległość: {distance_from_last_extremum} meczów")
#         win_probability = min(1, distance_from_last_extremum / avg_interval)  # Im dalej od minimum, tym większe szanse
# else:
#     win_probability = 0.5  # Neutralne prawdopodobieństwo, gdy brak danych

# print(f"Szacowane prawdopodobieństwo wygranej: {win_probability * 100:.2f}%")

# Przeanalizowanie wyników
print(f"Ogólny trend (regresja liniowa): {overall_trend}")
print(f"Ogólny trend (porównanie średnich): {trend_by_means}")

# Wizualizacja danych i wyników
plt.plot(smoothed_x, smoothed_elo, label='Elo (wygładzone)')
# Rysowanie linii dla punktów przejścia (bez powtarzania etykiety)
for i, point in enumerate(transition_points):
    plt.axvline(x=point, color='r', linestyle='--', label='Przejście trendu' if i == 0 else None)

plt.scatter(local_maxima, smoothed_elo[local_maxima], color='g', label='Lokalne maksima')
plt.scatter(local_minima, smoothed_elo[local_minima], color='b', label='Lokalne minima')

# Zaznaczanie segmentów na wykresie
for i in range(num_segments):
    start_idx = i * segment_size
    end_idx = start_idx + segment_size
    plt.axvspan(smoothed_x[start_idx], smoothed_x[min(end_idx, len(smoothed_elo)-1)], color='yellow', alpha=0.1, label='Segment' if i == 0 else None)

plt.xlabel('Numer meczu (wygładzone)')
plt.ylabel('Elo')
plt.title('Analiza zmian Elo gracza')
plt.legend()

# Zapisanie wykresu do pliku (np. w formacie PNG)
plt.savefig('elo_trend_analysis_smoothed.png')
plt.close()


import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import linregress

# **Stopień wielomianu - zmieniaj tylko tutaj!**
poly_degree = 3  # Możesz zmienić na dowolny stopień (np. 2, 3, 5, itd.)

# Dopasowanie wielomianu
coefficients = np.polyfit(smoothed_x, smoothed_elo, poly_degree)
polynomial = np.poly1d(coefficients)

# Obliczenie trendu na podstawie wielomianu
trend_line = polynomial(smoothed_x)

# Wizualizacja
plt.plot(smoothed_x, smoothed_elo, label='Elo (wygładzone)')
plt.plot(smoothed_x, trend_line, label=f'Trend (stopień {poly_degree})', linestyle='--', color='purple')
plt.xlabel('Numer meczu (wygładzone)')
plt.ylabel('Elo')
plt.title('Analiza trendu za pomocą regresji wielomianowej')
plt.legend()
plt.savefig('trend_analysis.png')
plt.close()

# Wypisanie współczynników trendu
print(f"Współczynniki wielomianu stopnia {poly_degree}:", coefficients)

# Rozpakowanie współczynników: ax^3 + bx^2 + cx + d
a, b, c, d = coefficients

# **1. Ogólna forma gracza (globalny trend)**
# if a > 0:
#     print("Ogólna forma: Silnie wzrostowa na dłuższą metę (dominacja trzeciego stopnia).")
# elif a < 0:
#     print("Ogólna forma: Silnie spadkowa na dłuższą metę (dominacja trzeciego stopnia).")
# else:
if b > 0:
    print("Ogólna forma: Wzrostowa (dominacja kwadratowego trendu).")
elif b < 0:
    print("Ogólna forma: Spadkowa (dominacja kwadratowego trendu).")
else:
    if c > 0:
        print("Ogólna forma: Liniowo wzrostowa.")
    elif c < 0:
        print("Ogólna forma: Liniowo spadkowa.")
    else:
        print("Ogólna forma: Neutralna.")

# **2. Ostatnia forma gracza (lokalny trend)**
# Analiza ostatnich 10 punktów
last_x = smoothed_x[-30:]
last_y = polynomial(last_x)

# Regresja liniowa na ostatnich punktach
slope, _, _, _, _ = linregress(last_x, last_y)
print(slope)
if slope > 0:
    print("Ostatnia forma: Wzrostowa (krótkoterminowo).")
elif slope < 0:
    print("Ostatnia forma: Spadkowa (krótkoterminowo).")
else:
    print("Ostatnia forma: Neutralna (krótkoterminowo).")