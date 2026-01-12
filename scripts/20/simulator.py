from pymongo import MongoClient
from datetime import datetime, timedelta

# Połączenie z bazą danych MongoDB
conn = MongoClient('')
db = conn['czech_liga_pro_test']
matches_coll = db['matches']
calendar_coll = db['match_calendar']
player_coll = db['players']



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

def calculate_elo_based_on_history(player_name, datetime):
    player_history = get_matches_history_of_player(player_name, datetime)
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
    if history:
        return history[-1]['elo_after']
    else:
        return 0

no_won = 0
no_lost = 0
suma = 0
# Zapytanie, aby pobrać X najnowszych rekordów według daty (od najnowszych)
matches = calendar_coll.find().sort('datetime', -1).limit(3500)

# Wypisanie wyników
for mecz in matches:
    # Pobieranie danych meczu
    if "home_odd" not in mecz:
        continue
    home_odd = mecz["home_odd"]
    away_odd = mecz["away_odd"]
    match_date = mecz["datetime"]
    home_name = list(player_coll.find({"player_id": mecz["home_id"]}, {'player_name': 1}))
    away_name = list(player_coll.find({"player_id": mecz["away_id"]}, {'player_name': 1}))
    if not home_name or not away_name:
        continue

    home_name = home_name[0]["player_name"]
    away_name = away_name[0]["player_name"]

    time_range_start = match_date - timedelta(minutes=60)
    time_range_end = match_date + timedelta(minutes=60)

    match_database_record = matches_coll.find_one({
        "home_name": home_name,
        "away_name": away_name,
        "datetime": {"$gte": time_range_start, "$lte": time_range_end}
    })

    if not match_database_record:
        print(f"[WARNING] Could not find matching record for: {home_name} vs {away_name}. Skipping.")
        continue

    home_elo = calculate_elo_based_on_history(home_name, match_database_record["datetime"])
    away_elo = calculate_elo_based_on_history(away_name, match_database_record["datetime"])

    if home_elo == 0 or away_elo == 0:
        continue

    if match_database_record["winner"] == 1 and home_elo > away_elo +50 and home_odd >= 1.6:
        no_won += 1
        suma += home_odd - 1
    elif match_database_record["winner"] == 0 and home_elo+50 < away_elo and away_odd >= 1.6:
        no_won += 1
        suma += away_odd - 1        
    elif match_database_record["winner"] == 1 and home_elo+50 < away_elo and away_odd >= 1.6:
        no_lost += 1
        suma -= 1
    elif match_database_record["winner"] == 0 and home_elo > away_elo+50 and home_odd >= 1.6:
        no_lost += 1
        suma -= 1    

print(f'Number of won {no_won}. Number of lost {no_lost}. Suma jednostek {suma}')



