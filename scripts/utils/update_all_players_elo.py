from pymongo import MongoClient

# Połączenie z bazą danych MongoDB
conn = MongoClient('')
db = conn['czech_liga_pro_test']
players_collection = db['players']
matches_collection = db['matches']

# Iteracja po każdym player_name w kolekcji players
for player in players_collection.find():
    player_name = player['player_name']
    
    # Pobieranie najnowszego wyniku (na podstawie daty) z kolekcji matches
    latest_match = matches_collection.find_one(
        {'$or': [
            {'home_name': player_name},
            {'away_name': player_name}
        ]},
        sort=[('datetime', -1)]  # Sortowanie po dacie malejąco (najnowsze na początku)
    )
    
    if latest_match:
        # Pobranie najnowszego score z kolekcji matches
        # Zależy od tego, czy gracz jest gospodarzem czy gościem
        if latest_match['home_name'] == player_name:
            if latest_match["match_category"] == "group":
                new_score = latest_match['home_elo'] + latest_match['delta_elo']
            if latest_match["match_category"] == "final":
                if latest_match['delta_elo'] > 0:
                    new_score = latest_match['home_elo'] + latest_match['delta_elo'] + 0.8
                else:
                    new_score = latest_match['home_elo'] + latest_match['delta_elo'] + 0.6
            if latest_match["match_category"] == "3rd":
                if latest_match['delta_elo'] > 0:
                    new_score = latest_match['home_elo'] + latest_match['delta_elo'] + 0.4
                else:
                    new_score = latest_match['home_elo'] + latest_match['delta_elo']
        else:
            if latest_match["match_category"] == "group":
                new_score = latest_match['away_elo'] - latest_match['delta_elo']
            if latest_match["match_category"] == "final":
                if latest_match['delta_elo'] < 0:
                    new_score = latest_match['away_elo'] + latest_match['delta_elo'] + 0.8
                else:
                    new_score = latest_match['away_elo'] + latest_match['delta_elo'] + 0.6
            if latest_match["match_category"] == "3rd":
                if latest_match['delta_elo'] < 0:
                    new_score = latest_match['away_elo'] + latest_match['delta_elo'] + 0.4
                else:
                    new_score = latest_match['away_elo'] + latest_match['delta_elo']
        
        # Aktualizacja pola score w kolekcji players
        players_collection.update_one(
            {'player_name': player_name}, 
            {'$set': {'elo': new_score, "last_elo_update": latest_match["datetime"]}}
        )
        print(f"Updated player {player_name} score to {new_score}")
    else:
        print(f"No matches found for player {player_name}")