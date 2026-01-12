import pymongo
import pytz
import numpy as np
from scipy import stats
import json
from telegram import Bot
from telegram.error import TelegramError
from datetime import datetime
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
from datetime import datetime, timedelta
import asyncio
from pymongo import DESCENDING
import random


from collections import defaultdict
from itertools import combinations
# Funkcja grupująca zakłady po dacie (tylko dzień i miesiąc są ważne, nie godzina)
def group_bets_by_date(bets):
    grouped_bets = defaultdict(list)
    for bet in bets:
        # Używamy tylko daty (bez godziny)
        date_str = bet["datetime"].date()
        grouped_bets[date_str].append(bet)
    return grouped_bets

# Funkcja do obliczania zysku z AKO2
def calculate_profit(odd1, odd2, result1, result2):
    if result1 == 1 and result2 == 1:
        return (odd1 * odd2 - 1)  # Zysk z AKO2
    return -1  # Jeżeli któryś zakład nie wszedł

def generate_symmetric_pairs(bets_on_date):
    num_bets = len(bets_on_date)
    if num_bets < 2:
        return []

    pairs = []
    for i in range(num_bets):
        bet1 = bets_on_date[i]
        bet2 = bets_on_date[(i + 1) % num_bets]  # Łączymy sąsiadujące zdarzenia w cyklu
        pairs.append((bet1, bet2))
    
    return pairs

class ValueEngine:
    def __init__(self):
        #region Connection Database 
        """
        Initializes the ValueEngine class by setting up database connections, loading configuration,
        and initializing the Telegram bot.
        """
        self.conn = pymongo.MongoClient('')
        self.db = self.conn['czech_liga_pro_test']
        self.calendar_coll = self.db['match_calendar']
        self.players_coll = self.db['players']
        self.betting_coll = self.db['betting_one']
        self.betting_coll_two = self.db['betting']
        self.matches_coll = self.db['matches']
        self.parameters_coll = self.db['parameters_new']
        self.betting_ai = self.db['test_marzec_model_xbs_away']
        self.betting_ai2 = self.db['test_marzec_model_home']
        self.betting_ai3 = self.db['betting_ai_three']
        self.database_parameters_coll = self.db['parameters_new']
        self.database_parameters_coll_away = self.db['parameters_new']

        
        #endregion

        # Load config
        self.config = self.load_config('/data/paletki/real-time/.creds')
        self.bot_token = self.config["BOT_TOKEN"]
        self.chat_id = self.config["CHAT_ID"]
        self.bot = Bot(token=self.bot_token)


    def load_config(self, file_path):
        """
        Loads configuration from a JSON file.

        Args:
            file_path (str): Path to the configuration file.

        Returns:
            dict: Parsed configuration as a dictionary.
        """
        with open(file_path, 'r') as f:
            return json.load(f)
        
    def get_last_inserted_id(self):
        """
        Retrieves the ID of the last inserted document in the `match_calendar` collection.

        Returns:
            ObjectId: The ID of the last inserted document, or None if the collection is empty.
        """
        #region Get_Last_Inserted_id
        last_doc = self.calendar_coll.find_one(sort=[('_id', -1)])
        return last_doc['_id'] if last_doc else None
    

    def player_matching(self, data):
        #region Player Matching
        """
        Matches players' data from the database based on the provided match data.

        Args:
            data (dict): Dictionary containing match data, including `datetime`, `home_id`, and `away_id`.

        Returns:
            tuple: A tuple containing localized match date, home player name, away player name,
                   home player's ELO, and away player's ELO.
        """
        date_for_match = data["datetime"]
        home_id = data["home_id"]
        away_id = data["away_id"]

        p_home = self.players_coll.find_one({"player_id": home_id})
        p_away = self.players_coll.find_one({"player_id": away_id})

        if p_home:
            elo_home = p_home.get('elo', 0)
            home_name_tt = p_home.get('player_name', "")
        else:
            elo_home = 0
            home_name_tt = ""

        if p_away:
            elo_away = p_away.get('elo', 0)
            away_name_tt = p_away.get('player_name', "")
        else:
            away_name_tt = ""
            elo_away = 0

        gmt_utc = pytz.timezone('UTC')
        date_for_match = gmt_utc.localize(date_for_match)
        gmt_plus_1 = pytz.timezone('Europe/Warsaw')
        date_for_match = date_for_match.astimezone(gmt_plus_1)

        return date_for_match, home_name_tt, away_name_tt, elo_home, elo_away

    def get_actual_form(self, home_player, away_player, intervals=[(100, 60), (60, 25), (25, 0)]):
        """
        Oblicza trendy dla przedziałów meczów określonych w intervals dla obu zawodników.
        
        Args:
            home_player (str): Nazwa gracza gospodarza.
            away_player (str): Nazwa gracza gościa.
            intervals (list): Lista krotek określających przedziały meczów, np. [(100, 60), (60, 25), (25, 0)].
        
        Returns:
            dict: Zawiera trendy dla obu graczy w formacie:
                {
                    "home": {"100-60": slope, "60-25": slope, "25-0": slope},
                    "away": {"100-60": slope, "60-25": slope, "25-0": slope}
                }
        """
        def calculate_trends_for_intervals(matches, player_name, intervals):
            """
            Oblicza trendy dla określonych przedziałów.
            """
            delta_elo = []
            for match in matches:
                # Obliczanie delta_elo w zależności od tego, czy gracz był "home" czy "away"
                delta = match.get('delta_elo', 0) if match.get('home_name') == player_name else -match.get('delta_elo', 0)
                match_category = match.get('match_category', '').lower()
                home_score, away_score = map(int, match["ft_result"].split(" : "))            
                home_is_winner = home_score > away_score == 1

                # Korekta delta_elo w zależności od kategorii meczu
                if match_category == '3rd':  # Mecz o 3. miejsce
                    if (home_is_winner and match.get('home_name') == player_name) or (not home_is_winner and match.get('away_name') == player_name):
                        delta += 0.4
                elif match_category == 'final':  # Mecz finałowy
                    if home_is_winner:
                        if match.get('home_name') == player_name:
                            delta += 0.8  # Wygrana w finale
                        else:
                            delta += 0.6  # Przegrana w finale, drugie miejsce
                    else:
                        if match.get('away_name') == player_name:
                            delta += 0.6  # Przegrana w finale
                delta_elo.append(delta)

            # Odwracanie kolejności, aby najnowsze mecze były na końcu
            delta_elo = np.flip(delta_elo)
            #print(f"[DEBUG] delta_elo for {player_name}: {delta_elo}")  # Debugowanie listy delta_elo

            trends = {}
            for start, end in intervals:
                if len(delta_elo) < start:
                    #print(f"[DEBUG] Not enough matches for interval {start}-{end}. Available: {len(delta_elo)}")
                    trends[f"{start}-{end}"] = 0
                    continue

                # Poprawne wyznaczenie zakresu w odwróconej liście
                limited_delta_elo = delta_elo[-start:-end or None]
                #print(f"[DEBUG] limited_delta_elo ({start}-{end}) for {player_name}: {limited_delta_elo}")  # Debugowanie zakresu

                if len(limited_delta_elo) < 2:
                    #print(f"[DEBUG] Not enough points in interval {start}-{end} for {player_name}.")
                    trends[f"{start}-{end}"] = 0
                    continue

                # Obliczanie cumulative elo i nachylenia
                cumulative_elo = np.cumsum(np.insert(limited_delta_elo, 0, 0))
                x = np.arange(len(cumulative_elo))
                slope, _, _, _, _ = stats.linregress(x, cumulative_elo)
                trends[f"{start}-{end}"] = slope

            return trends

        # Pipeline dla gracza "home_player"
        home_pipeline = [
            {"$match": {
                "$or": [
                    {"home_name": home_player},
                    {"away_name": home_player}
                ],
                "datetime": {
                    "$lt": self.match["datetime"]
                }
            }},
            {"$sort": {"datetime": -1}},
            {"$limit": max(interval[0] for interval in intervals)}
        ]
        home_matches = list(self.matches_coll.aggregate(home_pipeline))
        #print(f"[DEBUG] Total matches for {home_player}: {len(home_matches)}")  # Debugowanie liczby meczów

        # Pipeline dla gracza "away_player"
        away_pipeline = [
            {"$match": {
                "$or": [
                    {"home_name": away_player},
                    {"away_name": away_player}
                ],
                "datetime": {
                    "$lt": self.match["datetime"]
                }
            }},
            {"$sort": {"datetime": -1}},
            {"$limit": max(interval[0] for interval in intervals)}
        ]
        away_matches = list(self.matches_coll.aggregate(away_pipeline))
        #print(f"[DEBUG] Total matches for {away_player}: {len(away_matches)}")  # Debugowanie liczby meczów

        # Obliczanie trendów dla obu graczy
        home_trends = calculate_trends_for_intervals(home_matches, home_player, intervals)
        away_trends = calculate_trends_for_intervals(away_matches, away_player, intervals)

        #print(f"[DEBUG] Trends for {home_player}: {home_trends}")  # Debugowanie trendów home
        #print(f"[DEBUG] Trends for {away_player}: {away_trends}")  # Debugowanie trendów away

        return {"home": home_trends, "away": away_trends}


    def get_combined_form_chart(self, home_player, away_player, n_values=100):
        #region Wykresy Formy
        """
        Generates a chart comparing the cumulative ELO changes of two players over recent matches.

        Args:
            home_player (str): Name of the home player.
            away_player (str): Name of the away player.
            n_values (int): Number of recent matches to consider for each player.

        Returns:
            str: File path of the saved chart image.
        """
        pipeline = [
            {"$match": {
                "$or": [
                    {"home_name": home_player},
                    {"away_name": home_player}
                ],
                "datetime": {
                    "$lt": self.match["datetime"]
                }
            }},
            {"$sort": {"datetime": -1}},
            {"$limit": n_values}
        ]

        home_matches = list(self.matches_coll.aggregate(pipeline))
        # Ustalanie wartości delta_elo z uwzględnieniem, czy gracz był home czy away
        home_delta_elo = []
        for match in home_matches:
            delta = match.get('delta_elo', 0) if match.get('home_name') == home_player else -match.get('delta_elo', 0)
            match_category = match.get('match_category', '').lower()
            #winner = match.get('winner')
            home_score, away_score = map(int, match["ft_result"].split(" : "))            
            home_is_winner = home_score > away_score == 1

            if match_category == '3rd':
                if (home_is_winner and match.get('home_name') == home_player) or (not home_is_winner and match.get('away_name') == home_player):
                    delta += 0.4
            elif match_category == 'final':
                if home_is_winner:
                    if match.get('home_name') == home_player:
                        delta += 0.8
                    else:
                        delta += 0.6
            home_delta_elo.append(delta)

        home_delta_elo = np.flip(home_delta_elo)
        home_cumulative_elo = np.cumsum(home_delta_elo)
        # Dodanie początkowej wartości 0
        home_cumulative_elo = np.insert(home_cumulative_elo, 0, 0)

        # Pipeline dla gracza "away_player"
        pipeline[0]["$match"]["$or"] = [
            {"home_name": away_player},
            {"away_name": away_player}
        ]

        away_matches = list(self.matches_coll.aggregate(pipeline))
        # Ustalanie wartości delta_elo z uwzględnieniem, czy gracz był home czy away
        away_delta_elo = []
        for match in away_matches:
            delta = match.get('delta_elo', 0) if match.get('home_name') == away_player else -match.get('delta_elo', 0)
            match_category = match.get('match_category', '').lower()
            #winner = match.get('winner')
            home_score, away_score = map(int, match["ft_result"].split(" : ")) 
            home_is_winner = home_score > away_score == 1

            if match_category == '3rd':
                if (home_is_winner and match.get('home_name') == away_player) or (not home_is_winner and match.get('away_name') == away_player):
                    delta += 0.4
            elif match_category == 'final':
                if home_is_winner:
                    if match.get('home_name') == away_player:
                        delta += 0.8
                    else:
                        delta += 0.6
            away_delta_elo.append(delta)

        away_delta_elo = np.flip(away_delta_elo)
        away_cumulative_elo = np.cumsum(away_delta_elo)    
        away_cumulative_elo = np.insert(away_cumulative_elo, 0, 0)
        
        # Rysowanie wykresu
        plt.figure(figsize=(12, 8))
        plt.plot(home_cumulative_elo, marker='o', label=f'Form: {home_player}')
        plt.plot(away_cumulative_elo, marker='s', label=f'Form: {away_player}')
        plt.title("Forma zawodników (ostatnie 100 turniejów)")
        plt.xlabel("Mecze")
        plt.ylabel("Kumulatywne zmiany ELO")
        plt.grid()
        plt.legend()

        # Zapisanie wykresu do pliku w katalogu 'real-time'
        directory = 'png_trash'
        if not os.path.exists(directory):
            os.makedirs(directory)
        file_path = os.path.join(directory, f"{home_player}_vs_{away_player}_form_chart.png")
        plt.savefig(file_path)
        plt.close()
        #endregion
        return file_path, home_cumulative_elo, away_cumulative_elo



    async def delete_form_chart_png_files(self, directory):
        #region Delete PNG Files
        """
        Deletes all PNG files containing 'form_chat' in their filename from the specified directory.

        Args:
            directory (str): Path to the directory to scan for files.

        Returns:
            None
        """
        try:
            for filename in os.listdir(directory):
                if filename.endswith('.png') and 'form_chart' in filename:
                    file_path = os.path.join(directory, filename)
                    os.remove(file_path)
                    print(f"Deleted file: {file_path}")
        except Exception as e:
            print(f"Error while deleting files: {e}")


    
    def get_upcoming_matches(self, n, create_features=False):
        """
        Retrieves matches scheduled to occur within the next n hours.

        Returns:
            list: List of upcoming match documents from the `match_calendar` collection.
        """
        #region Upcoming Matches
        now = datetime.now(self.local_zone)  # Czas lokalny (GMT+1)
        if not create_features: 
            future_time = now - timedelta(days=n)
            future_time = datetime(2024, 12, 11, 6,59,0)
            now = datetime(2025, 3, 6, 11,30,0)

            # Pobierz mecze zaplanowane na kolejne n godziny
            matches = self.calendar_coll.find({
                "datetime": {"$lte": now.astimezone(self.utc_zone), "$gt": future_time.astimezone(self.utc_zone)}
            })
            return list(matches)
        else:
            future_time = now - timedelta(hours=n)

            # Pobierz mecze zaplanowane na kolejne n godziny
            matches = self.matches_coll.find({
                "datetime": {"$lte": now.astimezone(self.utc_zone), "$gt": future_time.astimezone(self.utc_zone)}
            })
            return list(matches)
    

    def is_duplicate(self, home_name_tt, away_name_tt, betting_collection, match_data):
        #region Duplikaty kolekcji
        """
        Checks if a match has already been processed in the betting collection.

        Args:
            home_name_tt (str): Home player's name.
            away_name_tt (str): Away player's name.
            betting_collection (MongoCollection): MongoDB collection to check for duplicates.
            match_data (dict): Match data containing at minimum keys: `datetime`, `home_name`, and `away_name`.

        Returns:
            bool: True if the match is a duplicate, False otherwise.
        """
        record_datetime = match_data.get("datetime")
        home_name_tt = home_name_tt.strip() if home_name_tt else match_data.get("home_name", "").strip()
        away_name_tt = away_name_tt.strip() if away_name_tt else match_data.get("away_name", "").strip()


        if isinstance(record_datetime, str):
            record_datetime = datetime.fromisoformat(record_datetime).replace(tzinfo=self.utc_zone).astimezone(self.local_zone)
        elif isinstance(record_datetime, datetime):
            if record_datetime.tzinfo is None:
                record_datetime = self.utc_zone.localize(record_datetime).astimezone(self.local_zone)
            else:
                record_datetime = record_datetime.astimezone(self.local_zone)

        record_datetime_str = record_datetime.strftime("%Y:%m:%d %H:%M")

        existing_bet = betting_collection.find_one({
            "home_name": home_name_tt,
            "away_name": away_name_tt,
            "processed": True
        })

        if existing_bet:
            existing_datetime = existing_bet.get("datetime")
            if existing_datetime == record_datetime_str:
                return True
        return False


    async def send_message_value_two(self, message):
        #region Send Telegram 2
        """
        Sends a formatted message and a chart image to a Telegram chat.

        Args:
            message (dict): Dictionary containing match details and the path to the chart image.

        Returns:
            int: Telegram message ID if the message was sent successfully, None otherwise.
        """
        message["datetime"] = message["datetime"].strftime("%Y:%m:%d %H:%M")
        bot = Bot(token=self.bot_token)

        # Przygotowanie wiadomości tekstowej
        formatted_message = (
            f"Data: {message['datetime']}\n"
            f"Mecz: {message['home_name']} vs {message['away_name']}\n"
            f"ELO: {message['home_elo']} vs {message['away_elo']}\n"
            f"Kursy: {message['odd_home']} vs {message['odd_away']}\n\n"
            f"Zakład: {message['bet']}"
        )

        attempts = 0
        while attempts < 3:
            try:
                # Wysyłanie wiadomości tekstowej
                sent_message = await bot.send_message(chat_id=self.chat_id, text=formatted_message)

                # Zapisanie ID wiadomości
                message_id = sent_message.message_id
                message["telegram_message_id"] = message_id  # Dodanie ID do rekordu

                # Wysyłanie wykresu
                with open(message["combined_form_chart"], "rb") as form_chart:
                    await bot.send_photo(chat_id=self.chat_id, photo=form_chart, caption="Forma obu zawodników")

                # Zwrócenie ID wiadomości
                return message_id

            except TelegramError as e:
                error_message = str(e)
                if "Flood control exceeded" in error_message:
                    retry_after = int(error_message.split("Retry in ")[1].split(" seconds")[0])
                    print(f"Flood control exceeded. Waiting {retry_after} seconds before retrying... (Attempt {attempts + 1}/3)")
                    await asyncio.sleep(retry_after)
                    attempts += 1
                else:
                    print(f"Błąd podczas wysyłania wiadomości: {e}")
                    return None

        print("Nie udało się wysłać wiadomości po 3 próbach.")
        return None
    

    async def update_finished_matches_value_two(self, betting_collection, betting_collection2):
        #region Update Telegram 2
        """
        Updates the results of finished matches in `betting_` and edits the corresponding Telegram messages.

        Returns:
            None
        """
        betting_coll2 = list(betting_collection.find())
        betting_coll = list(betting_collection2.find())


        bot = Bot(token=self.bot_token)
        won = 0
        lost = 0
        unit = 0
        all = 0
        kursy = []


        # Keys to check for duplicates
        keys_to_check = ["datetime", "home_name", "away_name"]
        not_duplicates_home = []
        not_duplicates_away = []
        # Set to store the seen identifiers
        seen = {}
        duplicates = []
        # Iterate through both lists
        for lst in [betting_coll, betting_coll2]:
            for item in lst:
                # Create a unique identifier (tuple) based on the selected keys
                identifier = tuple(item[key] for key in keys_to_check)
                
                # If the identifier is already in 'seen', it's a duplicate
                if identifier in seen:
                    # Add both the original and the duplicate item to the duplicates list
                    duplicates.append(seen[identifier])  # Append the original occurrence
                    duplicates.append(item)  # Append the current duplicate
                else:
                    # Otherwise, store the current dictionary in 'seen' with the identifier as the key
                    seen[identifier] = item

        # Remove duplicates by comparing the key-value pairs and ensuring uniqueness
        unique_duplicates = []
        seen_set = []

        # Convert each dictionary into a tuple of key-value pairs for comparison
        for item in duplicates:
            # Create a tuple of the key-value pairs (this is hashable and can be compared)
            item_tuple = tuple(sorted(item.items()))  # Sorting ensures consistency in ordering
            
            if item_tuple not in seen_set:
                seen_set.append(item_tuple)
                unique_duplicates.append(item)

        for i in betting_coll:
            if i not in unique_duplicates:
                not_duplicates_home.append(i)
        for i in betting_coll2:
            if i not in unique_duplicates:
                not_duplicates_away.append(i)

        all_bets = []
        for p_h in [0.9]: #np.arange(0.80, 0.95, 0.05):  # Jeden krok dla 0.8
            for p_a in [0.75]: #np.arange(0.70, 0.95, 0.05):  # Zakres od 0.70 do 0.95
                won = 0
                lost = 0
                unit = 0
                all = 0
                kursy = []
                for i in range(0, len(unique_duplicates), 2):
                    home_match = unique_duplicates[i]
                    away_match = unique_duplicates[i+1]
                    # Pobieranie danych meczu
                    match_date = home_match["datetime"]
                    home_name = home_match["home_name"]
                    away_name = home_match["away_name"]
                    found1 = False
                    found2 = False

                    # if (away_match["bet"] == 2 and away_match["p1_away_prob"] > p_a):
                    #     found1 = True
                    # elif (away_match["bet"] == 2 and away_match["p1_away_prob"] <= 0.98 and away_match["diff_last_tournaments"] > 4):
                    #     found1 = True
                    # elif (away_match["bet"] == 2 and away_match["p1_away_prob"] <= 0.98 and away_match["experience"] < 0):
                    #     found1 = True
                    # elif (away_match["bet"] == 2 and away_match["p1_away_prob"] <= 0.98 and away_match['diff_players_vs_opponents_50'] < -50):
                    #     found1 = True

                    # if (home_match["bet"] == 1 and home_match["p0_home_prob"] > p_h): # avg_opponents_home
                    #     found2 = True
                    # elif (home_match["bet"] == 1 and home_match["p0_home_prob"] <= 0.98 and home_match["diff_last_tournaments"] < -4):
                    #     found2 = True
                    # elif (home_match["bet"] == 1 and home_match["p0_home_prob"] <= 0.98 and home_match["experience"] > 0):
                    #     found2 = True
                    # elif (home_match["bet"] == 1 and home_match["p0_home_prob"] <= 0.98 and home_match['diff_saved'] < -0.06):
                    #     found2 = True #diff_players_vs_opponents_50
                    # elif (home_match["bet"] == 1 and home_match["p0_home_prob"] <= 0.98 and home_match['diff_players_vs_opponents_50'] > 50):
                    #     found2 = True 


                    if (away_match["bet"] == 2 and away_match["p1_away_prob"] > p_a):
                        found1 = True

                    if (home_match["bet"] == 1 and home_match["p0_home_prob"] > p_h):
                        found2 = True

                    if found1 and found2:
                        continue
                    
                    if not found1 and not found2:
                        continue
                    # Szukanie rekordu w `betting_collection` pasującego do meczu z tolerancją +/- 10 minut
                    if isinstance(match_date, str):
                        match_datetime = datetime.strptime(match_date, "%Y:%m:%d %H:%M")
                    else:
                        match_datetime = match_date
                    #if match_datetime.month <2 or match_datetime.month == 12:
                    #    continue
                    time_range_start = match_datetime - timedelta(minutes=120)
                    time_range_end = match_datetime + timedelta(minutes=120)

                    if "created_time" in home_match:
                        if home_match["created_time"]:
                            time_range_start = home_match["created_time"]

                    match_database_record = self.matches_coll.find_one({
                        "home_name": home_name,
                        "away_name": away_name,
                        "datetime": {"$gte": time_range_start, "$lte": time_range_end}
                    })

                    if not match_database_record:
                        #print(f"[WARNING] Could not find matching record for: {home_name} vs {away_name}. Skipping.")
                        continue

                    ft_result = match_database_record["ft_result"]
                    #print(f"[INFO] Matching betting record found for match: {home_name} vs {away_name}.")
                    a = int(ft_result.strip().split(":")[0].replace(" ", ""))
                    b = int(ft_result.strip().split(":")[1].replace(" ", ""))
                    if a > b:
                        winner = 1
                    else:
                        winner = 2
                    if found2:
                        if winner == home_match['bet']:
                            won += 1
                            unit += (home_match['odd_home']) - 1
                            kursy.append(home_match['odd_home'])
                            all_bets.append(
                                {
                                    "datetime": match_datetime,
                                    "result": 1,
                                    "odd": home_match['odd_home']
                                }
                            )
                        else:
                            unit -= 1
                            lost += 1 
                            all_bets.append(
                                {
                                    "datetime": match_datetime,
                                    "result": 0,
                                    "odd": -1
                                }
                            )
                        all += 1      
                    elif found1:
                        if winner == away_match['bet']:
                            unit += (away_match['odd_away']) - 1
                            kursy.append(away_match['odd_away'])
                            won += 1
                            all_bets.append(
                                {
                                    "datetime": match_datetime,
                                    "result": 1,
                                    "odd": away_match['odd_away']
                                }
                            )
                        else:
                            unit -= 1
                            lost += 1
                            all_bets.append(
                                {
                                    "datetime": match_datetime,
                                    "result": 0,
                                    "odd": -1
                                }
                            )
                        all += 1                        

                for match in not_duplicates_home:
                    # Pobieranie danych meczu
                    match_date = match["datetime"]
                    home_name = match["home_name"]
                    away_name = match["away_name"]
                    found = False

                    if (match["bet"] == 1 and match["p0_home_prob"] > p_h):
                        found = True


                    # if (match["bet"] == 1 and match["p0_home_prob"] > p_h): # avg_opponents_home
                    #     found = True
                    # elif (match["bet"] == 1 and match["p0_home_prob"] <= 0.98 and match["diff_last_tournaments"] < -4):
                    #     found = True
                    # elif (match["bet"] == 1 and match["p0_home_prob"] <= 0.98 and match["experience"] > 0):
                    #     found = True
                    # elif (match["bet"] == 1 and match["p0_home_prob"] <= 0.98 and match['diff_saved'] < -0.06):
                    #     found = True #diff_players_vs_opponents_50
                    # elif (match["bet"] == 1 and match["p0_home_prob"] <= 0.98 and match['diff_players_vs_opponents_50'] > 50):
                    #     found = True 

                    if not found:
                        continue

                    # Szukanie rekordu w `betting_collection` pasującego do meczu z tolerancją +/- 10 minut
                    if isinstance(match_date, str):
                        match_datetime = datetime.strptime(match_date, "%Y:%m:%d %H:%M")
                    else:
                        match_datetime = match_date
                    #if match_datetime.month <2 or match_datetime.month == 12:
                    #    continue
                    time_range_start = match_datetime - timedelta(minutes=120)
                    time_range_end = match_datetime + timedelta(minutes=120)

                    if "created_time" in match:
                        if match["created_time"]:
                            time_range_start = match["created_time"]

                    match_database_record = self.matches_coll.find_one({
                        "home_name": home_name,
                        "away_name": away_name,
                        "datetime": {"$gte": time_range_start, "$lte": time_range_end}
                    })

                    if not match_database_record:
                        #print(f"[WARNING] Could not find matching record for: {home_name} vs {away_name}. Skipping.")
                        continue

                    ft_result = match_database_record["ft_result"]
                    #print(f"[INFO] Matching betting record found for match: {home_name} vs {away_name}.")
                    a = int(ft_result.strip().split(":")[0].replace(" ", ""))
                    b = int(ft_result.strip().split(":")[1].replace(" ", ""))
                    if a > b:
                        winner = 1
                    else:
                        winner = 2
                    
                    if winner == match['bet']:
                        unit += (match['odd_home']) - 1
                        kursy.append(match['odd_home'])
                        won += 1
                        all_bets.append(
                            {
                                "datetime": match_datetime,
                                "result": 1,
                                "odd": match['odd_home']
                            }
                        )
                    else:
                        unit -= 1
                        lost += 1
                        all_bets.append(
                            {
                                "datetime": match_datetime,
                                "result": 0,
                                "odd": -1
                            }
                        )
                    all += 1

                for match in not_duplicates_away:
                    # Pobieranie danych meczu
                    match_date = match["datetime"]
                    home_name = match["home_name"]
                    away_name = match["away_name"]
                    found = False

                    if (match["bet"] == 2 and match["p1_away_prob"] > p_a):
                        found = True

                    # if (match["bet"] == 2 and match["p1_away_prob"] > p_a):
                    #     found = True
                    # elif (match["bet"] == 2 and match["p1_away_prob"] <= 0.98 and match["diff_last_tournaments"] > 4):
                    #     found = True
                    # elif (match["bet"] == 2 and match["p1_away_prob"] <= 0.98 and match["experience"] < 0):
                    #     found = True
                    # elif (match["bet"] == 2 and match["p1_away_prob"] <= 0.98 and match['diff_players_vs_opponents_50'] < -50):
                    #     found = True

                    if not found:
                        continue

                    # Szukanie rekordu w `betting_collection` pasującego do meczu z tolerancją +/- 10 minut
                    if isinstance(match_date, str):
                        match_datetime = datetime.strptime(match_date, "%Y:%m:%d %H:%M")
                    else:
                        match_datetime = match_date
                    #if match_datetime.month <2 or match_datetime.month == 12:
                    #    continue
                    time_range_start = match_datetime - timedelta(minutes=120)
                    time_range_end = match_datetime + timedelta(minutes=120)

                    if "created_time" in match:
                        if match["created_time"]:
                            time_range_start = match["created_time"]

                    match_database_record = self.matches_coll.find_one({
                        "home_name": home_name,
                        "away_name": away_name,
                        "datetime": {"$gte": time_range_start, "$lte": time_range_end}
                    })

                    if not match_database_record:
                        #print(f"[WARNING] Could not find matching record for: {home_name} vs {away_name}. Skipping.")
                        continue

                    ft_result = match_database_record["ft_result"]
                    #print(f"[INFO] Matching betting record found for match: {home_name} vs {away_name}.")
                    a = int(ft_result.strip().split(":")[0].replace(" ", ""))
                    b = int(ft_result.strip().split(":")[1].replace(" ", ""))
                    if a > b:
                        winner = 1
                    else:
                        winner = 2
                    
                    if winner == match['bet']:
                        unit += (match['odd_away']) - 1
                        kursy.append(match['odd_away'])
                        won += 1
                        all_bets.append(
                            {
                                "datetime": match_datetime,
                                "result": 1,
                                "odd": match['odd_away']
                            }
                        )
                    else:
                        unit -= 1
                        lost += 1
                        all_bets.append(
                            {
                                "datetime": match_datetime,
                                "result": 0,
                                "odd": -1
                            }
                        )
                    all += 1
                print(f'Home p:{round(p_h,2)}, Away p:{round(p_a,2)}\n Wins:{won}, Loses:{lost}\n Units:{round(unit,2)}, Yield:{round(unit/(won+lost),2)}, Accuracy:{round(won/(won+lost),2)}\n\n')

        # # Stawka (dla uproszczenia zakładamy, że stawka to 1 jednostka)
        # stake = 1

        # # Obliczamy zyski dla każdego dnia
        # ako_bets = {}
        # total_profit = 0
        # total_bets = 0
        # total_wins = 0
        # total_losses = 0
        # # Grupujemy zakłady po dacie
        # grouped_bets = group_bets_by_date(all_bets)

        # for date, bets_on_date in grouped_bets.items():
        #     ako2_combinations = generate_symmetric_pairs(bets_on_date)  # Tworzymy system symetryczny
            
        #     for bet1, bet2 in ako2_combinations:
        #         odd1 = bet1['odd']
        #         odd2 = bet2['odd']
        #         result1 = bet1['result']
        #         result2 = bet2['result']
                
        #         profit = calculate_profit(odd1, odd2, result1, result2)
        #         total_profit += profit
                
        #         if profit > 0:
        #             total_wins += 1
        #         else:
        #             total_losses += 1

        # total_bets = total_wins + total_losses
        # accuracy = (total_wins / total_bets) * 100 if total_bets > 0 else 0
        # yield_value = total_profit / total_bets if total_bets > 0 else 0

        # # Wyświetlamy wyniki
        # print(f"Całkowity zysk: {total_profit:.2f} jednostek")
        # print(f"Accuracy: {accuracy:.2f}%")
        # print(f"Yield: {yield_value:.2f} jednostek na zakład")


        srednia = sum(kursy) / len(kursy)
        s = 1

    async def update_finished_matches_value_three(self, betting_collection):
        #region Update Telegram 2
        """
        Updates the results of finished matches in `betting_ai_three` and edits the corresponding Telegram messages.

        Returns:
            None
        """
        betting_coll = list(betting_collection.find())


        bot = Bot(token=self.bot_token)
        won = 0
        lost = 0
        unit = 0
        all = 0
        kursy = []


        all_bets = []
        for p_h in [0.9]: #np.arange(0.80, 0.95, 0.05):  # Jeden krok dla 0.8
            for p_a in [0.75]: #np.arange(0.70, 0.95, 0.05):  # Zakres od 0.70 do 0.95               
                for match in betting_coll:
                    # Pobieranie danych meczu
                    match_date = match["datetime"]
                    home_name = match["home_name"]
                    away_name = match["away_name"]
                    found = False

                    if (match["bet"] == 1 and match["p0_home_prob"] > p_h) or (match["bet"] == 2 and match["p1_away_prob"] > p_a):
                        if (match["bet"] == 1 and match["p0_home_prob"] > match["p1_away_prob"] + 0.3) or (match["bet"] == 2 and match["p0_home_prob"] +0.3 < match["p1_away_prob"]):
                            found = True


                    # if (match["bet"] == 1 and match["p0_home_prob"] > p_h): # avg_opponents_home
                    #     found = True
                    # elif (match["bet"] == 1 and match["p0_home_prob"] <= 0.98 and match["diff_last_tournaments"] < -4):
                    #     found = True
                    # elif (match["bet"] == 1 and match["p0_home_prob"] <= 0.98 and match["experience"] > 0):
                    #     found = True
                    # elif (match["bet"] == 1 and match["p0_home_prob"] <= 0.98 and match['diff_saved'] < -0.06):
                    #     found = True #diff_players_vs_opponents_50
                    # elif (match["bet"] == 1 and match["p0_home_prob"] <= 0.98 and match['diff_players_vs_opponents_50'] > 50):
                    #     found = True 

                    if not found:
                        continue

                    # Szukanie rekordu w `betting_collection` pasującego do meczu z tolerancją +/- 10 minut
                    if isinstance(match_date, str):
                        match_datetime = datetime.strptime(match_date, "%Y:%m:%d %H:%M")
                    else:
                        match_datetime = match_date
                    #if match_datetime.month <2 or match_datetime.month == 12:
                    #    continue
                    time_range_start = match_datetime - timedelta(minutes=120)
                    time_range_end = match_datetime + timedelta(minutes=120)

                    if "created_time" in match:
                        if match["created_time"]:
                            time_range_start = match["created_time"]

                    match_database_record = self.matches_coll.find_one({
                        "home_name": home_name,
                        "away_name": away_name,
                        "datetime": {"$gte": time_range_start, "$lte": time_range_end}
                    })

                    if not match_database_record:
                        #print(f"[WARNING] Could not find matching record for: {home_name} vs {away_name}. Skipping.")
                        continue

                    ft_result = match_database_record["ft_result"]
                    #print(f"[INFO] Matching betting record found for match: {home_name} vs {away_name}.")
                    a = int(ft_result.strip().split(":")[0].replace(" ", ""))
                    b = int(ft_result.strip().split(":")[1].replace(" ", ""))
                    if a > b:
                        winner = 1
                    else:
                        winner = 2
                    
                    if winner == match['bet']:
                        if winner == 1:
                            unit += (match['odd_home']) - 1
                            kursy.append(match['odd_home'])
                            won += 1
                            all_bets.append(
                                {
                                    "datetime": match_datetime,
                                    "result": 1,
                                    "odd": match['odd_home']
                                }
                            )
                        else:
                            unit += (match['odd_away']) - 1
                            kursy.append(match['odd_away'])
                            won += 1
                            all_bets.append(
                                {
                                    "datetime": match_datetime,
                                    "result": 1,
                                    "odd": match['odd_away']
                                }
                            )                            
                    else:
                        unit -= 1
                        lost += 1
                        all_bets.append(
                            {
                                "datetime": match_datetime,
                                "result": 0,
                                "odd": -1
                            }
                        )
                    all += 1
                print(f'Home p:{round(p_h,2)}, Away p:{round(p_a,2)}\n Wins:{won}, Loses:{lost}\n Units:{round(unit,2)}, Yield:{round(unit/(won+lost),2)}, Accuracy:{round(won/(won+lost),2)}\n\n')

        # # Stawka (dla uproszczenia zakładamy, że stawka to 1 jednostka)
        # stake = 1

        # # Obliczamy zyski dla każdego dnia
        # ako_bets = {}
        # total_profit = 0
        # total_bets = 0
        # total_wins = 0
        # total_losses = 0
        # # Grupujemy zakłady po dacie
        # grouped_bets = group_bets_by_date(all_bets)

        # for date, bets_on_date in grouped_bets.items():
        #     ako2_combinations = generate_symmetric_pairs(bets_on_date)  # Tworzymy system symetryczny
            
        #     for bet1, bet2 in ako2_combinations:
        #         odd1 = bet1['odd']
        #         odd2 = bet2['odd']
        #         result1 = bet1['result']
        #         result2 = bet2['result']
                
        #         profit = calculate_profit(odd1, odd2, result1, result2)
        #         total_profit += profit
                
        #         if profit > 0:
        #             total_wins += 1
        #         else:
        #             total_losses += 1

        # total_bets = total_wins + total_losses
        # accuracy = (total_wins / total_bets) * 100 if total_bets > 0 else 0
        # yield_value = total_profit / total_bets if total_bets > 0 else 0

        # # Wyświetlamy wyniki
        # print(f"Całkowity zysk: {total_profit:.2f} jednostek")
        # print(f"Accuracy: {accuracy:.2f}%")
        # print(f"Yield: {yield_value:.2f} jednostek na zakład")


        srednia = sum(kursy) / len(kursy)
        s = 1


    async def calculate_and_notify_roi(self, betting_collection, n):
        #region Notify ROI Telegram
        """
        Calculates the ROI for the last `n` matches marked as `edited = True` and sends the result via Telegram.
        Only sends a message if there are at least `n` new matches.
        """
        # Step 1: Fetch the last matches where `edited = True`
        # Step 1: Fetch the last matches where `edited = True` and odds >= 1.8
        query = {
            "processed": True,
            "edited": True,
            "$or": [
                {"bet": 1, "odd_home": {"$gte": 1.7, "$lte": 9.99}},
                {"bet": 2, "odd_away": {"$gte": 1.7, "$lte": 9.99}}
            ]
        }


        #query = {"processed": True, "edited": True}
        if self.last_processed_match_id:
            query["_id"] = {"$gt": self.last_processed_match_id}  # Only fetch matches with IDs greater than the last processed

        filtered_matches = list(betting_collection.find(query).sort("_id", DESCENDING).limit(n))

        # Ensure we have at least `n` matches to process
        if len(filtered_matches) < n:
            print(f"[INFO] Not enough new matches found for ROI calculation. Found: {len(filtered_matches)}, Required: {n}.")
            return

        # Step 2: Initialize variables to calculate ROI
        total_bets = len(filtered_matches)
        total_investment = 0
        total_returns = 0

        player_outcomes = []  # To store the outcomes for each player

        for match in filtered_matches:
            bet = match["bet"]
            ft_result = match.get("ft_result")  # Final result in format "X : Y"

            if not ft_result:
                print(f"[WARNING] Match {match['_id']} has no final result. Skipping.")
                continue

            # Parse the final result to determine the winner
            try:
                home_score, away_score = map(int, ft_result.split(" : "))
            except ValueError:
                print(f"[ERROR] Invalid ft_result format for match {match['_id']}: {ft_result}. Skipping.")
                continue

            # Determine if the bet was successful
            winner = 1 if home_score > away_score else 2
            stake = 1  # Assuming a constant stake for all bets

            if winner == bet:
                # Bet was successful, calculate returns based on odds
                if winner == 1:
                    total_returns += match["odd_home"] * stake
                    player_outcomes.append(f"{match['home_name']} @{match['odd_home']} ✅")
                elif winner == 2:
                    total_returns += match["odd_away"] * stake
                    player_outcomes.append(f"{match['away_name']} @{match['odd_away']} ✅")
            else:
                # Bet was unsuccessful
                if bet == 1:
                    player_outcomes.append(f"{match['home_name']} @{match['odd_home']} ❌")
                elif bet == 2:
                    player_outcomes.append(f"{match['away_name']} @{match['odd_away']} ❌")

            # Add the stake to the total investment
            total_investment += stake

        # Step 3: Update the last processed match ID
        self.last_processed_match_id = filtered_matches[0]["_id"]  # Update to the most recent match ID

        # Step 4: Calculate ROI
        roi = ((total_returns - total_investment) / total_investment) * 100 if total_investment > 0 else 0

        # Step 5: Prepare detailed player outcomes
        player_outcomes_message = "\n".join(player_outcomes)
        print(message)
        # Step 6: Send ROI report to Telegram
        message = (
            f"📊 ROI Report for Last {n} Matches\n\n"
            f"📈 Total Matches: {total_bets}\n"
            f"💳 Total Investment: {total_investment:.2f} units\n"
            f"💰 Total Returns: {total_returns:.2f} units\n"
            f"🌐 ROI: {roi:.2f}%\n\n"
            f"Player Outcomes:\n\n{player_outcomes_message}"
        )

        try:
            async with self.bot:
                await self.bot.send_message(chat_id=self.chat_id, text=message)
            print("[INFO] ROI report sent to Telegram.")
        except TelegramError as e:
            print(f"[ERROR] Failed to send ROI report to Telegram: {e}")
        except Exception as e:
            print(f"[ERROR] Unexpected error: {e}")

    def get_last_matches_of_player(self, player_name):
        # Wyszukiwanie ostatniego meczu gracza (sprawdzamy zarówno home_name, jak i away_name)
        #region Last_Matches_Player
        last_match = self.matches_coll.find({
            "$or": [
                {"home_name": player_name},
                {"away_name": player_name}
            ],
            "datetime": {
                "$lt": self.match["datetime"]
            }
        }).sort("datetime", -1).limit(1)

        # Uzyskanie daty ostatniego meczu
        last_match_date = last_match[0]["datetime"].date()
        # Przygotowujemy zapytanie MongoDB
        query = {
            "$or": [
                {"home_name": player_name},  # Zastąp 'Gracz Nazwa' nazwą gracza
                {"away_name": player_name}   # Zastąp 'Gracz Nazwa' nazwą gracza
            ],
            "datetime": {
                "$gte": datetime.combine(last_match_date, datetime.min.time()),
                "$lt": self.match["datetime"]
            }
        }

        # Wykonujemy zapytanie
        matches = self.matches_coll.find(query)

        return matches

    def get_actual_player_elo(self, player_name, matches):
        #region Actual_Player_Elo
        try:
            # Jeśli dane nie są listą, przekonwertuj na listę
            if not isinstance(matches, list):
                #print(f"[DEBUG] Matches is not a list. Converting to list. Current value: {matches}")
                matches = [matches]  # Konwertuj na listę z jednym elementem

            actual_elo = 0
            delta_sum = 0

            for match in matches:
                # Walidacja: czy match jest słownikiem
                if not isinstance(match, dict):
                    print(f"[ERROR] Invalid match data. Expected dict, got {type(match)}: {match}")
                    continue

                # Walidacja: czy match zawiera wymagane klucze
                required_keys = {"home_name", "away_name", "home_elo", "away_elo", "delta_elo", "match_category"}
                if not required_keys.issubset(match.keys()):
                    print(f"[ERROR] Missing keys in match: {match}")
                    continue

                # Logika obliczeń
                if match["home_name"] == player_name:
                    actual_elo = match["home_elo"]
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
                    actual_elo = match["away_elo"]
                    if match["match_category"] == "final":
                        if match["delta_elo"] > 0:
                            delta_sum += 0.6
                        else:
                            delta_sum += 0.8
                    if match["match_category"] == "3rd":
                        if match["delta_elo"] < 0:
                            delta_sum += 0.4

            return actual_elo + delta_sum

        except Exception as e:
            print(f"[ERROR] Exception in get_actual_player_elo: {str(e)}")
            return 0


    # Funkcja do wysyłania powiadomienia na Telegram
    async def send_message(self, message):
        message["datetime"] = message["datetime"].strftime("%Y-%m-%d %H:%M")
        bot = Bot(token=self.bot_token)
        a = json.dumps(message)
        a = a.replace(r"{", "\\{").replace("}", "\\}")
        a = r'```json\n' + a + '\n```'             
        try:
            # Wysyłanie wiadomości
            response = await bot.send_message(
                chat_id=self.chat_id, 
                text=a,
                parse_mode='MarkdownV2'
            )
        except TelegramError as e:
            print(f"Błąd podczas wysyłania wiadomości: {e}")

    def compare_experience(self, home_cumulative_elo, away_cumulative_elo):
        #region Experience vs Noob
        """
        Compares the experience level of two players based on the total number of matches played.

        This method uses the number of recent matches for both
        the home player and the away player. A threshold of 100 matches is used to classify players as experienced or not.

        Args:
            home_cumulative_elo (list): List of the home player recent matches.
            away_cumulative_elo (list): List of the away player recent matches.

        Returns:
            int: A value indicating the comparison of experience levels:
                - 1: Home player is experienced and away player is inexperienced.
                - -1: Home player is inexperienced and away player is experienced.
                - 0: Both players have the same experience level (either both are experienced or both are inexperienced).
        """


        # Determine the experience level based on the number of matches
        home_experience = 1 if len(home_cumulative_elo) > 100 else 0
        away_experience = 1 if len(away_cumulative_elo) > 100 else 0

        # Calculate the experience comparison (home - away)
        return home_experience - away_experience
    

    def check_dominance(self, home_cumulative_elo, away_cumulative_elo):
        #region Check Dominance
        """
        Determines if one player's ELO progression chart is above the other's in at least 85% of the matches.

        This method uses the cumulative ELO data for both players
        and compares the values point-by-point. Skips the comparison if either player has fewer than 100 matches overall.

        Args:
            home_cumulative_elo (list): List of the home player recent matches.
            away_cumulative_elo (list): List of the away player recent matches.

        Returns:
            float, 2 values percentages: 
        """

        # Skip if either player has fewer than 100 matches
        #if len(home_cumulative_elo) <= 100 or len(away_cumulative_elo) <= 100:
        #    return 0,0

        # Ensure both have the same number of matches to compare
        min_length = min(len(home_cumulative_elo), len(away_cumulative_elo))
        home_cumulative_elo = home_cumulative_elo[:min_length]
        away_cumulative_elo = away_cumulative_elo[:min_length]


        # Count dominance points
        home_dominance = sum(h > a for h, a in zip(home_cumulative_elo, away_cumulative_elo))
        away_dominance = sum(a > h for h, a in zip(home_cumulative_elo, away_cumulative_elo))
        total_points = len(home_cumulative_elo)

        # Calculate dominance percentages
        home_dominance_percentage = home_dominance / total_points
        away_dominance_percentage = away_dominance / total_points

        return home_dominance_percentage, away_dominance_percentage



    def big_wins_losses(self, home_cumulative_elo, away_cumulative_elo):
        #region Big_Wins_Losses
        """
        Calculates the number of big wins and big losses for both players based on recent matches.

        Big wins are defined as positive delta elo values >= 8.
        Big losses are defined as negative delta elo values whose absolute value is >= 8.

        This method uses the data returned by the `get_combined_form_chart` method to avoid re-querying the database.

        Args:
            home_cumulative_elo (list): List of the home player recent matches.
            away_cumulative_elo (list): List of the away player recent matches.

        Returns:
            dict: A dictionary with big wins and losses for both players:
                {
                    'home': (n_big_wins_home, n_big_losses_home),
                    'away': (n_big_wins_away, n_big_losses_away)
                }
        """
        # Define the threshold for big wins and big losses
        threshold = 8  # Fixed threshold value

        # Extract raw delta elo values for home player
        home_delta_elo_values = [
            home_cumulative_elo[i] - home_cumulative_elo[i - 1] for i in range(1, len(home_cumulative_elo))
        ]

        # Extract raw delta elo values for away player
        away_delta_elo_values = [
            away_cumulative_elo[i] - away_cumulative_elo[i - 1] for i in range(1, len(away_cumulative_elo))
        ]

        # Count big wins and big losses for home player
        n_big_wins_home = sum(delta >= threshold for delta in home_delta_elo_values if delta > 0)
        n_big_losses_home = sum(abs(delta) >= threshold for delta in home_delta_elo_values if delta < 0)


        # Count big wins and big losses for away player
        n_big_wins_away = sum(delta >= threshold for delta in away_delta_elo_values if delta > 0)
        n_big_losses_away = sum(abs(delta) >= threshold for delta in away_delta_elo_values if delta < 0)

        return int(n_big_wins_home), int(n_big_wins_away), int(n_big_losses_home), int(n_big_losses_away)


    def analyze_sets(self, json_data):
        #region analyze_sets
        """
        Analizuje wynik jednego meczu, licząc comebacki i choke'y w setach.
        Args:
            json_data (dict): Dane pojedynczego meczu w formacie JSON.
        Returns:
            dict: Wyniki w formacie:
                {"home": {"3+": [comeback_count, choke_count], ...},
                "away": {"3+": [comeback_count, choke_count], ...}}.
        """

        def calculate_running_score(set_scores):
            """
            Oblicza bieżący wynik na podstawie listy punktów.
            Args:
                set_scores (list): Lista punktów w formacie ['A13', 'B8', ...].
            Returns:
                list: Lista bieżących wyników w formacie [(A_score, B_score), ...].
            """
            a_score = 0
            b_score = 0
            running_scores = []

            for score in set_scores:
                if 'A' in score:
                    a_score += 1
                elif 'B' in score:
                    b_score += 1
                running_scores.append((a_score, b_score))

            return running_scores

        def count_comebacks_and_chokes(scores, target_player):
            #region Count_comeback
            """
            Liczy przypadki, w których gracz:
            - Przegrywał 3+, 4+, 5+, 6+ punktami i wygrał seta.
            - Prowadził 3+, 4+, 5+, 6+ punktami i przegrał seta.
            Uwzględnia tylko największą przewagę w każdej kategorii.
            
            Args:
                scores (list): Lista wyników [(A_score, B_score), ...].
                target_player (str): 'A' lub 'B' - gracz, którego wyniki analizujemy.
            Returns:
                dict: Wyniki w formacie {"3+": [comeback_count, choke_count], "4+": [...], ...}.
            """
            results = {"3+": [0, 0], "4+": [0, 0], "5+": [0, 0], "6+": [0, 0]}
            target_index = 0 if target_player == 'A' else 1
            other_index = 1 if target_player == 'A' else 0

            for lead in [3, 4, 5, 6]:
                max_deficit = 0  # Największa strata target_player
                max_lead = 0     # Największa przewaga target_player

                for target_score, other_score in scores:
                    deficit = other_score - target_score
                    lead_advantage = target_score - other_score

                    # Śledzenie największej straty (deficytu)
                    if deficit >= lead:
                        max_deficit = max(max_deficit, deficit)

                    # Śledzenie największej przewagi
                    if lead_advantage >= lead:
                        max_lead = max(max_lead, lead_advantage)

                # Jeśli target_player odrobił największy deficyt i wygrał seta
                final_score = scores[-1]
                if max_deficit >= lead and final_score[target_index] > final_score[other_index]:
                    results[f"{lead}+"][0] += 1

                # Jeśli target_player stracił największą przewagę i przegrał seta
                if max_lead >= lead and final_score[target_index] < final_score[other_index]:
                    results[f"{lead}+"][1] += 1

            return results
        
        def count_set_points(running_scores, target_player):
            results = {"saved_set_points": [0], "wasted_set_points": [0]}
            saved_set_points = 0
            wasted_set_points = 0
            target_index = 0 if target_player == 'A' else 1
            other_index = 1 if target_player == 'A' else 0

            for i in range(len(running_scores) - 1):
                if running_scores[i][target_index] >= 10 and running_scores[i][target_index] > running_scores[i][other_index]:
                    if running_scores[i+1][target_index] == running_scores[i+1][other_index]:
                        wasted_set_points += 1
                if running_scores[i][other_index] >= 10 and running_scores[i][target_index] < running_scores[i][other_index]:
                    if running_scores[i+1][target_index] == running_scores[i+1][other_index]:
                        saved_set_points += 1

            results['saved_set_points'][0] = saved_set_points
            results['wasted_set_points'][0] = wasted_set_points
                
            return results


        # Obliczanie wyników dla meczu
        results = {
            "home": {"3+": [0, 0], "4+": [0, 0], "5+": [0, 0], "6+": [0, 0]},
            "away": {"3+": [0, 0], "4+": [0, 0], "5+": [0, 0], "6+": [0, 0]},
        }

        results_sets_points = {
            'home': {"saved_set_points": [0], "wasted_set_points": [0]},
            'away': {"saved_set_points": [0], "wasted_set_points": [0]} 
        }

        sets = [json_data.get(f"Set{i}") for i in range(1, 6)]
        home_player = 'A'
        away_player = 'B'

        for set_scores in sets:
            if set_scores is None:
                continue

            running_scores = calculate_running_score(set_scores)
            home_results = count_comebacks_and_chokes(running_scores, home_player)
            away_results = count_comebacks_and_chokes(running_scores, away_player)
            home_sets_points = count_set_points(running_scores, home_player)
            away_sets_points = count_set_points(running_scores, away_player)

            for lead_type in results["home"]:
                results["home"][lead_type][0] += home_results[lead_type][0]
                results["home"][lead_type][1] += home_results[lead_type][1]
                results["away"][lead_type][0] += away_results[lead_type][0]
                results["away"][lead_type][1] += away_results[lead_type][1]

            for lead_type in results_sets_points["home"]:
                results_sets_points["home"][lead_type][0] += home_sets_points[lead_type][0]
                results_sets_points["away"][lead_type][0] += away_sets_points[lead_type][0]

        return results, results_sets_points


    def analyze_last_50_matches_for_player(self, player_name):
        #region Analyze last 50
        """
        Analizuje ostatnie 50 meczów danego gracza.
        Liczy średnią liczbę comebacków i choke'ów w setach, 
        grupując wyniki według przewag 3+, 4+, 5+, 6+.
        Dodatkowo zlicza 'saved_set_points' i 'wasted_set_points',
        normalizując obie wartości "na set".
        Args:
            player_name (str): Nazwa analizowanego gracza.
        Returns:
            tuple: (analyzed_results, analyzed_results_sets_points) w formacie:
                analyzed_results = {"3+": [średnia_comeback, średnia_choke], ...}
                analyzed_results_sets_points = {"saved_set_points": [średnia], "wasted_set_points": [średnia]}
        """
        matches = list(
            self.matches_coll.find(
            {
                "$or": [
                    {"home_name": player_name},
                    {"away_name": player_name}
                ],
                "datetime": {
                    "$lt": self.match["datetime"]
                }
            })
            .sort("datetime", -1)
            .limit(50)
        )
        
        # Słownik na comebacki/choke'y
        analyzed_results = {
            "3+": [0, 0],
            "4+": [0, 0],
            "5+": [0, 0],
            "6+": [0, 0]
        }

        # Słownik na punkty setowe
        analyzed_results_sets_points = {
            "saved_set_points": [0],
            "wasted_set_points": [0] 
        }

        # Zmienna na liczbę wszystkich *rozegranych* setów
        played_sets = 0

        for match in matches:
            # Uzyskujemy wyniki z analyze_sets
            result, result_sets_points = self.analyze_sets(match)

            # Określamy, czy badany gracz był po stronie 'home' czy 'away'
            if match["home_name"] == player_name:
                player_results = result["home"]
                player_results_sets = result_sets_points["home"]
            else:
                player_results = result["away"]
                player_results_sets = result_sets_points["away"]

            # Sumujemy comebacki i choke’y (3+, 4+, 5+, 6+)
            for lead in ["3+", "4+", "5+", "6+"]:
                analyzed_results[lead][0] += player_results[lead][0]  # comebacki
                analyzed_results[lead][1] += player_results[lead][1]  # choke'y

            # Sumujemy set points (saved / wasted)
            analyzed_results_sets_points["saved_set_points"][0] += player_results_sets["saved_set_points"][0]
            analyzed_results_sets_points["wasted_set_points"][0] += player_results_sets["wasted_set_points"][0]

            # Zliczamy faktycznie rozegrane sety w tym meczu
            for i in range(1, 6):
                if match.get(f"Set{i}") is not None:
                    played_sets += 1

        # Jeśli rozegrano jakiekolwiek sety, normalizujemy (dzielimy) wszystko "na set"
        if played_sets > 0:
            # Normalizacja comebacków/choke’ów
            analyzed_results = {
                lead: [count / played_sets for count in values]
                for lead, values in analyzed_results.items()
            }
            # Normalizacja 'saved_set_points' i 'wasted_set_points'
            analyzed_results_sets_points["saved_set_points"][0] /= played_sets
            analyzed_results_sets_points["wasted_set_points"][0] /= played_sets


        return analyzed_results, analyzed_results_sets_points["saved_set_points"][0], analyzed_results_sets_points["wasted_set_points"][0]



    def get_days_since_last_match(self, player_name):
        #region Days_Diff_Last_Match
        """
        Returns the time in days (as a float) since the last match of the given player.

        Args:
            player_name (str): The name of the player to check.

        Returns:
            float: Days since the last match, or None if no matches found.
        """
        # Find the most recent match for the player
        last_match = self.matches_coll.find_one(
            {
                "$or": [
                    {"home_name": player_name},
                    {"away_name": player_name}
                ],
                "datetime": {
                    "$lt": self.match["datetime"]
                }
            },
            sort=[("datetime", DESCENDING)]
        )
        
        if not last_match:
            return None  # No matches found for the player

        # Calculate the difference in days
        last_match_time = last_match["datetime"]
        now = datetime.utcnow()
        time_difference = abs(now - last_match_time)
        days_difference = time_difference.total_seconds() / (24 * 3600)  # Convert seconds to days
        
        return round(days_difference, 2)

    def get_average_games_last_week(self, player_name):
        #region Av_Games_Last_Week
        """
        Calculates the average number of games per day for a given player over the last 7 days.

        Args:
            player_name (str): The name of the player.

        Returns:
            float: Average number of games per day in the last 7 days, or 0 if no games were found.
        """
        # Define the time range (last 7 days)
        #now = datetime.utcnow()
        now = self.match["datetime"]
        seven_days_ago = now - timedelta(days=7)

        # Find all matches for the player in the last 7 days
        matches = self.matches_coll.find({
            "$or": [
                {"home_name": player_name},
                {"away_name": player_name}
            ],
            "datetime": {
                "$gte": seven_days_ago,
                "$lt": now
            }
        })

        # Use len() to count the total number of matches
        total_games = len(list(matches))

        # Calculate the average games per day
        average_games_per_day = total_games / 7 if total_games > 0 else 0

        return round(average_games_per_day, 2)

    def calculate_h2h_last_three_months(self, player_a, player_b):
        #region H2H_Last_3months
        """
        Calculates the Head-to-Head (H2H) record of two players over the last 3 months.

        Args:
            player_a (str): Name of player A.
            player_b (str): Name of player B.

        Returns:
            tuple: A tuple containing two floats: (win_ratio_player_a, win_ratio_player_b),
                   or (0, 0) if no matches were found.
        """
        # Define the time range (last 3 months)
        #now = datetime.utcnow()
        now = self.match["datetime"]
        three_months_ago = now - timedelta(days=90)

        # Find all matches between the two players in the last 3 months
        matches = self.matches_coll.find({
            "$or": [
                {"home_name": player_a, "away_name": player_b},
                {"home_name": player_b, "away_name": player_a}
            ],
            "datetime": {
                "$gte": three_months_ago,
                "$lt": now
            }
        })

        # Initialize counters
        wins_player_a = 0
        wins_player_b = 0
        total_matches = 0

        # Count wins for each player
        for match in matches:
            home_score, away_score = map(int, match["ft_result"].split(" : ")) 

            winner = 1 if home_score > away_score else 0
            total_matches += 1
            if winner and match.get("home_name") == player_a:
                wins_player_a += 1
            elif winner == 0 and match.get("away_name") == player_a:
                wins_player_a += 1
            elif winner and match.get("home_name") == player_b:
                wins_player_b += 1
            elif winner == 0 and match.get("away_name") == player_b:
                wins_player_b += 1

        # Calculate ratios
        if total_matches == 0:
            return 0,0,0,0 # No matches found
        
        diff_h2h = wins_player_a - wins_player_b

        return diff_h2h, total_matches, wins_player_a/total_matches, wins_player_b/total_matches

    def calculate_elo_diff(self, player_a_name, player_b_name):
        #region Elo_diff
        """
        Calculates the ELO difference between two players with dynamic scaling.
        The significance of the ELO difference increases with the average ELO of the players.

        Args:
            player_a_name (str): Name of player A.
            player_b_name (str): Name of player B.

        Returns:
            tuple or None: A tuple containing (elo_diff, elo_diff_scaled, elo_a, elo_b) or None if calculation fails.
        """
        try:
            # Get last matches for both players
            matches_a = list(self.get_last_matches_of_player(player_a_name))  # Convert cursor to list
            matches_b = list(self.get_last_matches_of_player(player_b_name))  # Convert cursor to list

            # Check if matches were retrieved
            if len(matches_a) == 0:
                print(f"[DEBUG] (calculate_elo_diff - matches_a) No matches found for player: {player_a_name}")
                return None
            if len(matches_b) == 0:
                print(f"[DEBUG] (calculate_elo_diff - matches_b)No matches found for player: {player_b_name}")
                return None

            # Get actual ELO values for both players
            elo_a = self.get_actual_player_elo(player_a_name, matches_a)
            elo_b = self.get_actual_player_elo(player_b_name, matches_b)

            # Validate retrieved ELO values
            if elo_a is None or elo_b is None:
                print(f"[DEBUG] Could not calculate actual ELO for players: {player_a_name} or {player_b_name}")
                return None

            # Calculate absolute ELO difference
            elo_diff = abs(elo_a - elo_b)

            # Calculate average ELO between the two players
            avg_elo = (elo_a + elo_b) / 2

            # Apply dynamic scaling with subtler effect
            scaling_factor = 1 + (avg_elo - 300) / 1900  # Subtle scaling between 1.0 and ~1.5

            # Scale the ELO difference
            elo_diff_scaled = elo_diff * scaling_factor

            return round(elo_diff, 2), round(elo_diff_scaled, 2), elo_a, elo_b

        except Exception as e:
            print(f"[ERROR] Exception occurred while calculating ELO difference:")
            print(f"Player A: {player_a_name}, Player B: {player_b_name}")
            print(f"Error: {str(e)}")
            return None, None, None, None

    

    def get_last_50_matches(self, player_name):
        #region Get_last_50_player
        """
        Retrieves the last 50 matches of the specified player.

        Args:
            player_name (str): The name of the player.

        Returns:
            list: A list of the last 50 match documents involving the player.
        """
        try:
            # Debugging: Log the player name and intended query
            #print(f"[DEBUG] Fetching last 50 matches for player: {player_name}")
            
            query = {
                "$or": [
                    {"home_name": player_name},
                    {"away_name": player_name}
                ],
                "datetime": {
                    "$lt": self.match["datetime"]
                }
            }
            
            #print(f"[DEBUG] MongoDB query: {query}")

            # Query MongoDB to find matches where the player participated
            matches_cursor = self.matches_coll.find(query).sort("datetime", -1).limit(50)

            # Convert the cursor to a list of match documents
            matches = list(matches_cursor)

            # Debugging: Check connection to the collection
            if self.matches_coll is None:
                print("[ERROR] MongoDB collection `matches_coll` is not initialized!")
                return []

            # Debugging: Log the number of matches retrieved
            #print(f"[DEBUG] Number of matches retrieved: {len(matches)}")
            #print(f"[DEBUG] Matches retrieved: {matches}")
            
            # If fewer than 50 matches, log the results
            #if len(matches) < 50:
                #print(f"[DEBUG] Retrieved less than 50 matches. Total matches found: {len(matches)}")
                #print(f"[DEBUG] Matches retrieved: {matches}")

            return matches

        except Exception as e:
            # Handle and log any unexpected errors
            print(f"[ERROR] Exception occurred while retrieving last 50 matches for player: {player_name}")
            print(f"Error details: {str(e)}")
            return []



    def get_matches_history_of_player(self, player_name, match_datetime, days_back=365):
        #region Matches_history
        try:
            # Dolny i górny limit dat
            days_before = match_datetime - timedelta(days=days_back)

            match_filter = {
                "$or": [
                    {"home_name": player_name},
                    {"away_name": player_name}
                ],
                "datetime": {
                    "$gte": days_before,
                    "$lt": match_datetime
                }
            }

            projection = {
                "datetime": 1,
                "league_name": 1,
                "home_name": 1,
                "away_name": 1,
                "home_elo": 1,
                "away_elo": 1,
                "delta_elo": 1,
                "match_category": 1
            }

            sort = [("datetime", 1)]

            # Debugging
            #print(f"[DEBUG] MongoDB filter: {match_filter}")
            #print(f"[DEBUG] Projection: {projection}")

            # Pobranie danych z MongoDB
            raw_matches = list(self.matches_coll.find(match_filter, projection).sort(sort))
            #print(f"[DEBUG] matches found for player: {raw_matches} within timeframe.")

            if not raw_matches:
                #print(f"[DEBUG] (get_matches_history_of_player) No matches found for player: {player_name} within timeframe.")
                return []

            grouped_matches = {}
            for match in raw_matches:
                if "datetime" not in match or "league_name" not in match:
                    #print(f"[ERROR] Match data is missing critical fields: {match}")
                    continue
                #print(match)
                match_date = match["datetime"].strftime("%Y-%m-%d")
                league_name = match["league_name"]
                key = (match_date, league_name)

                if key not in grouped_matches:
                    grouped_matches[key] = []
                grouped_matches[key].append(match)

            result = [
                {
                    "_id": {"date": key[0], "league_name": key[1]},
                    "matches": matches
                }
                for key, matches in grouped_matches.items()
            ]

            # Debugging
            #print(f"[DEBUG] Grouped matches: {result}")
            return result

        except Exception as e:
            print(f"[ERROR] Exception in get_matches_history_of_player: {str(e)}")
            return []



    def calculate_elo_based_on_history(self, player_name, match_datetime, days_back=40):
        #region ELo_History
        try:
            # Pobranie historii gracza
            player_history = self.get_matches_history_of_player(player_name, match_datetime, days_back)
            #print("[DEBUG] Player history:", player_history)
            
            if not player_history:
                #print(f"[DEBUG] No match history found for player: {player_name}")
                return 0

            history = []
            for group in player_history:
                if not group.get('matches') or not isinstance(group['matches'], list):
                    print(f"[DEBUG] Invalid matches data in group: {group}")
                    continue

                # Walidacja danych w pierwszym meczu grupy
                first_match = group['matches'][0]
                if not all(key in first_match for key in ["home_name", "away_name", "home_elo", "away_elo"]):
                    print(f"[DEBUG] Missing keys in first match data: {first_match}")
                    continue

                #print(first_match)
                # Sprawdzenie, czy gracz jest gospodarzem czy gościem
                if first_match["home_name"] == player_name:
                    actual_elo = first_match.get("home_elo")
                else:
                    actual_elo = first_match.get("away_elo")

                if actual_elo is None:
                    print(f"[DEBUG] Missing ELO data in match: {first_match}")
                    continue

                # Procesowanie wszystkich meczów w grupie (tutaj jest blad gdzies w srodku!!)
                for match in group['matches']:
                    try:
                        actual_elo = self.get_actual_player_elo(player_name, match)
                    except Exception as e:
                        print(f"[ERROR] Unga Bunga, mam cie: {str(e)}")
                    history.append({'elo_after': round(actual_elo, 2)})

            # Sprawdzenie, czy historia zawiera dane (wczesniej jest blad, ciekawe..?)
            #print("##################")
            #print(history)
            #print("##################")
            if history:
                #print(f"[DEBUG] Final ELO history for player {player_name}: {history}")
                return history[-1]['elo_after']
            else:
                print(f"[DEBUG] No valid ELO data calculated for player: {player_name}")
                return 0

        except Exception as e:
            print(f"[ERROR] Exception in calculate_elo_based_on_history: {str(e)}")
            return 0

    def avg_opponents_last_50_matches(self, player_name):
        #region Av_opponents_50
        """
        Calculates the average ELO of opponents in the last 50 matches of the specified player.

        Args:
            player_name (str): The name of the player.

        Returns:
            float: The average ELO of the opponents, or 0 if no matches are found.
        """
        try:
            # Get the last 50 matches of the player
            last_matches = self.get_last_50_matches(player_name)

            if not last_matches:
                print(f"[DEBUG] (avg_opponents_last_50_matches) No matches found for player: {player_name}")
                return 0

            #print(f"[DEBUG] Retrieved {len(last_matches)} matches for player: {player_name}")
            opponent_elos = []

            for match in last_matches:
                match_datetime = match["datetime"]

                # Determine the opponent's name
                if match["home_name"] == player_name:
                    opponent_name = match["away_name"]
                else:
                    opponent_name = match["home_name"]

                #print(f"[DEBUG] Match datetime: {match_datetime}, Opponent: {opponent_name}")

                # Calculate the opponent's ELO at the time of the match
                opponent_elo = self.calculate_elo_based_on_history(opponent_name, match_datetime, 2)
                
                #print(f"[DEBUG] Opponent: {opponent_name}, ELO: {opponent_elo}")

                if opponent_elo > 0:  # Exclude invalid ELOs
                    opponent_elos.append(opponent_elo)

            # Calculate the average ELO of opponents]
            #print(f"Oppononts ELOS: {opponent_elos}")
            if opponent_elos:
                avg_elo = sum(opponent_elos) / len(opponent_elos)
                #print(f"[DEBUG] Average ELO of opponents: {avg_elo}")
                return round(avg_elo, 2)
            else:
                print(f"[DEBUG] No valid ELO data for opponents of player: {player_name}")
                return 0

        except Exception as e:
            # Log any unexpected errors
            print(f"[ERROR] Exception occurred while calculating average opponents' ELO for player: {player_name}")
            print(f"Error details: {str(e)}")
            return 0