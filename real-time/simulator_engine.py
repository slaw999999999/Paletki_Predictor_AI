from value_engine import ValueEngine
import pytz
from pymongo import MongoClient
from datetime import datetime, timedelta

class SimulatorEngine(ValueEngine):
    def __init__(self):
        super().__init__()
        self.utc_zone = pytz.utc
        self.local_zone = pytz.timezone('Europe/Warsaw')


    def get_last_100_matches_of_player(self, player_name, match_datetime):
        #region Last_100_matches_Player
        """
        Retrieves the number of matches a player played in the database.
        If the player played 100 or more matches, returns 100; otherwise, returns the actual count.
        Stops searching once 100 matches are found to save computational resources.

        Args:
            player_name (str): The name of the player.
            match_datetime (datetime): The datetime of the match for which ELO is calculated.

        Returns:
            int: Number of matches played (100 if matches are >= 100).
        """
        # Filtrowanie meczów zawodnika w całej bazie danych do momentu czasu meczu
        match_filter = {
            "$and": [
                {
                    "$or": [
                        {"home_name": player_name},
                        {"away_name": player_name}
                    ]
                },
                {
                    "datetime": {"$lt": match_datetime}  # Tylko mecze przed czasem meczu
                }
            ]
        }

        # Użycie kursora z limitem na wynikach
        match_count = 0
        cursor = self.matches_coll.find(match_filter, {"_id": 1}).batch_size(50)  # Pobieranie minimalnych danych

        # Iteracja przez mecze do momentu przekroczenia 100
        for _ in cursor:
            match_count += 1
            if match_count >= 100:
                cursor.close()  # Zamknięcie kursora
                return 100

        # Zwróć rzeczywistą liczbę meczów, jeśli mniej niż 100
        return match_count


    def get_matches_history_of_player(self, player_name, match_datetime, days_back=40):
        #region History_player
        """
        Retrieves the match history of a player within a time frame
        from X days before the match to 30 minutes before the match.

        Args:
            player_name (str): The name of the player.
            match_datetime (datetime): The datetime of the match for which ELO is calculated.
            days_back (int): Number of days to look back from the match_datetime.

        Returns:
            list: A list of match history grouped by date and league.
        """
        # Dolny limit: X dni przed meczem
        days_before = match_datetime - timedelta(days=days_back)
        # Górny limit: 30 minut przed meczem
        thirty_minutes_before = match_datetime - timedelta(minutes=30)

        # Filtrowanie meczów w zadanych ramach czasowych
        match_filter = {
            "$and": [
                {
                    "$or": [
                        {"home_name": player_name},
                        {"away_name": player_name}
                    ]
                },
                {
                    "datetime": {
                        "$gte": days_before,  # X dni przed meczem
                        "$lt": thirty_minutes_before  # Do 30 minut przed meczem
                    }
                }
            ]
        }

        # Wybór tylko niezbędnych pól
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

        # Sortowanie według daty
        sort = [("datetime", 1)]

        # Pobranie meczów z bazy danych
        raw_matches = list(self.matches_coll.find(match_filter, projection).sort(sort))

        # Grupowanie wyników w Pythonie
        grouped_matches = {}
        for match in raw_matches:
            match_date = match["datetime"].strftime("%Y-%m-%d")
            league_name = match["league_name"]
            key = (match_date, league_name)

            if key not in grouped_matches:
                grouped_matches[key] = []
            grouped_matches[key].append(match)

        # Konwersja wyników na odpowiedni format
        result = [
            {
                "_id": {"date": key[0], "league_name": key[1]},
                "matches": matches
            }
            for key, matches in grouped_matches.items()
        ]

        return result




    def get_actual_player_elo(self, player_name, match, actual_elo):
        #region actual_elo
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

    def calculate_elo_based_on_history(self, player_name, match_datetime, days_back=40):
        #region calculate_elo_history
        """
        Calculates the ELO of a player based on their match history
        within a specific timeframe (from days_back to 30 minutes before the match).

        Args:
            player_name (str): The name of the player.
            match_datetime (datetime): The datetime of the match for which ELO is calculated.
            days_back (int): Number of days to look back from the match_datetime.

        Returns:
            float: The calculated ELO for the player.
        """
        player_history = self.get_matches_history_of_player(player_name, match_datetime, days_back)
        history = []

        for group in player_history:
            if group['matches'][0]["home_name"] == player_name:
                actual_elo = group['matches'][0]["home_elo"]
            else:
                actual_elo = group['matches'][0]["away_elo"]

            for match in group['matches']:
                match["actual_elo"] = self.get_actual_player_elo(player_name, match, actual_elo)
                actual_elo = match["actual_elo"]
                history.append({'elo_after': round(actual_elo, 2)})

        if history:
            return history[-1]['elo_after']
        else:
            return 0



    def get_day_all_matches(self, date):
        #region all_Day_matches
        """
        Retrieves all matches from the database for a specific date.

        Args:
            date (str): The date for which matches should be retrieved in the format 'YYYY-MM-DD'.

        Returns:
            list: A list of match documents from the MongoDB collection.
        """
        try:
            # Parse the input date to a datetime object
            query_date = datetime.strptime(date, "%Y-%m-%d")

            # Calculate the range for the day in UTC
            start_of_day = query_date.replace(hour=0, minute=0, second=0, microsecond=0)
            end_of_day = start_of_day + timedelta(days=1) - timedelta(microseconds=1)

            # Query the matches collection for matches within the specified day
            matches = list(self.matches_coll.find({
                "datetime": {
                    "$gte": start_of_day,
                    "$lte": end_of_day
                }
            }))

            return matches

        except Exception as e:
            print(f"An error occurred while retrieving matches: {e}")
            return []

    def get_day_calendar_matches(self, date):
        #region calendar_matches
        """
        Retrieves all calendar matches from the database for a specific date.

        Args:
            date (str): The date for which calendar matches should be retrieved in the format 'YYYY-MM-DD'.

        Returns:
            list: A list of calendar match documents from the MongoDB collection.
        """
        try:
            # Parse the input date to a datetime object
            query_date = datetime.strptime(date, "%Y-%m-%d")

            # Calculate the range for the day in UTC
            start_of_day = query_date.replace(hour=0, minute=0, second=0, microsecond=0)
            end_of_day = start_of_day + timedelta(days=1) - timedelta(microseconds=1)

            # Query the calendar collection for matches within the specified day
            calendar_matches = list(self.calendar_coll.find({
                "datetime": {
                    "$gte": start_of_day,
                    "$lte": end_of_day
                }
            }))

            return calendar_matches

        except Exception as e:
            print(f"An error occurred while retrieving calendar matches: {e}")
            return []
        
    def add_players_CzechWebsiteName_and_ActualElo(self, calendar_matches, days_back=40):
        #region adding_data
        """
        Enriches match_calendar entries with player names and their actual ELO.

        Args:
            calendar_matches (list): List of match calendar entries.
            days_back (int): Number of days to look back for match history.

        Returns:
            list: A list of enriched match calendar entries.
        """
        try:
            enriched_matches = []

            for match in calendar_matches:
                home_player = self.players_coll.find_one({"player_id": match.get("home_id")})
                away_player = self.players_coll.find_one({"player_id": match.get("away_id")})

                # Dodaj dane zawodników do meczu
                match["home_player_name"] = home_player.get("player_name") if home_player else None
                match["away_player_name"] = away_player.get("player_name") if away_player else None

                # Oblicz aktualne ELO na podstawie historii w ramach czasowych
                match["home_actual_elo"] = self.calculate_elo_based_on_history(
                    match["home_player_name"], match["datetime"], days_back
                    )
                match["away_actual_elo"] = self.calculate_elo_based_on_history(
                    match["away_player_name"], match["datetime"], days_back
                    )

                match["home_last_100"] = self.get_last_100_matches_of_player(
                    match["home_player_name"], match["datetime"]
                    )
                
                match["away_last_100"] = self.get_last_100_matches_of_player(
                    match["away_player_name"], match["datetime"]
                    )

                enriched_matches.append(match)

            return enriched_matches

        except Exception as e:
            print(f"An error occurred while enriching calendar matches: {e}")
            return []


        
    def merge_matches_and_calendar_matches(self, matches_list, calendar_matches_modified):
        #region merge_collections
        """
        Merges matches list with the enriched calendar matches, updating the matches with enriched data if a match is found.
        Only matches that are successfully updated will be included in the returned list.

        Args:
            matches_list (list): List of matches from the database.
            calendar_matches_modified (list): List of enriched calendar matches.

        Returns:
            list: Updated matches list (only updated matches, with duplicates removed).
        """
        update = 0
        non_update = 0
        updated_matches = []  # List to store successfully updated matches

        for calendar_match in calendar_matches_modified:
            home_name = calendar_match["home_player_name"]
            away_name = calendar_match["away_player_name"]
            match_datetime = calendar_match["datetime"]

            # Define time range for matching
            time_range_start = match_datetime - timedelta(hours=4)
            time_range_end = match_datetime + timedelta(hours=4)

            # Search for a matching record in matches_list
            for match in matches_list:
                if (match["home_name"] == home_name and
                    match["away_name"] == away_name and
                    time_range_start <= match["datetime"] <= time_range_end):
                    # Filter out matches with missing or invalid odds
                    home_odd = calendar_match.get("home_odd")
                    away_odd = calendar_match.get("away_odd")
                    if home_odd is None or away_odd is None:
                        non_update += 1
                        continue

                    if home_odd == 0 or away_odd == 0:
                        non_update += 1
                        continue

                    if home_odd > 2 and away_odd > 2:
                        non_update += 1
                        continue

                    # Update the match with enriched data
                    match["home_actual_elo"] = calendar_match.get("home_actual_elo")
                    match["away_actual_elo"] = calendar_match.get("away_actual_elo")
                    match["home_player_name"] = calendar_match.get("home_player_name")
                    match["away_player_name"] = calendar_match.get("away_player_name")
                    match["home_last_100"] = calendar_match.get("home_last_100")
                    match["away_last_100"] = calendar_match.get("away_last_100")
                    match["home_odd"] = home_odd
                    match["away_odd"] = away_odd

                    updated_matches.append(match)  # Add updated match to the list
                    update += 1
                    break
            else:
                non_update += 1

        print(f"ZAKTUALIZOWANE: {update}\nNIEZAKTUALIZOWANE: {non_update}")

        # Remove unnecessary keys
        keys_to_remove = [
            "Set1", "Set2", "Set3", "Set4", "Set5",
            "odd_home", "odd_away", "elo_diff", "odds_url", 'home_id', 'delta_elo'
        ]
        for match in updated_matches:
            for key in keys_to_remove:
                match.pop(key, None)  # Remove key if it exists

        # Remove duplicates based on home_name, away_name, and datetime
        unique_matches = []
        seen = set()
        for match in updated_matches:
            match_identifier = (match["home_name"], match["away_name"], match["datetime"])
            if match_identifier not in seen:
                unique_matches.append(match)
                seen.add(match_identifier)

        return unique_matches


    def evaluate_strategy_1(self, updated_matches, date):
        #region evaluate_1
        """
        Evaluates the profitability of a betting strategy for a single day.

        Args:
            updated_matches (list): List of matches enriched with ELO and odds data.
            date (datetime): The date for which ROI is being calculated.

        Returns:
            dict: Dictionary containing the number of won/lost bets and ROI for the day.
        """
        # Initialize counters
        no_won = 0
        no_lost = 0
        lost_match = 0
        suma = 0  # Total profit/loss

        # Iterate through the matches
        #print(f"Processing matches for date: {date}")
        for match in updated_matches:
            #print(match)


            # Extract relevant data
            home_elo = match.get("home_actual_elo", 0)
            away_elo = match.get("away_actual_elo", 0)
            home_odd = match.get("home_odd", 0)
            away_odd = match.get("away_odd", 0)
            winner = match.get("winner")

            if home_odd == 0 or away_odd == 0:
                #print(f"Missing odds for match: {match}")
                lost_match += 1
                continue
                
             # Ensure odds are valid numbers
            if home_odd is None or away_odd is None:
                #print(f"Ignored match due to missing odds: {match}")
                lost_match += 1
                continue

            if home_odd > 2 and away_odd > 2:
                #print(f"Ignored match due to wrong odds: {match}")
                lost_match += 1
                continue

            if winner == 1 and away_elo + 50 >= home_elo >= away_elo + 35 and home_odd >= 2:
                #print(f"Won bet on home: {match}")
                no_won += 1
                suma += home_odd - 1
            elif winner == 0 and home_elo + 35 <= away_elo <= home_elo + 50 and away_odd >= 2:
                #print(f"Won bet on away: {match}")
                no_won += 1
                suma += away_odd - 1
            elif winner == 1 and home_elo + 35 <= away_elo <= home_elo + 50 and away_odd >= 2:
                #print(f"Lost bet on away: {match}")
                no_lost += 1
                suma -= 1
            elif winner == 0 and away_elo + 50 >= home_elo >= away_elo + 35 and home_odd >= 2:
                #print(f"Lost bet on home: {match}")
                no_lost += 1
                suma -= 1
            else:
                #print(f"Ignored match: {match}")
                lost_match += 1


        if int(no_lost) + int(no_won) == 0:
            return {
                "date": date.strftime("%Y-%m-%d"),
                "no_won": 0,
                "no_lost": 0,
                "units": 0,
                "ROI": 0,
                "ignored_matches": lost_match,
            }
        else:
            return {
                "date": date.strftime("%Y-%m-%d"),
                "no_won": no_won,
                "no_lost": no_lost,
                "units": suma,
                "ROI": suma/(no_won+no_lost),
                "ignored_matches": lost_match,
            }
    

    def evaluate_strategy_2(self, updated_matches, date):
        #region evaluate_2
        """
        Evaluates the profitability of a betting strategy for a single day.

        Args:
            updated_matches (list): List of matches enriched with ELO and odds data.
            date (datetime): The date for which ROI is being calculated.

        Returns:
            dict: Dictionary containing the number of won/lost bets and ROI for the day.
        """
        # Initialize counters
        no_won = 0
        no_lost = 0
        lost_match = 0
        suma = 0  # Total profit/loss

        # Iterate through the matches
        #print(f"Processing matches for date: {date}")
        for match in updated_matches:
            #print(match)


            # Extract relevant data
            if match.get("home_actual_elo", 0) == 0:
                home_elo = match.get("home_elo", 0)
                match["home_actual_elo"] = home_elo
            else:
                home_elo = match.get("home_actual_elo", 0)

            # Extract relevant data
            if match.get("away_actual_elo", 0) == 0:
                away_elo = match.get("away_elo", 0)
                match["away_actual_elo"] = away_elo
            else:
                away_elo = match.get("away_actual_elo", 0)

            home_odd = match.get("home_odd", 0)
            away_odd = match.get("away_odd", 0)
            winner = match.get("winner")
            home_100 = match.get('home_last_100')
            away_100 = match.get('away_last_100')

            if home_odd == 0 or away_odd == 0:
                #print(f"Missing odds for match: {match}")
                lost_match += 1
                continue
                
             # Ensure odds are valid numbers
            if home_odd is None or away_odd is None:
                #print(f"Ignored match due to missing odds: {match}")
                lost_match += 1
                continue

            if home_odd > 2 and away_odd > 2:
                #print(f"Ignored match due to wrong odds: {match}")
                lost_match += 1
                continue

            if winner == 1 and home_elo > away_elo + 1 and home_odd >= 1.7 and int(home_100) == 100 and int(away_100) < 100:
                #print(f"Won bet on home: {match}")
                no_won += 1
                suma += home_odd - 1
            elif winner == 0 and home_elo + 1 < away_elo and away_odd >= 1.7 and int(home_100) < 100 and int(away_100) == 100:
                #print(f"Won bet on away: {match}")
                no_won += 1
                suma += away_odd - 1
            elif winner == 1 and home_elo + 1 < away_elo and away_odd >= 1.7 and int(home_100) < 100 and int(away_100) == 100:
                #print(f"Lost bet on away: {match}")
                no_lost += 1
                suma -= 1
            elif winner == 0 and home_elo > away_elo + 1 and home_odd >= 1.7 and int(home_100) == 100 and int(away_100) < 100:
                #print(f"Lost bet on home: {match}")
                no_lost += 1
                suma -= 1
            else:
                #print(f"Ignored match: {match}")
                lost_match += 1


        if int(no_lost) + int(no_won) == 0:
            return {
                "date": date.strftime("%Y-%m-%d"),
                "no_won": 0,
                "no_lost": 0,
                "units": 0,
                "ROI": 0,
                "ignored_matches": lost_match,
            }
        else:
            return {
                "date": date.strftime("%Y-%m-%d"),
                "no_won": no_won,
                "no_lost": no_lost,
                "units": suma,
                "ROI": suma/(no_won+no_lost),
                "ignored_matches": lost_match,
            }


    def strategy_1(self, simulator_engine):
        #region strategy_1
        """
       """
        simulator = simulator_engine

        end_date = datetime.now()
        start_date = end_date - timedelta(days=28)

        all_results = []

        for day_offset in range(29):
            current_date = start_date + timedelta(days=day_offset)
            matches = simulator.get_day_all_matches(current_date.strftime("%Y-%m-%d"))
            calendar_matches = simulator.get_day_calendar_matches(current_date.strftime("%Y-%m-%d"))
            calendar_matches_modified = simulator.add_players_CzechWebsiteName_and_ActualElo(
                calendar_matches, days_back=40  # 40 dni wstecz
            )

            updated_matches = simulator.merge_matches_and_calendar_matches(matches, calendar_matches_modified)
            #print(updated_matches)
            #print("#####################################")
            daily_result = simulator.evaluate_strategy_1(updated_matches, current_date)
            all_results.append(daily_result)

            print(
                f"Date: {daily_result['date']}, "
                f"Lost: {daily_result['no_lost']}, Won: {daily_result['no_won']}, "
                f"Units: {daily_result['units']}  ROI: {daily_result['ROI']}, "
                f"Ignored Matches: {daily_result['ignored_matches']}"
            )

        total_won = sum(result["no_won"] for result in all_results)
        total_lost = sum(result["no_lost"] for result in all_results)
        total_units = sum(result["units"] for result in all_results)
        if total_lost != 0 and total_won != 0:

            total_roi = total_units/(total_lost+total_won)

            print(f"\nSUMMARY: Total Won: {total_won}, Total Lost: {total_lost}, Total units: {total_units},Total ROI: {total_roi}")


    def strategy_2(self, simulator_engine):
        #region Strategy_2
        """
      
        
        """
        simulator = simulator_engine

        end_date = datetime.now()
        start_date = end_date - timedelta(days=27)

        all_results = []

        for day_offset in range(28):
            current_date = start_date + timedelta(days=day_offset)
            matches = simulator.get_day_all_matches(current_date.strftime("%Y-%m-%d"))
            calendar_matches = simulator.get_day_calendar_matches(current_date.strftime("%Y-%m-%d"))
            calendar_matches_modified = simulator.add_players_CzechWebsiteName_and_ActualElo(
                calendar_matches, days_back=40  # 40 dni wstecz
            )

            updated_matches = simulator.merge_matches_and_calendar_matches(matches, calendar_matches_modified)
            #print(updated_matches)
            #print("#####################################")
            daily_result = simulator.evaluate_strategy_2(updated_matches, current_date)
            all_results.append(daily_result)

            print(
                f"Date: {daily_result['date']}, "
                f"Lost: {daily_result['no_lost']}, Won: {daily_result['no_won']}, "
                f"Units: {daily_result['units']}  ROI: {daily_result['ROI']}, "
                f"Ignored Matches: {daily_result['ignored_matches']}"
            )

        total_won = sum(result["no_won"] for result in all_results)
        total_lost = sum(result["no_lost"] for result in all_results)
        total_units = sum(result["units"] for result in all_results)
        if total_lost != 0 and total_won != 0:

            total_roi = total_units/(total_lost+total_won)

            print(f"\nSUMMARY: Total Won: {total_won}, Total Lost: {total_lost}, Total units: {total_units},Total ROI: {total_roi}")



def main():
    simulator = SimulatorEngine()
    simulator.strategy_1(simulator)

if __name__ == "__main__":
    main()



