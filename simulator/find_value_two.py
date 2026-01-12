import time
import asyncio
from datetime import datetime, timedelta
import pytz
import joblib

import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

from value_engine import ValueEngine

class FindValueTest(ValueEngine):
    def __init__(self):
        super().__init__()
        self.utc_zone = pytz.utc
        self.local_zone = pytz.timezone('Europe/Warsaw')

        # === 1. Ładowanie wytrenowanego modelu Gradient Boosting ===
        self.gb_model = joblib.load("trained_model_gradient_boosting.pkl")

        # Ustalone progi pewności (prawdopodobieństwo z predict_proba)/ bazowane na results_models.txt. Cel to minimum 80% accuracy
        self.threshold_away = 0.67  # jeśli p1 >= 0.67 => bet na gościa (klasa 1)
        self.threshold_home = 0.5  # jeśli p0 >= 0.7 => bet na gospodarza (klasa 0)

    
    def find_value_two(self, date_for_match, home, away, elo_home, elo_away, odd_home, odd_away,
                        elo_diff_scaled,diff_last_tournaments,diff_zmeczenie,h2h_home_scaled,diff_h2h, total_h2h,
                        diff_players_vs_opponents_50,diff_dominance_percentage,diff_closest_trends,
                        diff_saved,diff_wasted,diff_walecznosc_3):
        # Generowanie wspólnego wykresu formy
        combined_form_chart, actual_form_home, actual_form_away = self.get_combined_form_chart(home, away)
    
        list_home = actual_form_home.tolist()
        list_away = actual_form_away.tolist()

        """"
        Progi pewności:
          - p1 >= 0.60 => bet=2 (Away)
          - p0 >= 0.62 => bet=1 (Home)
        Zwraca dict z danymi o meczu, jeśli spełni warunki betu, w przeciwnym wypadku {}.
        """

        # 1. Tworzymy słownik z 11 cechami (kluczowe, by nazwy były takie same jak w modelu!)
        data_dict = {
            "elo_diff_scaled": elo_diff_scaled,
            "diff_last_tournaments": diff_last_tournaments,
            "diff_zmeczenie": diff_zmeczenie,
            "h2h_home_scaled": h2h_home_scaled,
            "diff_h2h": diff_h2h,
            "diff_players_vs_opponents_50": diff_players_vs_opponents_50,
            "diff_dominance_percentage": diff_dominance_percentage,
            "diff_closest_trends": diff_closest_trends,
            "diff_saved": diff_saved,
            "diff_wasted": diff_wasted,
            "diff_walecznosc_3": diff_walecznosc_3
        }

        # 2. Zamieniamy to na DataFrame z 1 wierszem
        df_input = pd.DataFrame([data_dict])

        # 3. Przewidujemy prawdopodobieństwa: p0 = home, p1 = away
        proba_gb = self.gb_model.predict_proba(df_input)[0]
        p0, p1 = proba_gb[0], proba_gb[1]

        # 4. Sprawdzamy progi pewności
        bet = None
        if (p1 >= self.threshold_away and odd_away >= 1.7 and elo_home <= elo_away - 35) or (p1 >= self.threshold_away and odd_away >= 1.7 and elo_home < elo_away and total_h2h >= 7 and h2h_home_scaled <= 0.37):
            return {
            "bet": 2,
            "datetime": date_for_match,
            "home_name": home,
            "away_name": away,
            "home_elo": float(elo_home),
            "away_elo": float(elo_away),
            "odd_home": odd_home,
            "odd_away": odd_away,
            "p0_home_prob": p0,
            "p1_away_prob": p1,
            "combined_form_chart": combined_form_chart,
            "form_home": list_home,
            "form_away": list_away
        }
        elif (p0 >= self.threshold_home and odd_home >= 1.7 and elo_home  - 35 >= elo_away) or (p0 >= self.threshold_home and odd_home >= 1.7 and elo_home > elo_away and total_h2h >= 7 and h2h_home_scaled >= 0.63):
            return {
            "bet": 1,
            "datetime": date_for_match,
            "home_name": home,
            "away_name": away,
            "home_elo": float(elo_home),
            "away_elo": float(elo_away),
            "odd_home": odd_home,
            "odd_away": odd_away,
            "p0_home_prob": p0,
            "p1_away_prob": p1,
            "combined_form_chart": combined_form_chart,
            "form_home": list_home,
            "form_away": list_away
        }
        
        return {}


    async def process_matches(self, betting_collection, n, last_seen_id=None):
        """
        Process matches asynchronously to avoid using `asyncio.run` within the event loop.
        """
        matches = self.get_upcoming_matches(n, last_seen_id) #n - liczba godzin do przodu

        for record in matches:
            last_seen_id = record["_id"]
            if 'home_odd' not in record or 'away_odd' not in record:
                continue

            # Match player details
            date_for_match, home_name_tt, away_name_tt, elo_home, elo_away = self.player_matching(record)

            
            if home_name_tt != '' and away_name_tt != '':
                # Check for duplicates
                if self.is_duplicate(home_name_tt, away_name_tt, betting_collection, record):
                    print("[INFO] Duplicate match found. Skipping processing.")
                    continue
                elo_diff, elo_diff_scaled, elo_home, elo_away = self.calculate_elo_diff(home_name_tt, away_name_tt)
                # Get match data for both players
                _, home_cumulative_elo, away_cumulative_elo = self.get_combined_form_chart(home_name_tt, away_name_tt)
                experience = self.compare_experience(home_cumulative_elo, away_cumulative_elo)
                         
                big_wins_home, big_losses_home, big_wins_away, big_losses_away = self.big_wins_losses(home_cumulative_elo, away_cumulative_elo)
                diff_big_wins = big_wins_home - big_wins_away
                diff_big_losses = big_losses_home - big_losses_away

                dominance_percentage_home, dominance_percentage_away = self.check_dominance(home_cumulative_elo, away_cumulative_elo)
                diff_dominance_percentage = dominance_percentage_home - dominance_percentage_away

                home_days = self.get_days_since_last_match(home_name_tt)
                away_days = self.get_days_since_last_match(away_name_tt)
                diff_last_tournament = home_days - away_days
                zmeczenie_home = self.get_average_games_last_week(home_name_tt)
                zmeczenie_away = self.get_average_games_last_week(away_name_tt)
                diff_zmeczenie = zmeczenie_home - zmeczenie_away
                diff_h2h, total_h2h, h2h_home_scaled, h2h_away_scaled = self.calculate_h2h_last_three_months(home_name_tt, away_name_tt)
                trends = self.get_actual_form(home_name_tt, away_name_tt)
                diff_closest_trends = trends['home']['25-0'] - trends['away']['25-0']
                avg_opponents_home = self.avg_opponents_last_50_matches(home_name_tt)
                avg_opponents_away  = self.avg_opponents_last_50_matches(away_name_tt)
                actual_elo_minus_opponents_50_home = elo_home - avg_opponents_home
                actual_elo_minus_opponents_50_away = elo_away - avg_opponents_away

                diff_players_vs_opponents_50 = actual_elo_minus_opponents_50_home - actual_elo_minus_opponents_50_away
                walecznosc_last_50_home, break_point_saved_home, break_point_wasted_home = self.analyze_last_50_matches_for_player(home_name_tt)
                walecznosc_last_50_away, break_point_saved_away, break_point_wasted_away = self.analyze_last_50_matches_for_player(away_name_tt)
                diff_saved = break_point_saved_home - break_point_saved_away
                diff_wasted  = break_point_wasted_home, break_point_wasted_away
                #print(walecznosc_last_50_home)
                diff_wasted_minus  = break_point_wasted_home - break_point_wasted_away #pozniej w proceisei uczenia robimy minus wiec tutaj zeby nie psuc po prostu stworze inna zmiennna
                # Rozpakowanie i obliczenie różnic dla zawodnika HOME
                diff_3_home = walecznosc_last_50_home["3+"][0] - walecznosc_last_50_home["3+"][1]
                diff_4_home = walecznosc_last_50_home["4+"][0] - walecznosc_last_50_home["4+"][1]
                diff_5_home = walecznosc_last_50_home["5+"][0] - walecznosc_last_50_home["5+"][1]
                diff_6_home = walecznosc_last_50_home["6+"][0] - walecznosc_last_50_home["6+"][1]

                # Rozpakowanie i obliczenie różnic dla zawodnika AWAY
                diff_3_away = walecznosc_last_50_away["3+"][0] - walecznosc_last_50_away["3+"][1]
                diff_4_away = walecznosc_last_50_away["4+"][0] - walecznosc_last_50_away["4+"][1]
                diff_5_away = walecznosc_last_50_away["5+"][0] - walecznosc_last_50_away["5+"][1]
                diff_6_away = walecznosc_last_50_away["6+"][0] - walecznosc_last_50_away["6+"][1]
                diff_walecznosc_3 = diff_3_home - diff_3_away
                

            

            odd_home = record['home_odd']
            odd_away = record['away_odd']

            if odd_home > 2 and odd_away > 2:
                print("[INFO] Both odds are above 2. Skipping processing.")
                continue


            if home_name_tt and away_name_tt:
                #print(walecznosc_last_50_home)
                diff_wasted_minus  = break_point_wasted_home - break_point_wasted_away #pozniej w proceisei uczenia robimy minus wiec tutaj zeby nie psuc po prostu stworze inna zmiennna
                
                result_one = self.find_value_two(date_for_match, home_name_tt, away_name_tt, elo_home, elo_away, odd_home, odd_away,
                                                 elo_diff_scaled,diff_last_tournament,diff_zmeczenie,h2h_home_scaled,diff_h2h, total_h2h,
                                                diff_players_vs_opponents_50,diff_dominance_percentage,diff_closest_trends,
                                                diff_saved,diff_wasted_minus,diff_walecznosc_3)
                if result_one:
                    # Check if the match already exists in the betting collection
                    print(f"[INFO] Processing match: {home_name_tt} vs {away_name_tt} at {date_for_match}")
                    additional_check = betting_collection.find_one({
                        "home_name": result_one["home_name"],
                        "away_name": result_one["away_name"],
                        "odd_home": result_one["odd_home"],
                        "odd_away": result_one["odd_away"],
                        "processed": True
                    })

                    if additional_check:
                        print("Identical match already in the collection. Skipping.")
                        continue

                    #miejsce na dodanie dodatkowych parametrow
                    result_one['elo_diff'] = elo_diff
                    result_one['elo_diff_scaled'] = elo_diff_scaled
                    result_one['experience'] = experience
                    result_one['home_days'] = home_days
                    result_one['away_days'] = away_days
                    result_one['diff_last_tournaments'] = diff_last_tournament
                    result_one['zmecznie_home'] = zmeczenie_home
                    result_one['zmeczenie_away'] = zmeczenie_away
                    result_one['diff_zmeczenie'] = diff_zmeczenie
                    result_one['h2h_home_scaled'] = h2h_home_scaled
                    result_one['h2h_away_scaled'] = h2h_away_scaled
                    result_one['diff_h2h'] = diff_h2h
                    result_one['total_h2h'] = total_h2h
                    result_one['avg_opponents_home'] = avg_opponents_home
                    result_one['avg_opponents_away'] = avg_opponents_away
                    result_one['diff_actual_elo_vs_opponents_50_home'] = actual_elo_minus_opponents_50_home
                    result_one['diff_actual_elo_vs_opponents_50_away'] = actual_elo_minus_opponents_50_away
                    result_one['diff_players_vs_opponents_50'] = diff_players_vs_opponents_50
                    result_one['big_wins_home'] = big_wins_home
                    result_one['big_losses_home'] = big_losses_home
                    result_one['big_wins_away'] = big_wins_away
                    result_one['big_losses_away'] = big_losses_away
                    result_one['diff_big_wins'] = diff_big_wins
                    result_one['diff_big_losses'] = diff_big_losses
                    result_one['dominance_percentage_home'] = dominance_percentage_home
                    result_one['dominance_percentage_away'] = dominance_percentage_away
                    result_one['trends_home'] = trends
                    result_one['diff_closest_trends'] = diff_closest_trends
                    result_one['break_point_saved_home'] = break_point_saved_home
                    result_one['break_point_saved_away'] = break_point_saved_away
                    result_one['break_point_wasted_home'] = break_point_wasted_home
                    result_one['break_point_wasted_away'] = break_point_wasted_away
                    result_one['diff_saved'] = diff_saved
                    result_one['diff_wasted'] = diff_wasted
                    result_one['diff_walecznosc'] = {
                        "3+": diff_3_home - diff_3_away,
                        "4+": diff_4_home - diff_4_away,
                        "5+": diff_5_home - diff_5_away,
                        "6+": diff_6_home - diff_6_away,
                    }
                    
       
                    # Use `await` directly instead of `asyncio.run`
                    message_id = await self.send_message_value_two(result_one)
                    if message_id:
                        result_one["telegram_message_id"] = message_id  # Assign message ID to the result
                    result_one.pop("combined_form_chart", None)  # Remove the chart file path
                    result_one.pop("form_home", None)  # Remove form_home
                    result_one.pop("form_away", None)  # Remove form_away
                    result_one.update({"processed": True, "edited": False})
                    betting_collection.insert_one(result_one)



def main():
    value_object = FindValueTest()
    LAST_SEEN_ID = None
    async def run_loop():
        while True:
            await value_object.process_matches(value_object.betting_ai ,5, LAST_SEEN_ID)  # Process next n hour matches
            await value_object.update_finished_matches_value_two(value_object.betting_ai )  # Update finished matches
            await value_object.calculate_and_notify_roi(value_object.betting_ai , 50)  # Calculate and notify ROI for the last (bets with result) n matches
            await value_object.delete_form_chart_png_files('png_trash')
            await asyncio.sleep(60)

    asyncio.run(run_loop())


if __name__ == "__main__":
    main()
