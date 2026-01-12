from value_engine import ValueEngine
import time
import asyncio

class FindValueOne(ValueEngine):
    def __init__(self):
        super().__init__()
    
    def find_value_one(self, date_for_match, home, away, elo_home, elo_away, odd_home, odd_away):
        # Drugi warunek - sprawdzam czy jest znaczna roznica w aktualnej formie
        actual_form_home = self.get_actual_form(home)
        actual_form_away = self.get_actual_form(away)
        list_home = []
        list_away = []
        for form in actual_form_home.values():
            list_home.append(form)
        for form in actual_form_away.values():
            list_away.append(form)
        # Pierwszy warunek - sprawdzam czy jest ELO wieksze co najmniej o 30
        if elo_home - elo_away >= 30 and odd_home >= odd_away:
            if sum(actual_form_home.values()) - sum(actual_form_away.values()) > 0.5:
                # Znalazlem value
                return {
                    "bet": 1,
                    "datetime": date_for_match,
                    "home_name": home,
                    "away_name": away,
                    "home_elo": elo_home,
                    "away_elo": elo_away,
                    "odd_home": odd_home,
                    "odd_away": odd_away,
                    "form_home": list_home,
                    "form_away": list_away
                }

        # Pierwszy warunek - sprawdzam czy jest ELO wieksze co najmniej o 30
        if elo_away - elo_home >= 30 and odd_away >= odd_home:
            if sum(actual_form_away.values()) - sum(actual_form_home.values()) > 0.5:
                return {
                    "bet": 2,
                    "datetime": date_for_match,
                    "home_name": home,
                    "away_name": away,
                    "home_elo": elo_home,
                    "away_elo": elo_away,
                    "odd_home": odd_home,
                    "odd_away": odd_away,
                    "form_home": list_home,
                    "form_away": list_away
                }

        return {}



    def find_value_two(self, date_for_match, home, away, elo_home, elo_away, odd_home, odd_away):
        # Drugi warunek - sprawdzam czy jest znaczna roznica w aktualnej formie
        actual_form_home = self.get_actual_form(home)
        actual_form_away = self.get_actual_form(away)


        list_home = []
        list_away = []
        for form in actual_form_home.values():
            list_home.append(form)
        for form in actual_form_away.values():
            list_away.append(form)


        # Pierwszy warunek - sprawdzam czy jest ELO wieksze co najmniej o 60
        if elo_home - elo_away >= 60 and odd_home >= 1.5:
            if sum(actual_form_home.values()) - sum(actual_form_away.values()) > 0.5:
                # Znalazlem value
                return {
                    "bet": 1,
                    "datetime": date_for_match,
                    "home_name": home,
                    "away_name": away,
                    "home_elo": elo_home,
                    "away_elo": elo_away,
                    "odd_home": odd_home,
                    "odd_away": odd_away,
                    "form_home": list_home,
                    "form_away": list_away
                }

        # Pierwszy warunek - sprawdzam czy jest ELO wieksze co najmniej o 30
        if elo_away - elo_home >= 60 and odd_away >= 1.5:
            if sum(actual_form_away.values()) - sum(actual_form_home.values()) > 0.5:
                return {
                    "bet": 2,
                    "datetime": date_for_match,
                    "home_name": home,
                    "away_name": away,
                    "home_elo": elo_home,
                    "away_elo": elo_away,
                    "odd_home": odd_home,
                    "odd_away": odd_away,
                    "form_home": list_home,
                    "form_away": list_away
                }

        return {}


def main():

    value_object = FindValueOne()

    last_id = value_object.get_last_inserted_id()


    while True:
        # Query for new documents that have a higher _id than the last one
        new_records = value_object.calendar_coll.find({'_id': {'$gt': last_id}}).sort('_id', 1)
        
        for record in new_records:
            if 'home_odd' not in record:
                continue
            last_id = record['_id']  # Update the last known _id
            date_for_match, home, away, elo_h, elo_a = value_object.player_matching(record)
            odd_home = record['home_odd']
            odd_away = record['away_odd']
            if home and away:
                result_one = value_object.find_value_one(date_for_match, home, away, elo_h, elo_a, odd_home, odd_away)
                result_two = value_object.find_value_two(date_for_match, home, away, elo_h, elo_a, odd_home, odd_away)

                # Add to DB betting and notify on tg
                if result_one:
                    asyncio.run(value_object.send_message(result_one))
                    value_object.betting_coll.insert_one(result_one)
                if result_two:
                    asyncio.run(value_object.send_message(result_two))
                    value_object.betting_coll.insert_one(result_two)

        # Wait a bit before checking again
        time.sleep(1)

if __name__ == "__main__":
    main()
