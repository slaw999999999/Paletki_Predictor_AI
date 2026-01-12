import sys
import os

sys.path.append(os.path.abspath('/data/paletki/real-time'))

import requests
import pymongo
import random
from proxy_list import proxy_list
import datetime
import pytz
import json
from fuzzywuzzy import fuzz


#region Connection to database settings
conn = pymongo.MongoClient('')
db = conn['czech_liga_pro_test']
matches_coll = db['matches']
player_coll = db['players']
#endregion
#region Get data request
session = requests.Session()
headers = {
    "accept": "application/vnd.xenvelop+json",
    "accept-language": "en_GB",
    "x-whence": "22",
    "x-referral": "339",
    "x-group": "1026",
    "x-bundleid": "org.fairpari.client",
    "appguid": "5254685f2327fd56_2",
    "x-fcountry": "147",
    "x-devicemanufacturer": "Genymobile",
    "x-devicemodel": "Pixel 3",
    "x-request-guid": "339_5254685f2327fd56_2_1734485308012_69_-1418210061",
    "x-country": "212",
    "content-type": "application/json; charset=utf-8",
    "user-agent": "org.fairpari.client-user-agent/fairpari-9(17262)",
    "version": "fairpari-9(17262)",
    "accept-encoding": "br, gzip",
}
#endregion
for k in range(9, 12):
    for j in range(1, 30):
        # Przykładowa data i godzina w formacie datetime
        dt_object = datetime.datetime(2023, k, j, 0,0,0)
        # Przypisanie strefy czasowej (np. GMT+1)
        timezone = pytz.timezone('Etc/GMT+1')
        # Przekształcenie datetime do strefy czasowej
        dt_object_with_tz = timezone.localize(dt_object)
        # Konwersja na timestamp
        start_date = int(dt_object_with_tz.timestamp())
        # Konwertowanie timestamp na datetime
        end_date = int(datetime.datetime.fromtimestamp(start_date+86399, tz=pytz.timezone('Etc/GMT+1')).timestamp())

        choose_one_region_proxy = random.choice(proxy_list)
        choose_one_proxy = random.choice(choose_one_region_proxy)
        proxy = {
            "http": choose_one_proxy,
            "https": choose_one_proxy,
        }
        session.proxies.update(proxy)
        try:
            response = session.get(f'https://qythugzzq.com/resultcoreservice/v1/games?champIds=2095165&dateFrom={str(start_date)}&dateTo={str(end_date)}&lng=en_GB&ref=339&gr=1026', headers=headers)
        except Exception as e:
            print(e)

        # Sprawdzenie, czy żądanie zakończyło się sukcesem
        if response.status_code == 200:
            # Odczytanie odpowiedzi w formacie JSON
            json_data = response.json()
        else:
            print(f"Nie udało się pobrać danych. Status code: {response.status_code}")
            exit(0)


        for i, match in enumerate(json_data["data"]["items"]):
            if "score" not in match:
                continue
            if not match["score"]:
                continue
            startDate = datetime.datetime.fromtimestamp(match["dateStart"])
            # if "Schauer" in match["opp1"]  or "Schauer" in match["opp2"]:
            #     a =1
            # Ustalamy początek i koniec dnia (00:00:00 do 23:59:59)
            start_of_day = datetime.datetime(startDate.year, startDate.month, startDate.day)
            end_of_day = start_of_day.replace(hour=23, minute=59, second=59)
            score = match["score"].split(" ")[0]
            detailed_score = match["score"].split(" ")[1].replace(",", " ").replace(":", "-")
            ft_score = score[0] + " : " + score[2]
            potential_matches = matches_coll.find(
                {
                    "result": detailed_score,
                    "datetime": {
                        "$gte": start_of_day,  # Większe lub równe początkowi dnia
                        "$lte": end_of_day     # Mniejsze lub równe końcowi dnia
                    }
                })
            for potential_match in potential_matches:
                home = potential_match["home_name"].split(" ")
                away = potential_match["away_name"].split(" ")
                home_name = home[1] + " " + home[0]
                away_name = away[1] + " " + away[0]
                m1 = fuzz.partial_ratio(home_name, match["opp1"])
                m2 = fuzz.partial_ratio(away_name, match["opp2"])
                if  m1 > 70 or m2 > 70:
                    if not player_coll.count_documents({ "player_id": match["opp1Ids"][0] }):
                        player_coll.update_one(
                            { "player_name": potential_match["home_name"] },  # Wyszukiwanie dokumentu po _id
                            { "$set": { "player_id": match["opp1Ids"][0] } }  # Dodanie nowego pola "grab"
                        )
                    if not player_coll.count_documents({ "player_id": match["opp2Ids"][0] }):
                        player_coll.update_one(
                            { "player_name": potential_match["away_name"] },  # Wyszukiwanie dokumentu po _id
                            { "$set": { "player_id": match["opp2Ids"][0] } }  # Dodanie nowego pola "grab"
                        )