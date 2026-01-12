import json
import requests
from datetime import datetime
import pymongo
import time
import random
import pytz
from proxy_list import proxy_list


#region Connection to database settings
conn = pymongo.MongoClient('')
db = conn['czech_liga_pro_test']
calendar_coll = db['match_calendar']
#endregion
#region Get data request
session = requests.Session()
headers = {
    "accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8",
    "accept-encoding": "gzip, deflate, br",
    "accept-language": "pl,en;q=0.9,pl-PL;q=0.8,en-US;q=0.7,et;q=0.6,pt;q=0.5,uz;q=0.4",
    "referer": "https://1xbit9.com/live/",
    "sec-ch-ua": '"Chromium";v="110", "Not A(Brand";v="24", "Google Chrome";v="110"',
    "sec-ch-ua-mobile": "?1",
    "sec-ch-ua-platform": '"Android"',
    "sec-fetch-dest": "empty",
    "sec-fetch-mode": "cors",
    "sec-fetch-site": "same-origin",
    #"user-agent": "Mozilla/5.0 (Linux; Android 6.0; Nexus 5 Build/MRA58N) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/110.0.0.0 Mobile Safari/537.36",
    "x-requested-with": "XMLHttpRequest"
}
# Main loop
while(True):
    choose_one_region_proxy = random.choice(proxy_list)
    choose_one_proxy = random.choice(choose_one_region_proxy)
    proxy = {
        "http": choose_one_proxy,
        "https": choose_one_proxy,
    }
    session.proxies.update(proxy)
    try:
        response = session.get("https://1xbit9.com/LineFeed/GetChampZip?champ=2095165&tz=1&tf=2200000&lng=pl&partner=65&country=147&gr=29", timeout=5)
    except Exception as e: 
        print(e)
        continue

    try:    
        response_json = json.loads(response.content.decode(response.encoding, errors="ignore"))
    except Exception as e:
        print(e)
        print(f"{response.status_code} {response.encoding} {datetime.now()}")
        time.sleep(10)
        continue

    if not response_json:
        time.sleep(10)
        continue
    if not response_json["Value"]:
        time.sleep(10)
        continue

    for match in response_json["Value"]["G"]:
        if not calendar_coll.count_documents({ 'match_id': match["CI"] }):
            match_data = {}
            # Konwertowanie timestampu na obiekt datetime w UTC
            date_utc = datetime.utcfromtimestamp(match["S"])

            # Przypisanie strefy czasowej UTC
            utc_zone = pytz.utc
            date_utc = pytz.utc.localize(date_utc)

            # Strefa czasowa Polski (CET/CEST)
            poland_zone = pytz.timezone('Europe/Warsaw')

            # Konwersja na czas polski
            date_poland = date_utc.astimezone(poland_zone)
            try:
                match_data = {
                    "datetime": date_poland,
                    "match_id": match["CI"],
                    "home_name": match["O1E"],
                    "home_id": match["O1I"],
                    "away_name": match["O2E"],
                    "away_id": match["O2I"]    
                }
            except Exception as e:
                print(e)
                continue
            if not calendar_coll.count_documents(
                {   
                    "datetime": match_data["datetime"], 
                    "home_id": match_data["home_id"],
                    "away_id": match_data["away_id"]  
                }):
          
                    choose_one_region_proxy = random.choice(proxy_list)
                    choose_one_proxy = random.choice(choose_one_region_proxy)
                    proxy = {
                        "http": choose_one_proxy,
                        "https": choose_one_proxy,
                    }
                    session.proxies.update(proxy)
                    try:
                        res_match = session.get(f"""https://1xbit9.com/LineFeed/GetGameZip?id={match["CI"]}&lng=en&isSubGames=true&GroupEvents=true&allEventsGroupSubGames=true&countevents=250
                            &partner=65&country=147&fcountry=147&marketType=1&gr=29&isNewBuilder=true""", timeout=5)
                    except Exception as e: 
                        print(e)
                        continue

                    try:    
                        match_details_json = json.loads(res_match.content.decode(res_match.encoding, errors="ignore"))
                    except Exception as e:
                        print(e)
                        print(f"{res_match.status_code} {res_match.encoding} {datetime.now()}")
                        time.sleep(10)
                        continue                

                    if not match_details_json:
                        time.sleep(10)
                        continue
                    if not match_details_json["Value"]:
                        time.sleep(10)
                        continue

                    # Pobierz aktualne kursy
                    try:
                        if match_details_json["Value"]["GE"][0]["E"][0][0]["T"] == 1:
                           home_odd = match_details_json["Value"]["GE"][0]["E"][0][0]["C"]
                           away_odd = match_details_json["Value"]["GE"][0]["E"][1][0]["C"]
                        else:
                           continue
                    except Exception as e:
                        print(e)
                        continue

                    if home_odd:
                        if home_odd > 2 and away_odd > 2:
                            continue 
                        
                        match_data["home_odd"] = home_odd
                        match_data["away_odd"] = away_odd

                        calendar_coll.insert_one(match_data)
                        time.sleep(2)         

    time.sleep(60)


