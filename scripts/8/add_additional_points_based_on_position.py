import sys
import os

sys.path.append(os.path.abspath('/data/paletki/real-time'))

import requests
import pymongo
import random
from proxy_list import proxy_list
from bs4 import BeautifulSoup


#region Connection to database settings
conn = pymongo.MongoClient('')
db = conn['czech_liga_pro_test']
matches_coll = db['matches']
#endregion
#region Get data request
session = requests.Session()
headers = {
    "accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8",
    "accept-encoding": "gzip, deflate, br",
    "accept-language": "pl,en;q=0.9,pl-PL;q=0.8,en-US;q=0.7,et;q=0.6,pt;q=0.5,uz;q=0.4",
    "sec-ch-ua": '"Chromium";v="110", "Not A(Brand";v="24", "Google Chrome";v="110"',
    "sec-ch-ua-mobile": "?1",
    "sec-ch-ua-platform": '"Android"',
    "sec-fetch-dest": "empty",
    "sec-fetch-mode": "cors",
    "sec-fetch-site": "same-origin",
    #"user-agent": "Mozilla/5.0 (Linux; Android 6.0; Nexus 5 Build/MRA58N) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/110.0.0.0 Mobile Safari/537.36",
    "x-requested-with": "XMLHttpRequest"
}
#endregion

urls = matches_coll.find({ "match_category": { "$exists": False } })

for i, url in enumerate(urls):
    choose_one_region_proxy = random.choice(proxy_list)
    choose_one_proxy = random.choice(choose_one_region_proxy)
    proxy = {
        "http": choose_one_proxy,
        "https": choose_one_proxy,
    }
    session.proxies.update(proxy)
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/116.0.0.0 Safari/537.36"
    }
    try:
        response = session.get(url["tt_url"], headers=headers)
    
        soup = BeautifulSoup(response.content, "lxml")
        div = soup.find_all("div", class_="score")
        final = div[0].find_all("h4", string=lambda text: "Final" in text if text else False)
        third = div[0].find_all("h4", string=lambda text: "3rd" in text if text else False)
        group = div[0].find_all("h4", string=lambda text: "group" in text if text else False)
        if final:
            matches_coll.update_one(
                { "tt_url": url["tt_url"] },  # Wyszukiwanie dokumentu po _id
                { "$set": { "match_category": "final" } }  # Dodanie nowego pola "grab"
            )
            continue
        elif third:
            matches_coll.update_one(
                { "tt_url": url["tt_url"] },  # Wyszukiwanie dokumentu po _id
                { "$set": { "match_category": "3rd" } }  # Dodanie nowego pola "grab"
            )
            continue
        else:
            matches_coll.update_one(
                { "tt_url": url["tt_url"] },  # Wyszukiwanie dokumentu po _id
                { "$set": { "match_category": "group" } }  # Dodanie nowego pola "grab"
            )
            continue
    except Exception as e:
        with open("games_update_problem.txt", "a+") as file:
            file.write(url["tt_url"])
        print(e)




