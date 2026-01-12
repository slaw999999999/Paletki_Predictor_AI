from proxy_list import proxy_list
import pymongo
import requests
import random
from datetime import datetime
import json
from bs4 import BeautifulSoup
import pytz
from fuzzywuzzy import fuzz


#region Connection to database settings
conn = pymongo.MongoClient('')
db = conn['czech_liga_pro_test']
players_coll = db['players']
calendar_coll = db['match_calendar']
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

# Funkcja do generowania URL dla danej daty
def generate_url_for_date(date):
    return f"https://tt.league-pro.com/tours/?year={date.year}&month={date.month:02d}&day={date.day:02d}"

def extract_links(soup):
    # Znajdowanie dat w elemencie span w div o klasie "node-desc item"
    return [
            "https://tt.league-pro.com/" + link["href"]
            for table in soup.find_all("table", class_="table")
            for link in table.find_all("a", string=lambda text: "Tournament" in text if text else False)
        ], [
            datetime.strptime(data.text.split('(')[0].strip()+data.text.split(')')[1],"%d %b %H:%M")
            for table in soup.find_all("table", class_="table")
            for data in table.find_all("td", class_="tournament-date")
        ]

def get_tournament_links_from_calendar_day(url):
    """
    Pobiera wszystkie linki do turniejów z kalendarza dnia na podanym URL.

    Args:
        url (str): URL strony z kalendarzem dnia.

    Returns:
        list: Lista linków do turniejów, jeśli strona została pomyślnie pobrana i sparsowana.
        str: Komunikat błędu w przypadku problemów z żądaniem HTTP.
    """
    try:
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
        response = session.get(url, headers=headers)
        if response.status_code == 200:
            content = response.content
            soup = BeautifulSoup(content, "lxml")
            links = extract_links(soup)
            return links
        else:
            return f"Request failed with status code {response.status_code} for URL {url}."            
    except requests.exceptions.RequestException as e:
        return f"An error occurred while fetching URL {url}: {e}"


def get_html_from_page(link):
    """
     Funkcja wykonuje zapytanie HTTP do podanego URL, pobiera zawartość strony
    i zwraca ją w postaci obiektu BeautifulSoup.

    Args:
        url (str): URL strony turnieju, z którego chcemy pobrać HTML.

    Returns:
        BeautifulSoup: Obiekt BeautifulSoup zawierający załadowany HTML turnieju,
                       umożliwiający dalszą analizę.
        str: Komunikat o błędzie, jeśli zapytanie zakończy się niepowodzeniem.
    """
    try:
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
        response = session.get(link, headers=headers)
        if response.status_code == 200:
            content = response.content
            soup = BeautifulSoup(content, "lxml")
            return soup
        else:
            return f"Request failed with status code {response.status_code} for URL {link}."
    except requests.exceptions.RequestException as e:
        print(f"An error occurred while fetching URL {link}: {e}")


def scrape_match_data(soup):
    """
    Scrapuje dane meczow ze strony zakonczonego turnieju
    
    Args:
        BeautifulSoup: Obiekt BeautifulSoup zawierający załadowany HTML turnieju,
                       umożliwiający dalszą analizę.

    Returns:
        list's: Tablice poszczegolnych parametrow, z brakujacym szczegolowym przebiem meczu.
    """
    
    content_div = soup.find_all("table", class_="games_list")
    table = content_div[0]
    rows = table.find_all("tr")
    
    data = []

    for row in rows[2:]:
        cells = row.find_all("td")
        if len(cells) == 7:
            data.append([cells[1].text, cells[5].text, int(cells[2].text), 
                        int(cells[4].text), datetime.strptime(cells[0].text, "%H:%M")])


    return data


def player_matching(data):

    date_for_match = data["datetime"]
    player_left_match = data["home_name"]
    player_right_match = data["away_name"]
    # Ustawienie strefy czasowej na GMT+0 (UTC)
    gmt_utc = pytz.timezone('UTC')
    date_for_match = gmt_utc.localize(date_for_match)

    gmt_plus_1 = pytz.timezone('Europe/Warsaw')  # Warszawa to strefa GMT+1 w zimie i GMT+2 w lecie
    date_for_match = date_for_match.astimezone(gmt_plus_1)
    url = generate_url_for_date(date_for_match)

    links, datatimes = get_tournament_links_from_calendar_day(url)

    for i, link in enumerate(links):
        if datatimes[i].hour > date_for_match.hour:
            continue
        soup = get_html_from_page(link)
        try:
            tour_matches_data = scrape_match_data(soup)
        except Exception as e:
            print(f"{e}")

        # Start matching
        for l, r, elo_l, elo_r, hour  in tour_matches_data:
            if hour.hour == date_for_match.hour and hour.minute == date_for_match.minute:
                if fuzz.partial_ratio(l, player_left_match) > 80 and \
                    fuzz.partial_ratio(r, player_right_match) > 80:
                        p_left = players_coll.find_one({"player_name": l})
                        if p_left:
                            e_l = p_left.get('elo', elo_l)
                        else:
                            e_l = elo_l
                        p_right = players_coll.find_one({"player_name": r})
                        if p_right:
                            e_r = p_right.get('elo', elo_r)
                        else:
                            e_r = elo_r
                        if abs(e_l - elo_l) < 100 and abs(e_r - elo_r) < 100:
                            # players matched
                            return p_left, p_right

def main():

    # Simulation - for now get last match offer from db
    # Fetch the last document based on the _id field
    last_match_from_offer = list(calendar_coll.find().sort('_id', -1).limit(1))[0]

    player_matching(last_match_from_offer)

if __name__ == "__main__":
    main()