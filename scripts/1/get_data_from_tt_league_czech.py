import sys
import os

sys.path.append(os.path.abspath('/data/paletki/real-time'))

from datetime import datetime, timedelta
from bs4 import BeautifulSoup
import requests
import re
import pymongo
import random
from proxy_list import proxy_list
import time

#region Connection to database settings
conn = pymongo.MongoClient('')
db = conn['czech_liga_pro_test']
matches_coll = db['matches']
players_coll = db['players']
leagues_coll = db['leagues']
session = requests.Session()
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
        list's: Tablice poszczegolnych parametrow, z brakujacym szczegolowym przebiem meczu. Jezeli turniej jest 4-osobowy,
                to dlugosc tablic to 8, jezeli 5-osobowy to dlugosc wynosi 14.
    """
    elo_lewego = []
    delta_elo = []
    godzina = []
    imie_lewego = []
    imie_prawego = []
    wynik_setowy = []
    exact_scores = [] 
    elo_prawego = []
    url_spotkania = []
    winner = []
    
    rows = soup.find_all("tr")
    
    divs = soup.find_all("div", class_="title")
    for title_div in divs:
        h1_tag = title_div.find('h1')  # Szukamy <h1> w danym <div>
        if h1_tag:
            name = h1_tag.text.strip()
    tournament_date = soup.find_all("span", class_="day")[0].text.strip()

    # Pętla po każdym wierszu
    for row in rows:
        # Sprawdzamy, czy wiersz zawiera dane o meczach (czy ma <a> z odpowiednimi danymi)
        time = row.find_all('a', class_='undr')
        players = row.find_all('a', href=True)
        score = row.find_all('a', class_='undrr bold')

        # Jeśli wiersz zawiera informacje o meczu (ma czas i dwóch graczy)
        if time and len(players) >= 4 and score:
            # Czas meczu (pierwszy <a> tag)
            match_time = time[0].text.strip()
            if match_time:
                parsed_datetime = parse_datetime(tournament_date, match_time)
                if parsed_datetime:
                    godzina.append(parsed_datetime)

            game_link = time[0]['href']  # Link do meczu z atrybutu href
            url_spotkania.append(f"https://tt.league-pro.com/{game_link}")

            # Wyciąganie graczy (drugi i czwarty <a> tag)
            player_left = players[1].text.strip()  # Gracz lewy
            imie_lewego.append(player_left)
            player_right = players[3].text.strip()  # Gracz prawy
            imie_prawego.append(player_right)

            # Wynik meczu (trzeci <a> tag)
            match_result = score[0].text.strip()
            wynik_setowy.append(match_result)

            # Wyciąganie ratingów
            ratings = row.find_all('b', class_='small')
            rating_left = rating_right = None
            change_left = change_right = exact_score = None
            
            if len(ratings) >= 2:
                # Ratingi
                rating_left = ratings[0].text.strip()
                elo_lewego.append(rating_left)
                rating_right = ratings[1].text.strip()
                elo_prawego.append(rating_right)

                # Zmiana ratingu (szukamy tagów <small> obok b)
                small_tags = row.find_all('small')
                if len(small_tags) >= 3:
                    # Zmiana ratingu dla lewego gracza
                    change_left = small_tags[0].text.strip()
                    delta_elo.append(change_left)
                    if float(change_left) > 0:
                        winner.append(1)
                    else:
                        winner.append(0)
                    # szczegolowy wynik setow
                    exact_score = small_tags[1].text.strip()
                    exact_scores.append(exact_score)
                    # Zmiana ratingu dla prawego gracza
                    #change_right = small_tags[2].text.strip()
    return godzina, name, url_spotkania, imie_lewego, elo_lewego, delta_elo, wynik_setowy, exact_scores, elo_prawego, imie_prawego, winner

def parse_datetime(date_str, time_str):
    """
    Łączy datę i godzinę w obiekt datetime, konwertując skrócone nazwy miesięcy na numery miesięcy.

    Args:
        date_str (str): Data w formacie np. '8 Dec 2024'.
        time_str (str): Godzina w formacie np. '16:00'.

    Returns:
        datetime: Obiekt datetime w formacie: 2024-12-08 16:00, lub None, jeśli wystąpił błąd.
    """
    try:
        MONTHS = {
            "Jan": "01", "Feb": "02", "Mar": "03", "Arp": "04", "May": "05", "Jun": "06",
            "Jul": "07", "Aug": "08", "Sep": "09", "Oct": "10", "Nov": "11", "Dec": "12"
        }

        date_parts = date_str.split()
        if len(date_parts) == 3:
            day, month, year = date_parts
            month_number = MONTHS.get(month.capitalize())
            if month_number:
                date_str = f"{year}-{month_number}-{day.zfill(2)}"
            else:
                raise ValueError(f"Nieznany skrócony miesiąc: {month}")

        combined_str = f"{date_str} {time_str}"
        return datetime.strptime(combined_str, "%Y-%m-%d %H:%M")
    except ValueError as e:
        print(f"Nie udało się sparsować daty i czasu: {e}")
        return None


def koduj_sety(cells_5, cells_6):
    """
    Funkcja pomocnicza do funkcji: get_sets_details. Jej zadaniem jest uporzadkowanie i konwersja surowych danych 
    do uporzadkowanej struktury.
    
    Args:
        2 tablice: cells_5 (Wynik punktowy w formie "x : y"), cells_6 (Czas w sekundach).

    Returns:
        dict: Słownik szczegolowego przebiegu dla kazdego z 5 setów.
        str: Komunikat błędu w przypadku problemów.
    """
    sets = [[] for _ in range(5)]
    current_set = 0
    
    previous_a_points = 0 
    previous_b_points = 0

    
    for i in range(len(cells_5)):
        wynik = cells_5[i]   # Wynik punktowy w formie "x : y"
        czas = cells_6[i]    # Czas w sekundach
        
        if wynik and czas:
            points = wynik.split(" : ")
            player_a_points = int(points[0])
            player_b_points = int(points[1])

            # Zidentyfikuj, kto zdobył punkt, patrząc na różnicę w punktach
            if player_a_points > previous_a_points:
                player = 'A'
            else:
                player = 'B'

            # Zakodowanie punktu w formacie A5 dla gracza A, który zdobył punkt w 5 sekundzie
            encoded_point = f"{player}{czas.split()[0]}"

            # Dodaj punkt do aktualnego seta
            sets[current_set].append(encoded_point)

            # Zaktualizuj poprzedni stan meczu
            previous_a_points = player_a_points
            previous_b_points = player_b_points

            # Sprawdzenie, czy set się zakończył
            if (player_a_points >= 11 or player_b_points >= 11) and abs(player_a_points - player_b_points) >= 2:
                # Resetuj poprzednie punkty na początku nowego seta
                previous_a_points = 0
                previous_b_points = 0
                current_set += 1  # Przechodzimy do kolejnego seta
                if current_set >= len(sets):  # Jeśli mamy już 5 setów, kończymy
                    break

    # Uzupełniamy brakujące sety, jeśli jest ich mniej niż 5
    for i in range(len(sets)):
        if not sets[i]:
            sets[i] = None  # Wypełniamy None w przypadku braku danych

    return sets


def get_sets_details(game_link):
    """
    Pobiera szczegóły przeiegu meczu z podanego linku: Kto po kolei i w jakim czasie zdobyl punkt w danym secie.
    Format(ciag znakow): A12B5 - Zawodnik A(home) zdobyl punkt w czasie 15 sekund. Nastepny punkt zdobyty 
                        przez B(away) w czasie 5 sekund. Obowiazuje kolejnosc zdobycia punktow od lewej do prawej.

    Args:
        game_link (str): URL strony z detalami meczu.

    Returns:
        dict: Słownik z wynikami setów.
        str: Komunikat błędu w przypadku problemów.
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
        response = session.get(game_link, headers=headers)
        response.raise_for_status()
        content = response.content
        soup = BeautifulSoup(content, "lxml")

        #get info about match category
        div = soup.find_all("div", class_="score")
        final = div[0].find_all("h4", string=lambda text: "Final" in text if text else False)
        third = div[0].find_all("h4", string=lambda text: "3rd" in text if text else False)
        group = div[0].find_all("h4", string=lambda text: "group" in text if text else False)
        match_category = ""
        if final:
            match_category = "final"
        elif third:
            match_category = "3rd"
        else:
            match_category = "group"

        tables = soup.find_all("table", class_="bordered-table")
        if not tables:
            raise ValueError("Nie znaleziono tabel z klasą 'bordered-table points'.")
        
        last_table = tables[-1]
        cells_5 = []
        cells_6 = []

        rows = last_table.find_all("tr")
        for row in rows:
            cells = row.find_all("td")
            
            if len(cells) >= 6:
                cells_5.append(cells[4].get_text(strip=True))
                cells_6.append(cells[5].get_text(strip=True))
        
        cells_5 = [cell for cell in cells_5 if not re.search(r'[a-zA-Z]', cell) and cell.strip() != '']
        cells_6 = [cell for cell in cells_6 if cell.strip() != '']
        
        if len(cells_5) != len(cells_6):
            raise ValueError(f"Długości tablic cells_5 i cells_6 różnią się: {len(cells_5)} != {len(cells_6)}")
        
        sets = koduj_sety(cells_5, cells_6)
        set_dict = {f'set_{i+1}': set_points for i, set_points in enumerate(sets)}
        
        return set_dict, match_category
        
    except requests.exceptions.RequestException as e:
        return f"An error occurred while fetching the game details: {e}"
    except ValueError as e:
        return f"Error processing game details: {e}"
    except Exception as e:
        return f"An unexpected error occurred: {e}"


def create_final_JSON(index, data_without_match_details, set_dict):
    """
    Tworzy finalny słownik JSON zawierający szczegóły meczu.
    
    Returns:
        dict: Słownik zawierający szczegóły meczu w ustrukturalizowanej formie.
              Wartości są przekształcone do odpowiednich typów danych (datetime,str, int, float).
              
    Struktura zwróconego słownika:
    {
        "datetime": datetime,    # Data i godzina meczu (obiekt datetime)
        "league_name": str,      # Nazwa ligi
        "tt_url": str,           # URL do szczegółów meczu
        "odds_url": None,        # Brak danych o kursach
        "home_name": str,        # Nazwa gospodarza
        "home_id": None,         # Brak ID gospodarza
        "home_elo": int,         # ELO gospodarza (int)
        "delta_elo": float,      # Różnica ELO (float)
        "ft_result": str,        # Pełny wynik meczu
        "result": str,           # Krótki wynik meczu
        "away_elo": int,         # ELO gościa (int)
        "away_name": str,        # Nazwa gościa
        "winner": int,           # Zwycięzca meczu (1 = gospodarz, 0 = gość)
        "elo_diff": int,         # Różnica ELO między zawodnikami
        "odd_home": None,        # Brak danych o kursach gospodarza
        "odd_away": None,        # Brak danych o kursach gościa
        "Set1": str,             # Przebieg pierwszego seta
        "Set2": str,             # Przebieg drugiego seta
        "Set3": str,             # Przebieg trzeciego seta
        "Set4": str,             # Przebieg czwartego seta
        "Set5": str,             # Przebieg piątego seta
    }
    """
    
    final_JSON = {
        "datetime": data_without_match_details[0][index],
        "league_name": str(data_without_match_details[1]), 
        "tt_url": str(data_without_match_details[2][index]), 
        "odds_url": None,
        "home_name": str(data_without_match_details[3][index]), 
        "home_id": None, 
        "home_elo": int(data_without_match_details[4][index]),
        "delta_elo": float(data_without_match_details[5][index]),
        "ft_result": str(data_without_match_details[6][index]), 
        "result": str(data_without_match_details[7][index]), 
        "away_elo": int(data_without_match_details[8][index]), 
        "away_name": str(data_without_match_details[9][index]), 
        "winner": int(data_without_match_details[10][index]),
        "elo_diff": int(data_without_match_details[4][index]) - int(data_without_match_details[8][index]),  
        "odd_home": None,
        "odd_away": None,  
        "Set1": set_dict['set_1'], 
        "Set2": set_dict['set_2'], 
        "Set3": set_dict['set_3'],  
        "Set4": set_dict['set_4'],  
        "Set5": set_dict['set_5'],  
    }
    return final_JSON

def add_player_to_collection(player_name, player_elo, last_update):
    if not players_coll.count_documents({ 'player_name': player_name }):
        player_data = {
            "player_name": player_name,
            "player_id": "",
            "elo": player_elo,
            "last_elo_update": last_update,
        }
        players_coll.insert_one(player_data)
    else:
        # check if ELO need to update
        player = players_coll.find_one({"player_name": player_name})
        update_in_db = player.get('last_elo_update', None)
        if last_update > update_in_db:
            players_coll.update_one(
                {"player_name": player_name},
                {"$set": {"elo": player_elo, "last_elo_update": last_update}}
            )

def add_league_to_collection(league_name):
    if not leagues_coll.count_documents({ 'league_name': league_name }):
        league_data = {
            "league_name": league_name,
        }
        leagues_coll.insert_one(league_data)

##############################
###########  MAIN  ###########
##############################

def main():
    """
    Główna funkcja programu do pobierania i przetwarzania danych o turniejach.
    """

    # Ustawienie zakresu dat
    today = datetime.today()
    start_date = today - timedelta(days=1)

    current_date = today
    while current_date >= start_date:
        url = generate_url_for_date(start_date)
        print(f"Scraping URL: {url}")
        
        #today = datetime.today()
        #current_date = today
        #url = generate_url_for_date(current_date)

        links = get_tournament_links_from_calendar_day(url)

        for link in links:
            soup = get_html_from_page(link)
            try:
                data_without_match_details = scrape_match_data(soup)
            except Exception as e:
                print(f"Błąd podczas scrapowania match data {link}: {e}")
                with open("games_problem.txt", "a+") as file:
                    file.write(link)
                continue
            ids_to_pop = []
            for l, ft_res in enumerate(data_without_match_details[6]):
                if "3" not in ft_res:
                    ids_to_pop.append(l)
                    continue
            
            # Sortowanie indeksów malejąco, aby uniknąć problemów z przesuwaniem elementów
            ids_to_pop.sort(reverse=True)
            for ids in ids_to_pop:
                for el in data_without_match_details:
                    if isinstance(el, list):
                        try:
                            el.pop(ids)
                        except:
                            continue
            links_to_single_torunament_games_details = data_without_match_details[2]
            #print(links_to_single_torunament_games_details)
            for index, game_link in enumerate(links_to_single_torunament_games_details):
                if not matches_coll.count_documents({ 'tt_url': game_link }):
                    try:
                        set_dict, match_category = get_sets_details(game_link)
                        final_JSON = create_final_JSON(index, data_without_match_details, set_dict)
                        final_JSON["match_category"] = match_category
                        add_league_to_collection(final_JSON["league_name"])
                        add_player_to_collection(final_JSON["home_name"], final_JSON["home_elo"], final_JSON["datetime"])
                        add_player_to_collection(final_JSON["away_name"], final_JSON["away_elo"], final_JSON["datetime"])
                        
                        matches_coll.insert_one(final_JSON)
                        
                    except Exception as e:
                        print(f"Błąd podczas przetwarzania meczu {game_link}: {e}")
        #time.sleep(1800)
        start_date += timedelta(days=1)


if __name__ == "__main__":
    main()