import psutil
import requests
import time
import json

# Funkcja do ładowania danych konfiguracyjnych z pliku JSON
def load_config(file_path):
    with open(file_path, 'r') as f:
        return json.load(f)

# Ładujemy dane konfiguracyjne z pliku config.json
config = load_config('.creds')

# Wczytanie tokena bota i ID czatu
BOT_TOKEN = config["BOT_TOKEN"]
CHAT_ID = ""

# Nazwa procesu, który chcesz monitorować
PROCESSES = ["get_actual_offer_from_bookie", "get_latest_history_from_tt_league_czech", "find_value_two"]

# Funkcja do wysyłania powiadomienia na Telegram
def send_telegram_message(message):
    url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage"
    params = {
        'chat_id': CHAT_ID,
        'text': message
    }
    response = requests.get(url, params=params)
    return response.json()

# Funkcja do sprawdzania, czy proces działa
def check_process(process_name):
    for proc in psutil.process_iter(['pid', 'name']):
        try:
            # Sprawdzamy pełną linię komend procesu (np. "python3 test.py")
            cmdline = ' '.join(proc.cmdline()).lower()
            if process_name.lower() in cmdline:
                return True
        except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
            # Obsługuje błędy związane z dostępem lub zakończonymi procesami
            pass
    return False
# Monitorowanie procesów
def monitor_process():
    while True:
        for process_name in PROCESSES:
            if not check_process(process_name):
                message = f"Uwaga! Proces {process_name} nie działa."
                send_telegram_message(message)
        
        # Czekamy 60 sekund przed kolejnym sprawdzeniem
        time.sleep(60)

if __name__ == "__main__":
    monitor_process()
