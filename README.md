Celem projektu jest wykonywanie predykcji meczów dla czeskiej ligi tenisa stołowego.

![Diagram architektury projektu](Blank%20diagram.png)

Kluczowe procesy i gdzie ich szukać:

1. Pobieranie / scraping danych — `scripts/`, `real-time/`
2. ETL i zapis do MongoDB — `scripts/`, `utils/`
3. Trening i ewaluacja modeli — `ai_training/`
4. Generowanie predykcji i eksport wyników — `real-time/`, `simulator/`
5. Utrzymanie 24/7 (monitoring, restart) — skrypty w `real-time/`

Krótka mapa plików:

- `ai_training/` — trening, eksperymenty, przykłady zapisu/ładowania modeli
- `real-time/` — integracje, pobieranie ofert, generowanie predykcji
- `simulator/` — symulacje i testy wpływu modeli na decyzje
- `scripts/`, `utils/` — ETL, pomocnicze skrypty do pozyskiwania danych
- `notebooks/` — analizy i eksperymenty reproducible
- `requirements.txt` — lista zależności
