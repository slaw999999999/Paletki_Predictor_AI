from pymongo import MongoClient
import time

# Połączenie z bazą danych MongoDB
conn = MongoClient('')
db = conn['czech_liga_pro_test']
collection_a = db['match_calendar']
collection_b = db['test_calendar']

# Krok 1: Pobranie najnowszych 30 rekordów z kolekcji A, posortowanych po dacie malejąco
latest_documents = collection_a.find().sort('datetime', -1).limit(30)

# Krok 2: Skopiowanie najnowszych 30 rekordów do kolekcji B
for doc in latest_documents:
    # Usuwamy pole '_id', aby uniknąć konfliktów, jeśli dokumenty z tym samym '_id' byłyby wstawiane
    doc.pop('_id', None)
    collection_b.insert_one(doc)
print("Najnowsze 30 rekordów zostało skopiowanych z kolekcji A do kolekcji B.")