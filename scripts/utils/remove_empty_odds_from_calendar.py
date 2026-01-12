from pymongo import MongoClient

# Połączenie z MongoDB
client = MongoClient('')
db = client['czech_liga_pro_test']
matches_collection = db['match_calendar']

# znajdz puste oddsy
unique_matches = matches_collection.find({"away_odd": {"$exists": False}})

# Usuwamy puste oddsy
for item in unique_matches:
    matches_collection.delete_many({"_id": item["_id"]})

print("Puste oddsy zostały usunięte.")