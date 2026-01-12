from pymongo import MongoClient

# Połączenie z MongoDB
client = MongoClient('')
db = client['czech_liga_pro_test']
matches_collection = db['matches']

# Agregacja, aby znaleźć duplikaty
unique_urls = matches_collection.aggregate([
    {"$group": {
        "_id": "$tt_url",        # Grupowanie po tt_url
        "count": {"$sum": 1},     # Zliczanie wystąpień
        "docs": {"$push": "$_id"} # Lista ID dokumentów
    }},
    {"$match": {"count": {"$gt": 1}}}  # Filtrujemy tylko duplikaty
])

# Usuwamy duplikaty
for item in unique_urls:
    doc_ids = item['docs']
    matches_collection.delete_many({"_id": {"$in": doc_ids[1:]}})

print("Duplikaty zostały usunięte.")