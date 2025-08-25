# app/utils/db.py
from pymongo import MongoClient
import os

MONGO_URI = os.getenv("MONGO_URI", "mongodb://localhost:27017")
client = MongoClient(MONGO_URI)
print("db.py loaded")

db = client['your_database_name']   # Change 'your_database_name' accordingly
