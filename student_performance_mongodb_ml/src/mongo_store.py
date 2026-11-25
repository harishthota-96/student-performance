from pymongo import MongoClient
from pymongo.errors import PyMongoError
from config import MONGODB_URI, MONGODB_DATABASE

def get_collection():
    client = MongoClient(MONGODB_URI, serverSelectionTimeoutMS=5000)
    client.admin.command("ping")
    return client[MONGODB_DATABASE]["students"], client

def import_csv(csv_path):
    import pandas as pd
    collection, client = get_collection()
    try:
        df = pd.read_csv(csv_path)
        records = df.to_dict("records")
        if records:
            collection.create_index("student_id", unique=True)
            for record in records:
                collection.update_one({"student_id": record["student_id"]},
                                      {"$set": record}, upsert=True)
        print(f"Imported/upserted {len(records)} student records.")
    finally:
        client.close()

def export_records():
    collection, client = get_collection()
    try:
        docs = list(collection.find({}, {"_id": 0}))
        return docs
    finally:
        client.close()
