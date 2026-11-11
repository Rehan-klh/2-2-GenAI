import os
from dotenv import load_dotenv
from pymongo import MongoClient

load_dotenv(override=True)

client = MongoClient(
    os.getenv("MONGODB_URI"),
    serverSelectionTimeoutMS=5000
)

try:
    client.admin.command("ping")
    print("MongoDB connection successful!")
    print("Cluster is reachable.")
except Exception as e:
    print("MongoDB connection failed:", e)
finally:
    client.close()