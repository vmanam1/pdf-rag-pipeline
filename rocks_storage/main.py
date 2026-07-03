import json
from pathlib import Path
from rocksdict import Rdict

BASE_DIR = Path(__file__).resolve().parent

# Load JSON file
file_path = BASE_DIR / "sample.json"
with open(file_path, "r", encoding="utf-8") as file:
    data = json.load(file)

# Open RocksDB instance
db_path = BASE_DIR / "db"
db = Rdict(db_path)

# Insert data into RocksDB
for paper_id, paper_data in data.items():
    db[paper_id] = json.dumps(paper_data)

# Verify and Retrieve data
retrieved_data = db["paper_1"]
print(json.loads(retrieved_data))

# Close RocksDB
db.close()
