import os
import json
from pathlib import Path
from rocksdict import Rdict

BASE_DIR = Path(__file__).resolve().parents[1]

# Paths
file_path = BASE_DIR / "rocks_storage" / "sample.json"
db_path = BASE_DIR / "rocks_storage" / "db"

# Ensure the sample JSON file exists
if not os.path.exists(file_path):
    raise FileNotFoundError(f"JSON file {file_path} not found. Please ensure the file exists.")

# Load JSON data
with open(file_path, "r", encoding="utf-8") as file:
    data = json.load(file)

# Open RocksDB instance
db = Rdict(db_path)

try:
    # Insert data into RocksDB
    for paper_id, paper_data in data.items():
        db[paper_id] = json.dumps(paper_data)
    print("Data successfully inserted into RocksDB.")

    # Retrieve and verify data
    print("\nVerifying data...")
    for paper_id in data.keys():
        retrieved_data = json.loads(db[paper_id])
        print(f"Retrieved for {paper_id}: {retrieved_data}")
        assert retrieved_data == data[paper_id], f"Mismatch for {paper_id}"

    print("\nAll data verified successfully.")
finally:
    # Close RocksDB
    db.close()
    print("RocksDB closed.")
