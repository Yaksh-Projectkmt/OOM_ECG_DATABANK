import pandas as pd
from pymongo import MongoClient

# ==== CONFIG ====
csv_file = "/home/system/ecgdatabank_new/scripts/unique_patient_ids.csv"   # your CSV file path
id_column = "patientId"    # column name in CSV containing patient IDs
mongo_uri = "mongodb://192.168.1.65:27017/"  # update if needed
db_names = ["ecgarrhythmias", "Patients"]   # <- list both DBs here
delete = True  # ⚠️ Set to False for dry-run

# ==== LOAD PATIENT IDS ====
df = pd.read_csv(csv_file)
print(f"CSV columns: {df.columns.tolist()}")
patient_ids = df[id_column].dropna().unique().tolist()

if not patient_ids:
    print("No patient IDs found in CSV.")
    exit()

print(f"Loaded {len(patient_ids)} patient IDs from CSV.")

# ==== CONNECT TO MONGO ====
client = MongoClient(mongo_uri)

# ==== LOOP OVER DATABASES ====
for db_name in db_names:
    db = client[db_name]
    print(f"\n=== Processing Database: {db_name} ===")
    
    for collection_name in db.list_collection_names():
        collection = db[collection_name]
        count = collection.count_documents({"PatientID": {"$in": patient_ids}})
        if delete:
            result = collection.delete_many({"PatientID": {"$in": patient_ids}})
            print(f"Deleted {result.deleted_count} from '{collection_name}'")
        else:
            print(f"Would delete {count} from '{collection_name}'")

if not delete:
    print("Dry-run completed. Set delete=True to actually delete.")
else:
    print("Deletion completed.")
