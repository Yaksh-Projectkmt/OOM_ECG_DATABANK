import pandas as pd
from pymongo import MongoClient

# ==== CONFIG ====
csv_file = r"/home/system/ecgdatabank_new/scripts/testfile.csv"  # CSV file path
id_column = "PatientID"                             # column name in CSV containing patient IDs
mongo_uri = "mongodb://192.168.1.65:27017/"          # Mongo URI
db_name = "ecgarrhythmias"                                 # single DB name
field_name = "server"                          # field name to add/update
field_value = "Test"                            # value to set for the field
update = True                                        # False = dry-run

# ==== LOAD PATIENT IDS ====
df = pd.read_csv(csv_file)
print(f"CSV columns: {df.columns.tolist()}")

if id_column not in df.columns:
    raise ValueError(f"Column '{id_column}' not found in CSV")

patient_ids = df[id_column].dropna().astype(str).unique().tolist()

if not patient_ids:
    print("No patient IDs found in CSV.")
    exit()

print(f"Loaded {len(patient_ids):,} patient IDs from CSV.")

# ==== CONNECT TO MONGO ====
client = MongoClient(mongo_uri)
db = client[db_name]
print(f"\n=== Processing Database: {db_name} ===")

# ==== LOOP OVER COLLECTIONS ====
for collection_name in db.list_collection_names():
    collection = db[collection_name]

    count = collection.count_documents({"PatientID": {"$in": patient_ids}})

    if count == 0:
        continue  # skip if no matches

    if update:
        # bulk update all matching docs
        result = collection.update_many(
            {"PatientID": {"$in": patient_ids}},
            {"$set": {field_name: field_value}}
        )
        print(f"Updated {result.modified_count:,} docs in '{collection_name}' "
              f"with {field_name}='{field_value}'")
    else:
        print(f"Would update {count:,} docs in '{collection_name}' "
              f"with {field_name}='{field_value}'")

if not update:
    print("\n Dry-run completed. Set update=True to actually update.")
else:
    print("\n Update completed.")
