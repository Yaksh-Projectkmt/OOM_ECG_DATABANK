# import pandas as pd
# from pymongo import MongoClient

# # ==== CONFIG ====
# csv_file = "D:\\hitesh\\unique_patient_ids_dev.csv"   # CSV file path
# id_column = "patient_id"                               # CSV column with patient IDs
# mongo_uri = "mongodb://192.168.1.65:27017/"           # Mongo URI
# delete = True  # ⚠️ Set False for dry-run

# # Specify database & collection pairs
# db_collections = [
#     ("ecgarrhythmias", "ShortPause"),
#     ("Patients", "ShortPause")
# ]

# # ==== LOAD PATIENT IDS ====
# df = pd.read_csv(csv_file)
# print(f"CSV columns: {df.columns.tolist()}")
# patient_ids = df[id_column].dropna().unique().tolist()

# if not patient_ids:
#     print("No patient IDs found in CSV.")
#     exit()

# print(f"Loaded {len(patient_ids)} patient IDs from CSV.")

# # ==== CONNECT TO MONGO ====
# client = MongoClient(mongo_uri)

# # ==== LOOP OVER DB-COLLECTION PAIRS ====
# for db_name, collection_name in db_collections:
#     db = client[db_name]
#     collection = db[collection_name]

#     print(f"\n=== Processing {db_name}.{collection_name} ===")
#     count = collection.count_documents({"PatientID": {"$in": patient_ids}})

#     if delete:
#         result = collection.delete_many({"PatientID": {"$in": patient_ids}})
#         print(f"Deleted {result.deleted_count} documents from '{collection_name}'")
#     else:
#         print(f"Would delete {count} documents from '{collection_name}'")

# if not delete:
#     print("✅ Dry-run completed. Set delete=True to actually delete.")
# else:
#     print("✅ Deletion completed.")
import pandas as pd
from pymongo import MongoClient

# ==== CONFIG ====
csv_file = "/home/system/ecgdatabank_new/scripts/unique_patient_ids_dev.csv""
id_column = "patient_id"
mongo_uri = "mongodb://192.168.1.65:27017/"
delete = True  # Set False for dry-run
chunk_size = 1000  # Split patient IDs into chunks for faster deletion

# DB-collection pairs
db_collections = [
    ("ecgarrhythmias", "WIDE-QRS"),
    ("Patients", "WIDE-QRS")
]

# ==== LOAD PATIENT IDS ====
df = pd.read_csv(csv_file)
patient_ids = df[id_column].dropna().unique().tolist()
print(f"Loaded {len(patient_ids)} patient IDs from CSV.")

if not patient_ids:
    print("No patient IDs found in CSV.")
    exit()

# ==== CONNECT TO MONGO ====
client = MongoClient(mongo_uri)

# ==== FUNCTION TO DELETE IN CHUNKS ====
def delete_in_chunks(collection, ids, chunk_size=1000):
    total_deleted = 0
    for i in range(0, len(ids), chunk_size):
        chunk = ids[i:i+chunk_size]
        if delete:
            result = collection.delete_many({"PatientID": {"$in": chunk}})
            total_deleted += result.deleted_count
        else:
            count = collection.count_documents({"PatientID": {"$in": chunk}})
            total_deleted += count
    return total_deleted

# ==== LOOP OVER DB-COLLECTIONS ====
for db_name, collection_name in db_collections:
    db = client[db_name]
    collection = db[collection_name]
    
    print(f"\n=== Processing {db_name}.{collection_name} ===")
    deleted_count = delete_in_chunks(collection, patient_ids, chunk_size)
    
    if delete:
        print(f"Deleted {deleted_count} documents from '{collection_name}'")
    else:
        print(f"Would delete {deleted_count} documents from '{collection_name}'")

if not delete:
    print("Dry-run completed. Set delete=True to actually delete.")
else:
    print("Deletion completed.")

