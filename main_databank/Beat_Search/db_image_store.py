from pymongo import MongoClient
from datetime import datetime
import os 
mongo_uri = os.getenv("MONGO_HOST")
mongo_client = MongoClient(mongo_uri)
db = mongo_client["Beat_search"]


# ---------------------------
# Get label collection
# ---------------------------
def get_label_collection(label: str):
    if not label or not isinstance(label, str):
        raise ValueError("terminology is required")

    safe_label = label.strip()
    if not safe_label:
        raise ValueError("terminology cannot be blank")

    return db[safe_label]


# ---------------------------
# Create base CSV document
# ---------------------------
def create_csv_document(batch_id, csv_name, terminology, hex_payload=None):
    collection = get_label_collection(terminology)

    update_doc = {
        "$setOnInsert": {
            "batch_id": batch_id,
            "csv_name": csv_name,
            "uploaded_at": datetime.utcnow(),
            "leads": []
        }
    }

    if hex_payload:
        update_doc["$set"] = hex_payload

    collection.update_one(
        {
            "batch_id": batch_id,
            "csv_name": csv_name
        },
        update_doc,
        upsert=True
    )


# ---------------------------
# Save one lead result (NEW)
# ---------------------------
def save_lead_result(
    batch_id,
    csv_name,
    lead,
    terminology,
    payload   # ← this is exactly your result_payload
):
    collection = get_label_collection(terminology)

    collection.update_one(
        {
            "batch_id": batch_id,
            "csv_name": csv_name
        },
        {
            "$push": {
                "leads": {
                    "lead": lead,
                    **payload,
                    "created_at": datetime.utcnow()
                }
            }
        },
        upsert=True
    )


# ---------------------------
# Fetch all leads for CSV
# ---------------------------
def get_csv_results(batch_id, csv_name, terminology):
    collection = get_label_collection(terminology)

    doc = collection.find_one(
        {"batch_id": batch_id, "csv_name": csv_name},
        {"_id": 0}
    )

    if not doc:
        return None

    return doc
