# db_image_store.py
import gridfs
from pymongo import MongoClient
from django.conf import settings
import os 

mongo_uri = os.getenv("MONGO_HOST")
mongo_client = MongoClient(mongo_uri)
db = mongo_client["Beat_search"]

fs = gridfs.GridFS(db)


def save_image_to_db(
    image_path,
    batch_id,
    csv_name,
    lead,
    meta=None
):
    with open(image_path, "rb") as f:
        return fs.put(
            f,
            image_path=image_path.split("\\")[-1],
            batch_id=batch_id,
            csv_name=csv_name,
            lead=lead,
            metadata=meta or {}
        )


def get_images_by_batch(batch_id):
    return fs.find({"batch_id": batch_id})


def get_image(image_id):
    return fs.get(image_id)