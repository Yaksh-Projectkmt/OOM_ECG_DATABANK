import gridfs
from django.utils import timezone
from django.utils.crypto import get_random_string
from pymongo import MongoClient
import os
# Connect to MongoDB
mongo_uri = os.getenv("MONGO_HOST")

# Create client
mongo_client = MongoClient(mongo_uri)

admin_db = mongo_client["admin"]  # admin DB
user_files = gridfs.GridFS(admin_db)

# Connect to MongoDB
def save_to_gridfs(file, username, label):
    if not file:
        return None
    
    file_id = user_files.put(
        file.read(),
        filename=f"{username}_{label}",
        content_type=file.content_type,
        upload_date=timezone.localtime(timezone.now()).strftime("%Y-%m-%d %H:%M:%S")
    )
    return str(file_id)
def generate_session_token():
    return get_random_string(64)
