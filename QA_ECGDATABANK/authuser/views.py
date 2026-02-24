from django.shortcuts import render, redirect
from django.contrib.auth.hashers import make_password, check_password
from django.conf import settings
from django.contrib import messages
from pymongo import MongoClient
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_protect,csrf_exempt
from datetime import datetime
from django.utils import timezone
from django.views.decorators.cache import never_cache
from bson.objectid import ObjectId
from authuser.email_Send import send_welcome_email, send_documents_processing_email,send_approved_email,send_rejected_email
from django.contrib.auth import get_user_model, login as django_login
import os, uuid, gridfs, re,base64, json
from reportlab.lib.pagesizes import A4
from datetime import datetime
from django.http import JsonResponse
import re

# Connect to MongoDB
# Connect to MongoDB
mongo_uri = os.getenv("MONGO_HOST")
Dev_MONGO_URI = os.getenv("DEV_HOST")
Dev_DB_NAME = os.getenv("DEV_MONGO_DB")
Databank_db = os.getenv("MONGO_DB")

# Create client
mongo_client = MongoClient(mongo_uri)
dev_mongo_client = MongoClient(Dev_MONGO_URI)

#Database
db = mongo_client['ecgarrhythmias1']
admin_db = mongo_client["admin"]
patients_db = mongo_client['Patients']
media_db = mongo_client["Download_files"]
Dev_DB =dev_mongo_client[Dev_DB_NAME]
#collections
users_collection = admin_db["users"]
sessions_collection = admin_db["sessions"] 
contact_collection = admin_db['contact_messages']
manage_currency_collection = admin_db["currency"]
payment_history_collection = admin_db["payment_history"]
Download_history_collection=admin_db["Download_history"]

download_fs = gridfs.GridFS(media_db, collection="downloads")
    
def home(request):
    if 'user_session' in request.session:
        return redirect('/Beat_Search/')
    else:
        return redirect('/auth/login/')

def help(request):
    return render(request, 'authuser/help.html')

@csrf_protect
@never_cache
def login(request):

    # Already logged in via Django session
    if request.user.is_authenticated:
        return redirect('/Beat_Search/')
    
    if request.method == "POST":
        username = request.POST.get('username')
        password = request.POST.get('password')

        # 1. Check Mongo User
        user = users_collection.find_one({"username": username})
        if not user:
            messages.error(request, "User not found.")
            return redirect('login')

        # 2. Password validation
        if not check_password(password, user['password']):
            messages.error(request, "Invalid password.")
            return redirect('login')

        now = timezone.localtime(timezone.now())
        sessions_collection.insert_one({
            "user_id": str(user["_id"]),
            "username": user["username"],
            "email": user.get("email"),
            "login_time": now.strftime("%Y-%m-%d %H:%M:%S"),
        })
        response = redirect('/Beat_Search/')
        return response

    # ----------------------
    # GET REQUEST → Show login
    # ----------------------
    request.session["show_status_message"] = True

    response = render(request, 'authuser/login.html')
    response['Cache-Control'] = 'no-store, no-cache, must-revalidate, max-age=0'
    response['Pragma'] = 'no-cache'
    response['Expires'] = '0'
    return response
@never_cache
def logout(request):

    response = redirect("login")
    messages.success(request, "Logged out successfully.")
    return response
def get_patients_data(request):
    doc1 = Dev_DB.dashboardcounts
    doc = Dev_DB.dashboardcounts.find_one(
        {"key": "allConts"},
        {
            "_id": 0,
            "patients": 1,
            "activePatients": 1,
            "recordingHours": 1
        }
    )
    if not doc:
        return JsonResponse({
            "count_patients": 0,
            "active_patients": 0,
            "recording_hours": 0
        })

    return JsonResponse({
        "count_patients": doc.get("patients", 0),
        "active_patients": doc.get("activePatients", 0),
        "recording_hours": round(doc.get("recordingHours", 0), 2)
    })