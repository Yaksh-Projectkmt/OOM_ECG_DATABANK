from django.shortcuts import render, redirect
from django.contrib.auth.hashers import make_password, check_password
from django.conf import settings
from django.contrib import messages
from pymongo import MongoClient
from django.http import JsonResponse
from concurrent.futures import ThreadPoolExecutor
from django.views.decorators.csrf import csrf_protect
import json
from datetime import datetime, timedelta
import threading
from django.utils import timezone
import uuid
# Connect to MongoDB
mongo_client = MongoClient("mongodb://192.168.1.65:27017/")
db = mongo_client['ecgarrhythmias']
patients_db = mongo_client['Patients']
users_collection = db["users"]
sessions_collection = db["sessions"] 
def home(request):
    if 'user_session' in request.session:
        return redirect('/ommecgdata/')
    else:
        return redirect('/auth/login/')  # explicitly use auth prefix
  # use the login route you already made

collections = [
    'Myocardial Infarction',
    'Atrial Fibrillation & Atrial Flutter',
    'HeartBlock',
    'Junctional Rhythm',
    'Premature Atrial Contraction',
    'Premature Ventricular Contraction',
    'Ventricular Fibrillation and Asystole',
    'Noise',
    'LBBB',
    'Artifacts',
    'SINUS-ARR',
    'ShortPause',
    'TC',
    'WIDE-QRS',
    'Abnormal',
    'Normal',
    'Others'
]
# User Registration
@csrf_protect
def register(request):
    if request.method == "POST":
        username = request.POST.get('username')
        email = request.POST.get('email')  # case-sensitive key!
        phone = request.POST.get('phone')
        password = request.POST.get('password')
        hashed_password = make_password(password)

        # Check if user already exists
        if users_collection.find_one({"username": username}):
            messages.error(request, "Username already exists.")
            return redirect('register')

        # Insert user into MongoDB
        users_collection.insert_one({
            "username": username,
            "email": email,
            "phone": phone,
            "password": hashed_password
        })

        messages.success(request, "Registration successful! You can log in now.")
        return redirect('login')  # Make sure 'login' is a valid URL name in your urls.py

    return render(request, 'authuser/register.html')

# User Login
def login(request):
    # If user is already logged in ? redirect
    if 'user_session' in request.session:
        return redirect('/ommecgdata/')

    if request.method == "POST":
        username = request.POST.get('username')
        password = request.POST.get('password')

        user = users_collection.find_one({"username": username})

        if user and check_password(password, user['password']):
            # Generate a unique session ID
            session_id = str(uuid.uuid4())

            # Store session in Django session
            request.session['user_session'] = {
                "session_id": session_id,
                "user_id": str(user['_id']),
                "username": user['username'],
                "email": user['email'],
                "phone": user.get('phone'),
            }

            # Store session in DB
            sessions_collection.insert_one({
                "_id": session_id,
                "user_id": str(user['_id']),
                "username": user['username'],
                "login_time": timezone.now(),
            })

            return redirect('/ommecgdata/')

        messages.error(request, "Invalid username or password.")
        return redirect('login')

    return render(request, 'authuser/login.html')


def profile(request):
    if 'user_session' not in request.session:
        messages.error(request, "You need to log in first.")
        return redirect('login')

    user_session = request.session['user_session']
    context = {
        "username": user_session['username'],
        "email": user_session['email'],
#        "phone":user_session['phone']
    }
    return render(request, 'authuser/profile.html', context)

@csrf_protect
def change_password(request):
    if 'user_session' not in request.session:
        return JsonResponse({"error": "You need to log in first."}, status=401)

    if request.method != "POST":
        return JsonResponse({"error": "Invalid request method."}, status=405)

    try:
        data = json.loads(request.body.decode('utf-8'))
        current_password = data.get('currentPassword', '').strip()
        new_password = data.get('newPassword', '').strip()

        if not current_password or not new_password:
            return JsonResponse({"error": "All password fields are required."}, status=400)

        user_session = request.session['user_session']
        user = users_collection.find_one({"username": user_session['username']})

        if not user or not check_password(current_password, user['password']):
            return JsonResponse({"error": "Current password is incorrect."}, status=400)

        if check_password(new_password, user['password']):
            return JsonResponse({"error": "New password must be different from the current password."}, status=400)

        hashed_new_password = make_password(new_password)
        users_collection.update_one(
            {"username": user_session['username']},
            {"$set": {"password": hashed_new_password}}
        )

        return JsonResponse({"message": "Password changed successfully."}, status=200)

    except json.JSONDecodeError:
        return JsonResponse({"error": "Invalid JSON data."}, status=400)
    except Exception as e:
        return JsonResponse({"error": str(e)}, status=500)


# User Dashboard
def dashboard(request):
    return redirect('/ommecgdata/')

# User Logout
def logout(request):
    request.session.flush()  # Clear session
    messages.success(request, "Logged out successfully.")
    return redirect('login')



def process_collection_data(collection_name):
    """Process collection data directly from Patients DB (fast)."""
    collection = patients_db[collection_name]

    # Get all patients in this collection
    cursor = collection.find({}, {"PatientID": 1, "total_records": 1, "total_time": 1})

    unique_patient_ids = set()
    total_documents = 0
    patient_data = {}
    for doc in cursor:
        pid = doc["PatientID"]
        unique_patient_ids.add(pid)
        total_documents += doc.get("total_records", 0)
        patient_data[pid] = {
            "document_count": doc.get("total_records", 0),
            "total_data_length": doc.get("total_time", 0),
        }

    # Recent patients (last 10)
    recent_docs = list(
        collection.find({}, {"PatientID": 1, "total_records": 1, "total_time": 1})
                  .sort("_id", -1)
                  .limit(10)
    )

    return total_documents, list(unique_patient_ids), patient_data, recent_docs

@csrf_protect
def update_profile(request):
    if 'user_session' not in request.session:
        return JsonResponse({"error": "You need to log in first."}, status=401)

    if request.method == "POST":
        try:
            data = json.loads(request.body)
            username = data.get('username')
            email = data.get('email')
            phone = data.get('phone')

            # Required fields
            if not username or not email:
                return JsonResponse({"error": "Username and email are required."}, status=400)

            # Validate email with regex
            email_pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
            if not re.match(email_pattern, email):
                return JsonResponse({"error": "Invalid email format."}, status=400)

            # ?? Validate phone (optional, but if provided must be 10 digits)
            if phone:
                if not re.fullmatch(r'\d{10}', str(phone)):
                    return JsonResponse({"error": "Phone number must be exactly 10 digits."}, status=400)
                phone = int(phone)
            else:
                phone = None

            current_username = request.session['user_session']['username']

            # Prepare update data
            update_data = {
                "username": username,
                "email": email,
            }
            if phone is not None:
                update_data["phone"] = phone

            # Update user in database
            users_collection.update_one(
                {"username": current_username},
                {"$set": update_data}
            )

            # Update session
            request.session['user_session'] = {
                "username": username,
                "email": email,
                "phone": phone if phone is not None else request.session['user_session'].get('phone'),
            }

            return JsonResponse({
                "success": True,
                "message": "Profile updated successfully.",
                "profile": {
                    "username": username,
                    "email": email,
                    "phone": phone if phone is not None else request.session['user_session'].get('phone')
                }
            })

        except Exception as e:
            return JsonResponse({"error": str(e)}, status=500)

    return JsonResponse({"error": "Invalid request method."}, status=405)