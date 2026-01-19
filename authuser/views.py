from django.shortcuts import render, redirect
from django.contrib.auth.hashers import make_password, check_password
from django.conf import settings
from django.views.decorators.http import require_GET
from django.contrib import messages
from pymongo import MongoClient
from django.http import JsonResponse,HttpResponse
from django.views.decorators.csrf import csrf_protect,csrf_exempt
from datetime import datetime,timedelta
from django.utils import timezone
from django.views.decorators.cache import never_cache
from bson.binary import Binary
from bson.objectid import ObjectId
from collections import defaultdict
from report.views import collections
from authuser.email_Send import send_welcome_email, send_documents_processing_email,send_approved_email,send_rejected_email,send_password_change_email
from django.contrib.auth.decorators import login_required
from django.contrib.admin.views.decorators import staff_member_required
from django.contrib.auth import get_user_model, login as django_login
from subscription.utils import sync_subscription_from_mongo
from django.contrib.admin.views.decorators import staff_member_required
import os, uuid, gridfs, re,base64, razorpay, json
from subscription.models import Plan
from .models import Wallet, CustomUser  
from .utils import save_to_gridfs, generate_session_token
from subscription.utils import get_download_price, create_download_history
from django.http import HttpResponse
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
from datetime import datetime

# Connect to MongoDB
mongo_uri = os.getenv("MONGO_HOST")

# Create client
mongo_client = MongoClient(mongo_uri)
client = razorpay.Client(auth=(settings.RAZORPAY_KEY_ID, settings.RAZORPAY_KEY_SECRET))

#Database
db = mongo_client['ecgarrhythmias']
admin_db = mongo_client["admin"]
patients_db = mongo_client['Patients']
media_db = mongo_client["Download_files"]

#collections
users_collection = admin_db["users"]
sessions_collection = admin_db["sessions"] 
contact_collection = admin_db['contact_messages']
manage_currency_collection = admin_db["currency"]
payment_history_collection = admin_db["payment_history"]
Download_history_collection=admin_db["Download_history"]


    
download_fs = gridfs.GridFS(media_db, collection="downloads")

#chunk wise store
UserModel = get_user_model()
fs = gridfs.GridFS(admin_db)



@staff_member_required
def registration_requests_admin(request):
    return render(request, "authuser/registration_requests.html")
    
def home(request):
    if 'user_session' in request.session:
        return redirect('/ommecgdata/')
    else:
        return redirect('/auth/login/')
        
def all_download_history(request):
    return render(request, "authuser/all_download_history.html")

def custom_csrf_failure(request, reason=""):
    return render(request, "authuser/csrf_error.html", status=403)

def session_expired(request):
    return render(request,'authuser/session_expired.html')

def patient_list(request):
    return render(request, 'authuser/patient_list.html')

def help(request):
    return render(request, 'authuser/help.html')

def payment_failed(request):
    return render(request, "authuser/payment_failed.html")
    
@require_GET
def get_wallet_balance(request):
    user_session = request.session.get("user_session")

    if not user_session:
        return JsonResponse({"balance": "0.00"})

    email = user_session.get("email")
    if not email:
        return JsonResponse({"balance": "0.00"})

    # Fetch user from MongoDB
    user = users_collection.find_one({"email": email})
    if not user:
        return JsonResponse({"balance": "0.00"})

    # Direct field
    balance = user.get("wallet_balance", "0.00")

    return JsonResponse({"balance": balance})

@never_cache
@csrf_protect
def register(request):
    if request.method != "POST":
        return render(request, "authuser/register.html")

    # 1. Inputs
    userType = request.POST.get("userType")
    username = request.POST.get("username", "").strip()
    email = request.POST.get("email", "").strip().lower()
    country_code = request.POST.get("countryCode", "").strip()
    phone = request.POST.get("phone", "").strip()
    password = request.POST.get("password")
    doctor_id = request.POST.get("doctorId", "").strip()
    professionBox = request.POST.get("professionInput", "").strip()

    student_id_file = request.FILES.get("studentId")
    address_proof_file = request.FILES.get("addressProof")
    doctor_cert_file = request.FILES.get("doctorCert")
    address_proof_other_file = request.FILES.get("addressProofOther")

    # 2. Required validation
    if not (username and email and password):
        messages.error(request, "Please fill all required fields.")
        return redirect("register")

    if userType == "student":
        if not student_id_file or not address_proof_file:
            messages.error(request, "Student ID and Address Proof required.")
            return redirect("register")

    elif userType == "doctor":
        if not doctor_id or not doctor_cert_file:
            messages.error(request, "Doctor ID and Certificate required.")
            return redirect("register")

    elif userType == "other":
        if not professionBox or not address_proof_other_file:
            messages.error(request, "Profession and Address Proof required.")
            return redirect("register")

    else:
        messages.error(request, "Invalid user type.")
        return redirect("register")

    # 3. Duplicate check in MongoDB
    if users_collection.find_one({"username": username}):
        messages.error(request, "Username already exists!")
        return redirect("register")

    if users_collection.find_one({"email": email}):
        messages.error(request, "Email already exists!")
        return redirect("register")

    # 3.1 Duplicate check Django DB
    if UserModel.objects.filter(username=username).exists():
        messages.error(request, "Username already exists in system!")
        return redirect("register")

    if UserModel.objects.filter(email=email).exists():
        messages.error(request, "Email already exists in system!")
        return redirect("register")

    # 4. Country + Currency
    if country_code == "+91":
        country, currency = "India", "INR"
    elif country_code == "+1":
        country, currency = "United States", "USD"
    else:
        country, currency = "Unknown", "USD"

    default_package = "Free"
    default_plan_obj = Plan.objects.filter(name__iexact="Free").first()

    # 5. Save Files in GridFS
    student_id_path = save_to_gridfs(student_id_file, username, "student_id")
    address_proof_path = save_to_gridfs(address_proof_file, username, "address_proof")
    doctor_cert_path = save_to_gridfs(doctor_cert_file, username, "doctor_certificate")
    other_address_path = save_to_gridfs(address_proof_other_file, username, "other_address")

    # 6. Create MongoDB User
    hashed_password = make_password(password)

    user_doc = {
        "role": userType.lower(),
        "username": username,
        "email": email,
        "country_code": country_code,
        "country": country,
        "phone": phone,
        "password": hashed_password,
        "doctorId": doctor_id if userType == "doctor" else None,
        "profession": professionBox if userType == "other" else None,
        "status": "pending",
        "package": default_package,
        "admin_comment": "",
        "register_time": timezone.localtime(timezone.now()).strftime("%Y-%m-%d %H:%M:%S"),
        "files": {
            "student_id": student_id_path,
            "address_proof": address_proof_path or other_address_path,
            "doctor_certificate": doctor_cert_path,
        },
    }

    inserted_user = users_collection.insert_one(user_doc)
    mongo_user_id = str(inserted_user.inserted_id)

    # 7. Create Django admin user
    django_user = UserModel.objects.create_user(
        username=username,
        email=email,
        password=password,
        role=userType.lower(),
        package=default_package.lower(),
        plan=default_plan_obj,
    )

    Wallet.objects.create(user=django_user)

    # 8. Currency Settings Insert
    manage_currency_collection.insert_one({
        "user_id": mongo_user_id,
        "username": username,
        "country_code": country_code,
        "country": country,
        "currency": currency,
        "created_at": timezone.localtime(timezone.now()).strftime("%Y-%m-%d %H:%M:%S"),
    })
    django_login(request, django_user)

    # 9. Create Session (Mongo + Cookie)
    session_id = str(uuid.uuid4())
    token = generate_session_token()
    now = timezone.localtime(timezone.now())
    expiry = now + timezone.timedelta(seconds=settings.CUSTOM_SESSION_EXPIRY_SECONDS)
    sessions_collection.insert_one({
        "_id": session_id,
        "token": token,
        "user_id": mongo_user_id,
        "username": username,
        "email": email,
        "role": userType.lower(),
        "login_time": now.strftime("%Y-%m-%d %H:%M:%S"),
        "expires_at": expiry.strftime("%Y-%m-%d %H:%M:%S"),
        "package": default_package,
        "status": "pending",
    })

    # Store in Django session (optional)
    request.session["user_session"] = {
        "session_id": session_id,
        "token": token,
        "user_id": mongo_user_id,
        "role": userType.lower(),
        "email": email,
        "package": default_package,
    }
    request.session.modified = True
    # 10. Set Secure Cookie
    response = redirect("/ommecgdata/")
    response.set_cookie(
        "session_token",
        token,
        max_age=settings.CUSTOM_SESSION_EXPIRY_SECONDS,
        secure=False,  # Change to True in production HTTPS
        httponly=True,
        samesite="Lax",
    )

    # 11. Send Emails
    send_welcome_email(username, email)
    send_documents_processing_email(username, email)

    return response

def read_file_base64(file_id):
    if not file_id:
        return None
    try:
        f = fs.get(ObjectId(file_id))
        encoded = base64.b64encode(f.read()).decode("utf-8")
        return {
            "filename": f.filename,
            "content_type": f.content_type,
            "data": encoded
        }
    except:
        return None

def get_registrations(request):
    try:
        registrations = []

        for doc in users_collection.find():

            if doc.get("role", "").lower() == "admin":
                continue

            files = doc.get("files", {})
            
            registrations.append({
                "id": str(doc["_id"]),
                "user_name": doc.get("username", ""),
                "email": doc.get("email", ""),
                "phone_number": doc.get("phone", ""),
                "role": doc.get("role", "").lower(),
                "status": doc.get("status", ""),

                # FIX: USE CORRECT KEYS
                "student_id_file": read_file_base64(files.get("student_id")),
                "address_proof_file": read_file_base64(files.get("address_proof")),
                "doctor_certificate_file": read_file_base64(files.get("doctor_certificate")),

                "doctor_id": doc.get("doctorId"),
                "created_at": doc.get("register_time")
            })

        return JsonResponse(registrations, safe=False)

    except Exception as e:
        return JsonResponse({"error": str(e)}, status=500)

#admin
@csrf_exempt
def update_registration_status(request):
    if request.method != "POST":
        return JsonResponse({"error": "Invalid request"}, status=405)
    try:
        data = json.loads(request.body)
        user_id = data.get("id")
        status = data.get("status")
        comment = data.get("comment", "")

        if not user_id or status not in ["approved", "rejected"]:
            return JsonResponse({"error": "Invalid data"}, status=400)

        # Update status in DB
        users_collection.update_one(
            {"_id": ObjectId(user_id)},
            {"$set": {"status": status, "admin_comment": comment}}
        )

        # Fetch user details to get email & username
        user_doc = users_collection.find_one({"_id": ObjectId(user_id)})
        if user_doc:
            username = user_doc.get("username", "User")
            receiver_email = user_doc.get("email")
           
           # Compose email based on status
            if status == "approved":
                send_approved_email(username,receiver_email)
            else:  # rejected
                send_rejected_email(username,receiver_email,comment)

        return JsonResponse({"success": True, "message": f"User {status} successfully"})
    except Exception as e:
        return JsonResponse({"error": str(e)}, status=500)

#login page Backend  
@csrf_protect
@never_cache
def login(request):

    # Already logged in via Django session
    if request.user.is_authenticated:
        return redirect('/ommecgdata/')

    # Already logged in via custom token
    # Auto-login only if token exists AND matches a valid session in Mongo
    token = request.COOKIES.get("session_token")
    if token:
        session = sessions_collection.find_one({"token": token})
        if session:
            return redirect('/ommecgdata/')

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

        # 3. Account Status Validation
        user_status = user.get("status", "pending").lower()

        if user_status == "rejected":
            messages.error(request, "Your account request has been rejected.")
            return redirect('login')

        # if user_status == "pending":
        #     messages.error(request, "Your account is under verification.")
        #     return redirect('login')

        # 4. Sync to Django CustomUser
        User = get_user_model()
        django_user, created = User.objects.get_or_create(
            username=user["username"],
            defaults={
                "email": user.get("email", ""),
                "password": make_password(None),
                "role": user.get("role", "user"),
                "package": user.get("package", "Free").lower(),
            }
        )

        sync_subscription_from_mongo(django_user)

        # Django login (auth middleware)
        django_login(request, django_user)

        # 5. Token Session
        session_id = str(uuid.uuid4())
        token = generate_session_token()
        now = timezone.localtime(timezone.now())
        expiry = now + timezone.timedelta(seconds=settings.CUSTOM_SESSION_EXPIRY_SECONDS)
        sessions_collection.insert_one({
            "_id": session_id,
            "token": token,
            "user_id": str(user["_id"]),
            "username": user["username"],
            "email": user.get("email"),
            "role": user.get("role", "user"),
            "login_time": now.strftime("%Y-%m-%d %H:%M:%S"),
            "expires_at": expiry.strftime("%Y-%m-%d %H:%M:%S"),
            "package": user.get("package", "Free"),
            "status": user_status,
        })

        request.session["user_session"] = {
            "session_id": session_id,
            "token": token,
            "user_id": str(user["_id"]),
            "username": user["username"],
            "email": user.get("email"),
            "phone": user.get("phone"),
            "role": user.get("role", "user"),
            "package": user.get("package", "Free"),
            "status": user_status,
            "admin_comment": user.get("admin_comment", ""),
            "features": user.get("package", "Free")
        }

        # 6. Cookie
        response = redirect('/ommecgdata/')
        response.set_cookie(
            "session_token",
            token,
            max_age=settings.CUSTOM_SESSION_EXPIRY_SECONDS,
            httponly=True,
            secure=True,  # True in HTTPS
            samesite="Lax"
        )
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

    # Correct way to detect session_expired flag
    session_expired = request.GET.get("session_expired", "").strip()

    if session_expired == "1":
        messages.warning(request, "Your session has expired. Please log in again.")
    else:
        messages.success(request, "Logged out successfully.")

    # Remove your custom session cookie
    response.delete_cookie("session_token")

    # Clear Django session
    request.session.flush()

    return response
#query releted service data store Backend
def save_contact(request):
  if request.method == "POST":
      name = request.POST.get("name", "").strip()
      email = request.POST.get("email", "").strip()
      message_text = request.POST.get("message", "").strip()

      # --- Validation ---
      if not name or not email or not message_text:
          messages.error(request, "Please fill out all fields before submitting.")
          return redirect('help')

      # --- Optional: basic email validation ---
      if "@" not in email or "." not in email:
          messages.error(request, "Please enter a valid email address.")
          return redirect('help')

      # --- Save message to MongoDB ---
      contact_data = {
          "name": name,
          "email": email,
          "message": message_text,
          "created_at": datetime.utcnow()
      }

      try:
          contact_collection.insert_one(contact_data)
          messages.success(request, " Your message has been sent successfully! We'll contact you soon.")
          return redirect('/ommecgdata/')
      except Exception as e:
          messages.error(request, f" Something went wrong while saving your message. Please try again later.")

      return redirect('help')

  # If method not POST, redirect safely
  return redirect('/ommecgdata/')

def get_patient_arrhythmia_records(request):
    try:
        # --- Step 1: Get arrhythmia collections ---
        arrhythmia_collections = patients_db.list_collection_names()

        patient_data = defaultdict(lambda: {
            "patient_id": "",
            "total_records": 0,
            "total_time": 0,
            "arrhythmias": [],
            "live": False  # default
        })

        for collection_name in arrhythmia_collections:
            collection = patients_db[collection_name]

            for doc in collection.find({}, {"_id": 0, "PatientID": 1, "total_records": 1, "total_time": 1}):
                patient_id = doc.get("PatientID")
                if not patient_id:
                    continue

                total_records = doc.get("total_records", 0)
                total_time_min = doc.get("total_time", 0) or 0
                total_time_sec = round(total_time_min * 60, 2)

                patient = patient_data[patient_id]
                patient["patient_id"] = patient_id
                patient["total_records"] += total_records
                patient["total_time"] += total_time_sec
                patient["arrhythmias"].append({
                    "type": collection_name,
                    "duration": total_time_sec,
                    "records": total_records
                })

        # --- Step 2: Connect to Live DB and mark live patients ---
        for pid, p in patient_data.items():
            merged = {}

            for arr in p["arrhythmias"]:
                t = arr["type"]
                if t not in merged:
                    merged[t] = {
                        "type": t,
                        "duration": arr["duration"],
                        "records": arr["records"]
                    }
                else:
                    merged[t]["duration"] += arr["duration"]
                    merged[t]["records"] += arr["records"]

            # replace arrhythmias with merged list
            p["arrhythmias"] = list(merged.values())
        try:
            LIVE_URI = "mongodb://readonly_user:9ikJ4Qn1YmG1l1EVF1OQ@192.168.2.131:27017/?authSource=admin"
            DB_NAME = "ecgs"

            live_client = MongoClient(LIVE_URI)
            live_db = live_client[DB_NAME]
            live_collection = live_db["patients"]

            # collect live patient IDs (either `patientId` or `PatientID`)
            live_ids = {
                str(doc.get("patientId") or doc.get("PatientID"))
                for doc in live_collection.find({}, {"_id": 0, "patientId": 1, "PatientID": 1})
                if doc.get("patientId") or doc.get("PatientID")
            }

            # mark live patients in our list
            for pid in live_ids:
                if pid in patient_data:
                    patient_data[pid]["live"] = True

        except Exception as live_err:
            print("Live DB error:", live_err)

        # --- Step 3: Sort and return ---
        patients = sorted(patient_data.values(), key=lambda x: x["patient_id"])

        return JsonResponse({"status": "success", "data": patients}, safe=False)

    except Exception as e:
        return JsonResponse({"status": "error", "message": str(e)})
def get_patients_data(request):
    total_patients = set()
    total_time = 0.0

    for name in collections:
        coll = patients_db[name]
        cursor = coll.find({}, {"PatientID": 1, "total_time": 1})

        group_time = 0.0
        for doc in cursor:
            pid = str(doc.get("PatientID", "")).strip()
            if pid:
                total_patients.add(pid)
            group_time += float(doc.get("total_time", 0))

        total_time += group_time

    totals = {
        "count_patients": len(total_patients),
        "total_time": round(total_time, 2),
    }

    return JsonResponse(totals)

#subscripation Backend[create_order,payment_status,get_user_status]
@csrf_exempt
def create_order(request):
    if request.method == "POST":

        # -------------------------------
        # CHECK USER STATUS FIRST
        # -------------------------------
        username = request.POST.get("name")
        user = users_collection.find_one({"username": username}, {"status": 1})
    
        if not user:
            return JsonResponse({"error": "User not found"}, status=404)

        status = user.get("status", "").lower()

        #If Rejected → Block + Redirect
        if status == "rejected":
            return JsonResponse({
                "error": "rejected",
                "message": "Your request was rejected."
            }, status=403)
        

        # If Faild → Block
        if status == "Faild":
            return JsonResponse({
                "error": "Faild",
                "message": "Your request is still pending approval."
            }, status=403)

        # -------------------------------
        # If Approved → CONTINUE
        # -------------------------------
        plan_name = request.POST.get("plan_name")
        name = request.POST.get("name")
        email = request.POST.get("email")
        amount = int(request.POST.get("amount")) * 100  # paise

        payment = client.order.create({
            "amount": amount,
            "currency": "INR",
            "payment_capture": "1"
        })

        payment_history_collection.insert_one({
            "username": name,
            "email": email,
            "plan_name": plan_name,
            "amount": amount // 100,
            "order_id": payment['id'],
            "paid": False,
        })

        context = {
            "payment": payment,
            "plan_name": plan_name,
            "amount": amount // 100,
            "username": name,
            "email": email,
            "razorpay_key": settings.RAZORPAY_KEY_ID
        }

        return render(request, "authuser/payment.html", context)

@csrf_exempt       
def payment_status(request):
  if request.method == "POST":
    data = request.POST
    
    try:
       # Verify Razorpay signature
       client.utility.verify_payment_signature({
           'razorpay_order_id': data.get('razorpay_order_id'),
           'razorpay_payment_id': data.get('razorpay_payment_id'),
           'razorpay_signature': data.get('razorpay_signature')
       })
        
       order_id = data.get('razorpay_order_id')
       payment_id = data.get('razorpay_payment_id')
                   # FIX START DATE
       start_dt = timezone.localtime(timezone.now())     # datetime object
       start_date = start_dt.strftime("%Y-%m-%d %H:%M:%S")
  
        # FIX END DATE
       end_dt = start_dt + timedelta(days=31)
       end_date = end_dt.strftime("%Y-%m-%d %H:%M:%S")
       
       # Update payment document in MongoDB
       result = payment_history_collection.update_one(
           {"order_id": order_id},
           {"$set": {
               "paid": True,
               "payment_id": payment_id,
               "updated_at":start_date,
               "Expiry":end_date
           }}
       )
    
       if result.modified_count > 0:
    
           # Fetch payment info
           payment_doc = payment_history_collection.find_one({"order_id": order_id})
           if payment_doc:
               user_email = payment_doc.get("email")
               plan_name = payment_doc.get("plan_name")
              
    
               # Update user's package and add plan history
               user_result = users_collection.update_one(
                   {"email": user_email},
                   {
                       "$set": {"package": plan_name, "updated_at": start_date},
                       "$push": {
                           "plan_history": {
                               "plan_name": plan_name,
                               "start_date": start_date,
                               "Expiry": end_date
                           }
                       }
                   }
               )
               if user_result.modified_count > 0:                    
                   # Add a success popup message
                   messages.success(request, "Payment successful! Please re-login to activate your upgraded package.")
                   request.session.flush()
                   return render(request, 'authuser/login.html')  # redirect to login page
               else:
                   return render(request, 'authuser/login.html')
       else:
           messages.error(request, "Payment not found. Please contact support.")
           return redirect('/ommecgdata/')
    
    except Exception as e:
       messages.error(request, "Payment verification failed. Please try again.")
       return render(request, "authuser/payment_failed.html")

  return redirect('/ommecgdata/')

@login_required
def get_user_status(request):
    try:
        username = request.GET.get("username")
        if not username and request.user.is_authenticated:
            username = request.user.username
            
        if not username:
            return JsonResponse({"error": "Username required"}, status=400)

        user = users_collection.find_one({"username": username}, {"status": 1})
        if not user:
            return JsonResponse({"error": "User not found"}, status=404)

        return JsonResponse({"status": user.get("status", "unknown")})
    except Exception as e:
        return JsonResponse({"error": str(e)}, status=500)

# wallet Backend[wallet_add_money,payment_status_add_money]
@csrf_exempt
def wallet_add_money(request):
    if request.method == "POST":
        try:
            amount = int(request.POST.get("amount")) * 100  # convert to paisa
            name = request.POST.get("name")
            email = request.POST.get("email")

            order = client.order.create({
                "amount": amount,
                "currency": "INR",
                "payment_capture": "1"
            })

            # Save Faild order in MongoDB
            Download_history_collection.insert_one({
                "order_id": order["id"],
                "email": email,
                "amount": amount / 100,  # store in rupees
                "status": "Faild",
                "created_at":timezone.localtime(timezone.now()).strftime("%Y-%m-%d %H:%M:%S")
            })

            context = {
                "payment": order,
                "razorpay_key": settings.RAZORPAY_KEY_ID,
                "amount": amount / 100,
                "name": name,
                "email": email,
            }
            return render(request, "authuser/wallet.html", context)

        except Exception as e:
            return redirect('profile')

    return redirect('profile')

@csrf_exempt
def payment_status_add_money(request):
    if request.method == "POST":
        data = request.POST
        try:
            # Verify signature
            client.utility.verify_payment_signature({
                'razorpay_order_id': data.get('razorpay_order_id'),
                'razorpay_payment_id': data.get('razorpay_payment_id'),
                'razorpay_signature': data.get('razorpay_signature')
            })

            order_id = data.get('razorpay_order_id')
            payment_id = data.get('razorpay_payment_id')

            # Update payment history
            result = Download_history_collection.update_one(
                {"order_id": order_id},
                {"$set": {
                    "payment_id": payment_id,
                    "status": "success",
                    "updated_at":timezone.localtime(timezone.now()).strftime("%Y-%m-%d %H:%M:%S")
                }}
            )

            if result.modified_count > 0:
                wallet_doc = Download_history_collection.find_one({"order_id": order_id})
                if wallet_doc:
                    email = wallet_doc["email"]
                    amount = float(wallet_doc["amount"])

                    # Update user's wallet balance
                    users_collection.update_one(
                        {"email": email},
                        {"$inc": {"wallet_balance": amount}}
                    )
                    return redirect('profile')  # redirect to profile page

            return redirect('profile')

        except Exception as e:
            return render(request, "authuser/payment_failed.html")

    return redirect('profile')
    
def has_successful_download(email, Data_ObjectID):
    return Download_history_collection.find_one({
        "email": email,
        "Data_ObjectID": Data_ObjectID,
        "status": "Success"
    }) is not None

def deduct_wallet_for_download(
    user,
    file_type,
    Data_ObjectId,
    arrhythmia,
    Lead,
    patient_id,
    DownloadfileId
):
    #STEP 0: Check free download
    if has_successful_download(user.email, Data_ObjectId):
        create_download_history(
            user=user,
            file_type=file_type,
            DownloadfileId=DownloadfileId,
            Data_ObjectId=Data_ObjectId,
            Arrhythmia=arrhythmia,
            PatientID=patient_id,
            Lead=Lead,
            price=0,
            status="Success"
        )
        return True, "Free download (already purchased)"

    # STEP 1: Get price
    price = get_download_price(user, file_type)

    # STEP 2: Get wallet
    wallet_doc = users_collection.find_one({"email": user.email})
    if not wallet_doc:
        return False, "User wallet not found"

    wallet_before = float(wallet_doc.get("wallet_balance", 0))

    # STEP 3: Check balance
    if wallet_before < price:
        create_download_history(
            user=user,
            file_type=file_type,
            DownloadfileId=DownloadfileId,
            Data_ObjectId=Data_ObjectId,
            Arrhythmia=arrhythmia,
            PatientID=patient_id,
            Lead=Lead,
            price=price,
            status="Insufficient Balance"
        )
        return False, "Insufficient wallet balance"

    # STEP 4: Deduct wallet
    users_collection.update_one(
        {"email": user.email},
        {"$inc": {"wallet_balance": -price}}
    )

    wallet_after = wallet_before - price

    # STEP 5: Log success
    create_download_history(
        user=user,
        file_type=file_type,
        DownloadfileId=DownloadfileId,
        Data_ObjectId=Data_ObjectId,
        Arrhythmia=arrhythmia,
        PatientID=patient_id,
        Lead=Lead,
        price=price,
        status="Success"
    )

    return True, "Wallet deduction successful"

def get_download_history(request):

    session = request.session.get("user_session", {})
    user_email = session.get("email")

    if not user_email:
        return JsonResponse({"history": []})

    user = users_collection.find_one({"email": user_email})
    if not user:
        return JsonResponse({"history": []})

    raw_history = list(
        Download_history_collection.find(
            {"email": user_email},
            {"_id": 0}
        )
    )

    # Convert string dates -> datetime for proper sorting
    history = []
    for h in raw_history:
        dt = h.get("download_at")

        if isinstance(dt, str) and dt.strip():
            try:
                h["download_at_dt"] = datetime.strptime(dt, "%Y-%m-%d %H:%M:%S")
            except:
                h["download_at_dt"] = datetime.min  # fallback
        elif isinstance(dt, datetime):
            h["download_at_dt"] = dt
        else:
            h["download_at_dt"] = datetime.min  # fallback for None or invalid

        history.append(h)

    # Sort properly by actual datetime
    history.sort(key=lambda x: x["download_at_dt"], reverse=True)

    # Remove temporary field
    for h in history:
        h.pop("download_at_dt", None)

    return JsonResponse({"history": history})
#profile details show Backend
def profile(request):
    # -------------------------------
    # Check user session
    # -------------------------------
    if 'user_session' not in request.session:
        messages.error(request, "You need to log in first.")
        return redirect('login')

    user_session = request.session['user_session']
    username = user_session.get('username')
    email = user_session.get('email')

    # -------------------------------
    # Fetch payment history
    # -------------------------------
    payment_history = list(
        admin_db.payment_history.find({
            "$or": [{"email": email}, {"username": username}]
        }).sort("updated_at", -1)
    )

    formatted_payments = []
    for p in payment_history:
        formatted_payments.append({
            "order_id": p.get("order_id", "-"),
            "plan_name": p.get("plan_name", "Unknown Plan"),
            "amount": p.get("amount", 0),
            "date": str(p.get("updated_at", "")),
            "Expiry": str(p.get("Expiry", "")),
            "status": "Paid" if p.get("paid") else "Faild",
        })
    # -------------------------------
    # Fetch wallet balance
    # -------------------------------
    user_doc = admin_db.users.find_one({"email": email})
    wallet_balance = float(user_doc.get("wallet_balance", 0.0)) if user_doc else 0.0

    # -------------------------------
    # Fetch wallet transactions
    # -------------------------------
    wallet_cursor = admin_db.Download_history.find({"email": email}).sort("created_at", -1)

    wallet_transactions = []
    for w in wallet_cursor:
        txn_type = "credit" if w.get("status") == "success" else "Faild"
        download_at_raw = w.get("download_at")

        if isinstance(download_at_raw, str):
            try:
                download_at = datetime.strptime(download_at_raw, "%Y-%m-%d %H:%M:%S")
            except:
                download_at = timezone.now()  # fallback
        else:
            download_at = download_at_raw  # already datetime
        wallet_transactions.append({
            "txn_id": str(w.get("_id")),
            "date": download_at,
            "amount": float(w.get("amount", 0)),
            "type": txn_type,
            "status": w.get("status", "Faild")
        })
    # -------------------------------
    # Context
    # -------------------------------
    context = {
        "user_session": user_session,
        "payment_history": formatted_payments,
        "wallet": {
            "balance": f"{wallet_balance:.2f}",
            "transactions": wallet_transactions
        }
    }

    return render(request, 'authuser/profile.html', context)

#Password change Backend
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
        email=user.get('email')
        username=user.get('username')
        if not user or not check_password(current_password, user['password']):
            return JsonResponse({"error": "Current password is incorrect."}, status=400)

        # Disallow same as current password
        if check_password(new_password, user['password']):
            return JsonResponse({"error": "New password must be different from the current password."}, status=400)

        # ---- Password history check ----
        password_history = user.get('password_history', [])
        for old_hash in password_history[-3:]:  # check last 3 passwords
            if check_password(new_password, old_hash):
                return JsonResponse({"error": "New password cannot match any of your last 3 passwords."}, status=400)

        # ---- Update password ----
        hashed_new_password = make_password(new_password)
        updated_history = (password_history + [hashed_new_password])[-3:]  # keep only last 3

        users_collection.update_one(
            {"username": user_session['username']},
            {"$set": {
                "password": hashed_new_password,
                "password_history": updated_history
            }}
        )
        send_password_change_email(username,email)
        return JsonResponse({"message": "Password changed successfully."}, status=200)

    except json.JSONDecodeError:
        return JsonResponse({"error": "Invalid JSON data."}, status=400)
    except Exception as e:
        return JsonResponse({"error": str(e)}, status=500)
        
#update user profile Backend
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
def download_history_file(request, DownloadfileId):
    try:
        history_doc = admin_db["Download_history"].find_one({"DownloadfileId": DownloadfileId})
          
        if not history_doc:
            return JsonResponse({"error": "History record not found"}, status=404)

        current_status = history_doc.get("status", "Success")
        email = history_doc.get("email")
        #CHECK USER WALLET EVERY TIME
        wallet_doc = users_collection.find_one({"email": email})
        balance = wallet_doc.get("wallet_balance", 0)

        cost = history_doc.get("amount")  # example cost

        # CASE 1: Already paid earlier
        if current_status == "Success":
            return download_file(media_db, download_fs, DownloadfileId)

        # CASE 2: Not paid earlier → check current wallet
        if balance >= cost:
            # Deduct now
            users_collection.update_one(
                    {"email": email},
                    {"$inc": {"wallet_balance": -cost}}
                )

            # Update status to success
            admin_db["Download_history"].update_one(
                {"DownloadfileId": DownloadfileId},
                {"$set": {"status": "Success"}}
            )

            # Now allow download
            return download_file(media_db, download_fs, DownloadfileId)

        # CASE 3: Balance still not enough
        return JsonResponse({
            "error": "Insufficient wallet balance",
            "status": "failed",
            "message": "Your wallet does not have enough balance. Please recharge and try again."
        }, status=402)

    except Exception as e:
        return JsonResponse({"error": str(e)}, status=500)


def download_file(media_db, download_fs, DownloadfileId):
    file_doc = media_db["downloads.files"].find_one({"DownloadfileId": DownloadfileId})
    if not file_doc:
        return JsonResponse({"error": "File not found"}, status=404)

    file_obj = download_fs.get(file_doc["_id"])

    filename = file_doc.get("filename", "download.ecg")
    content_type = file_doc.get("contentType", "application/octet-stream")

    response = HttpResponse(file_obj.read(), content_type=content_type)
    response["Content-Disposition"] = f"attachment; filename={filename}"
    return response
@login_required
@csrf_exempt
def check_paid_status(request):

    if request.method != "POST":
        return JsonResponse({"error": "Invalid request"}, status=405)

    data = json.loads(request.body)
    object_id = data.get("object_id")

    if not object_id:
        return JsonResponse({"error": "Missing object_id"}, status=400)

    paid = Download_history_collection.find_one({
        "Data_ObjectID": object_id,
        "email": request.user.email,
        "status": "Success"         
    })
    return JsonResponse({
        "is_paid": bool(paid)
    })
def download_all_receipt(request):
    user = request.user

    history = list(Download_history_collection.find({
        "email": user.email
    }).sort("download_at", -1))

    response = HttpResponse(content_type="application/pdf")
    response["Content-Disposition"] = 'attachment; filename="All_Download_Files_Report.pdf"'

    p = canvas.Canvas(response, pagesize=A4)
    width, height = A4
    y = height - 50

    # ---------- HEADER ----------
    p.setFont("Helvetica-Bold", 16)
    p.drawString(50, y, "Download History Report")
    y -= 30

    p.setFont("Helvetica", 10)
    p.drawString(50, y, f"User Email: {user.email}")
    y -= 15
    p.drawString(50, y, f"Generated On: {datetime.now().strftime('%d-%m-%Y %H:%M')}")
    y -= 25

    # ---------- TABLE HEADER ----------
    p.setFont("Helvetica-Bold", 9)
    p.drawString(40, y, "Date")
    p.drawString(110, y, "Transaction ID")
    p.drawString(230, y, "Patient ID")
    p.drawString(310, y, "Arrhythmia")
    p.drawString(400, y, "Channel")
    p.drawString(440, y, "Type")
    p.drawString(500, y, "Status")
    y -= 10
    p.line(40, y, 550, y)
    y -= 14

    # ---------- TABLE DATA ----------
    p.setFont("Helvetica", 9)

    if not history:
        p.drawString(50, y, "No download history found.")
    else:
        for item in history:
            if y < 80:
                p.showPage()
                y = height - 50
                p.setFont("Helvetica", 9)

            p.drawString(40, y, str(item.get("download_at", "NA"))[:10])
            p.drawString(110, y, str(item.get("transaction_id", "NA"))[:14])
            p.drawString(230, y, str(item.get("PatientID", "NA")))
            p.drawString(310, y, str(item.get("Arrhythmia", "NA"))[:18])
            p.drawString(400, y, str(item.get("Lead", "NA")))
            p.drawString(440, y, str(item.get("file_type", "NA")))
            p.drawString(500, y, str(item.get("status", "NA")))

            y -= 14

    p.showPage()
    p.save()
    return response