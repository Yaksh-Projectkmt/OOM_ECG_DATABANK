# from django.shortcuts import render, redirect
# from django.contrib.auth.hashers import make_password, check_password
# from django.conf import settings
# from django.contrib import messages
# from pymongo import MongoClient
# from django.http import JsonResponse,HttpResponse
# from django.views.decorators.csrf import csrf_protect,csrf_exempt
# import json
# from datetime import datetime,timedelta
# from django.utils import timezone
# import uuid
# from django.views.decorators.cache import never_cache
# from bson.binary import Binary
# import razorpay
# from bson.objectid import ObjectId
# import base64
# from collections import defaultdict
# from report.views import collections
# from authuser.email_Send import send_welcome_email, send_documents_processing_email,send_approved_email,send_rejected_email
# from django.contrib.auth.decorators import login_required
# import os
# from django.contrib.admin.views.decorators import staff_member_required
# from django.contrib.auth import get_user_model, login as django_login
# from subscription.utils import sync_subscription_from_mongo
# from django.contrib.auth.hashers import check_password
# from django.shortcuts import render
# from django.contrib.admin.views.decorators import staff_member_required
# import re

# # Connect to MongoDB
# mongo_client = MongoClient("mongodb://192.168.1.65:27017/")
# db = mongo_client['ecgarrhythmias']
# manage_db = mongo_client["manage"]# manage DB
# patients_db = mongo_client['Patients']
# users_collection = db["users"]
# sessions_collection = db["sessions"] 
# contact_collection = db['contact_messages']
# manage_currency_collection = manage_db["currency"]
# payment_history_collection = manage_db["payment_history"]
# manage_ui_collection=manage_db["manage_ui"]
# wallet_history_collection=manage_db["wallet_history"]
# client = razorpay.Client(auth=(settings.RAZORPAY_KEY_ID, settings.RAZORPAY_KEY_SECRET))

# @staff_member_required
# def registration_requests_admin(request):
#     return render(request, "authuser/registration_requests.html")

# def get_plan_features(plan_name):
#     """Fetch features from manage_ui collection based on plan name"""
#     plan = manage_ui_collection.find_one({"plan_name": plan_name})
#     if plan and "features" in plan:
#         return plan["features"]
#     return []
# def home(request):
#     if 'user_session' in request.session:
#         return redirect('/ommecgdata/')
#     else:
#         return redirect('/auth/login/')  # explicitly use auth prefix


# def patient_list(request):
#     return render(request, 'authuser/patient_list.html')

# def help(request):
#     return render(request, 'authuser/help.html')

# def payment_failed(request):
#     return render(request, "authuser/payment_failed.html")

# # User Registration[register,get_registrations,update_registration_status]
# @never_cache
# @csrf_protect
# # def register(request):
# #     if request.method == "POST":
# #         userType = request.POST.get('userType')
# #         username = request.POST.get('username', '').strip()
# #         email = request.POST.get('email', '').strip().lower()
# #         country_code = request.POST.get('countryCode', '').strip()
# #         phone = request.POST.get('phone', '').strip()
# #         password = request.POST.get('password')
# #         doctor_id = request.POST.get('doctorId', '').strip()

# #         student_id_file = request.FILES.get('studentId')
# #         address_proof_file = request.FILES.get('addressProof')
# #         doctor_cert_file = request.FILES.get('doctorCert')
# #         address_proof_other_file = request.FILES.get("addressProofOther")

# #         # Basic validation
# #         if not (username and email and password):
# #             messages.error(request, "Please fill all required fields.")
# #             return redirect('register')

# #         if userType == "student":
# #             if not student_id_file or not address_proof_file:
# #                 messages.error(request, "Both Student ID and Address Proof are required for student registration.")
# #                 return redirect('register')
# #         elif userType == "doctor":
# #             if not doctor_id or not doctor_cert_file:
# #                 messages.error(request, "Doctor ID and Certificate are required for doctor registration.")
# #                 return redirect('register')
# #         elif userType == "other":
# #             if not address_proof_other_file:
# #                 messages.error(request, "Address Proof is required for Other users.")
# #                 return redirect('register')

# #         else:
# #             messages.error(request, "Invalid user type.")
# #             return redirect('register')
        
# #         # Check if user exists
# #         existing_user = users_collection.find_one({
# #             "$or": [{"username": username}, {"email": email}]
# #         })
# #         if existing_user:
# #             messages.error(request, "Username or Email already exists.")
# #             return redirect('register')

# #         hashed_password = make_password(password)

# #         # --- Country Detection ---
# #         if country_code == "+91":
# #             country = "India"
# #             currency = "INR"
# #         elif country_code == "+1":
# #             country = "United States"
# #             currency = "USD"
# #         else:
# #             country = "Unknown"
# #             currency = "USD"

# #         default_package = "Free"
# #         allowed_features = get_plan_features(default_package)

# #         # --- Save uploaded files ---
# #         def save_file(file, folder):
# #             upload_dir = os.path.join(settings.MEDIA_ROOT, folder)
# #             os.makedirs(upload_dir, exist_ok=True)
# #             filename = f"{uuid.uuid4()}_{file.name}"
# #             file_path = os.path.join(upload_dir, filename)
# #             with open(file_path, 'wb+') as destination:
# #                 for chunk in file.chunks():
# #                     destination.write(chunk)
# #             return f"{folder}/{filename}"  # relative path for easy access in templates

        
# #         # SAVE FILES BASED ON USER TYPE -------------------
# #         student_id_path = None
# #         address_proof_path = None
# #         doctor_cert_path = None
# #         if userType == "student":
# #             student_id_path = save_file(student_id_file, 'student_ids')
# #             address_proof_path = save_file(address_proof_file, 'address_proofs')
# #         elif userType == "doctor":
# #             doctor_cert_path = save_file(doctor_cert_file, 'doctor_certificates')

# #         elif userType == "other":
# #             address_proof_path = save_file(address_proof_other_file, 'other_address_proofs')

# #         # CREATE USER DOCUMENT ----------------------------
# #         user_doc = {
# #             "role": userType,
# #             "username": username,
# #             "email": email,
# #             "country_code": country_code,
# #             "country": country,
# #             "phone": phone,
# #             "password": hashed_password,
# #             "doctorId": doctor_id if userType == "doctor" else None,
# #             "register_time": timezone.now(),
# #             "status": "pending",
# #             "package": default_package,
# #             "files": {
# #                 "student_id_path": student_id_path,
# #                 "address_proof_path": address_proof_path,
# #                 "doctor_certificate_path": doctor_cert_path
# #             }
# #         }

# #         inserted_user = users_collection.insert_one(user_doc)
# #         user_id = str(inserted_user.inserted_id)

# #         manage_currency_collection.insert_one({
# #             "user_id": user_id,
# #             "username": username,
# #             "country_code": country_code,
# #             "country": country,
# #             "currency": currency,
# #             "created_at": timezone.now(),
# #         })

# #         session_id = str(uuid.uuid4())
# #         request.session['user_session'] = {
# #             "session_id": session_id,
# #             "user_id": user_id,
# #             "username": username,
# #             "email": email,
# #             "role": userType,
# #             "Package": default_package,
# #             "features": allowed_features
# #         }

# #         sessions_collection.insert_one({
# #             "_id": session_id,
# #             "user_id": user_id,
# #             "username": username,
# #             "role": userType,
# #             "login_time": timezone.now(),
# #             "Package": default_package
# #         })

# #         send_welcome_email(username, email)
# #         send_documents_processing_email(username, email)

# #         return redirect('/ommecgdata/')

# #     return render(request, 'authuser/register.html')
# def register(request):
#     if request.method == "POST":
#         userType = request.POST.get('userType')
#         username = request.POST.get('username', '').strip()
#         email = request.POST.get('email', '').strip().lower()
#         country_code = request.POST.get('countryCode', '').strip()
#         phone = request.POST.get('phone', '').strip()
#         password = request.POST.get('password')
#         doctor_id = request.POST.get('doctorId', '').strip()

#         student_id_file = request.FILES.get('studentId')
#         address_proof_file = request.FILES.get('addressProof')
#         doctor_cert_file = request.FILES.get('doctorCert')
#         address_proof_other_file = request.FILES.get("addressProofOther")

#         # Basic validation
#         if not (username and email and password):
#             messages.error(request, "Please fill all required fields.")
#             return redirect('register')

#         if userType == "student":
#             if not student_id_file or not address_proof_file:
#                 messages.error(request, "Both Student ID and Address Proof are required for student registration.")
#                 return redirect('register')
#         elif userType == "doctor":
#             if not doctor_id or not doctor_cert_file:
#                 messages.error(request, "Doctor ID and Certificate are required for doctor registration.")
#                 return redirect('register')
#         elif userType == "other":
#             if not address_proof_other_file:
#                 messages.error(request, "Address Proof is required for Other users.")
#                 return redirect('register')

#         else:
#             messages.error(request, "Invalid user type.")
#             return redirect('register')
        
#         # Check if user exists
#         # --- CHECK EMAIL & USERNAME ---
#         if users_collection.find_one({"username": username}):
#             return JsonResponse({"status": "username_exists"})

#         if users_collection.find_one({"email": email}):
#             return JsonResponse({"status": "email_exists"})

#         hashed_password = make_password(password)

#         # --- Country Detection ---
#         if country_code == "+91":
#             country = "India"
#             currency = "INR"
#         elif country_code == "+1":
#             country = "United States"
#             currency = "USD"
#         else:
#             country = "Unknown"
#             currency = "USD"

#         default_package = "Free"
#         allowed_features = get_plan_features(default_package)

#         # --- Save uploaded files ---
#         def save_file(file, folder):
#             upload_dir = os.path.join(settings.MEDIA_ROOT, folder)
#             os.makedirs(upload_dir, exist_ok=True)
#             filename = f"{uuid.uuid4()}_{file.name}"
#             file_path = os.path.join(upload_dir, filename)
#             with open(file_path, 'wb+') as destination:
#                 for chunk in file.chunks():
#                     destination.write(chunk)
#             return f"{folder}/{filename}"  # relative path for easy access in templates

        
#         # SAVE FILES BASED ON USER TYPE -------------------
#         student_id_path = None
#         address_proof_path = None
#         doctor_cert_path = None
#         if userType == "student":
#             student_id_path = save_file(student_id_file, 'student_ids')
#             address_proof_path = save_file(address_proof_file, 'address_proofs')
#         elif userType == "doctor":
#             doctor_cert_path = save_file(doctor_cert_file, 'doctor_certificates')

#         elif userType == "other":
#             address_proof_path = save_file(address_proof_other_file, 'other_address_proofs')

#         # CREATE USER DOCUMENT ----------------------------
#         user_doc = {
#             "role": userType,
#             "username": username,
#             "email": email,
#             "country_code": country_code,
#             "country": country,
#             "phone": phone,
#             "password": hashed_password,
#             "doctorId": doctor_id if userType == "doctor" else None,
#             "register_time": timezone.now(),
#             "status": "pending",
#             "package": default_package,
#             "files": {
#                 "student_id_path": student_id_path,
#                 "address_proof_path": address_proof_path,
#                 "doctor_certificate_path": doctor_cert_path
#             }
#         }

#         inserted_user = users_collection.insert_one(user_doc)
#         user_id = str(inserted_user.inserted_id)

#         manage_currency_collection.insert_one({
#             "user_id": user_id,
#             "username": username,
#             "country_code": country_code,
#             "country": country,
#             "currency": currency,
#             "created_at": timezone.now(),
#         })

#         session_id = str(uuid.uuid4())
#         request.session['user_session'] = {
#             "session_id": session_id,
#             "user_id": user_id,
#             "username": username,
#             "email": email,
#             "role": userType,
#             "Package": default_package,
#             "features": allowed_features
#         }

#         sessions_collection.insert_one({
#             "_id": session_id,
#             "user_id": user_id,
#             "username": username,
#             "role": userType,
#             "login_time": timezone.now(),
#             "Package": default_package
#         })

#         send_welcome_email(username, email)
#         send_documents_processing_email(username, email)

#         return redirect('/ommecgdata/')

#     return render(request, 'authuser/register.html')

# #admin
# def get_registrations(request):
#     try:
#         registrations = []
#         base_url = request.build_absolute_uri('/')[:-1]

#         for doc in users_collection.find():
#             if doc.get("role", "").lower() == "admin":
#                 continue

#             file_data = doc.get("files", {})
            
#             student_id_path = file_data.get("student_id_path")
#             address_proof_path = file_data.get("address_proof_path")
#             doctor_cert_path = file_data.get("doctor_certificate_path")

#             # Convert file path ? full media URL
#             def make_url(path):
#                 if path:
#                     return f"{base_url}{settings.MEDIA_URL}{path}"
#                 return None

#             registrations.append({
#                 "id": str(doc["_id"]),
#                 "user_name": doc.get("username", ""),
#                 "email": doc.get("email", ""),
#                 "phone_number": doc.get("phone", ""),
#                 "role": doc.get("role", "").lower(),
#                 "status": doc.get("status", ""),
#                 "student_id_url": make_url(student_id_path),
#                 "address_proof_url": make_url(address_proof_path),
#                 "doctor_certificate_url": make_url(doctor_cert_path),
#                 "doctor_id": doc.get("doctorId"),
#                 "created_at": doc.get("register_time"),
#             })

#         return JsonResponse(registrations, safe=False)

#     except Exception as e:
#         print("Error:", e)
#         return JsonResponse({"error": str(e)}, status=500)
# #admin
# @csrf_exempt
# def update_registration_status(request):
#     if request.method != "POST":
#         return JsonResponse({"error": "Invalid request"}, status=405)
#     try:
#         data = json.loads(request.body)
#         user_id = data.get("id")
#         status = data.get("status")
#         comment = data.get("comment", "")

#         if not user_id or status not in ["approved", "rejected"]:
#             return JsonResponse({"error": "Invalid data"}, status=400)

#         # Update status in DB
#         users_collection.update_one(
#             {"_id": ObjectId(user_id)},
#             {"$set": {"status": status, "admin_comment": comment}}
#         )

#         # Fetch user details to get email & username
#         user_doc = users_collection.find_one({"_id": ObjectId(user_id)})
#         if user_doc:
#             username = user_doc.get("username", "User")
#             receiver_email = user_doc.get("email")
            
#             # Compose email based on status
#             if status == "approved":
#                 send_approved_email(username,receiver_email)
#             else:  # rejected
#                 send_rejected_email(username,receiver_email,comment)

#         return JsonResponse({"success": True, "message": f"User {status} successfully"})
#     except Exception as e:
#         return JsonResponse({"error": str(e)}, status=500)

# #login page Backend  
# @csrf_protect
# @never_cache
# def login(request):
#     # If already logged in (Django session)
#     if request.user.is_authenticated:
#         return redirect('/ommecgdata/')

#     # If old session exists
#     if 'user_session' in request.session:
#         return redirect('/ommecgdata/')

#     if request.method == "POST":
#         username = request.POST.get('username')
#         password = request.POST.get('password')
        
#         # Fetch from MongoDB
#         user = users_collection.find_one({"username": username})
#         handle_plan_expiry(user)
#         if not user:
#             messages.error(request, "User not found.")
#             return redirect('login')

#         if not check_password(password, user['password']):
#             messages.error(request, "Invalid password.")
#             return redirect('login')

#         user_status = user.get("status", "pending").lower()

#         # ----------------------------
#         # REJECTED ACCOUNT
#         # ----------------------------
#         if user_status == "rejected":
#             messages.error(request, "Your account request has been rejected.")
#             return redirect('login')

#         # ----------------------------
#         # APPROVED ACCOUNT → Proceed Login
#         # ----------------------------

#         User = get_user_model()
#         django_user, created = User.objects.get_or_create(
#             username=user["username"],
#             defaults={
#                 "email": user.get("email", ""),
#                 "password": make_password(None),
#             }
#         )

#         sync_subscription_from_mongo(django_user)

#         django_login(request, django_user)

#         # SAVE status in session so JS can read it
#         session_id = str(uuid.uuid4())
#         request.session["user_session"] = {
#             "session_id": session_id,
#             "user_id": str(user["_id"]),
#             "username": user["username"],
#             "email": user["email"],
#             "phone": user.get("phone"),
#             "userType": user.get("userType", "user"),
#             "status": user_status,              # ADDED
#             "Package": user.get("Package", "Free"),
#             "features": get_plan_features(user.get("Package", "Free")),
#         }

#         # Save session log to MongoDB
#         sessions_collection.insert_one({
#             "_id": session_id,
#             "user_id": str(user["_id"]),
#             "username": user["username"],
#             "userType": user.get("userType", "user"),
#             "login_time": timezone.now(),
#             "Package": user.get("Package", "Free"),
#             "status": user_status,              # ADDED
#         })

#         return redirect('/ommecgdata/')

#     response = render(request, 'authuser/login.html')
#     response['Cache-Control'] = 'no-store, no-cache, must-revalidate, max-age=0'
#     response['Pragma'] = 'no-cache'
#     response['Expires'] = '0'
#     return response

# #profile details show Backend
# def profile(request):
#     # -------------------------------
#     # Check user session
#     # -------------------------------
#     if 'user_session' not in request.session:
#         messages.error(request, "You need to log in first.")
#         return redirect('login')

#     user_session = request.session['user_session']
#     username = user_session.get('username')
#     email = user_session.get('email')

#     # -------------------------------
#     # Fetch payment history
#     # -------------------------------
#     payment_history = list(
#         manage_db.payment_history.find({
#             "$or": [{"email": email}, {"username": username}]
#         }).sort("updated_at", -1)
#     )

#     formatted_payments = []
#     for p in payment_history:
#         formatted_payments.append({
#             "order_id": p.get("order_id", "-"),
#             "plan_name": p.get("plan_name", "Unknown Plan"),
#             "amount": p.get("amount", 0),
#             "date": p.get("updated_at", datetime.now()),
#             "Expiry":p.get("Expiry", datetime.now()),
#             "status": "Paid" if p.get("paid") else "Faild",
#         })

#     # -------------------------------
#     # Fetch wallet balance
#     # -------------------------------
#     user_doc = db.users.find_one({"email": email})
#     wallet_balance = float(user_doc.get("wallet_balance", 0.0)) if user_doc else 0.0

#     # -------------------------------
#     # Fetch wallet transactions
#     # -------------------------------
#     wallet_cursor = manage_db.wallet_history.find({"email": email}).sort("created_at", -1)

#     wallet_transactions = []
#     for w in wallet_cursor:
#         txn_type = "credit" if w.get("status") == "success" else "Faild"
#         wallet_transactions.append({
#             "txn_id": str(w.get("_id")),
#             "date": w.get("created_at", timezone.now()),
#             "amount": float(w.get("amount", 0)),
#             "type": txn_type,
#             "status": w.get("status", "Faild")
#         })

#     # -------------------------------
#     # Context
#     # -------------------------------
#     context = {
#         "user_session": user_session,
#         "payment_history": formatted_payments,
#         "wallet": {
#             "balance": f"{wallet_balance:.2f}",
#             "transactions": wallet_transactions
#         }
#     }

#     return render(request, 'authuser/profile.html', context)

# #Password change Backend
# @csrf_protect
# def change_password(request):
#     if 'user_session' not in request.session:
#         return JsonResponse({"error": "You need to log in first."}, status=401)

#     if request.method != "POST":
#         return JsonResponse({"error": "Invalid request method."}, status=405)

#     try:
#         data = json.loads(request.body.decode('utf-8'))
#         current_password = data.get('currentPassword', '').strip()
#         new_password = data.get('newPassword', '').strip()

#         if not current_password or not new_password:
#             return JsonResponse({"error": "All password fields are required."}, status=400)

#         user_session = request.session['user_session']
#         user = users_collection.find_one({"username": user_session['username']})

#         if not user or not check_password(current_password, user['password']):
#             return JsonResponse({"error": "Current password is incorrect."}, status=400)

#         # Disallow same as current password
#         if check_password(new_password, user['password']):
#             return JsonResponse({"error": "New password must be different from the current password."}, status=400)

#         # ---- Password history check ----
#         password_history = user.get('password_history', [])
#         for old_hash in password_history[-3:]:  # check last 3 passwords
#             if check_password(new_password, old_hash):
#                 return JsonResponse({"error": "New password cannot match any of your last 3 passwords."}, status=400)

#         # ---- Update password ----
#         hashed_new_password = make_password(new_password)
#         updated_history = (password_history + [hashed_new_password])[-3:]  # keep only last 3

#         users_collection.update_one(
#             {"username": user_session['username']},
#             {"$set": {
#                 "password": hashed_new_password,
#                 "password_history": updated_history
#             }}
#         )

#         return JsonResponse({"message": "Password changed successfully."}, status=200)

#     except json.JSONDecodeError:
#         return JsonResponse({"error": "Invalid JSON data."}, status=400)
#     except Exception as e:
#         return JsonResponse({"error": str(e)}, status=500)
        
# # User Logout
# @never_cache
# def logout(request):
#     request.session.flush()
#     messages.success(request, "Logged out successfully.")

#     response = redirect('login')
#     response['Cache-Control'] = 'no-store, no-cache, must-revalidate, max-age=0'
#     response['Pragma'] = 'no-cache'
#     response['Expires'] = '0'
#     return response


# #update user profile Backend
# @csrf_protect
# def update_profile(request):
#     if 'user_session' not in request.session:
#         return JsonResponse({"error": "You need to log in first."}, status=401)

#     if request.method == "POST":
#         try:
#             data = json.loads(request.body)
#             username = data.get('username')
#             email = data.get('email')
#             phone = data.get('phone')

#             # Required fields
#             if not username or not email:
#                 return JsonResponse({"error": "Username and email are required."}, status=400)

#             # Validate email with regex
#             email_pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
#             if not re.match(email_pattern, email):
#                 return JsonResponse({"error": "Invalid email format."}, status=400)

#             # ?? Validate phone (optional, but if provided must be 10 digits)
#             if phone:
#                 if not re.fullmatch(r'\d{10}', str(phone)):
#                     return JsonResponse({"error": "Phone number must be exactly 10 digits."}, status=400)
#                 phone = int(phone)
#             else:
#                 phone = None

#             current_username = request.session['user_session']['username']

#             # Prepare update data
#             update_data = {
#                 "username": username,
#                 "email": email,
#             }
#             if phone is not None:
#                 update_data["phone"] = phone

#             # Update user in database
#             users_collection.update_one(
#                 {"username": current_username},
#                 {"$set": update_data}
#             )

#             # Update session
#             request.session['user_session'] = {
#                 "username": username,
#                 "email": email,
#                 "phone": phone if phone is not None else request.session['user_session'].get('phone'),
#             }

#             return JsonResponse({
#                 "success": True,
#                 "message": "Profile updated successfully.",
#                 "profile": {
#                     "username": username,
#                     "email": email,
#                     "phone": phone if phone is not None else request.session['user_session'].get('phone')
#                 }
#             })

#         except Exception as e:
#             return JsonResponse({"error": str(e)}, status=500)

#     return JsonResponse({"error": "Invalid request method."}, status=405)

# #query releted service data store Backend
# def save_contact(request):
#   if request.method == "POST":
#       name = request.POST.get("name", "").strip()
#       email = request.POST.get("email", "").strip()
#       message_text = request.POST.get("message", "").strip()

#       # --- Validation ---
#       if not name or not email or not message_text:
#           messages.error(request, "Please fill out all fields before submitting.")
#           return redirect('help')

#       # --- Optional: basic email validation ---
#       if "@" not in email or "." not in email:
#           messages.error(request, "Please enter a valid email address.")
#           return redirect('help')

#       # --- Save message to MongoDB ---
#       contact_data = {
#           "name": name,
#           "email": email,
#           "message": message_text,
#           "created_at": datetime.utcnow()
#       }

#       try:
#           contact_collection.insert_one(contact_data)
#           messages.success(request, " Your message has been sent successfully! We'll contact you soon.")
#           return redirect('/ommecgdata/')
#       except Exception as e:
#           messages.error(request, f" Something went wrong while saving your message. Please try again later.")

#       return redirect('help')

#   # If method not POST, redirect safely
#   return redirect('/ommecgdata/')

# #All Patient backend Show[get_patient_arrhythmia_records,get_patients_data]   
# def get_patient_arrhythmia_records(request):
#     try:
#         # --- Step 1: Get arrhythmia collections ---
#         arrhythmia_collections = patients_db.list_collection_names()

#         patient_data = defaultdict(lambda: {
#             "patient_id": "",
#             "total_records": 0,
#             "total_time": 0,
#             "arrhythmias": [],
#             "live": False  # default
#         })

#         for collection_name in arrhythmia_collections:
#             collection = patients_db[collection_name]

#             for doc in collection.find({}, {"_id": 0, "PatientID": 1, "total_records": 1, "total_time": 1}):
#                 patient_id = doc.get("PatientID")
#                 if not patient_id:
#                     continue

#                 total_records = doc.get("total_records", 0)
#                 total_time_min = doc.get("total_time", 0) or 0
#                 total_time_sec = round(total_time_min * 60, 2)

#                 patient = patient_data[patient_id]
#                 patient["patient_id"] = patient_id
#                 patient["total_records"] += total_records
#                 patient["total_time"] += total_time_sec
#                 patient["arrhythmias"].append({
#                     "type": collection_name,
#                     "duration": total_time_sec,
#                     "records": total_records
#                 })

#         # --- Step 2: Connect to Live DB and mark live patients ---
#         try:
#             LIVE_URI = "mongodb://readonly_user:9ikJ4Qn1YmG1l1EVF1OQ@192.168.2.131:27017/?authSource=admin"
#             DB_NAME = "ecgs"

#             live_client = MongoClient(LIVE_URI)
#             live_db = live_client[DB_NAME]
#             live_collection = live_db["patients"]

#             # collect live patient IDs (either `patientId` or `PatientID`)
#             live_ids = {
#                 str(doc.get("patientId") or doc.get("PatientID"))
#                 for doc in live_collection.find({}, {"_id": 0, "patientId": 1, "PatientID": 1})
#                 if doc.get("patientId") or doc.get("PatientID")
#             }

#             # mark live patients in our list
#             for pid in live_ids:
#                 if pid in patient_data:
#                     patient_data[pid]["live"] = True

#         except Exception as live_err:
#             print("Live DB error:", live_err)

#         # --- Step 3: Sort and return ---
#         patients = sorted(patient_data.values(), key=lambda x: x["patient_id"])

#         return JsonResponse({"status": "success", "data": patients}, safe=False)

#     except Exception as e:
#         return JsonResponse({"status": "error", "message": str(e)})

# def get_patients_data(request):
#     total_patients = set()
#     total_time = 0.0

#     for name in collections:
#         coll = patients_db[name]
#         cursor = coll.find({}, {"PatientID": 1, "total_time": 1})

#         group_time = 0.0
#         for doc in cursor:
#             pid = str(doc.get("PatientID", "")).strip()
#             if pid:
#                 total_patients.add(pid)
#             group_time += float(doc.get("total_time", 0))

#         total_time += group_time

#     totals = {
#         "count_patients": len(total_patients),
#         "total_time": round(total_time, 2),
#     }

#     return JsonResponse(totals)

# # def create_order(request):
# #     if request.method == "POST":
# #         plan_name = request.POST.get("plan_name")
# #         name = request.POST.get("name")
# #         email = request.POST.get("email")
# #         amount = int(request.POST.get("amount")) * 100  # paise
# #         payment = client.order.create({
# #             "amount": amount,
# #             "currency": "INR",
# #             "payment_capture": "1"
# #         })
# #         order = payment_history_collection.insert_one({
# #             "username":name,
# #             "email":email,
# #             "plan_name":plan_name,
# #             "amount":amount // 100,
# #             "order_id":payment['id'],
# #             "paid":False,
# #         })
# #         context = {
# #             "payment": payment,
# #             "plan_name": plan_name,
# #             "amount": amount // 100,
# #             "username": name,
# #             "email": email,
# #             "razorpay_key": settings.RAZORPAY_KEY_ID
# #         }
# #         return render(request, "authuser/payment.html", context)

# #subscripation Backend[create_order,payment_status,get_user_status]
# @csrf_exempt
# def create_order(request):
#     if request.method == "POST":

#         # -------------------------------
#         # CHECK USER STATUS FIRST
#         # -------------------------------
#         username = request.POST.get("name")
#         user = users_collection.find_one({"username": username}, {"status": 1})

#         if not user:
#             return JsonResponse({"error": "User not found"}, status=404)

#         status = user.get("status", "").lower()

#         #If Rejected → Block + Redirect
#         if status == "rejected":
#             return JsonResponse({
#                 "error": "rejected",
#                 "message": "Your request was rejected."
#             }, status=403)
        

#         # If Faild → Block
#         if status == "Faild":
#             return JsonResponse({
#                 "error": "Faild",
#                 "message": "Your request is still pending approval."
#             }, status=403)

#         # -------------------------------
#         # If Approved → CONTINUE
#         # -------------------------------
#         plan_name = request.POST.get("plan_name")
#         name = request.POST.get("name")
#         email = request.POST.get("email")
#         amount = int(request.POST.get("amount")) * 100  # paise

#         payment = client.order.create({
#             "amount": amount,
#             "currency": "INR",
#             "payment_capture": "1"
#         })

#         payment_history_collection.insert_one({
#             "username": name,
#             "email": email,
#             "plan_name": plan_name,
#             "amount": amount // 100,
#             "order_id": payment['id'],
#             "paid": False,
#         })

#         context = {
#             "payment": payment,
#             "plan_name": plan_name,
#             "amount": amount // 100,
#             "username": name,
#             "email": email,
#             "razorpay_key": settings.RAZORPAY_KEY_ID
#         }

#         return render(request, "authuser/payment.html", context)

# # def get_updated_plan_dates(user_doc):
# #     """
# #     Determines the correct start_date and expiry_date for a newly purchased plan.
# #     Also blocks user if they already bought 3 plans in the same month.
# #     """

# #     plan_history = user_doc.get("plan_history", [])

# #     now = timezone.now()
# #     current_month = now.month
# #     current_year = now.year

# #     # Count purchases in current month
# #     monthly_purchases = [
# #         p for p in plan_history
# #         if p.get("buy_date") and  # UPDATED
# #            p["buy_date"].month == current_month and
# #            p["buy_date"].year == current_year
# #     ]

# #     if len(monthly_purchases) >= 3:
# #         return None, None, "LIMIT_REACHED"

# #     # BUY DATE is always NOW
# #     buy_date = now

# #     # CASE 1: User has NO previous plan → start today
# #     if not plan_history:
# #         start_date = now
# #         expiry_date = start_date + timedelta(days=31)
# #         return buy_date, start_date, expiry_date, "OK"

# #     # CASE 2: User has history → get last plan expiry
# #     last_plan = plan_history[-1]
# #     last_expiry = last_plan.get("expiry_date") or last_plan.get("Expiry")

# #     if last_expiry:
# #         start_date = last_expiry + timedelta(days=1)
# #     else:
# #         start_date = now

# #     expiry_date = start_date + timedelta(days=31)

# #     return buy_date, start_date, expiry_date, "OK"

# @csrf_exempt       
# def payment_status(request):
#   if request.method == "POST":
#     data = request.POST
#     try:
#         # Verify Razorpay signature
#         client.utility.verify_payment_signature({
#             'razorpay_order_id': data.get('razorpay_order_id'),
#             'razorpay_payment_id': data.get('razorpay_payment_id'),
#             'razorpay_signature': data.get('razorpay_signature')
#         })

#         order_id = data.get('razorpay_order_id')
#         payment_id = data.get('razorpay_payment_id')
#         start_date = timezone.now()
#         end_date = start_date + timedelta(days=31)
#         # Update payment document in MongoDB
#         result = payment_history_collection.update_one(
#             {"order_id": order_id},
#             {"$set": {
#                 "paid": True,
#                 "payment_id": payment_id,
#                 "updated_at":start_date,
#                 "Expiry":end_date
#             }}
#         )

#         if result.modified_count > 0:

#             # Fetch payment info
#             payment_doc = payment_history_collection.find_one({"order_id": order_id})
#             if payment_doc:
#                 user_email = payment_doc.get("email")
#                 plan_name = payment_doc.get("plan_name")
                

#                 # Update user's package and add plan history
#                 user_result = users_collection.update_one(
#                     {"email": user_email},
#                     {
#                         "$set": {"Package": plan_name, "updated_at": start_date},
#                         "$push": {
#                             "plan_history": {
#                                 "plan_name": plan_name,
#                                 "start_date": start_date,
#                                 "Expiry": end_date
#                             }
#                         }
#                     }
#                 )

#                 if user_result.modified_count > 0:                    
#                     # Add a success popup message
#                     messages.success(request, "Payment successful! Please re-login to activate your upgraded package.")
#                     request.session.flush()
#                     return render(request, 'authuser/login.html')  # redirect to login page
#                 else:
#                     return render(request, 'authuser/login.html')
#         else:
#             messages.error(request, "Payment not found. Please contact support.")
#             return redirect('/ommecgdata/')

#     except Exception as e:
#         messages.error(request, "Payment verification failed. Please try again.")
#         return render(request, "authuser/payment_failed.html")

#   return redirect('/ommecgdata/')
# # @csrf_exempt
# # def payment_status(request):
# #     if request.method == "POST":
# #         data = request.POST
# #         try:
# #             client.utility.verify_payment_signature({
# #                 'razorpay_order_id': data.get('razorpay_order_id'),
# #                 'razorpay_payment_id': data.get('razorpay_payment_id'),
# #                 'razorpay_signature': data.get('razorpay_signature')
# #             })

# #             order_id = data.get('razorpay_order_id')
# #             payment_id = data.get('razorpay_payment_id')

# #             # Fetch payment document
# #             payment_doc = payment_history_collection.find_one({"order_id": order_id})
# #             if not payment_doc:
# #                 messages.error(request, "Payment not found!")
# #                 return redirect('/ommecgdata/')

# #             user_email = payment_doc.get("email")
# #             plan_name = payment_doc.get("plan_name")

# #             # Fetch user
# #             user_doc = users_collection.find_one({"email": user_email})
# #             if not user_doc:
# #                 messages.error(request, "User not found!")
# #                 return redirect('/ommecgdata/')

# #             # Determine correct start/end dates + limit check
# #             start_date, end_date, status = get_updated_plan_dates(user_doc)

# #             if status == "LIMIT_REACHED":
# #                 messages.error(request, "You can purchase only 3 plans in a month.")
# #                 return redirect('/ommecgdata/')

# #             # Update payment entry as paid
# #             payment_history_collection.update_one(
# #                 {"order_id": order_id},
# #                 {"$set": {
# #                     "paid": True,
# #                     "payment_id": payment_id,
# #                     "updated_at": timezone.now(),
# #                     "Expiry": end_date
# #                 }}
# #             )

# #             # Update user’s current package + push to history
# #             users_collection.update_one(
# #                 {"email": user_email},
# #                 {
# #                     "$set": {"Package": plan_name},
# #                     "$push": {
# #                         "plan_history": {
# #                             "plan_name": plan_name,
# #                             "start_date": start_date,
# #                             "Expiry": end_date
# #                         }
# #                     }
# #                 }
# #             )

# #             messages.success(request, "Payment successful! Re-login to apply your new plan.")
# #             request.session.flush()
# #             return render(request, 'authuser/login.html')

# #         except Exception as e:
# #             print("Payment error:", str(e))
# #             messages.error(request, "Payment verification failed.")
# #             return render(request, "authuser/payment_failed.html")

# #     return redirect('/ommecgdata/')

# def get_user_status(request):
#     try:
#         username = request.GET.get("username")
#         if not username:
#             return JsonResponse({"error": "Username required"}, status=400)

#         user = users_collection.find_one({"username": username}, {"status": 1})
#         if not user:
#             return JsonResponse({"error": "User not found"}, status=404)

#         return JsonResponse({"status": user.get("status", "unknown")})
#     except Exception as e:
#         return JsonResponse({"error": str(e)}, status=500)

# # wallet Backend[wallet_add_money,payment_status_add_money]
# @csrf_exempt
# def wallet_add_money(request):
#     if request.method == "POST":
#         try:
#             amount = int(request.POST.get("amount")) * 100  # convert to paisa
#             name = request.POST.get("name")
#             email = request.POST.get("email")

#             order = client.order.create({
#                 "amount": amount,
#                 "currency": "INR",
#                 "payment_capture": "1"
#             })

#             # Save Faild order in MongoDB
#             wallet_history_collection.insert_one({
#                 "order_id": order["id"],
#                 "email": email,
#                 "amount": amount / 100,  # store in rupees
#                 "status": "Faild",
#                 "created_at": timezone.now()
#             })

#             context = {
#                 "payment": order,
#                 "razorpay_key": settings.RAZORPAY_KEY_ID,
#                 "amount": amount / 100,
#                 "name": name,
#                 "email": email,
#             }
#             return render(request, "authuser/wallet.html", context)

#         except Exception as e:
#             print("Error in wallet_add_money:", e)
#             return redirect('profile')

#     return redirect('profile')

# @csrf_exempt
# def payment_status_add_money(request):
#     if request.method == "POST":
#         data = request.POST
#         try:
#             # Verify signature
#             client.utility.verify_payment_signature({
#                 'razorpay_order_id': data.get('razorpay_order_id'),
#                 'razorpay_payment_id': data.get('razorpay_payment_id'),
#                 'razorpay_signature': data.get('razorpay_signature')
#             })

#             order_id = data.get('razorpay_order_id')
#             payment_id = data.get('razorpay_payment_id')

#             # Update payment history
#             result = wallet_history_collection.update_one(
#                 {"order_id": order_id},
#                 {"$set": {
#                     "payment_id": payment_id,
#                     "status": "success",
#                     "updated_at": timezone.now()
#                 }}
#             )

#             if result.modified_count > 0:
#                 wallet_doc = wallet_history_collection.find_one({"order_id": order_id})
#                 if wallet_doc:
#                     email = wallet_doc["email"]
#                     amount = float(wallet_doc["amount"])

#                     # Update user's wallet balance
#                     users_collection.update_one(
#                         {"email": email},
#                         {"$inc": {"wallet_balance": amount}}
#                     )
#                     return redirect('profile')  # redirect to profile page

#             return redirect('profile')

#         except Exception as e:
#             return render(request, "authuser/payment_failed.html")

#     return redirect('profile')

# def handle_plan_expiry(user_doc):
#     """Checks if the user's active plan has expired and switches to the next queued plan."""

#     plan_history = user_doc.get("plan_history", [])
#     if not plan_history:
#         return "NO_PLAN"

#     now = timezone.now()

#     # Sort history by start date (ensure correct order)
#     plan_history = sorted(plan_history, key=lambda x: x["start_date"])

#     # Get active plan (first one that is not expired AND whose start_date <= now)
#     active_plan = None
#     for plan in plan_history:
#         if plan["start_date"] <= now and plan["Expiry"] >= now:
#             active_plan = plan
#             break

#     # CASE 1: Active plan exists → nothing to do
#     if active_plan:
#         return "ACTIVE_OK"

#     # CASE 2: No active plan → find next FUTURE plan
#     future_plans = [p for p in plan_history if p["start_date"] > now]
#     if future_plans:
#         next_plan = future_plans[0]

#         # Activate this plan
#         users_collection.update_one(
#             {"_id": user_doc["_id"]},
#             {"$set": {"Package": next_plan["plan_name"]}}
#         )
#         return "NEXT_ACTIVATED"

#     # CASE 3: No future plans → move to free plan
#     users_collection.update_one(
#         {"_id": user_doc["_id"]},
#         {"$set": {"Package": "free"}}
#     )

#     return "SET_FREE"
#=========================================================above is the local script and below is live script========================================================================
from django.shortcuts import render, redirect
from django.contrib.auth.hashers import make_password, check_password
from django.conf import settings
from django.contrib import messages
from pymongo import MongoClient
from django.http import JsonResponse,HttpResponse
from django.views.decorators.csrf import csrf_protect,csrf_exempt
import json
from datetime import datetime,timedelta
from django.utils import timezone
import uuid
from django.views.decorators.cache import never_cache
from bson.binary import Binary
import razorpay
from bson.objectid import ObjectId
import base64
from collections import defaultdict
from report.views import collections
from authuser.email_Send import send_welcome_email, send_documents_processing_email,send_approved_email,send_rejected_email
from django.contrib.auth.decorators import login_required
import os
from django.contrib.admin.views.decorators import staff_member_required
from django.contrib.auth import get_user_model, login as django_login
from subscription.utils import sync_subscription_from_mongo
from django.contrib.auth.hashers import check_password
from django.shortcuts import render
from django.contrib.admin.views.decorators import staff_member_required
import re

# Connect to MongoDB
mongo_client = MongoClient("mongodb://192.168.1.65:27017/")
db = mongo_client['ecgarrhythmias']
manage_db = mongo_client["manage"]# manage DB
patients_db = mongo_client['Patients']
users_collection = db["users"]
sessions_collection = db["sessions"] 
contact_collection = db['contact_messages']
manage_currency_collection = manage_db["currency"]
payment_history_collection = manage_db["payment_history"]
manage_ui_collection=manage_db["manage_ui"]
wallet_history_collection=manage_db["wallet_history"]
download_history_collection = db["download_history"]

client = razorpay.Client(auth=(settings.RAZORPAY_KEY_ID, settings.RAZORPAY_KEY_SECRET))

@staff_member_required
def registration_requests_admin(request):
    return render(request, "authuser/registration_requests.html")

def get_plan_features(plan_name):
    """Fetch features from manage_ui collection based on plan name"""
    plan = manage_ui_collection.find_one({"plan_name": plan_name})
    if plan and "features" in plan:
        return plan["features"]
    return []
def home(request):
    if 'user_session' in request.session:
        return redirect('/ommecgdata/')
    else:
        return redirect('/auth/login/')  # explicitly use auth prefix


def patient_list(request):
    return render(request, 'authuser/patient_list.html')

def help(request):
    return render(request, 'authuser/help.html')

def payment_failed(request):
    return render(request, "authuser/payment_failed.html")

# User Registration[register,get_registrations,update_registration_status]
@never_cache
@csrf_protect
# def register(request):
#     if request.method == "POST":
#         userType = request.POST.get('userType')
#         username = request.POST.get('username', '').strip()
#         email = request.POST.get('email', '').strip().lower()
#         country_code = request.POST.get('countryCode', '').strip()
#         phone = request.POST.get('phone', '').strip()
#         password = request.POST.get('password')
#         doctor_id = request.POST.get('doctorId', '').strip()

#         student_id_file = request.FILES.get('studentId')
#         address_proof_file = request.FILES.get('addressProof')
#         doctor_cert_file = request.FILES.get('doctorCert')
#         address_proof_other_file = request.FILES.get("addressProofOther")

#         # Basic validation
#         if not (username and email and password):
#             messages.error(request, "Please fill all required fields.")
#             return redirect('register')

#         if userType == "student":
#             if not student_id_file or not address_proof_file:
#                 messages.error(request, "Both Student ID and Address Proof are required for student registration.")
#                 return redirect('register')
#         elif userType == "doctor":
#             if not doctor_id or not doctor_cert_file:
#                 messages.error(request, "Doctor ID and Certificate are required for doctor registration.")
#                 return redirect('register')
#         elif userType == "other":
#             if not address_proof_other_file:
#                 messages.error(request, "Address Proof is required for Other users.")
#                 return redirect('register')

#         else:
#             messages.error(request, "Invalid user type.")
#             return redirect('register')
        
#         # Check if user exists
#         existing_user = users_collection.find_one({
#             "$or": [{"username": username}, {"email": email}]
#         })
#         if existing_user:
#             messages.error(request, "Username or Email already exists.")
#             return redirect('register')

#         hashed_password = make_password(password)

#         # --- Country Detection ---
#         if country_code == "+91":
#             country = "India"
#             currency = "INR"
#         elif country_code == "+1":
#             country = "United States"
#             currency = "USD"
#         else:
#             country = "Unknown"
#             currency = "USD"

#         default_package = "Free"
#         allowed_features = get_plan_features(default_package)

#         # --- Save uploaded files ---
#         def save_file(file, folder):
#             upload_dir = os.path.join(settings.MEDIA_ROOT, folder)
#             os.makedirs(upload_dir, exist_ok=True)
#             filename = f"{uuid.uuid4()}_{file.name}"
#             file_path = os.path.join(upload_dir, filename)
#             with open(file_path, 'wb+') as destination:
#                 for chunk in file.chunks():
#                     destination.write(chunk)
#             return f"{folder}/{filename}"  # relative path for easy access in templates

        
#         # SAVE FILES BASED ON USER TYPE -------------------
#         student_id_path = None
#         address_proof_path = None
#         doctor_cert_path = None
#         if userType == "student":
#             student_id_path = save_file(student_id_file, 'student_ids')
#             address_proof_path = save_file(address_proof_file, 'address_proofs')
#         elif userType == "doctor":
#             doctor_cert_path = save_file(doctor_cert_file, 'doctor_certificates')

#         elif userType == "other":
#             address_proof_path = save_file(address_proof_other_file, 'other_address_proofs')

#         # CREATE USER DOCUMENT ----------------------------
#         user_doc = {
#             "role": userType,
#             "username": username,
#             "email": email,
#             "country_code": country_code,
#             "country": country,
#             "phone": phone,
#             "password": hashed_password,
#             "doctorId": doctor_id if userType == "doctor" else None,
#             "register_time": timezone.now(),
#             "status": "pending",
#             "package": default_package,
#             "files": {
#                 "student_id_path": student_id_path,
#                 "address_proof_path": address_proof_path,
#                 "doctor_certificate_path": doctor_cert_path
#             }
#         }

#         inserted_user = users_collection.insert_one(user_doc)
#         user_id = str(inserted_user.inserted_id)

#         manage_currency_collection.insert_one({
#             "user_id": user_id,
#             "username": username,
#             "country_code": country_code,
#             "country": country,
#             "currency": currency,
#             "created_at": timezone.now(),
#         })

#         session_id = str(uuid.uuid4())
#         request.session['user_session'] = {
#             "session_id": session_id,
#             "user_id": user_id,
#             "username": username,
#             "email": email,
#             "role": userType,
#             "Package": default_package,
#             "features": allowed_features
#         }

#         sessions_collection.insert_one({
#             "_id": session_id,
#             "user_id": user_id,
#             "username": username,
#             "role": userType,
#             "login_time": timezone.now(),
#             "Package": default_package
#         })

#         send_welcome_email(username, email)
#         send_documents_processing_email(username, email)

#         return redirect('/ommecgdata/')

#     return render(request, 'authuser/register.html')
def register(request):
    if request.method == "POST":
        userType = request.POST.get('userType')
        username = request.POST.get('username', '').strip()
        email = request.POST.get('email', '').strip().lower()
        country_code = request.POST.get('countryCode', '').strip()
        phone = request.POST.get('phone', '').strip()
        password = request.POST.get('password')
        doctor_id = request.POST.get('doctorId', '').strip()

        student_id_file = request.FILES.get('studentId')
        address_proof_file = request.FILES.get('addressProof')
        doctor_cert_file = request.FILES.get('doctorCert')
        address_proof_other_file = request.FILES.get("addressProofOther")

        # Basic validation
        if not (username and email and password):
            messages.error(request, "Please fill all required fields.")
            return redirect('register')

        if userType == "student":
            if not student_id_file or not address_proof_file:
                messages.error(request, "Both Student ID and Address Proof are required for student registration.")
                return redirect('register')
        elif userType == "doctor":
            if not doctor_id or not doctor_cert_file:
                messages.error(request, "Doctor ID and Certificate are required for doctor registration.")
                return redirect('register')
        elif userType == "other":
            if not address_proof_other_file:
                messages.error(request, "Address Proof is required for Other users.")
                return redirect('register')

        else:
            messages.error(request, "Invalid user type.")
            return redirect('register')
        
        # Check if user exists
        # --- CHECK EMAIL & USERNAME ---
        if users_collection.find_one({"username": username}):
            return JsonResponse({"status": "username_exists"})

        if users_collection.find_one({"email": email}):
            return JsonResponse({"status": "email_exists"})

        hashed_password = make_password(password)

        # --- Country Detection ---
        if country_code == "+91":
            country = "India"
            currency = "INR"
        elif country_code == "+1":
            country = "United States"
            currency = "USD"
        else:
            country = "Unknown"
            currency = "USD"

        default_package = "Free"
        allowed_features = get_plan_features(default_package)

        # --- Save uploaded files ---
        def save_file(file, folder):
            upload_dir = os.path.join(settings.MEDIA_ROOT, folder)
            os.makedirs(upload_dir, exist_ok=True)
            filename = f"{uuid.uuid4()}_{file.name}"
            file_path = os.path.join(upload_dir, filename)
            with open(file_path, 'wb+') as destination:
                for chunk in file.chunks():
                    destination.write(chunk)
            return f"{folder}/{filename}"  # relative path for easy access in templates

        
        # SAVE FILES BASED ON USER TYPE -------------------
        student_id_path = None
        address_proof_path = None
        doctor_cert_path = None
        if userType == "student":
            student_id_path = save_file(student_id_file, 'student_ids')
            address_proof_path = save_file(address_proof_file, 'address_proofs')
        elif userType == "doctor":
            doctor_cert_path = save_file(doctor_cert_file, 'doctor_certificates')

        elif userType == "other":
            address_proof_path = save_file(address_proof_other_file, 'other_address_proofs')

        # CREATE USER DOCUMENT ----------------------------
        user_doc = {
            "role": userType,
            "username": username,
            "email": email,
            "country_code": country_code,
            "country": country,
            "phone": phone,
            "password": hashed_password,
            "doctorId": doctor_id if userType == "doctor" else None,
            "register_time": timezone.now(),
            "status": "pending",
            "package": default_package,
            "files": {
                "student_id_path": student_id_path,
                "address_proof_path": address_proof_path,
                "doctor_certificate_path": doctor_cert_path
            }
        }

        inserted_user = users_collection.insert_one(user_doc)
        user_id = str(inserted_user.inserted_id)

        manage_currency_collection.insert_one({
            "user_id": user_id,
            "username": username,
            "country_code": country_code,
            "country": country,
            "currency": currency,
            "created_at": timezone.now(),
        })

        session_id = str(uuid.uuid4())
        request.session['user_session'] = {
            "session_id": session_id,
            "user_id": user_id,
            "username": username,
            "email": email,
            "role": userType,
            "package": default_package,
            "features": allowed_features
        }

        sessions_collection.insert_one({
            "_id": session_id,
            "user_id": user_id,
            "username": username,
            "role": userType,
            "login_time": timezone.now(),
            "package": default_package
        })

        send_welcome_email(username, email)
        send_documents_processing_email(username, email)

        return redirect('/ommecgdata/')

    return render(request, 'authuser/register.html')

#admin
def get_registrations(request):
    try:
        registrations = []
        base_url = request.build_absolute_uri('/')[:-1]

        for doc in users_collection.find():
            if doc.get("role", "").lower() == "admin":
                continue

            file_data = doc.get("files", {})
            
            student_id_path = file_data.get("student_id_path")
            address_proof_path = file_data.get("address_proof_path")
            doctor_cert_path = file_data.get("doctor_certificate_path")

            # Convert file path ? full media URL
            def make_url(path):
                if path:
                    return f"{base_url}{settings.MEDIA_URL}{path}"
                return None

            registrations.append({
                "id": str(doc["_id"]),
                "user_name": doc.get("username", ""),
                "email": doc.get("email", ""),
                "phone_number": doc.get("phone", ""),
                "role": doc.get("role", "").lower(),
                "status": doc.get("status", ""),
                "student_id_url": make_url(student_id_path),
                "address_proof_url": make_url(address_proof_path),
                "doctor_certificate_url": make_url(doctor_cert_path),
                "doctor_id": doc.get("doctorId"),
                "created_at": doc.get("register_time"),
            })

        return JsonResponse(registrations, safe=False)

    except Exception as e:
        print("Error:", e)
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
    # If already logged in (Django session)
    if request.user.is_authenticated:
        return redirect('/ommecgdata/')

    # If old session exists
    if 'user_session' in request.session:
        return redirect('/ommecgdata/')

    if request.method == "POST":
        username = request.POST.get('username')
        password = request.POST.get('password')
        
        # Fetch from MongoDB
        user = users_collection.find_one({"username": username})
        handle_plan_expiry(user)
        if not user:
            messages.error(request, "User not found.")
            return redirect('login')

        if not check_password(password, user['password']):
            messages.error(request, "Invalid password.")
            return redirect('login')

        user_status = user.get("status", "pending").lower()

        # ----------------------------
        # REJECTED ACCOUNT
        # ----------------------------
        if user_status == "rejected":
            messages.error(request, "Your account request has been rejected.")
            return redirect('login')

        # ----------------------------
        # APPROVED ACCOUNT → Proceed Login
        # ----------------------------

        User = get_user_model()
        django_user, created = User.objects.get_or_create(
            username=user["username"],
            defaults={
                "email": user.get("email", ""),
                "password": make_password(None),
            }
        )

        sync_subscription_from_mongo(django_user)

        django_login(request, django_user)

        # SAVE status in session so JS can read it
        session_id = str(uuid.uuid4())
        request.session["user_session"] = {
            "session_id": session_id,
            "user_id": str(user["_id"]),
            "username": user["username"],
            "email": user["email"],
            "phone": user.get("phone"),
            "userType": user.get("userType", "user"),
            "status": user_status,              # ADDED
            "package": user.get("package", "Free"),
            "features": get_plan_features(user.get("Package", "Free")),
        }

        # Save session log to MongoDB
        sessions_collection.insert_one({
            "_id": session_id,
            "user_id": str(user["_id"]),
            "username": user["username"],
            "userType": user.get("userType", "user"),
            "login_time": timezone.now(),
            "package": user.get("package", "Free"),
            "status": user_status,              # ADDED
        })

        return redirect('/ommecgdata/')

    response = render(request, 'authuser/login.html')
    response['Cache-Control'] = 'no-store, no-cache, must-revalidate, max-age=0'
    response['Pragma'] = 'no-cache'
    response['Expires'] = '0'
    return response

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
        manage_db.payment_history.find({
            "$or": [{"email": email}, {"username": username}]
        }).sort("updated_at", -1)
    )

    formatted_payments = []
    for p in payment_history:
        formatted_payments.append({
            "order_id": p.get("order_id", "-"),
            "plan_name": p.get("plan_name", "Unknown Plan"),
            "amount": p.get("amount", 0),
            "date": p.get("updated_at", datetime.now()),
            "Expiry":p.get("Expiry", datetime.now()),
            "status": "Paid" if p.get("paid") else "Faild",
        })

    # -------------------------------
    # Fetch wallet balance
    # -------------------------------
    user_doc = db.users.find_one({"email": email})
    wallet_balance = float(user_doc.get("wallet_balance", 0.0)) if user_doc else 0.0

    # -------------------------------
    # Fetch wallet transactions
    # -------------------------------
    wallet_cursor = manage_db.wallet_history.find({"email": email}).sort("created_at", -1)

    wallet_transactions = []
    for w in wallet_cursor:
        txn_type = "credit" if w.get("status") == "success" else "Faild"
        wallet_transactions.append({
            "txn_id": str(w.get("_id")),
            "date": w.get("created_at", timezone.now()),
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

        return JsonResponse({"message": "Password changed successfully."}, status=200)

    except json.JSONDecodeError:
        return JsonResponse({"error": "Invalid JSON data."}, status=400)
    except Exception as e:
        return JsonResponse({"error": str(e)}, status=500)
        
# User Logout
@never_cache
def logout(request):
    request.session.flush()
    messages.success(request, "Logged out successfully.")

    response = redirect('login')
    response['Cache-Control'] = 'no-store, no-cache, must-revalidate, max-age=0'
    response['Pragma'] = 'no-cache'
    response['Expires'] = '0'
    return response


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

#All Patient backend Show[get_patient_arrhythmia_records,get_patients_data]   
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

# def create_order(request):
#     if request.method == "POST":
#         plan_name = request.POST.get("plan_name")
#         name = request.POST.get("name")
#         email = request.POST.get("email")
#         amount = int(request.POST.get("amount")) * 100  # paise
#         payment = client.order.create({
#             "amount": amount,
#             "currency": "INR",
#             "payment_capture": "1"
#         })
#         order = payment_history_collection.insert_one({
#             "username":name,
#             "email":email,
#             "plan_name":plan_name,
#             "amount":amount // 100,
#             "order_id":payment['id'],
#             "paid":False,
#         })
#         context = {
#             "payment": payment,
#             "plan_name": plan_name,
#             "amount": amount // 100,
#             "username": name,
#             "email": email,
#             "razorpay_key": settings.RAZORPAY_KEY_ID
#         }
#         return render(request, "authuser/payment.html", context)

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

#def get_updated_plan_dates(user_doc):
#    """
#    Determines the correct start_date and expiry_date for a newly purchased plan.
#    Also blocks user if they already bought 3 plans in the same month.
#    """
#
#    plan_history = user_doc.get("plan_history", [])
#
#    now = timezone.now()
#    current_month = now.month
#    current_year = now.year
#
#    # Count purchases in current month
#    monthly_purchases = [
#        p for p in plan_history
#        if p.get("buy_date") and  # UPDATED
#           p["buy_date"].month == current_month and
#           p["buy_date"].year == current_year
#    ]
#
#    if len(monthly_purchases) >= 3:
#        return None, None, "LIMIT_REACHED"
#
#    # BUY DATE is always NOW
#    buy_date = now
#
#    # CASE 1: User has NO previous plan → start today
#    if not plan_history:
#        start_date = now
#        expiry_date = start_date + timedelta(days=31)
#        return buy_date, start_date, expiry_date, "OK"
#
#    # CASE 2: User has history → get last plan expiry
#    last_plan = plan_history[-1]
#    last_expiry = last_plan.get("expiry_date") or last_plan.get("Expiry")
#
#    if last_expiry:
#        start_date = last_expiry + timedelta(days=1)
#    else:
#        start_date = now
#
#    expiry_date = start_date + timedelta(days=31)
#
#    return buy_date, start_date, expiry_date, "OK"

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
       start_date = timezone.now()
       end_date = start_date + timedelta(days=31)
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
#@csrf_exempt
#def payment_status(request):
#    if request.method == "POST":
#        data = request.POST
#        try:
#            client.utility.verify_payment_signature({
#                'razorpay_order_id': data.get('razorpay_order_id'),
#                'razorpay_payment_id': data.get('razorpay_payment_id'),
#                'razorpay_signature': data.get('razorpay_signature')
#            })
#
#            order_id = data.get('razorpay_order_id')
#            payment_id = data.get('razorpay_payment_id')
#
#            # Fetch payment document
#            payment_doc = payment_history_collection.find_one({"order_id": order_id})
#            if not payment_doc:
#                messages.error(request, "Payment not found!")
#                return redirect('/ommecgdata/')
#
#            user_email = payment_doc.get("email")
#            plan_name = payment_doc.get("plan_name")
#
#            # Fetch user
#            user_doc = users_collection.find_one({"email": user_email})
#            if not user_doc:
#                messages.error(request, "User not found!")
#                return redirect('/ommecgdata/')
#
#            # Determine correct start/end dates + limit check
#            start_date, end_date, status = get_updated_plan_dates(user_doc)
#
#            if status == "LIMIT_REACHED":
#                messages.error(request, "You can purchase only 3 plans in a month.")
#                return redirect('/ommecgdata/')
#
#            # Update payment entry as paid
#            payment_history_collection.update_one(
#                {"order_id": order_id},
#                {"$set": {
#                    "paid": True,
#                    "payment_id": payment_id,
#                    "updated_at": timezone.now(),
#                    "Expiry": end_date
#                }}
#            )
#
#            # Update user’s current package + push to history
#            users_collection.update_one(
#                {"email": user_email},
#                {
#                    "$set": {"Package": plan_name},
#                    "$push": {
#                        "plan_history": {
#                            "plan_name": plan_name,
#                            "start_date": start_date,
#                            "Expiry": end_date
#                        }
#                    }
#                }
#            )
#
#            messages.success(request, "Payment successful! Re-login to apply your new plan.")
#            request.session.flush()
#            return render(request, 'authuser/login.html')
#
#        except Exception as e:
#            print("Payment error:", str(e))
#            messages.error(request, "Payment verification failed.")
#            return render(request, "authuser/payment_failed.html")
#
#    return redirect('/ommecgdata/')

def get_user_status(request):
    try:
        username = request.GET.get("username")
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
            wallet_history_collection.insert_one({
                "order_id": order["id"],
                "email": email,
                "amount": amount / 100,  # store in rupees
                "status": "Faild",
                "created_at": timezone.now()
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
            print("Error in wallet_add_money:", e)
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
            result = wallet_history_collection.update_one(
                {"order_id": order_id},
                {"$set": {
                    "payment_id": payment_id,
                    "status": "success",
                    "updated_at": timezone.now()
                }}
            )

            if result.modified_count > 0:
                wallet_doc = wallet_history_collection.find_one({"order_id": order_id})
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

from django.utils import timezone
from django.utils.timezone import make_aware, is_naive

def make_dt_aware(dt):
    if dt and is_naive(dt):
        return make_aware(dt)
    return dt

def handle_plan_expiry(user_doc):
    """Checks if the user's active plan has expired and switches to the next queued plan."""

    plan_history = user_doc.get("plan_history", [])
    if not plan_history:
        return "NO_PLAN"

    now = timezone.now()

    # Make all dates timezone-aware
    for plan in plan_history:
        plan["start_date"] = make_dt_aware(plan["start_date"])
        plan["Expiry"] = make_dt_aware(plan["Expiry"])

    # Sort by start date
    plan_history = sorted(plan_history, key=lambda x: x["start_date"])

    # Find active plan
    active_plan = None
    for plan in plan_history:
        if plan["start_date"] <= now and plan["Expiry"] >= now:
            active_plan = plan
            break

    if active_plan:
        return "ACTIVE_OK"

    # No active → check future plan
    future_plans = [p for p in plan_history if p["start_date"] > now]
    if future_plans:
        next_plan = future_plans[0]

        users_collection.update_one(
            {"_id": user_doc["_id"]},
            {"$set": {"package": next_plan["plan_name"]}}
        )
        return "NEXT_ACTIVATED"

    # No future → switch to free
    users_collection.update_one(
        {"_id": user_doc["_id"]},
        {"$set": {"package": "free"}}
    )

    return "SET_FREE"