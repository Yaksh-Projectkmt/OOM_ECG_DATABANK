from django.shortcuts import render, redirect
from django.http import JsonResponse,FileResponse,HttpResponse
from bson import ObjectId
import pymongo
import matplotlib.pyplot as plt
from io import BytesIO
from django.views.decorators.http import require_http_methods
from django.views.decorators.csrf import csrf_exempt  
from pymongo import MongoClient
import json
import pandas as pd
from django.core.files.storage import default_storage
from django.conf import settings
from scipy import signal
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import os
import math
import plotly.graph_objects as go
import plotly.io as pio
import traceback
import io
from .PQRST_detection_model import check_r_index, check_qs_index, check_pt_index, r_index_model, pt_index_model
import random
from django.views.decorators.http import require_POST
from django.core.paginator import Paginator
from .get_Data import get_the_data
from django.core.files.base import ContentFile
import base64
import matplotlib
import datetime
from datetime import datetime, timezone
matplotlib.use('Agg')

client = pymongo.MongoClient("mongodb://192.168.1.65:27017/")
db = client["ecgarrhythmias"]
patient_db=client['Patients']

Morphology_data=client["Morphology_data"]
Morphology_patient_db=client["Morphology_Patients"]

Analysis_data=client["Analysis_data"]
Analysis_data_patient=client["Analysis_patients"]

Queues = client["Queue"]
logs_collection = Queues["multiple_segments"]

arrhythmias_dict = {
'Myocardial Infarction': ['T-wave abnormalities', 'Inferior MI', 'Lateral MI'],
'Atrial Fibrillation & Atrial Flutter': ['AFIB', 'Aflutter', 'AFL'],
'HeartBlock': ['I DEGREE', 'MOBITZ-I', 'MOBITZ-II', 'III Degree'],
'Junctional Rhythm': ['Junctional Bradycardia', 'Junctional Rhythm'],
'Premature Atrial Contraction': ['PAC-Isolated', 'PAC-Bigeminy', 'PAC-Couplet', 'PAC-Triplet',
                                'SVT', 'PAC-Trigeminy', 'PAC-Quadrigeminy'],
'Premature Ventricular Contraction': ['AIVR', 'PVC-Bigeminy', 'PVC-Couplet', 'PVC-Isolated',
                                                'PVC-Quadrigeminy', 'NSVT', 'PVC-Trigeminy',
                                                'PVC-Triplet', 'IVR', 'VT'],
'Ventricular Fibrillation and Asystole': ['VFIB', 'VFL', 'ASYSTOLE'],
'Noise':['Noise'],
'LBBB':['LBBB','RBBB'],
'Artifacts': ['Artifacts'],
'SINUS-ARR': ['SINUS-ARR'],
'ShortPause': ['Short Pause', 'Long Pause'],
'TC': ['TC'],
'WIDE-QRS': ['WIDE-QRS'],
'Abnormal': ['ABNORMAL'],
'Normal':['Normal'],
'Others':['Others'],
}

# Dashboard View
def index(request):
    return render(request, 'oom_ecg_data/ecg_data_index.html', {'arrhythmias_dict':arrhythmias_dict})#,data=data

def select_arrhythmia(request):
    selected_arrhythmia = request.GET.get('arrhythmia', '')  # Get the selected key
    return render(request, "your_template.html", {
        "arrhythmias_dict": arrhythmias_dict,
        "selected_arrhythmia": selected_arrhythmia,
    })

def update_patient_db(patient_id, arrhythmia, frequency, datalength):
    """
    Update or insert patient summary in patient_db.<arrhythmia> collection.
    """
    patient_col = patient_db[arrhythmia]

    # time in minutes
    time_minutes = (datalength / int(frequency)) / 60

    # increment if exists, else insert
    patient_col.update_one(
        {"PatientID": patient_id},
        {
            "$inc": {
                "total_records": 1,
                "total_time": time_minutes
            },
            "$setOnInsert": {
                "PatientID": patient_id
            }
        },
        upsert=True
    )
# Mapping between main DB and patient DB
PATIENT_DB_MAP = {
    "ecgarrhythmias": client["Patients"],
    "Morphology_data": client["Morphology_Patients"],
    "Analysis_data": client["Analysis_patients"],
}

def update_patient_on_edit(patient_id, old_arrhythmia, new_arrhythmia, datalength, frequency, source_db):
    """
    Move a record from old_arrhythmia new_arrhythmia inside the correct patient DB.
    source_db: the DB where the record was found (ecgarrhythmias, Morphology_data, etc.)
    """
    db_name = source_db.name
    if db_name not in PATIENT_DB_MAP:
        raise ValueError(f"No patient DB mapping found for {db_name}")

    patient_db = PATIENT_DB_MAP[db_name]
    time_minutes = (datalength / frequency) / 60.0

    # --- Decrement from old arrhythmia ---
    old_col = patient_db[old_arrhythmia]
    old_col.update_one(
        {"PatientID": patient_id},
        {"$inc": {"total_records": -1, "total_time": -time_minutes}}
    )

    # Cleanup if records hit 0 or below
    doc = old_col.find_one({"PatientID": patient_id})
    if doc and doc.get("total_records", 0) <= 0:
        old_col.delete_one({"PatientID": patient_id})

    # --- Increment in new arrhythmia ---
    new_col = patient_db[new_arrhythmia]
    new_col.update_one(
        {"PatientID": patient_id},
        {
            "$inc": {"total_records": 1, "total_time": time_minutes},
            "$setOnInsert": {"PatientID": patient_id}
        },
        upsert=True
    )
    
def update_patient_on_delete(patient_id, arrhythmia, datalength, frequency, source_db):
    """
    When deleting a record, decrement patient counters in the correct patient DB.
    """
    db_name = source_db.name
    if db_name not in PATIENT_DB_MAP:
        raise ValueError(f"No patient DB mapping found for {db_name}")

    patient_db_ref = PATIENT_DB_MAP[db_name]
    time_minutes = (datalength / frequency) / 60.0
    col = patient_db_ref[arrhythmia]
    col.update_one(
        {"PatientID": patient_id},
        {"$inc": {"total_records": -1, "total_time": -time_minutes}}
    )

    # Remove patient entry if empty
    doc = col.find_one({"PatientID": patient_id})
    if doc and doc.get("total_records", 0) <= 0:
        col.delete_one({"PatientID": patient_id})

@csrf_exempt
def new_insert_data(request):
    if request.method == "POST":
        try:
            if "csv_file" not in request.FILES:
                return JsonResponse({"status": "error", "message": "File is not uploaded..."})

            patient_id = request.POST.get("patientId")
            arrhythmia_mi = request.POST.getlist("arrhythmia[]")
            sub_arrhythmia = request.POST.getlist("subArrhythmia[]")
            frequency = int(request.POST.get("newfrequency"))
            lead_type = request.POST.get("lead")
            lead = int(lead_type.split(' ')[0])

            if len(arrhythmia_mi) != len(sub_arrhythmia):
                return JsonResponse({
                    "status": "error",
                    "message": "Mismatch between arrhythmia and sub-arrhythmia selections."
                })

            collections_to_insert = [
                (arrhythmia_mi[i].strip(), sub_arrhythmia[i].strip())
                for i in range(len(arrhythmia_mi))
            ]

            file = request.FILES["csv_file"]
            file_path = os.path.join(settings.MEDIA_ROOT, "temp", file.name)
            default_storage.save(file_path, file)

            all_lead_data = pd.read_csv(file_path)
            column_count = all_lead_data.shape[1]

              # Validate lead-column match
            if lead_type == "2" and column_count > 2:
                return JsonResponse({"status": "error", "message": "Incorrect lead selected."})
            elif lead_type == "7" and column_count not in [7, 8]:
                return JsonResponse({"status": "error", "message": "Incorrect lead selected."})
            elif lead_type == "12" and column_count not in [12, 13]:
                return JsonResponse({"status": "error", "message": "Incorrect lead selected."})

            all_lead_data.columns = all_lead_data.columns.str.upper()

            # Drop time column if exists
            if lead_type == "12" and all_lead_data.shape[1] == 13:
                all_lead_data = all_lead_data.iloc[:, 1:13]
            elif lead_type == "7" and all_lead_data.shape[1] == 8:
                all_lead_data = all_lead_data.iloc[:, 1:8]
            elif lead_type == "2" and all_lead_data.shape[1] > 1:
                all_lead_data = all_lead_data.iloc[:, 1:2]

            # Handle header rows with alphabet values
            if any(str(_).isalpha() for _ in all_lead_data.iloc[0, :].values):
                if lead_type == "2":
                    all_lead_data = pd.read_csv(file_path, skiprows=1, usecols=[0]).fillna(0)
                    all_lead_data.columns = ['II']
                elif lead_type == "7":
                    col_indices = [0, 1, 2, 3, 4, 5, 6]
                    all_lead_data = pd.read_csv(file_path, skiprows=1, usecols=col_indices).fillna(0)
                    all_lead_data.columns = ['I', 'II', 'III', 'aVR', 'aVL', 'aVF', 'V5']
                elif lead_type == "12":
                    col_indices = list(range(12))
                    all_lead_data = pd.read_csv(file_path, skiprows=1, usecols=col_indices).fillna(0)
                    all_lead_data.columns = ['I', 'II', 'III', 'aVR', 'aVL', 'aVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']
            else:
                # Rename to standard
                if lead_type == "2":
                    all_lead_data = all_lead_data.rename(columns={all_lead_data.columns[0]: 'II'})
                elif lead_type == "7":
                    all_lead_data = all_lead_data.rename(columns={
                        all_lead_data.columns[0]: 'I',
                        all_lead_data.columns[1]: 'II',
                        all_lead_data.columns[2]: 'III',
                        all_lead_data.columns[3]: 'aVR',
                        all_lead_data.columns[4]: 'aVL',
                        all_lead_data.columns[5]: 'aVF',
                        all_lead_data.columns[6]: 'V5'
                    })
                elif lead_type == "12":
                    all_lead_data = all_lead_data.rename(columns={
                        all_lead_data.columns[0]: 'I',
                        all_lead_data.columns[1]: 'II',
                        all_lead_data.columns[2]: 'III',
                        all_lead_data.columns[3]: 'aVR',
                        all_lead_data.columns[4]: 'aVL',
                        all_lead_data.columns[5]: 'aVF',
                        all_lead_data.columns[6]: 'V1',
                        all_lead_data.columns[7]: 'V2',
                        all_lead_data.columns[8]: 'V3',
                        all_lead_data.columns[9]: 'V4',
                        all_lead_data.columns[10]: 'V5',
                        all_lead_data.columns[11]: 'V6'
                    })

            # Check for duplicates
            for coll_name, sub_arr in collections_to_insert:
                collection = db[coll_name]
                exists = collection.count_documents({
                    'PatientID': patient_id,
                    'Arrhythmia': sub_arr,
                    'Lead': int(lead),
                    'Frequency': int(frequency)
                }) > 0
                if exists:
                    if os.path.exists(file_path):
                        os.remove(file_path)
                    return JsonResponse({
                        "status": "error",
                        "message": f"Duplicate data found in collection '{coll_name}' with sub-arrhythmia '{sub_arr}'."
                    })
            chunk_size = 2000
            total_rows = len(all_lead_data)
            num_chunks = math.ceil(total_rows / chunk_size)
            columns = list(all_lead_data.columns)

            for chunk_index in range(num_chunks):
                start = chunk_index * chunk_size
                end = min((chunk_index + 1) * chunk_size, total_rows)
                chunk = all_lead_data.iloc[start:end]

                data_dict = {col.upper(): chunk[col].tolist() for col in columns}
                datalength = len(chunk)  # <- rows count per chunk

                for coll_name, sub_arr in collections_to_insert:
                    db_insert_data = {
                        'PatientID': patient_id,
                        'Arrhythmia': sub_arr,
                        'Lead': lead,
                        'Frequency': frequency,
                        'datalength': datalength,
                        "server":"Local",
                        "created_at": datetime.now(timezone.utc),
                        'Data': data_dict
                    }
                    db[coll_name].insert_one(db_insert_data)
                    # --- Update patient_db after insert ---
                    update_patient_db(patient_id, coll_name, frequency, datalength)
            if os.path.exists(file_path):
                os.remove(file_path)

            return JsonResponse({
                "status": "success",
                "message": f"{num_chunks} chunks inserted successfully for PatientID {patient_id}."
            })

        except Exception as e:
            return JsonResponse({"status": "error", "message": str(e)})

    return JsonResponse({"status": "error", "message": "Invalid request!"})

 # API view for ECG data (Example)

def api_ecg_data(request):
    data = {"message": "ECG data API response"}
    return JsonResponse(data)

def new_table_data(request):
    return render(request,'oom_ecg_data/table.html')

def submit_form(request):
    # Your form handling logic here
    return render(request, 'oom_ecg_data/table.html')


def get_collection_with_data(patient_query, arrhythmia):
    # List of DBs to search (priority order)
    db_candidates = [db, Morphology_data, Analysis_data]
    
    for candidate in db_candidates:
        if arrhythmia in candidate.list_collection_names():
            collection = candidate[arrhythmia]
            total_count = collection.count_documents(patient_query)
            if total_count > 0:
                return collection, total_count
    return None, 0
@csrf_exempt
def fetch_ecg_data(request):
    if request.method == 'POST':
        try:
            patient_id = request.POST.get('patientId')
            lead_type = int(request.POST.get('leadType'))
            arrhythmia = request.POST.get('arrhythmia')
            frequency = int(request.POST.get('frequency'))
        except Exception as e:
            return JsonResponse({"status": "error", "message": "Invalid input: " + str(e)}, status=400)

        query = {
            "PatientID": patient_id,
            "Lead": lead_type,
            "Frequency": frequency
        }

        #  Search in arrhythmia-wise collections across multiple DBs
        collection, total_count = get_collection_with_data(query, arrhythmia)

        if not collection:
            return JsonResponse({
                "status": "error",
                "message": "No ECG data found for the given criteria",
                "total_count": 0
            })

        total_pages = (total_count + 9) // 10

        # Fetch first 10 records
        pipeline = [
            {"$match": query},
            {"$limit": 10},
            {"$project": {
                "PatientID": 1,
                "Arrhythmia": 1,
                "Lead": 1,
                "Frequency": 1,
                "samples_taken": {
                    "$cond": {
                        "if": {"$isArray": "$Data.II"},
                        "then": {"$size": "$Data.II"},
                        "else": 0
                    }
                }
            }}
        ]
        first_page = list(collection.aggregate(pipeline))

        page_obj = [
            {
                "object_id": str(r["_id"]),
                "PatientID": r.get("PatientID", ""),
                "Arrhythmia": r.get("Arrhythmia", ""),
                "Lead": r.get("Lead", ""),
                "Frequency": r.get("Frequency", ""),
                "duration": int(round(r.get("samples_taken", 0) / int(r.get("Frequency", 1)), 2))
            }
            for r in first_page
        ]

        # Save session for pagination
        request.session["ecg_query"] = {
            "patient_id": patient_id,
            "lead_type": lead_type,
            "frequency": frequency,
            "arrhythmia": arrhythmia
        }
        request.session["total_pages"] = total_pages
        request.session.modified = True

        return JsonResponse({
            "status": "success",
            "data": page_obj,
            "total_pages": total_pages,
            "total_records": total_count,
            "arrhythmia": arrhythmia
        })

    # Handle AJAX GET (pagination)
    if request.headers.get('x-requested-with') == 'XMLHttpRequest':
        page_str = request.GET.get("page")
        if not page_str or not page_str.isdigit():
            return JsonResponse({"error": "Missing or invalid page parameter"}, status=400)

        page = int(page_str)
        ecg_query = request.session.get("ecg_query")
        if not ecg_query:
            return JsonResponse({"error": "No query stored"}, status=400)

        query = {
            "PatientID": ecg_query["patient_id"],
            "Lead": ecg_query["lead_type"],
            "Frequency": ecg_query["frequency"]
        }
        collection = db[ecg_query["arrhythmia"]]
        total_count = collection.count_documents(query)
        total_pages = (total_count + 9) // 10
        skip_count = (page - 1) * 10

        # Use aggregation for pagination too
        pipeline = [
            {"$match": query},
            {"$skip": skip_count},
            {"$limit": 10},
            {"$project": {
                "PatientID": 1,
                "Arrhythmia": 1,
                "Lead": 1,
                "Frequency": 1,
                "samples_taken": {
                    "$cond": {
                        "if": {"$isArray": "$Data.II"},
                        "then": {"$size": "$Data.II"},
                        "else": 0
                    }
                }
            }}
        ]
        page_records = list(collection.aggregate(pipeline))

        page_data = [
            {
                "object_id": str(r["_id"]),
                "PatientID": r.get("PatientID", ""),
                "Arrhythmia": r.get("Arrhythmia", ""),
                "Lead": r.get("Lead", ""),
                "Frequency": r.get("Frequency", ""),
                "duration": int(round(r.get("samples_taken", 0) / int(r.get("Frequency", 1)), 2)) 
            }
            for r in page_records
        ]

        return JsonResponse({
            "data": page_data,
            "total_records": total_count,
            "total_pages": total_pages
        })

    return JsonResponse({"error": "Invalid request"}, status=400)

def fetch_random_ecg_data(request, arrhythmia):
    collection = db[arrhythmia]

    # Pagination
    page = int(request.GET.get("page", 1))
    page_size = 10
    skip = (page - 1) * page_size

    # Filters
    patient_id = request.GET.get("patientId", "").strip()
    lead = request.GET.get("lead", "").strip()
    arrhythmia_param = request.GET.get("arrhythmia", "").strip()
    frequency = request.GET.get("frequency", "").strip()

    query = {}
    if patient_id:
        query["PatientID"] = {"$regex": patient_id, "$options": "i"}
    if lead:
        query["Lead"] = int(lead) if lead.isdigit() else (2 if lead.upper() == "II" else lead)
    if arrhythmia_param:
        query["Arrhythmia"] = {"$regex": arrhythmia_param, "$options": "i"}
    if frequency:
        query["Frequency"] = {"$regex": frequency, "$options": "i"}

    # Count
    total_count = collection.count_documents(query) if query else collection.estimated_document_count()
    total_pages = (total_count + page_size - 1) // page_size

    # Aggregation
    pipeline = [
        {"$match": query},
        {"$skip": skip},
        {"$limit": page_size},
        {"$project": {
            "PatientID": 1,
            "Arrhythmia": 1,
            "Lead": 1,
            "Frequency": 1,
            "samples_taken": {
                "$cond": {
                    "if": {"$isArray": "$Data.II"},
                    "then": {"$size": "$Data.II"},
                    "else": 0
                }
            }
        }}
    ]
    records = list(collection.aggregate(pipeline))

    page_obj = [
        {
            "object_id": str(r["_id"]),
            "PatientID": r.get("PatientID", ""),
            "Arrhythmia": r.get("Arrhythmia", ""),
            "Lead": "II" if r.get("Lead", "") == 2 else r.get("Lead", ""),
            "LeadNumeric": r.get("Lead", ""),
            "Frequency": r.get("Frequency", ""),
            "collection_name": arrhythmia,
            "duration": int(round(r.get("samples_taken", 0) / int(r.get("Frequency", 1)), 2)) 
        }
        for r in records
    ]

    if request.headers.get("X-Requested-With") == "XMLHttpRequest":
        return JsonResponse({
            "status": "success",
            "data": page_obj,
            "total_records": total_count,
            "total_pages": total_pages,
            "current_page": page
        })

    return render(request, "oom_ecg_data/ecg_details.html", {
        "arrhythmia": arrhythmia,
        "page_obj": page_obj,
        "total_pages": total_pages,
        "current_page": page,
        "card_name": arrhythmia
    })


def get_object_id(request):
    if request.method != "POST":
        return JsonResponse({"error": "Invalid request method"}, status=405)

    try:
        data = json.loads(request.body)
        patient_id = data.get("patientId")
        lead_type = str(data.get("lead"))
        arrhythmia_raw = data.get("selectedArrhythmia", "").strip()
        objectID = data.get("objectId")
        samples_taken = int(data.get("samplesTaken"))

        # Support multiple arrhythmias
        arrhythmia_list = [a.strip() for a in arrhythmia_raw.split(",") if a.strip()]

        if not objectID or not arrhythmia_list:
            return JsonResponse({"error": "Missing required parameters"}, status=400)

        objid = ObjectId(objectID)
        result = None
        found_collection = None

        # List of DBs to search
        db_candidates = [db, Morphology_data, Analysis_data]

        # Try each arrhythmia across all DBs
        for arrhythmia in arrhythmia_list:
            for candidate in db_candidates:
                for col_name in candidate.list_collection_names():
                    # Match ignoring case and underscores/spaces
                    if col_name.lower().replace("_", " ") == arrhythmia.lower().replace("_", " "):
                        collection = candidate[col_name]
                        result = collection.find_one({"_id": objid})
                        if result:
                            found_collection = f"{candidate.name}.{col_name}"
                            break
                if result:
                    break
            if result:
                break

        if not result or "Data" not in result:
            return JsonResponse({"error": "ECG data not found"}, status=404)

        ecg_data_dict = result["Data"]

        # Normalize lead keys
        standard_leads = ['I', 'II', 'III', 'aVR', 'aVL', 'aVF',
                          'V1', 'V2', 'V3', 'V4', 'V5', 'V6']
        lead_mapping = {lead.lower(): lead for lead in standard_leads}

        normalized_ecg_data = {}
        for key, value in ecg_data_dict.items():
            norm_key = lead_mapping.get(str(key).lower().strip(), key)
            normalized_ecg_data[norm_key] = value

        # Return lead data
        if lead_type == "2":
            if "II" not in normalized_ecg_data:
                return JsonResponse({"error": "Lead II data not found"}, status=404)
            ecg_data = normalized_ecg_data["II"][:samples_taken]
            return JsonResponse({"x": list(range(len(ecg_data))), "ecgData": ecg_data})

        elif lead_type in ["7", "12"]:
            lead_sets = {
                "7": ["I", "II", "III", "aVR", "aVL", "aVF", "V5"],
                "12": ["I", "II", "III", "aVR", "aVL", "aVF",
                       "V1", "V2", "V3", "V4", "V5", "V6"]
            }
            selected_leads = lead_sets.get(lead_type, [])
            extracted = {lead: normalized_ecg_data[lead][:samples_taken]
                         for lead in selected_leads if lead in normalized_ecg_data}
            if not extracted:
                return JsonResponse({"error": f"No valid leads found for {lead_type}-lead ECG"}, status=404)
            return JsonResponse({"ecgData": extracted})

        elif lead_type in normalized_ecg_data:
            ecg_data = normalized_ecg_data[lead_type][:samples_taken]
            return JsonResponse({"x": list(range(len(ecg_data))), "ecgData": ecg_data})

        else:
            return JsonResponse({"error": f"Lead {lead_type} data not found"}, status=404)

    except Exception as e:
        return JsonResponse({"error": str(e)}, status=400)

@csrf_exempt
def edit_datas(request):
    db_candidates = [client["ecgarrhythmias"], Morphology_data, Analysis_data]

    if request.method != "POST":
        return JsonResponse({'status': 'error', 'message': 'Invalid request method'}, status=405)

    try:
        data = json.loads(request.body)
    except json.JSONDecodeError:
        return JsonResponse({'status': 'error', 'message': 'Invalid JSON'}, status=400)

    # Helper function to normalize collection names
    def normalize(name: str):
        return name.lower().replace("_", " ").strip()

    # CASE 1: Check if collection exists in ANY DB
    if 'collection_name' in data and len(data) == 1:
        for candidate in db_candidates:
            for col_name in candidate.list_collection_names():
                if normalize(col_name) == normalize(data['collection_name']):
                    return JsonResponse({'exists': True})
        return JsonResponse({'exists': False})

    # CASE 2: Proceed to update
    patient_id = data.get('PatientID')
    object_id = data.get('object_id')
    old_collection_name = data.get('old_collection')
    new_collection_name = data.get('new_collection')
    lead = data.get('lead')

    if not all([object_id, old_collection_name, new_collection_name, lead]):
        return JsonResponse({'status': 'error', 'message': 'Missing required fields'}, status=400)

    try:
        lead = int(lead)
    except ValueError as e:
        return JsonResponse({'status': 'error', 'message': f'Invalid Lead value: {e}'}, status=400)

    try:
        obj_id = ObjectId(object_id)
    except Exception as e:
        return JsonResponse({'status': 'error', 'message': f'Invalid ObjectId: {e}'}, status=400)

    fetched_data = None
    find_collection = None
    source_db = None

    # Search for document across all DBs in old_collection_name
    for candidate in db_candidates:
        for col_name in candidate.list_collection_names():
            if normalize(col_name) == normalize(old_collection_name):
                coll = candidate[col_name]
                doc = coll.find_one({'_id': obj_id})
                if doc:
                    fetched_data = doc
                    find_collection = coll
                    source_db = candidate
                    break
        if fetched_data:
            break

    if not fetched_data:
        return JsonResponse({'status': 'error', 'message': 'Data not found'}, status=404)

    if not patient_id:
        patient_id = fetched_data.get('PatientID')

    # Locate target collection (normalize too)
    new_collection = None
    for col_name in source_db.list_collection_names():
        if normalize(col_name) == normalize(new_collection_name):
            new_collection = source_db[col_name]
            break

    if not new_collection:
        new_collection = source_db[new_collection_name]

    # Check for duplicates
    duplicate = new_collection.find_one({'Data': fetched_data.get('Data')})
    if duplicate:
        return JsonResponse({
            'status': 'error',
            'message': "Duplicate data: This patient's ECG with same lead and arrhythmia already exists."
        }, status=409)

    # Prepare new document
    update_arrhy_data = {
        'PatientID': patient_id,
        'Arrhythmia': new_collection_name,
        'Lead': lead,
        'Frequency': fetched_data.get('Frequency'),
        'leadtype': fetched_data.get('leadtype'),
        'datalength': fetched_data.get('datalength'),
        'Version': fetched_data.get('Version'),
        'title_lines': fetched_data.get('title_lines'),
        'summar': fetched_data.get('**metrics'),
        'server': fetched_data.get('server', 'Local'),
        'created_at': fetched_data.get('created_at', datetime.utcnow()),
        'Data': fetched_data.get('Data')
    }

    insert_result = new_collection.insert_one(update_arrhy_data)
    find_collection.delete_one({'_id': obj_id})

    datalength = fetched_data.get("datalength")
    freq = fetched_data.get('Frequency', 200)
    update_patient_on_edit(patient_id, old_collection_name, new_collection_name, datalength, freq, source_db)

    request.session["ecg_query"] = {
        "patient_id": patient_id,
        "lead_type": lead,
        "frequency": freq,
        "arrhythmia": new_collection_name
    }
    request.session.modified = True

    return JsonResponse({
        'status': 'success',
        'message': 'Data edited and saved successfully',
        'new_object_id': str(insert_result.inserted_id)
    })

@csrf_exempt
def delete_data(request):
    if request.method != "POST":
        return JsonResponse({'status': 'error', 'message': 'Invalid request method'}, status=405)

    try:
        object_id = request.POST.get('object_id')
        if not object_id:
            return JsonResponse({'status': 'error', 'message': 'Missing object_id'}, status=400)

        obj_id = ObjectId(object_id)

        # Candidate DBs to search
        db_candidates = [client["ecgarrhythmias"], Morphology_data, Analysis_data]

        fetched_data = None
        find_collection = None
        source_db = None
        collection_name = None

        # Search across all DBs
        for candidate in db_candidates:
            for cname in candidate.list_collection_names():
                coll = candidate[cname]
                doc = coll.find_one({'_id': obj_id})
                if doc:
                    fetched_data = doc
                    find_collection = coll
                    source_db = candidate
                    collection_name = cname
                    break
            if fetched_data:
                break

        if not fetched_data:
            return JsonResponse({'status': 'error', 'message': 'Data not found'}, status=404)

        # Delete the record from the main DB
        result = find_collection.delete_one({'_id': obj_id})
        if result.deleted_count == 0:
            return JsonResponse({'status': 'error', 'message': 'Delete failed'}, status=500)

        # ---- Update patient DB ----
        patient_id = fetched_data.get("PatientID")
        datalength = fetched_data.get("datalength", len(fetched_data.get("Data", {}).get("II", [])))
        freq = fetched_data.get("Frequency", 200)

        update_patient_on_delete(patient_id, collection_name, datalength, freq, source_db)

        return JsonResponse({'status': 'success', 'message': 'Data deleted successfully'})

    except Exception as e:
        return JsonResponse({'status': 'error', 'message': str(e)}, status=500)
    
@csrf_exempt
def process_and_return_ecg(request):
    """
    Receives ECG data (x, y format), applies low-pass filtering and baseline correction,
    scales the signal between 0 and 4, and returns the processed ECG data as JSON.
    """
    try:
        # Ensure request is POST and contains JSON data
        if request.method != "POST":
            return JsonResponse({'error': 'Invalid request method'}, status=400)

        data = json.loads(request.body)  # Parse JSON request
        # Extract x and y values
        x_values = data.get("x", [])
        raw_ecg_data = data.get("y", [])  # Extract y values (ECG data)
        
        # Validate input
        if not raw_ecg_data or not x_values or len(x_values) != len(raw_ecg_data):
            return JsonResponse({'error': 'Invalid or missing ECG data'}, status=400)

        # Convert ECG data to NumPy array
        ecg_signal = np.array(raw_ecg_data, dtype=float)
        # Apply low-pass filter (cutoff frequency: 40 Hz)
        fs = 500  # Sampling frequency (adjust if needed)
        cutoff = 40
        b, a = signal.butter(3, cutoff / (fs / 2), btype='lowpass', analog=False)
        low_passed = signal.filtfilt(b, a, ecg_signal)

        # Baseline correction using median filter
        baseline = signal.medfilt(low_passed, kernel_size=131)
        corrected_ecg = low_passed - baseline

        # Scale the signal to range 0 to 4
        min_val = np.min(corrected_ecg)
        max_val = np.max(corrected_ecg)
        if max_val - min_val == 0:
            scaled_ecg = np.zeros_like(corrected_ecg)  # If constant signal, set to 0
        else:
            scaled_ecg = 4 * (corrected_ecg - min_val) / (max_val - min_val)
        
        # Return processed ECG data with original x-values
        return JsonResponse({'x': x_values, 'y': scaled_ecg.tolist()})

    except Exception as e:
        return JsonResponse({'error': str(e)}, status=400)
    
def get_pqrst_data(request):
    if request.method != 'POST':
        return JsonResponse({"status": "error", "message": "Invalid request method"}, status=405)

    try:
        data = json.loads(request.body)
        object_id = data.get("object_id")
        arrhythmia_raw = data.get("arrhythmia", "").strip()
        lead_config = data.get("lead_config")  # "2_lead", "7_lead", "12_lead"

        if not object_id or not arrhythmia_raw or not lead_config:
            return JsonResponse({
                "status": "error",
                "message": "Missing parameters: object_id, arrhythmia, or lead_config."
            }, status=400)

        if lead_config not in ["2_lead", "7_lead", "12_lead"]:
            return JsonResponse({
                "status": "error",
                "message": "Invalid lead_config. Must be '2_lead', '7_lead', or '12_lead'."
            }, status=400)

        arrhythmia_list = [a.strip() for a in arrhythmia_raw.split(",") if a.strip()]
        record = None
        found_collection = None

        # Search across all DB candidates
        db_candidates = [client["ecgarrhythmias"], Morphology_data, Analysis_data]

        for arr in arrhythmia_list:
            for candidate in db_candidates:
                for col_name in candidate.list_collection_names():
                    # Normalize: ignore case and treat spaces/underscores as same
                    if col_name.lower().replace("_", " ") == arr.lower().replace("_", " "):
                        collection = candidate[col_name]
                        doc = collection.find_one({"_id": ObjectId(object_id)})
                        if doc and "Data" in doc:
                            record = doc
                            found_collection = f"{candidate.name}.{col_name}"
                            break
                if record:
                    break
            if record:
                break


        if not record or "Data" not in record:
            return JsonResponse({"status": "error", "message": "Invalid or missing ECG data."}, status=404)

        frequency = int(record.get("Frequency", 200))

        # Normalize Data keys
        standard_leads = ['I', 'II', 'III', 'aVR', 'aVL', 'aVF',
                          'V1', 'V2', 'V3', 'V4', 'V5', 'V6']
        lead_mapping = {lead.lower(): lead for lead in standard_leads}

        normalized_data = {}
        for key, value in record["Data"].items():
            norm_key = lead_mapping.get(str(key).lower().strip(), key)
            normalized_data[norm_key] = value

        # Expected leads by config
        if lead_config == "2_lead":
            expected_leads = ["II"]
        elif lead_config == "7_lead":
            expected_leads = ["I", "II", "III", "aVR", "aVL", "aVF", "V5"]
        else:  # 12_lead
            expected_leads = ["I", "II", "III", "aVR", "aVL", "aVF",
                              "V1", "V2", "V3", "V4", "V5", "V6"]

        available_leads = [lead for lead in expected_leads if lead in normalized_data]
        if not available_leads:
            return JsonResponse({
                "status": "error",
                "message": f"No valid leads found. Expected: {expected_leads}"
            }, status=404)

        # Create DataFrame
        lead_data = {lead: normalized_data[lead] for lead in available_leads}
        df = pd.DataFrame(lead_data)

        # Run detection models
        r_index = check_r_index(df, lead_config, frequency, r_index_model)
        s_index, q_index = check_qs_index(df, r_index, lead_config)
        t_index, p_index, _, _, _ = check_pt_index(df, lead_config, r_index)

        return JsonResponse({
            "status": "success",
            "r_peaks": [int(i) for i in r_index],
            "q_points": [int(i) for i in q_index],
            "s_points": [int(i) for i in s_index],
            "p_points": [int(i) for i in p_index],
            "t_points": [int(i) for i in t_index],
        })

    except Exception as e:
        traceback.print_exc()
        return JsonResponse({"status": "error", "message": str(e)}, status=500)

@csrf_exempt
def selecteddownload(request):
    try:
        data = json.loads(request.body)
        if not data:
            return JsonResponse({'error': 'No data received'}, status=400)

        # Initialize buffer for CSV
        buffer = io.StringIO()

        # Handle 2-lead ECG (single lead)
        if 'x' in data and 'y' in data:
            df = pd.DataFrame({
                'TimeIndex': data['x'],
                'II': data['y']
            })
            df.to_csv(buffer, index=False, encoding='utf-8')

        # Handle 7/12-lead ECG (multiple leads)
        elif 'leadDict' in data:
            lead_dict = data['leadDict']
            if not lead_dict:
                return JsonResponse({'error': 'No lead data provided'}, status=400)

            # Create DataFrame with all leads
            lead_names = list(lead_dict.keys())
            first_lead = lead_names[0]
            df_data = {
                'TimeIndex': lead_dict[first_lead]['x']
            }
            for lead in lead_names:
                if len(lead_dict[lead]['x']) == len(lead_dict[first_lead]['x']):
                    df_data[lead] = lead_dict[lead]['y']
                else:
                    return JsonResponse({'error': f'Inconsistent data length for lead {lead}'}, status=400)

            df = pd.DataFrame(df_data)
            df.to_csv(buffer, index=False, encoding='utf-8')

        else:
            return JsonResponse({'error': 'Invalid data format'}, status=400)

        buffer.seek(0)
        response = HttpResponse(buffer.getvalue(), content_type='text/csv')
        response['Content-Disposition'] = 'attachment; filename="selected_ecg_data.csv"'
        return response

    except Exception as e:
        return JsonResponse({'error': str(e)}, status=500)  

@csrf_exempt
def insert_db_Data(request):
    if request.method != "POST":
        return JsonResponse({"status": "error", "message": "Only POST requests allowed"})

    patient_id = request.POST.get("patientId")
    source = request.POST.get("source")  # expected integer (e.g., 2 for live)

    if not patient_id or not source:
        return JsonResponse({"status": "error", "message": "Missing patientId or source"})
    try:
        # Call your data retrieval + insertion logic
        get_the_data(
            patientID=patient_id,
            source=int(source),
            datetime_format=24
        )

        return JsonResponse({
            "status": "success",
            "message": f"All data for {patient_id} inserted."
        })

    except Exception as e:
        return JsonResponse({"status": "error", "message": str(e)})

@csrf_exempt
def upload_plot(request):
    """
    Handle ECG file uploads for sharing: plot PNG, raw data CSV, PQRST CSV.
    """
    if request.method != "POST":
        return JsonResponse({"status": "error", "message": "Invalid request method"}, status=405)

    try:
        data = json.loads(request.body)
        file_type = data.get("type")           # "plot_png", "raw_data", "pqrst_csv"
        content = data.get("content")          # Base64 for images, CSV text for CSV
        patient_id = data.get("patientId", "unknown")

        if not content or not file_type:
            return JsonResponse({"status": "error", "message": "Missing content or type"}, status=400)

        # Determine directory & file extension
        if file_type == "plot_png":
            ext = "png"
            subdir = "ecg_plots"
        elif file_type in ["raw_data", "pqrst_csv",'selected_data']:
            ext = "csv"
            subdir = "ecg_csv"
        else:
            return JsonResponse({"status": "error", "message": f"Unsupported file type: {file_type}"}, status=400)

        # Create folder if not exists
        dir_path = os.path.join(settings.MEDIA_ROOT, subdir)
        os.makedirs(dir_path, exist_ok=True)

        file_name = f"{file_type}_{patient_id}.{ext}"
        file_path = os.path.join(dir_path, file_name)

        # Save file
        if ext == "png":
            # Expecting Base64 string for image
            format, imgstr = content.split(';base64,')
            with open(file_path, "wb") as f:
                f.write(base64.b64decode(imgstr))
        else:
            # CSV text
            with open(file_path, "w", encoding="utf-8") as f:
                f.write(content)

        # Build public URL
        file_url = request.build_absolute_uri(os.path.join(settings.MEDIA_URL, subdir, file_name))
        return JsonResponse({"status": "success", "url": file_url})

    except Exception as e:
        return JsonResponse({"status": "error", "message": str(e)}, status=500)

@csrf_exempt
@require_POST
def get_multiple_segments(request):
    try:
        data = json.loads(request.body)  
        lead = int(data.get("lead"))
        frequency = int(data.get("frequency"))
        arrhythmia_data = data.get("arrhythmiaData", [])


        username = "anonymous"
        if 'user_session' in request.session and 'username' in request.session['user_session']:
            username = request.session['user_session']['username']

        # --- Insert initial log with pending status ---
        log_doc = {
            "user": username,
            "lead": lead,
            "frequency": frequency,
            "arrhythmiaData": arrhythmia_data,
            "raw_request": data,
            "status": "pending",
            "created_at": datetime.now(timezone.utc),
            "updated_at": None,
            "error_message": None
        }
        log_id = logs_collection.insert_one(log_doc).inserted_id

        session_query_list = []
        channels_map = {
            2: ["II"],
            7: ["I", "II", "III", "aVR", "aVL", "aVF", "v5"],
            12: ["I", "II", "III", "AVR", "AVL", "AVF",
                 "V1", "V2", "V3", "V4", "V5", "V6"]
        }
        required_channels = channels_map.get(lead)
        if not required_channels:
            # --- Update log on error ---
            logs_collection.update_one(
                {"_id": log_id},
                {"$set": {"status": "error", "error_message": "Invalid lead", "updated_at": datetime.now(timezone.utc)}}
            )
            return JsonResponse({"status": "error", "message": "Invalid lead", "data": []})

        for group in arrhythmia_data:
            arrhythmia = group.get("arrhythmia")
            duration = int(group.get("duration"))   # seconds
            sample_count = frequency * duration     # total samples needed

            selected_segments = []
            try:
                collection = db[arrhythmia]

                # Always combine smaller docs, ignore docs bigger than needed
                query = {"Lead": lead, "datalength": {"$gt": 0, "$lte": sample_count}}
                projection = {
                    "PatientID": 1,
                    "Arrhythmia": 1,
                    "Lead": 1,
                    "datalength": 1,
                    "Data": 1
                }

                cursor = collection.find(query, projection).limit(200)

                total_collected = 0
                for record in cursor:
                    if total_collected >= sample_count:
                        break

                    available_channels = [ch for ch in required_channels if ch in record["Data"]]
                    if not available_channels:
                        continue

                    need = sample_count - total_collected
                    have = record.get("datalength", 0)
                    take = min(have, need)
                    if take <= 0:
                        continue

                    seg = {
                        "object_id": str(record.get("_id")),
                        "PatientID": record.get("PatientID", "N/A"),
                        "Arrhythmia": record.get("Arrhythmia", arrhythmia),
                        "Lead": "II" if lead == 2 else lead,
                        "LeadNumeric": lead,
                        "Frequency": frequency,
                        "SamplesTaken": take,
                        "collection_name": arrhythmia,
                        "channels_used": available_channels
                    }
                    selected_segments.append(seg)
                    total_collected += take

            except Exception as inner_e:
                print(f"Error while processing {arrhythmia}: {inner_e}")

            session_query_list.append({
                "arrhythmia": arrhythmia,
                "lead": lead,
                "frequency": frequency,
                "duration": duration,
                "segments": selected_segments
            })

        # -------- Save in session --------
        request.session["multi_ecg_query"] = session_query_list
        request.session["segment_mode"] = True
        request.session.modified = True

        # -------- Flatten response --------
        flattened_segments = []
        for group in session_query_list:
            for seg in group.get("segments", []):
                samples_taken = seg.get("SamplesTaken", 0)
                freq = seg.get("Frequency", 1)
                duration_sec = round(samples_taken / freq, 2) if freq > 0 else 0
                flattened_segments.append({
                    "patient_id": seg.get("PatientID"),
                    "arrhythmia": group.get("arrhythmia"),
                    "lead": seg.get("Lead"),
                    "frequency": freq,
                    "samples_taken": samples_taken,
                    "duration": duration_sec,
                    "channels_used": seg.get("channels_used", [])
                })

        # --- Update log with final status ---
        logs_collection.update_one(
            {"_id": log_id},
            {"$set": {
                "status": "complete" if flattened_segments else "error",
                "updated_at": datetime.now(timezone.utc),
                "processed_segments": session_query_list,
                "total_records": len(flattened_segments)
            }}
        )

        return JsonResponse({
            "status": "success" if flattened_segments else "error",
            "message": "No matching records found" if not flattened_segments else "OK",
            "data": flattened_segments,
            "total_pages": 1 if flattened_segments else 0,
            "total_records": len(flattened_segments),
            "arrhythmia": session_query_list[0]['arrhythmia'] if session_query_list else ""
        })

    except Exception as e:
        username = "anonymous"
        if 'user_session' in request.session and 'username' in request.session['user_session']:
            username = request.session['user_session']['username']

        Queues["multiple_segments"].insert_one({
            "user": username,
            "status": "error",
            "error_message": str(e),
            "created_at": datetime.now(timezone.utc),
            "raw_request": request.body.decode("utf-8", errors="ignore")
        })
        return JsonResponse({"status": "error", "message": str(e), "data": []})

def ecg_details(request, arrhythmia):   
    normalized_arrhythmia = arrhythmia.strip().replace("_", " ")

    full_data = []
    
    ecg_data = request.session.get("multi_ecg_query", [])
    is_segment_mode = request.session.get("segment_mode", False)

    if is_segment_mode and ecg_data:
        # --- Segment Mode (from get_multiple_segments) ---
        arrhythmia_data = [
            entry for entry in ecg_data
            if entry.get("arrhythmia", "").lower() == normalized_arrhythmia
        ]
        for entry in arrhythmia_data:
            for seg in entry.get("segments", []):
                samples_taken = seg.get("SamplesTaken", 0)
                if not samples_taken:
                    continue
                samples_taken = seg.get("SamplesTaken", 0)
                freq = seg.get("Frequency", 1)
                duration_sec = int(round(samples_taken / freq, 2))
                full_data.append({
                    "object_id": seg.get("object_id", ""),
                    "PatientID": seg.get("PatientID", ""),
                    "Arrhythmia": seg.get("Arrhythmia", ""),
                    "Lead": seg.get("Lead", ""),
                    "LeadNumeric": seg.get("LeadNumeric", ""),
                    "Frequency": seg.get("Frequency", ""),
                    "duration":duration_sec,
                    "collection_name": seg.get("collection_name", normalized_arrhythmia),
                    "samples_taken": samples_taken
                })
        # No pagination in segment mode
        page_data = full_data
        total_pages = 1
        current_page = 1

    else:
        # --- Normal Query Mode ---
        ecg_query = request.session.get("ecg_query", {})
        if not ecg_query:
            return render(request, "oom_ecg_data/ecg_details.html", {
                "page_obj": [], "total_pages": 1, "arrhythmia": arrhythmia, "show_alert": True
            })

        query = {
            "PatientID": ecg_query["patient_id"],
            "Lead": ecg_query["lead_type"],
            "Frequency": ecg_query["frequency"]
        }

        collection = db[normalized_arrhythmia]
        results = list(collection.find(query).limit(1000))
        for r in results:
            samples_taken = r.get("SamplesTaken", 0)
            if not samples_taken:
                continue
            full_data.append({
                "object_id": str(r["_id"]),
                "PatientID": r.get("PatientID", ""),
                "Arrhythmia": r.get("Arrhythmia", arrhythmia),
                "Lead": "II" if r.get("Lead", "") == 2 else r.get("Lead", ""),
                "Frequency": r.get("Frequency", ""),
                "collection_name": normalized_arrhythmia,
                "samples_taken": samples_taken
            })

        # Pagination: only here we apply 10-per-page
        paginator = Paginator(full_data, 10)
        page = int(request.GET.get("page", 1))
        try:
            page_data = paginator.page(page)
        except:
            page_data = paginator.page(1)

        total_pages = paginator.num_pages
        current_page = page

    card_name = full_data[0]['collection_name'] if full_data else normalized_arrhythmia

    # AJAX: Return JSON
    if request.headers.get("X-Requested-With") == "XMLHttpRequest":
        response_data = {
            "status": "success",
            "data": list(page_data) if not is_segment_mode else page_data,  # Handle both cases
            "total_records": len(full_data),
            "total_pages": total_pages,
            "current_page": current_page
        }
        return JsonResponse(response_data)

    # Normal HTML render
    return render(request, "oom_ecg_data/ecg_details.html", {
        "page_obj": page_data,
        "total_pages": total_pages,
        "current_page": current_page,
        "arrhythmia": arrhythmia,
        "card_name": card_name,
    })