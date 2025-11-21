# views.py
from django.shortcuts import render
from django.http import JsonResponse,HttpResponse
from pymongo import MongoClient
from django.views.decorators.csrf import csrf_exempt
from datetime import datetime, timedelta
from bson.objectid import ObjectId
import threading
from django.shortcuts import render
from django.http import Http404
import pymongo
from bson import ObjectId
from django.utils.safestring import mark_safe
import json
import io
import pandas as pd
from django.conf import settings
import base64
import os
from oom_ecg_data.PQRST_detection_model import check_r_index, check_qs_index, check_pt_index, r_index_model, pt_index_model
import traceback
# ======================== THREAD LOCK ========================
refresh_lock = threading.Lock()

# ======================== DB CONNECTIONS ========================
client = MongoClient("mongodb://192.168.1.65:27017/")
patient_db = client['Patients']
ecg_db = client['ecgarrhythmias']

# ECG DB collections (arrhythmia groups)
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
# ======================== DASHBOARD VIEW ========================
def index(request):
    pie_data, totals = get_patients_data()
    initial_pie_data = get_pie_chart_data()

    return render(request, "report/report_index.html", {
        "initial_pie_data": initial_pie_data,
        "totals": totals
    })
def get_pie_chart_data():
    """Compute pie chart (per group time in minutes)."""
    collection_data_count = {}
    for collection_name in collections:  # Abnormal, Normal, Others
        coll = patient_db[collection_name]
        cursor = coll.find({}, {"total_time": 1})

        total_time = 0.0
        for doc in cursor:
            total_time += doc.get("total_time", 0)

        if total_time > 0:
            collection_data_count[collection_name] = round(total_time, 2)

    return collection_data_count

# ======================== PATIENTS DB AGGREGATION ========================
def get_patients_data():
    total_patients = set()
    total_records = 0
    total_time = 0.0
    pie_data = {}

    for name in collections:
        coll = patient_db[name]
        cursor = coll.find({}, {"PatientID": 1, "total_records": 1, "total_time": 1})

        group_patients = set()
        group_records = 0
        group_time = 0.0

        for doc in cursor:
            patient_id = doc.get("PatientID")
            if patient_id is not None:
                pid = str(patient_id).strip()
                total_patients.add(pid)
                group_patients.add(pid)
            group_records += int(doc.get("total_records", 0))
            group_time += float(doc.get("total_time", 0))

        total_records += group_records
        total_time += group_time

        if group_records > 0 or group_time > 0:
            pie_data[name] = {
                "patients": len(group_patients),
                "records": group_records,
                "time": round(group_time, 2),
            }

    totals = {
        "total_patients": len(total_patients),
        "total_records": total_records,
        "total_time": round(total_time, 2),
    }

    return pie_data, totals

def patients_totals_api(request):
    """Return total patients, records, and time (live, no cache)."""
    _, totals = get_patients_data()
    return JsonResponse(totals)


def refresh_patients_cache():
    with refresh_lock:
        _, totals = get_patients_data()   # ignore pie_data here
        patient_db["stats"].update_one(
            {"_id": "patients_totals"},
            {"$set": {**totals, "updated_at": datetime.utcnow()}},
            upsert=True
        )

def pie_chart_api(request):
    data = get_pie_chart_data()
    return JsonResponse(data)
   

def refresh_pie_cache():
    with refresh_lock:
        data = get_pie_chart_data()
        patient_db["stats"].update_one(
            {"_id": "pie_chart"},
            {"$set": {"data": data, "updated_at": datetime.utcnow()}},
            upsert=True
        )

# ======================== RECENT ECG RECORDS (ecgarrhythmias DB) ========================
def recent_records_api(request):
    records = []
    for collection_name in collections:
        coll = ecg_db[collection_name]
        docs = list(
            coll.find({}, {"_id": 1, "PatientID": 1, "Data": 1, "Arrhythmia": 1})
               .sort("_id", -1).limit(1)
        )
        for doc in docs:
            records.append({
                "_id": str(doc["_id"]),
                "PatientID": doc.get("PatientID", "Unknown"),
                "Lead": "II" if isinstance(doc.get("Data"), dict) and "II" in doc["Data"] else "N/A",
                "Arrhythmia": doc.get("Arrhythmia", "Unknown"),
                "Duration": (
                    f"{round(len(doc.get('Data', {}).get('II', [])) / 200, 2)} sec"
                    if isinstance(doc.get("Data"), dict) else "0 sec"
                ),
                "collection": collection_name,  # needed later
            })

    records = sorted(records, key=lambda x: ObjectId(x["_id"]).generation_time, reverse=True)[:5]
    return JsonResponse({"records": records})

def waveform_by_id(request, collection_name, record_id):
    try:
        record = ecg_db[collection_name].find_one({"_id": ObjectId(record_id)})
    except Exception:
        raise Http404("Invalid record ID")

    if not record:
        raise Http404("ECG record not found")

    # Extract ECG signal from Data.II (defaulting to empty array)
    data = []
    if isinstance(record.get("Data"), dict) and "II" in record["Data"]:
        data = record["Data"]["II"]

    # Metadata
    patient_id = record.get("PatientID", "Unknown")
    arrhythmia = record.get("Arrhythmia", "Unknown")
    lead = "II" if "II" in record.get("Data", {}) else "N/A"
    frequency = 200 if lead in ["2", "II"] else 250 if lead in ["7", "12"] else 200
    duration = round(len(data) / frequency, 2) if data else 0

    return render(request, "report/waveforms.html", {
        "patient_id": patient_id,
        "lead": lead,
        "arrhythmia": arrhythmia,
        "duration": duration,
        "frequency": frequency,
        "collection": collection_name,
        "record_id": str(record["_id"]),
        # Send as safe JSON for Plotly
        "ecg_signal": mark_safe(json.dumps(data)),
    })
    
def update_patient_on_edit(patient_id, old_arrhythmia, new_arrhythmia, datalength, frequency):
    time_minutes = (datalength / frequency) / 60

    # Decrement from old arrhythmia
    old_col = patient_db[old_arrhythmia]
    old_col.update_one(
        {"PatientID": patient_id},
        {"$inc": {"total_records": -1, "total_time": -time_minutes}}
    )

    # If total_records becomes 0 ? remove patient entry
    doc = old_col.find_one({"PatientID": patient_id})
    if doc and doc.get("total_records", 0) <= 0:
        old_col.delete_one({"PatientID": patient_id})

    # Increment in new arrhythmia
    new_col = patient_db[new_arrhythmia]
    new_col.update_one(
        {"PatientID": patient_id},
        {
            "$inc": {"total_records": 1, "total_time": time_minutes},
            "$setOnInsert": {"PatientID": patient_id}
        },
        upsert=True
    )

def get_parent_collection(subtype_name):
    for parent, subtypes in arrhythmias_dict.items():
        if subtype_name in subtypes or subtype_name == parent:
            return parent
    return None

def edit_data(request):
    db = client["ecgarrhythmias"]

    if request.method != "POST":
        return JsonResponse({'status': 'error', 'message': 'Invalid request method'}, status=405)

    try:
        data = json.loads(request.body)
    except json.JSONDecodeError:
        return JsonResponse({'status': 'error', 'message': 'Invalid JSON'}, status=400)
   
    
    patient_id = data.get('PatientID')
    object_id = data.get('object_id')
    old_collection = get_parent_collection(data.get('old_collection'))
    new_collection = get_parent_collection(data.get('new_collection'))
    lead = data.get('lead')
    if not all([object_id, old_collection, new_collection, lead]):
        return JsonResponse({'status': 'error', 'message': 'Missing required fields'}, status=400)

    try:
        obj_id = ObjectId(object_id)
        lead = int(lead)
    except Exception as e:
        return JsonResponse({'status': 'error', 'message': f'Invalid parameters: {e}'}, status=400)

    # Actual Mongo collections (from dropdown value, not just display string)
    find_collection = db[old_collection]
    insert_collection = db[new_collection]
    fetched_data = find_collection.find_one({'_id': obj_id})
    if not fetched_data:
        return JsonResponse({'status': 'error', 'message': 'Data not found in old collection'}, status=404)

    if not patient_id:
        patient_id = fetched_data.get('PatientID')

    # Check duplicates: by PatientID + Lead + Data length
    duplicate = insert_collection.find_one({
        'PatientID': patient_id,
        'Lead': lead,
        'Data': fetched_data.get('Data')
    })
    if duplicate:
        return JsonResponse({'status': 'error', 'message': 'Duplicate ECG data already exists.'}, status=409)

    # Build new document
    new_doc = {
        'PatientID': patient_id,
        'Arrhythmia': new_collection,   # update arrhythmia label
        'Lead': lead,
        'Frequency': fetched_data.get('Frequency', 200),
        'Data': fetched_data.get('Data'),
        'datalength': fetched_data.get('datalength', len(fetched_data.get("Data", {}).get("II", [])))
    }
    # Insert into new collection
    insert_result = insert_collection.insert_one(new_doc)

    # Delete only after successful insert
    find_collection.delete_one({'_id': obj_id})

    # Update patient stats
    datalength = new_doc['datalength']
    freq = new_doc['Frequency']
    update_patient_on_edit(patient_id, old_collection, new_collection, datalength, freq)

    # Update session
    request.session["ecg_query"] = {
        "patient_id": patient_id,
        "lead_type": lead,
        "frequency": freq,
        "arrhythmia": new_collection
    }
    request.session.modified = True

    return JsonResponse({
        'status': 'success',
        'message': 'Data edited and saved successfully',
        'new_object_id': str(insert_result.inserted_id)
    })

@csrf_exempt
def upload_plot(request):
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
        elif file_type in ['raw_data','selected_data']:
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
def get_pqrst_data(request):

    if request.method != 'POST':
        return JsonResponse({"status": "error", "message": "Invalid request method"}, status=405)

    try:
        data = json.loads(request.body)

        object_id = data.get("object_id")
        arrhythmia_raw = data.get("arrhythmia", "").strip()
        lead_config = data.get("lead_config")

        if not object_id or not arrhythmia_raw or not lead_config:
            return JsonResponse({"status": "error",
                                 "message": "Missing parameters: object_id, arrhythmia, or lead_config."}, status=400)

        if lead_config not in ["2_lead", "7_lead", "12_lead"]:
            return JsonResponse({"status": "error",
                                 "message": "Invalid lead_config. Must be '2_lead', '7_lead', or '12_lead'."}, status=400)

        arrhythmia_list = [a.strip() for a in arrhythmia_raw.split(",") if a.strip()]

        record = None

        db = client["ecgarrhythmias"]

        for arr in arrhythmia_list:
            if arr in db.list_collection_names():
                collection = db[arr]
                doc = collection.find_one({"_id": ObjectId(object_id)})
                if doc and "Data" in doc:
                    record = doc
                    break
            else:
                print(f"Collection {arr} not found in ecg_db")

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
        else:
            expected_leads = ["I", "II", "III", "aVR", "aVL", "aVF",
                              "V1", "V2", "V3", "V4", "V5", "V6"]

        available_leads = [lead for lead in expected_leads if lead in normalized_data]

        if not available_leads:
            return JsonResponse({"status": "error",
                                 "message": f"No valid leads found. Expected: {expected_leads}"}, status=404)

        # Create DataFrame
        lead_data = {lead: normalized_data[lead] for lead in available_leads}
        df = pd.DataFrame(lead_data)

        # Run detection models
        r_index_dic = check_r_index(df, lead_config, frequency, r_index_model)
        s_index, q_index = check_qs_index(df, r_index_dic, lead_config)
        t_index, p_index, _, _, _ = check_pt_index(df, r_index_dic, lead_config)

        return JsonResponse({
            "status": "success",
            "R": {lead: [int(i) for i in r_index_dic[lead]] for lead in r_index_dic},
            "Q": {lead: [int(i) for i in q_index[lead]] for lead in q_index},
            "S": {lead: [int(i) for i in s_index[lead]] for lead in s_index},
            "P": {lead: [int(i) for i in p_index[lead]] for lead in p_index},
            "T": {lead: [int(i) for i in t_index[lead]] for lead in t_index},
        })

    except Exception as e:
        traceback.print_exc()
        return JsonResponse({"status": "error", "message": str(e)}, status=500)

refresh_lock = threading.Lock()

def total_data(request):
    _, totals = get_patients_data()

    if request.headers.get("X-Requested-With") == "XMLHttpRequest":
        return JsonResponse(totals)
    return render(request, "authuser/login.html", totals)

def refresh_totals_cache():
    with refresh_lock:
        total_patients = set()
        total_records = 0
        total_time = 0.0

        for name in collections:
            coll = patient_db[name]
            cursor = coll.find({}, {"PatientID": 1, "total_records": 1, "total_time": 1})

            for doc in cursor:
                total_patients.add(doc["PatientID"])
                total_records += doc.get("total_records", 0)
                total_time += doc.get("total_time", 0)

        stats_doc = {
            "_id": "patients_totals",
            "total_patients": len(total_patients),
            "total_records": total_records,
            "total_time": round(total_time, 2),
            "updated_at": datetime.utcnow()
        }
        patient_db["stats"].update_one({"_id": "patients_totals"}, {"$set": stats_doc}, upsert=True)