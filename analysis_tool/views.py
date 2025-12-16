from django.shortcuts import render
from django.http import JsonResponse,FileResponse,Http404
import pandas as pd
import matplotlib.pyplot as plt
from pymongo import MongoClient
import numpy as np
from .models import Image  # Ensure your Image model is imported
from django.views.decorators.csrf import csrf_exempt 
from pdf2image import convert_from_path
from PyPDF2 import PdfMerger
from django.core.files.base import ContentFile
from scipy import signal
from sklearn.preprocessing import MinMaxScaler
from urllib.parse import unquote
from django.utils import timezone
import os, shutil, zipfile, json, logging
from PyPDF2 import PdfMerger
from django.conf import settings
from django.http import FileResponse, HttpResponse
from gridfs import GridFS
from bson.objectid import ObjectId

# Ai Models
from .Scripts import afib_alf_model_check
from .Scripts import block_model_check
from .Scripts import mi_model_check
from .Scripts import pac_model_check
from .Scripts import pac_junc_model_check
from .Scripts import pvc_model_check
from .Scripts import vifib_vfl_model_check 
from .Scripts import ALL_Arrhythmia
from .oea.OEA_arrhy_mi_detection import predict_grid_type
from .oea.OEA_arrhy_mi_detection import check_noise  
from .oea import OEA_arrhy_mi_detection

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Connect to MongoDB
mongo_uri = os.getenv("MONGO_HOST")

# Create client
mongo_client = MongoClient(mongo_uri)
media_db = mongo_client["Download_files"]
admin_db = mongo_client["admin"]

download_fs = GridFS(media_db, collection="downloads")
download_logs = admin_db["download_logs"]

#database
db=mongo_client['Analysis_data']
analysis_csv_patient = mongo_client['Analysis_Patients']

def index(request):
    images = Image.objects.all()  # Fetch all uploaded images
    return render(request, 'analysis_tool/analysis_index.html', {'images': images})

# API view for ECG data (Example)
def api_ecg_data(request):
    data = {"message": "ECG data API response"}
    return JsonResponse(data)

def analysis_img_data_insert_db(patient_id, arrhythmia, datalength,frequency=240):
    patient_col = analysis_csv_patient[arrhythmia]
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

def image_data_insert_db(upload_file_path, arrhythmia_for, title_lines, metrics):
    # Always use 12-Lead collection
    data_collection = db[arrhythmia_for]

    Csvname = os.path.basename(upload_file_path)
    patient_id = Csvname.split('.')[0]

    # Load CSV
    all_lead_data = pd.read_csv(upload_file_path).fillna(0)

    # Map for 12-lead
    col_map = {
        0: 'I', 1: 'II', 2: 'III', 3: 'aVR', 4: 'aVL', 5: 'aVF',
        6: 'V1', 7: 'V2', 8: 'V3', 9: 'V4', 10: 'V5', 11: 'V6'
    }

    # If CSV already has headers
    if any(str(_).isalpha() for _ in all_lead_data.columns):
        available_leads = [lead for lead in col_map.values() if lead in all_lead_data.columns]
        all_lead_data = all_lead_data[available_leads].fillna(0)
    else:
        # If CSV has numeric columns
        available_indices = [i for i in col_map.keys() if i in all_lead_data.columns]
        selected_col_map = {i: col_map[i] for i in available_indices}
        all_lead_data = all_lead_data[available_indices].rename(columns=selected_col_map)

    datalength = len(all_lead_data)

    # Always 12-lead  frequency 240
    ecg_data_dic = {
        'PatientID': patient_id,
        'Arrhythmia': arrhythmia_for,
        'Lead': 12,
        'Frequency': 240,
        'server': "local",
        'title_lines': title_lines,   # stored as array
        **metrics,                   #  flattened HR, RRInterval, etc.
        'datalength': datalength,
        'created_At': timezone.localtime(timezone.now()).strftime("%Y-%m-%d %H:%M:%S"),
        'Data': {}
    }

    # Insert image analysis helper (if you need it)
    analysis_img_data_insert_db(patient_id, arrhythmia_for, datalength)

    # Add ECG signals
    for lead in all_lead_data.columns:
        ecg_data_dic['Data'][lead] = all_lead_data[lead].tolist()

    data_collection.insert_one(ecg_data_dic)
    return "Data inserted in db."

def analysis_csv_data_insert_db(patient_id, arrhythmia, datalength,lead):
    frequency = 200 if lead in [2,7] else 240
    patient_col = analysis_csv_patient[arrhythmia]
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

def uploads_file(request):
    if request.method == 'POST' and request.FILES.get('file'):
        uploaded_file = request.FILES['file']
        file_extension = uploaded_file.name.split('.')[-1].lower()

        # Save original full CSV to media/uploads
        media_upload_folder = os.path.join(settings.MEDIA_ROOT, 'analysis_tool', 'uploads')
        os.makedirs(media_upload_folder, exist_ok=True)

        file_path = os.path.join(media_upload_folder, uploaded_file.name)
        with open(file_path, 'wb+') as destination:
            for chunk in uploaded_file.chunks():
                destination.write(chunk)

        if file_extension in ['jpg', 'jpeg', 'png']:
            return JsonResponse({'message': 'Image uploaded successfully', 'filename': 'uploads/' + uploaded_file.name})
        if file_extension == 'pdf':
            return JsonResponse({
                'message': 'PDF uploaded successfully',
                'filename': 'uploads/' + uploaded_file.name,
                'file_type': 'pdf'
            })
        elif file_extension == 'csv':
            try:
                df = pd.read_csv(file_path)
                lead_names = [col for col in df.columns if col != 'Index']
                lead_count = len(lead_names)

                # Standard supported ECG leads
                standard_leads = ['I', 'II', 'III', 'aVR', 'aVL', 'aVF',
                                'V1', 'V2', 'V3', 'V4', 'V5', 'V6']

                # Mapping lowercase ? canonical lead names
                lead_mapping = {lead.lower(): lead for lead in standard_leads}

                # Add special mapping: ECG ? II
                lead_mapping['ecg'] = 'II'

                normalized_columns = {}

                for col in lead_names:
                    col_clean = str(col).lower().strip()

                    # If column is "ecg" ? convert to "II"
                    if col_clean == "ecg":
                        normalized_columns[col] = "II"
                        continue

                    # If column maps to any standard lead
                    if col_clean in lead_mapping:
                        normalized_columns[col] = lead_mapping[col_clean]
                    else:
                        # Keep original name if no match
                        normalized_columns[col] = col

                # Rename DataFrame columns
                df.rename(columns=normalized_columns, inplace=True)


                # Handle supported ECG formats
                if lead_count in [2, 7, 12]:
                    # Extract data for all available standard leads (up to 2000 points)
                    extracted = {
                        lead: df[lead].dropna().iloc[:2000].tolist()
                        for lead in standard_leads if lead in df.columns
                    }
                    if not extracted:
                        return JsonResponse({'error': f'No valid leads found for {lead_count}-lead ECG'}, status=404)

                    # Find the minimum length of extracted leads to ensure uniform x-values
                    data_lengths = [len(data) for data in extracted.values()]
                    if not data_lengths:
                        return JsonResponse({'error': 'No valid data found for any leads'}, status=404)
                    data_length = min(data_lengths)
                    x_values = list(range(data_length))

                    return JsonResponse({
                        'message': 'CSV uploaded',
                        'show_cards': True,
                        'filename': 'uploads/' + uploaded_file.name,
                        'lead_count': lead_count,
                        'ecgData': extracted,
                        'x': x_values
                    })

                return JsonResponse({
                    'message': 'CSV uploaded',
                    'show_cards': True,
                    'filename': 'uploads/' + uploaded_file.name,
                    'lead_count': lead_count
                })

            except Exception as e:
                return JsonResponse({'error': f'Error reading CSV: {str(e)}'}, status=500)

        return JsonResponse({'error': 'Invalid file type'}, status=400)

    return render(request, 'analysis_tool/analysis_index.html')

# run oea analysis
def process_img(request, img):
    return check_oea_analysis(img)

# oea analysis
def check_oea_analysis(img):
    image_path = os.path.join(settings.MEDIA_ROOT, 'analysis_tool','uploads', img)
    file_name = os.path.basename(image_path)
    img_new = file_name.split('.')[0]
    # 1. Check if image exists
    if not os.path.exists(image_path):
        return JsonResponse({'error': f'File {img} not found in uploads.'}, status=404)

    # 2. Predict grid type to detect "No ECG"
    _, grid_type = predict_grid_type(image_path)
    if grid_type == "No ECG":
        return JsonResponse({'error': 'No ECG detected in the uploaded image.'}, status=400)

    # 3. Run arrhythmia detection
    oea_result = OEA_arrhy_mi_detection.signal_extraction_and_arrhy_detection(image_path)

    # 4. Convert NumPy to JSON-safe formats
    def convert_numpy(obj):
        if isinstance(obj, (np.ndarray, list)):
            return list(obj)
        elif isinstance(obj, (np.integer, np.int32, np.int64)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj, dict):
            return {k: convert_numpy(v) for k, v in obj.items()}
        else:
            return obj

    oea_result = convert_numpy(oea_result)

    # 5. Artifact detection from OEA result
    detections = oea_result.get("detections", [])
    for d in detections:
        if d.get("detect", "").strip().upper() == "ARTIFACTS":
            return JsonResponse({'error': 'Artifacts detected in the ECG.'}, status=400)

    # 6. Prepare result ZIP
    processed_image_path = os.path.join(settings.MEDIA_ROOT, 'analysis_tool','uploads', img_new +".jpg")
    csv_filename = f"{os.path.splitext(img)[0]}.csv"
    csv_path = os.path.join(settings.MEDIA_ROOT, 'analysis_tool','uploads', csv_filename)

    files_to_zip = [] 
    summary = oea_result.get("summary_text", {})
    title_lines = summary.get("title_lines", [])
    bottom_label = summary.get("bottom_label", "")

    # extract arrhythmia from title_lines
    arrhythmia_for = "Unknown"
    for line in title_lines:
        if "Arrhythmia:" in line:
            arrhythmia_for = line.replace("Arrhythmia:", "").strip().strip(",")
            break

    # parse bottom_label into dictionary
    def parse_bottom_label(label):
        metrics = {}
        parts = label.replace('"', '').split(',')
        for part in parts:
            if ':' in part:
                k, v = part.split(':', 1)
                k = k.strip()
                v = v.strip()
                try:
                    v = float(v) if '.' in v else int(v)
                except ValueError:
                    pass
                metrics[k] = v
        return metrics

    metrics = parse_bottom_label(bottom_label)

    if os.path.exists(processed_image_path):
        files_to_zip.append(processed_image_path)

    if os.path.exists(csv_path):
        image_data_insert_db(
            upload_file_path=csv_path,
            arrhythmia_for=arrhythmia_for,
            title_lines=title_lines,
            metrics=metrics
        )

def lowpass(signal_data):
    b, a = signal.butter(3, 0.4, btype='lowpass', analog=False)
    return signal.filtfilt(b, a, signal_data)

def baseline_construction_200(ecg_signal, kernel_Size=101):
    s_corrected = signal.detrend(ecg_signal)
    return s_corrected - signal.medfilt(s_corrected, kernel_Size)
    
@csrf_exempt
def plot_csv_view(request):
    """
    Receives a CSV file containing ECG data with multiple leads (up to 7 or 12),
    applies low-pass filtering, baseline correction, and scaling to each lead,
    and returns the processed data as JSON.
    """
    if request.method == "POST":
        try:
            uploaded_file = request.FILES.get("ecg_file")
            if not uploaded_file:
                return JsonResponse({"error": "No file uploaded"}, status=400)

            # Validate file type
            if not uploaded_file.name.endswith('.csv'):
                return JsonResponse({"error": "Only CSV files are supported"}, status=400)

            # Save to /media/uploads/
            upload_folder = os.path.join(settings.MEDIA_ROOT, 'analysis_tool', 'uploads')
            os.makedirs(upload_folder, exist_ok=True)

            file_path = os.path.join(upload_folder, uploaded_file.name)
            with open(file_path, 'wb+') as f:
                for chunk in uploaded_file.chunks():
                    f.write(chunk)

            # Read CSV and normalize column names
            df = pd.read_csv(file_path)
            df.columns = [col.strip().lower() for col in df.columns]

            # Define standard ECG lead names (up to 12 leads)
            valid_leads = ['i', 'ii', 'iii', 'avr', 'avl', 'avf', 'v1', 'v2', 'v3', 'v4', 'v5', 'v6', 'v7', 'v8', 'v9', 'v10', 'v11', 'v12', 'ecg']
            found_leads = [col for col in df.columns if col in valid_leads]

            if not found_leads:
                return JsonResponse({"error": "No valid ECG lead columns found (e.g., 'i', 'ii', 'v1', ..., 'v12', 'ecg')"}, status=400)

            # Process each lead
            result = {}
            for lead in found_leads:
                # Extract and process ECG data
                ecg_signal = np.array(df[lead], dtype=float)

                # Apply low-pass filter
                low_passed = lowpass(ecg_signal)

                # Apply baseline correction
                corrected_ecg = baseline_construction_200(low_passed)

                # Scale the signal to range 0 to 4 (consistent with original)
                scaler = MinMaxScaler(feature_range=(0, 4))
                scaled_ecg = scaler.fit_transform(corrected_ecg.reshape(-1, 1)).flatten()

                # Truncate to 2000 points (consistent with original)
                scaled_ecg = scaled_ecg[:2000]
                x_values = list(range(len(scaled_ecg)))

                result[lead] = {"x": x_values, "y": scaled_ecg.tolist()}

            return JsonResponse({"leads": result})

        except Exception as e:
            return JsonResponse({"error": str(e)}, status=500)

    return JsonResponse({"error": "Invalid request method"}, status=405)


def data_insert_db(is_lead_for, upload_file_path, arrhythmia_for, filename):
    data_collection = db[arrhythmia_for]
    Csvname = os.path.basename(upload_file_path)
    patient_id= Csvname.split('.')[0]
    all_lead_data = pd.read_csv(upload_file_path).fillna(0)

    # Define expected leads for each type
    expected_leads = {
        "2_Lead": ['II'],
        "7_Lead": ['I', 'II', 'III', 'aVR', 'aVL', 'aVF', 'v5'],
        "12_Lead": ['I', 'II', '(III', 'aVR', 'aVL', 'aVF', 'v1', 'v2', 'v3', 'v4', 'v5', 'v6']
    }

    all_lead_data = pd.read_csv(upload_file_path).fillna(0)
    # Ensure numeric columns are integers
    all_lead_data.columns = [int(c) if str(c).isdigit() else c for c in all_lead_data.columns]
    
    if any(isinstance(_, str) and _.isalpha() for _ in all_lead_data.columns):
        if "ECG" in all_lead_data.columns:
            all_lead_data = all_lead_data.rename(columns={"ECG": "II"})
            available_leads = ["II"]
        else:
            available_leads = [lead for lead in expected_leads[is_lead_for] if lead in all_lead_data.columns]
        all_lead_data = all_lead_data[available_leads].fillna(0)

    else:
        col_map = {
            "2_Lead": {0:'II'},
            "7_Lead": {0:'I',1:'II',2:'III',3:'aVR',4:'aVL',5:'aVF',6:'v5'},
            "12_Lead": {0:'I',1:'II',2:'III',3:'aVR',4:'aVL',5:'aVF',6:'v1',7:'v2',8:'v3',9:'v4',10:'v5',11:'v6'}
        }
        available_indices = [i for i in col_map[is_lead_for].keys() if i in all_lead_data.columns]
        selected_col_map = {i: col_map[is_lead_for][i] for i in available_indices}
        all_lead_data = all_lead_data[available_indices].rename(columns=selected_col_map)

    datalength = len(all_lead_data)

    # Decide frequency properly
    if is_lead_for == '2_Lead':
        lead=2
        frequency = 200
    elif is_lead_for == '7_Lead':
        lead=7
        frequency = 200
    else:  # 12 lead
        lead=12
        frequency = 240

    ecg_data_dic = {
        'PatientID': patient_id,
        'Arrhythmia': arrhythmia_for,
        'Lead': lead,
        'Frequency': frequency,
        "server": "local",
        "Version":filename,
        'datalength': datalength,
        "created_At": timezone.localtime(timezone.now()).strftime("%Y-%m-%d %H:%M:%S"),
        'Data': {}
    }
    analysis_csv_data_insert_db(patient_id,arrhythmia_for,datalength,lead)
    for lead in all_lead_data.columns:
        ecg_data_dic['Data'][lead] = all_lead_data[lead].tolist()

    data_collection.insert_one(ecg_data_dic)
    return "Data inserted in db."

# Process arrhythmia
@csrf_exempt
def run_model_arrhythmia(request,category,filename):
    if request.method != "POST":
        return JsonResponse({"error": "Invalid request"}, status=405)

    csv_name = request.POST.get("csv_name")
    is_lead = request.POST.get("is_lead")
    if not csv_name or not is_lead:
        return JsonResponse({"error": "Missing csv_name or is_lead"}, status=400)

    upload_file_path = os.path.join(
        settings.MEDIA_ROOT, "analysis_tool", "uploads", csv_name
    )

    if not os.path.exists(upload_file_path):
        return JsonResponse({"error": f"{csv_name} not found"}, status=404)
    # Run model
    result = check_arrhythmia_model(
        category,
        upload_file_path,
        is_lead,
        filename,
    )

    return JsonResponse({"result": result})
# Check arrhythmia model
def check_arrhythmia_model(category, upload_file_path, is_lead_for, filename):

    save_result = os.path.join(settings.MEDIA_ROOT, 'analysis_tool', 'analysis_result/')
    msg = 'Not Analysis.......'
    result_dic = None  
    if category == 'afib_afl':
        file_id = afib_alf_model_check.model_check_for_ecg_data(
            upload_file_path,
            is_lead_for
        )
        # AFIB handled via PDF → no result_dic
        return "AFIB & Flutter Result generated successfully"

    elif category == 'block':
        file_id = block_model_check.model_check_for_ecg_data(
            upload_file_path, is_lead_for
        )
        return "Block Result generated successfully"

    elif category == 'mi':
        file_id = mi_model_check.model_check_for_ecg_data(
            upload_file_path, is_lead_for
        )
        return "MI Result generated successfully"

    elif category == 'pvc':
        file_id = pvc_model_check.model_check_for_ecg_data(
            upload_file_path, is_lead_for
        )
        return "PVC Result generated successfully"
    elif category == 'pac':
        file_id = pac_model_check.model_check_for_ecg_data(
            upload_file_path, is_lead_for
        )
        return "PAC Result generated successfully"

    elif category == 'pac_jn':
        file_id = pac_junc_model_check.model_check_for_ecg_data(
            upload_file_path, is_lead_for
        )
        return "JUNCTIONAL Result generated successfully"
    elif category == 'all_arrhythmia':
        file_id = ALL_Arrhythmia.model_check_for_ecg_data(
            upload_file_path, is_lead_for
        )
        return "ALL Result generated successfully"
    elif category == 'vifib_vfl':
        file_id = vifib_vfl_model_check.model_check_for_ecg_data(
            upload_file_path, is_lead_for
        )
        return "VIFIB Result generated successfully"
    # ---------------------------
    # FINAL SAFETY CHECK
    # ---------------------------
    if not result_dic:
        return "Analysis failed (empty result)"

    if result_dic.get('is_error'):
        return "Something went wrong"

    return msg



def check_tmt_analysis(img):

    image_path = os.path.join(settings.MEDIA_ROOT, 'analysis_tool','uploads', img)

    if not os.path.exists(image_path):
        return []

    oea_result = OEA_arrhy_mi_detection.signal_extraction_and_arrhy_detection(image_path)

    def convert_numpy(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, dict):
            return {k: convert_numpy(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_numpy(i) for i in obj]
        else:
            return obj

    oea_result = convert_numpy(oea_result)

    processed_image_path = os.path.join(settings.MEDIA_ROOT, 'analysis_tool','uploads', img)
    csv_filename = f"{os.path.splitext(img)[0]}.csv"
    csv_path = os.path.join(os.path.dirname(__file__), "analysis_result", csv_filename)

    files_to_return = []

    if os.path.exists(processed_image_path):
        files_to_return.append(processed_image_path)

    if os.path.exists(csv_path):
        all_lead_data = pd.read_csv(csv_path).fillna(0)
        ecg_data_dict = {col: all_lead_data[col].tolist() for col in all_lead_data.columns}
#        data_insert_db(is_lead_for="12_Lead", upload_file_path=csv_path, arrhythmia_for="IMAGE")
        files_to_return.append(csv_path)

    additional_files_path = os.path.join(settings.MEDIA_ROOT, "additional_files")
    if os.path.exists(additional_files_path):
        for file_name in os.listdir(additional_files_path):
            file_path = os.path.join(additional_files_path, file_name)
            if os.path.isfile(file_path):
                files_to_return.append(file_path)

    return files_to_return
    
@csrf_exempt
def download_tmt_file(request, filename):
    #filename = unquote(filename)  # decode %20 into spaces
    file_path = os.path.join(settings.MEDIA_ROOT, 'analysis_tool','uploads', filename)
    if os.path.exists(file_path):
        return FileResponse(open(file_path, 'rb'), as_attachment=True)
    else:
        raise Http404("File not found.")

    
@csrf_exempt
def upload_tmt_pdf(request):
    
    if request.method == 'POST' and request.FILES.get('file'):
        pdf_file = request.FILES['file']
        pdf_name_without_ext = os.path.splitext(pdf_file.name)[0]

        # Create folder
        pdf_folder = os.path.join(settings.MEDIA_ROOT, 'analysis_tool', 'uploads')
        os.makedirs(pdf_folder, exist_ok=True)

        # FULL PDF PATH
        pdf_path = os.path.join(pdf_folder, pdf_file.name)

        # SAVE PDF CORRECTLY
        with open(pdf_path, 'wb') as destination:
            for chunk in pdf_file.chunks():
                destination.write(chunk)

        # Convert pages 6�15
        images = convert_from_path(pdf_path, first_page=6, last_page=15)

        result_files = []
        image_filenames = []

        for idx, img in enumerate(images, start=6):
            image_filename = f"{pdf_name_without_ext}_page_{idx}.jpg"
            image_path = os.path.join(pdf_folder, image_filename)

            img.save(image_path, "JPEG")
            image_filenames.append(image_filename)

            # Your processing
            files = check_tmt_analysis(image_filename)
            result_files.extend(files)

        # Create ZIP
        zip_filename = f"{pdf_name_without_ext}_tmt_analysis.zip"
        zip_path = os.path.join(pdf_folder, zip_filename)

        with zipfile.ZipFile(zip_path, 'w') as zipf:
            for f in result_files:
                if os.path.exists(f):
                    zipf.write(f, arcname=os.path.basename(f))

        # Return JSON
        return JsonResponse({
            "zip_file": zip_filename,
            "images": image_filenames   # send ALL preview images
        })

    return JsonResponse({'error': 'Invalid request'}, status=400)


def download_patient_pdf(request):
    raw_patient_id = request.GET.get("patient_id")
    patient_id, _ = os.path.splitext(raw_patient_id)
    print(patient_id)
    grid_file = download_fs.find_one({
        "meta.patient_id": patient_id
    })

    if not grid_file:
        return HttpResponse("PDF not found", status=404)

    # LOG DOWNLOAD HERE (BEST PLACE)
    download_logs.insert_one({
        "patient_id": patient_id,
        "file_id": grid_file._id,
        "file_type": "PDF",
        "downloaded_by": request.user.username if request.user.is_authenticated else "anonymous",
        "downloaded_at": timezone.now()
    })

    response = HttpResponse(
        grid_file.read(),
        content_type="application/pdf"
    )
    response["Content-Disposition"] = (
        f'attachment; filename="{grid_file.filename}"'
    )

    return response