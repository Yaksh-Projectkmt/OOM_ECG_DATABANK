from django.shortcuts import render
from django.http import JsonResponse,FileResponse,Http404,StreamingHttpResponse
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
from threading import Lock
import gc
from io import BytesIO
from pdf2image import convert_from_bytes
import uuid
from bson import ObjectId
from PIL import Image as PILImage
from PIL import ImageDraw
from datetime import datetime
# Ai Models
from Scripts_Models.Scripts import afib_alf_model_check
from Scripts_Models.Scripts import block_model_check
from Scripts_Models.Scripts import mi_model_check
from Scripts_Models.Scripts import pac_model_check
from Scripts_Models.Scripts import pac_junc_model_check
from Scripts_Models.Scripts import pvc_model_check
from Scripts_Models.Scripts import vifib_vfl_model_check 
from Scripts_Models.Scripts import ALL_Arrhythmia
from Scripts_Models.Scripts import OEA_arrhy_mi_detection
from Scripts_Models.Scripts.OEA_arrhy_mi_detection import predict_grid_type
from Scripts_Models.Scripts.OEA_arrhy_mi_detection import check_noise  

from django.http import HttpResponse
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
from datetime import datetime

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Connect to MongoDB
mongo_uri = os.getenv("MONGO_HOST")

# Create client
mongo_client = MongoClient(mongo_uri)
media_db = mongo_client["Download_files"]
Queues = mongo_client["Queue"]

analysis_tasks=Queues['analysis_tasks']

download_fs = GridFS(media_db, collection="downloads")
download_logs = media_db["download_logs"]

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

def analyzing_history(request):
    history = list(
        analysis_tasks.find(
            {"user": request.user.username if request.user.is_authenticated else "anonymous"}
        ).sort("created_at", -1)
    )
    return render(request, 'analysis_tool/analyzing_history.html', {
        "history": history
    })

def uploads_file(request):
    if request.method == 'POST' and request.FILES.get('file'):
        uploaded_file = request.FILES['file']
        file_extension = uploaded_file.name.split('.')[-1].lower()

        task_id = None

        # ONLY IMAGES → temp/<task_id>/
        if file_extension in ['jpg', 'jpeg', 'png']:
            task_id = f"IMG-{timezone.now().strftime('%Y%m%d%H%M%S%f')}"
            base_dir = os.path.join(
                settings.MEDIA_ROOT,
                'analysis_tool',
                'temp',
                task_id
            )
            os.makedirs(base_dir, exist_ok=True)
            file_path = os.path.join(base_dir, uploaded_file.name)

        # CSV + PDF → normal uploads
        else:
            media_upload_folder = os.path.join(
                settings.MEDIA_ROOT,
                'analysis_tool',
                'uploads'
            )
            os.makedirs(media_upload_folder, exist_ok=True)
            file_path = os.path.join(media_upload_folder, uploaded_file.name)

        # Save file
        with open(file_path, 'wb+') as destination:
            for chunk in uploaded_file.chunks():
                destination.write(chunk)

        # ================= IMAGE =================
        if file_extension in ['jpg', 'jpeg', 'png']:
            return JsonResponse({
                'message': 'Image uploaded successfully',
                'filename': uploaded_file.name,
                'task_id': task_id,
                'file_type': 'image'
            })

        # ================= CSV =================
        if file_extension == 'csv':
            try:
                df = pd.read_csv(file_path)
                lead_names = [col for col in df.columns if col != 'Index']
                lead_count = len(lead_names)

                return JsonResponse({
                    'message': 'CSV uploaded successfully',
                    'filename': 'uploads/' + uploaded_file.name,
                    'lead_count': lead_count,
                    'file_type': 'csv'
                })

            except Exception as e:
                return JsonResponse(
                    {'error': f'Error reading CSV: {str(e)}'},
                    status=500
                )

        # ================= PDF =================
        if file_extension == 'pdf':
            return JsonResponse({
                'message': 'PDF uploaded successfully',
                'filename': 'uploads/' + uploaded_file.name,
                'file_type': 'pdf'
            })

        return JsonResponse({'error': 'Invalid file type'}, status=400)

    return render(request, 'analysis_tool/analysis_index.html')
# run oea analysis
def process_img(request, img,task_id):
    if request.method != "POST":
        return JsonResponse({"error": "Invalid request"}, status=405)

    analysis_tasks.insert_one({
        "task_id": task_id,
        "user": request.user.username if request.user.is_authenticated else "anonymous",
        "file_name": img,
        "file_type": "image",
        "lead_type":12,
        "status": "pending",
        "file_id": None,
        "created_at": timezone.now(),
        "completed_at": None,
        "error": None
    })

    try:
        result = check_oea_analysis(request, img, task_id)

        #  ERROR PATH — only JsonResponse has status_code
        if isinstance(result, JsonResponse):
            analysis_tasks.update_one(
                {"task_id": task_id},
                {"$set": {
                    "status": "failed",
                    "error": result.content.decode(),
                    "completed_at": timezone.now()
                }}
            )
            return result

        # SUCCESS PATH — result is a tuple
        oea_result, image_file_id = result
        file_id = str(image_file_id)

        analysis_tasks.update_one(
            {"task_id": task_id},
            {"$set": {
                "status": "success",
                "file_id": file_id,
                "completed_at": timezone.now()
            }}
        )

        return JsonResponse({
            "status": "success",
            "task_id": task_id,
            "file_id": file_id,
            "arrhythmia": oea_result.get("final_arrhythmia", "Unknown")
        })

    except Exception as e:
        analysis_tasks.update_one(
            {"task_id": task_id},
            {"$set": {
                "status": "failed",
                "error": str(e),
                "completed_at": timezone.now()
            }}
        )
        return JsonResponse({"error": str(e)}, status=500)

def check_oea_analysis(request, img, task_id):
    temp_dir = os.path.join(settings.MEDIA_ROOT,'analysis_tool','temp',task_id)

    image_path = os.path.join(temp_dir, img)

    print("TEMP DIR:", temp_dir)
    print("IMAGE PATH:", image_path)

    if not os.path.exists(image_path):
        return JsonResponse(
            {'error': f'File {img} not found in temp folder.'},
            status=404
        )

    _, grid_type = predict_grid_type(image_path)
    if grid_type == "No ECG":
        return JsonResponse(
            {'error': 'No ECG detected.'},
            status=400
        )

    oea_result, image_file_id = (
        OEA_arrhy_mi_detection.signal_extraction_and_arrhy_detection(
            image_path=image_path,
            task_id=task_id
        )
    )

    def convert_numpy(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (np.integer, np.int64)):
            return int(obj)
        if isinstance(obj, (np.floating, np.float64)):
            return float(obj)
        if isinstance(obj, dict):
            return {k: convert_numpy(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [convert_numpy(v) for v in obj]
        return obj

    oea_result = convert_numpy(oea_result)

    for d in oea_result.get("detections", []):
        if d.get("detect", "").upper() == "ARTIFACTS":
            return JsonResponse(
                {'error': 'Artifacts detected.'},
                status=400
            )

    return oea_result, image_file_id

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

# Process arrhythmia
@csrf_exempt
def run_model_arrhythmia(request, category, filename):
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

    # -----------------------------
    # CREATE TASK (PENDING)
    # -----------------------------
    task_id = f"CSV-{timezone.now().strftime('%Y%m%d%H%M%S')}"
    is_lead = is_lead.split('_')[0]
    analysis_tasks.insert_one({
        "task_id": task_id,
        "user": request.user.username if request.user.is_authenticated else "anonymous",
        "file_name": csv_name,
        "file_type": "csv",
        "arrhythmia": category,
        "lead_type": is_lead,
        "status": "pending",
        "file_id": None,
        "created_at": timezone.now(),
        "completed_at": None
    })

    try:
        # -----------------------------
        # RUN MODEL (MODEL STORES PDF IN GRIDFS)
        # -----------------------------
        grid_file_id = check_arrhythmia_model(
            category,
            upload_file_path,
            is_lead,
            filename
        )

        if not grid_file_id:
            raise Exception("Result PDF not generated")

        # -----------------------------
        # UPDATE TASK → SUCCESS
        # -----------------------------
        analysis_tasks.update_one(
            {"task_id": task_id},
            {"$set": {
                "status": "success",
                "file_id": str(grid_file_id),
                "completed_at": timezone.now()
            }}
        )

        return JsonResponse({
            "status": "success",
            "task_id": task_id,
            "file_id": str(grid_file_id),
            "message": "Analysis completed successfully"
        })

    except Exception as e:
        # -----------------------------
        # UPDATE TASK → FAILED
        # -----------------------------
        analysis_tasks.update_one(
            {"task_id": task_id},
            {"$set": {
                "status": "failed",
                "error": str(e),
                "completed_at": timezone.now()
            }}
        )
        return JsonResponse({"error": str(e)}, status=500)

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
        return file_id

    elif category == 'block':
        file_id = block_model_check.model_check_for_ecg_data(
            upload_file_path, is_lead_for
        )
        return file_id

    elif category == 'mi':
        file_id = mi_model_check.model_check_for_ecg_data(
            upload_file_path, is_lead_for
        )
        return file_id

    elif category == 'pvc':
        file_id = pvc_model_check.model_check_for_ecg_data(
            upload_file_path, is_lead_for
        )
        return file_id
    elif category == 'pac':
        file_id = pac_model_check.model_check_for_ecg_data(
            upload_file_path, is_lead_for
        )
        return file_id

    elif category == 'pac_jn':
        file_id = pac_junc_model_check.model_check_for_ecg_data(
            upload_file_path, is_lead_for
        )
        return file_id
    elif category == 'all_arrhythmia':
        file_id = ALL_Arrhythmia.model_check_for_ecg_data(
            upload_file_path, is_lead_for
        )
        return file_id
    elif category == 'vifib_vfl':
        file_id = vifib_vfl_model_check.model_check_for_ecg_data(
            upload_file_path, is_lead_for
        )
        return file_id
    # ---------------------------
    # FINAL SAFETY CHECK
    # ---------------------------
    if not result_dic:
        return "Analysis failed (empty result)"

    if result_dic.get('is_error'):
        return "Something went wrong"

    return msg
    
def check_tmt_full_analysis(pdf_name, task_id):
    pdf_path = os.path.join(settings.MEDIA_ROOT, 'analysis_tool', 'uploads', pdf_name)
    if not os.path.exists(pdf_path):
        raise FileNotFoundError("TMT PDF not found")

    temp_dir = os.path.join(settings.MEDIA_ROOT, 'analysis_tool', 'temp', task_id)
    os.makedirs(temp_dir, exist_ok=True)

    #TMT range: pages 6 to 15
    pages = convert_from_path(
        pdf_path,
        dpi=300,
        first_page=6,
        last_page=7
    )

    pdf_images = []

    for idx, page in enumerate(pages, start=6):  # start=6 keeps real page number
        img_path = os.path.join(temp_dir, f"page_{idx}.jpg")
        page.save(img_path, "JPEG")

        #Call OEA
        _, image_file_id = OEA_arrhy_mi_detection.signal_extraction_and_arrhy_detection(
            image_path=img_path,
            task_id=f"{task_id}"
        )

        # #Read output from GridFS
        img_bytes = download_fs.get(image_file_id).read()
        pil_img = PILImage.open(BytesIO(img_bytes)).convert("RGB")
        pdf_images.append(pil_img)

    if not pdf_images:
        raise Exception("No TMT outputs generated")

    final_pdf = os.path.join(temp_dir, f"{task_id}.pdf")
    pdf_images[0].save(
        final_pdf,
        save_all=True,
        append_images=pdf_images[1:]
    )

    with open(final_pdf, "rb") as f:
        pdf_file_id = download_fs.put(
            f,
            filename=f"{task_id}.pdf",
            contentType="application/pdf",
            metadata={
                "task_id": task_id,
                "type": "tmt",
                "page_range": "6-7",
                "created_at": datetime.utcnow()
            }
        )

    shutil.rmtree(temp_dir, ignore_errors=True)

    return {
        "pages_processed": len(pdf_images),
        "range": "6-7"
    }, pdf_file_id
    
@csrf_exempt
def process_tmt_pdf(request):
    if request.method != "POST":
        return JsonResponse({"error": "Invalid request"}, status=405)
        
    uploaded_pdf = request.FILES["file"]
    pdf_name = uploaded_pdf.name
    
    task_id = f"TMT-{timezone.now().strftime('%Y%m%d%H%M%S%f')}"

    analysis_tasks.insert_one({
        "task_id": task_id,
        "user": request.user.username if request.user.is_authenticated else "anonymous",
        "file_name": pdf_name,
        "file_type": "tmt",
        "lead_type": 12,
        "status": "pending",
        "file_id": None,
        "created_at": timezone.now(),
        "completed_at": None,
        "error": None
    })

    try:
        result, pdf_file_id = check_tmt_full_analysis(pdf_name, task_id)

        analysis_tasks.update_one(
            {"task_id": task_id},
            {"$set": {
                "status": "success",
                "file_id": str(pdf_file_id),
                "completed_at": timezone.now()
            }}
        )

        return JsonResponse({
            "status": "success",
            "task_id": task_id,
            "file_id": str(pdf_file_id)
        })

    except Exception as e:
        analysis_tasks.update_one(
            {"task_id": task_id},
            {"$set": {
                "status": "failed",
                "error": str(e),
                "completed_at": timezone.now()
            }}
        )
        return JsonResponse({"error": str(e)}, status=500)
from django.http import StreamingHttpResponse, HttpResponse
from bson import ObjectId

CHUNK_SIZE = 8192  # 8KB

def gridfs_iterator(grid_file):
    while True:
        chunk = grid_file.read(CHUNK_SIZE)
        if not chunk:
            break
        yield chunk


def download_by_file_id(request, file_id):
    try:
        grid_file = download_fs.get(ObjectId(file_id))
    except Exception:
        return HttpResponse("File not found", status=404)

    # log download
    download_logs.insert_one({
        "file_id": grid_file._id,
        "downloaded_by": request.user.username if request.user.is_authenticated else "anonymous",
        "downloaded_at": timezone.now()
    })

    content_type = grid_file.content_type or "application/octet-stream"

    response = StreamingHttpResponse(
        gridfs_iterator(grid_file),
        content_type=content_type
    )

    response["Content-Disposition"] = (
        f'attachment; filename="{grid_file.filename}"'
    )
    response["Content-Length"] = grid_file.length

    return response

def get_analysis_history(request):
    history = list(
        analysis_tasks.find(
            {"user": request.user.username}
        ).sort("created_at", -1)
    )

    for h in history:
        h["_id"] = str(h["_id"])
        if "file_id" in h and h["file_id"]:
            h["file_id"] = str(h["file_id"])
        if "created_at" in h:
            h["created_at"] = h["created_at"].strftime("%Y-%m-%d %H:%M")
    return JsonResponse({"history": history})
def download_all_receipt(request):
    user = request.user
    history = list(
        analysis_tasks.find(
            {"user": user.username}
        ).sort("created_at", -1)
    )

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
    p.drawString(50, y, f"User: {user.username}")
    y -= 15
    p.drawString(50, y, f"Generated On: {datetime.now().strftime('%d-%m-%Y %H:%M')}")
    y -= 25

    # ---------- TABLE HEADER ----------
    p.setFont("Helvetica-Bold", 9)
    p.drawString(40, y, "Date")
    p.drawString(100, y, "Task ID")
    p.drawString(200, y, "File Name")
    p.drawString(350, y, "Channel")
    p.drawString(400, y, "Type")
    p.drawString(450, y, "Status")
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

            created_at = item.get("created_at")
            date_str = created_at.strftime("%d-%m-%Y") if created_at else "NA"

            p.drawString(40, y, date_str)
            p.drawString(100, y, item.get("task_id", "NA")[:16])
            p.drawString(200, y, item.get("file_name", "NA")[:25])
            p.drawString(350, y, str(item.get("lead_type", "NA")))
            p.drawString(400, y, item.get("file_type", "NA"))
            p.drawString(450, y, item.get("status", "NA"))

            y -= 14

    p.showPage()
    p.save()
    return response