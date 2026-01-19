from django.shortcuts import render
from django.http import JsonResponse, HttpResponse
from django.views.decorators.csrf import csrf_exempt
import numpy as np
import json, os, time, threading
from Scripts_Models.Scripts.Beat_Search import run_ecg_batch
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
from PIL import Image
from django.conf import settings
from urllib.parse import quote
from bson import ObjectId
from .db_image_store import get_images_by_batch, get_image
import io
from django.http import HttpResponse
from reportlab.lib.pagesizes import A10
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Paragraph, Image as RLImage, Spacer, PageBreak
from reportlab.lib.units import inch
from reportlab.lib.enums import TA_LEFT, TA_CENTER
from PIL import Image
import io
from reportlab.lib.utils import ImageReader

from django.http import HttpResponse, Http404
def index(request):
    return render(request, "Beat_Search/Beat_Search.html")


# =========================
# START BATCH
# =========================
@csrf_exempt
def start_batch_search(request):
    if request.method != "POST":
        return JsonResponse({"error": "Invalid method"}, status=405)

    files = request.FILES.getlist("files")
    if not files:
        return JsonResponse({"error": "No CSV files received"}, status=400)

    # ---- REQUIRED SESSION DATA ----
    required_keys = [
        "reference_beat",
        "reference_range",
        "reference_raw_full"
    ]

    for k in required_keys:
        if k not in request.session:
            return JsonResponse({
                "error": f"Missing reference data: {k}"
            }, status=400)

    # ---- LOAD REFERENCE ----
    reference_beat = np.array(
        request.session["reference_beat"], dtype=np.float32
    )
    reference_idx = request.session["reference_range"]

    reference_raw_full = np.array(
        request.session["reference_raw_full"], dtype=np.float32
    )

    reference_r = np.array(
        request.session.get("reference_r", []), dtype=np.int32
    )
    reference_q = np.array(
        request.session.get("reference_q", []), dtype=np.int32
    )
    reference_s = np.array(
        request.session.get("reference_s", []), dtype=np.int32
    )
    # ---- CREATE BATCH FOLDER ----
    batch_id = f"batch_{int(time.time())}"
    # batch_dir = os.path.join(BASE_RESULTS_DIR, batch_id)
    batch_dir = os.path.join(settings.MEDIA_ROOT, "Beat_Search", 'ecg_results', batch_id)
    os.makedirs(batch_dir, exist_ok=True)

    # ---- SAVE UPLOADED CSVs ----
    for f in files:
        with open(os.path.join(batch_dir, f.name), "wb+") as dst:
            for chunk in f.chunks():
                dst.write(chunk)

    # ---- START BACKGROUND THREAD ----
    threading.Thread(
        target=run_ecg_batch,
        args=(
            batch_dir,
            reference_beat,
            reference_idx,
            reference_raw_full,
            reference_r,
            reference_q,
            reference_s
        ),
        daemon=True
    ).start()

    return JsonResponse({
        "status": "started",
        "batch_id": batch_id,
        "message": "ECG batch started successfully"
    })

def get_batch_images(request, batch_id):
    images = []

    for f in get_images_by_batch(batch_id):
        images.append({
            "id": str(f._id),
            "filename": f.filename,
            "csv": f.csv_name,
            "lead": f.lead,
            "meta": f.metadata
        })

    return JsonResponse({
        "count": len(images),
        "images": images
    })

def serve_image(request, image_id):
    try:
        file = get_image(ObjectId(image_id))
    except Exception as e:
        print("GridFS error:", e)
        raise Http404("Image not found")

    return HttpResponse(file.read(), content_type="image/png")


def download_batch_pdf(request, batch_id):
    files = list(get_images_by_batch(batch_id))

    if not files:
        return HttpResponse("No images", status=404)

    response = HttpResponse(content_type="application/pdf")
    response["Content-Disposition"] = f'attachment; filename="{batch_id}.pdf"'

    pdf = None

    for i, f in enumerate(files):
        img_bytes = io.BytesIO(f.read())
        pil_img = Image.open(img_bytes)

        # Image size in pixels
        width_px, height_px = pil_img.size

        # ReportLab uses points (1 px ≈ 0.75 pt at 96 dpi)
        width_pt = width_px * 0.75
        height_pt = height_px * 0.75

        if pdf is None:
            pdf = canvas.Canvas(response, pagesize=(width_pt, height_pt))
        else:
            pdf.setPageSize((width_pt, height_pt))

        pdf.drawImage(
            ImageReader(pil_img),
            0, 0,
            width=width_pt,
            height=height_pt
        )

        pdf.showPage()

    pdf.save()
    return response
@csrf_exempt
def save_reference_pattern(request):
    if request.method != "POST":
        return JsonResponse({"error": "Invalid method"}, status=405)

    try:
        data = json.loads(request.body)

        reference_beat = np.array(data["reference_beat"], dtype=np.float32)
        reference_range = data["reference_range"]
        reference_raw_full = np.array(data["reference_raw_full"], dtype=np.float32)

        reference_r = np.array(data.get("reference_r", []), dtype=np.int32)
        reference_q = np.array(data.get("reference_q", []), dtype=np.int32)
        reference_s = np.array(data.get("reference_s", []), dtype=np.int32)

    except Exception as e:
        return JsonResponse(
            {"error": f"Invalid reference data: {str(e)}"},
            status=400
        )

    request.session["reference_beat"] = reference_beat.tolist()
    request.session["reference_range"] = reference_range
    request.session["reference_raw_full"] = reference_raw_full.tolist()

    request.session["reference_r"] = reference_r.tolist()
    request.session["reference_q"] = reference_q.tolist()
    request.session["reference_s"] = reference_s.tolist()

    request.session.modified = True

    return JsonResponse({
        "status": "ok",
        "message": "Reference pattern saved successfully"
    })
def check_batch_status(request, batch_id):
    batch_dir = os.path.join(
        settings.MEDIA_ROOT,
        "Beat_Search",
        "ecg_results",
        batch_id
    )
    done_flag = os.path.join(batch_dir, "__DONE__")

    return JsonResponse({
        "completed": os.path.exists(done_flag)
    })