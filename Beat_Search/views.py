# import os
# import io
# import json
# import time
# import threading
# import numpy as np

# from bson import ObjectId
# from PIL import Image
# from reportlab.pdfgen import canvas
# from reportlab.lib.utils import ImageReader

# from django.conf import settings
# from django.http import (
#     JsonResponse,
#     HttpResponse,
#     Http404,
#     StreamingHttpResponse
# )
# from django.views.decorators.csrf import csrf_exempt
# from Scripts_Models.Scripts.Beat_Search import run_ecg_batch
# from .db_image_store import get_image, get_csv_images
# from django.shortcuts import render

# # =====================================
# # 1. Save reference beat (unchanged)
# # =====================================
# def index(request):
#     return render(request, "Beat_Search/Beat_Search.html")

# @csrf_exempt
# def save_reference_pattern(request):
#     if request.method != "POST":
#         return JsonResponse({"error": "Invalid method"}, status=405)

#     try:
#         data = json.loads(request.body)

#         request.session["reference_beat"] = data["reference_beat"]
#         request.session["reference_range"] = data["reference_range"]
#         request.session["reference_raw_full"] = data["reference_raw_full"]

#         request.session["reference_r"] = data.get("reference_r", [])
#         request.session["reference_q"] = data.get("reference_q", [])
#         request.session["reference_s"] = data.get("reference_s", [])

#         request.session.modified = True

#         return JsonResponse({"status": "ok"})

#     except Exception as e:
#         return JsonResponse({"error": str(e)}, status=400)

# def start_batch_search(request):
#     if request.method != "POST":
#         return JsonResponse({"error": "Invalid method"}, status=405)

#     files = request.FILES.getlist("files")
#     terminology = request.POST.get("terminology")

#     if not files:
#         return JsonResponse({"error": "No files"}, status=400)
#     if not terminology:
#         return JsonResponse({"error": "Missing terminology"}, status=400)

#     batch_id = f"batch_{int(time.time())}"
#     batch_dir = os.path.join(
#         settings.MEDIA_ROOT,
#         "Beat_Search",
#         "ecg_results",
#         batch_id
#     )
#     os.makedirs(batch_dir, exist_ok=True)

#     for f in files:
#         with open(os.path.join(batch_dir, f.name), "wb+") as dst:
#             for chunk in f.chunks():
#                 dst.write(chunk)

#     return JsonResponse({"batch_id": batch_id})



# # =====================================
# # 3. SSE STREAM (core of your goal)
# # =====================================

# from django.http import StreamingHttpResponse
# import json
# import threading
# import queue

# def stream_batch_results(request, batch_id):
#     terminology = request.GET.get("terminology")

#     # ---- Load session data ----
#     reference_beat = np.array(request.session["reference_beat"], dtype=np.float32)
#     reference_range = request.session["reference_range"]
#     reference_raw_full = np.array(request.session["reference_raw_full"], dtype=np.float32)

#     reference_r = np.array(request.session.get("reference_r", []), dtype=np.int32)
#     reference_q = np.array(request.session.get("reference_q", []), dtype=np.int32)
#     reference_s = np.array(request.session.get("reference_s", []), dtype=np.int32)

#     batch_dir = os.path.join(
#         settings.MEDIA_ROOT,
#         "Beat_Search",
#         "ecg_results",
#         batch_id
#     )

#     # Thread-safe queue for streaming
#     q = queue.Queue()

#     # Worker thread
#     def worker():
#         for item in run_ecg_batch(
#             batch_dir,
#             reference_beat,
#             reference_range,
#             reference_raw_full,
#             reference_r,
#             reference_q,
#             reference_s,
#             terminology
#         ):
#             q.put(item)

#     threading.Thread(target=worker, daemon=True).start()

#     # SSE stream
#     def event_stream():
#         while True:
#             item = q.get()   # blocks until new result

#             if "done" in item:
#                 yield f"data: {json.dumps({'done': True})}\n\n"
#                 break

#             payload = {
#                 "id": str(item["image_id"]),
#                 "lead": item["lead"],
#                 "csv": item["csv"],
#                 "matches": item["matches"]
#             }

#             yield f"data: {json.dumps(payload)}\n\n"

#     return StreamingHttpResponse(
#         event_stream(),
#         content_type="text/event-stream"
#     )


# # =====================================
# # 4. Serve image
# # =====================================

# def serve_image(request, image_id):
#     try:
#         file = get_image(ObjectId(image_id))
#         return HttpResponse(file.read(), content_type="image/jpeg")
#     except Exception:
#         raise Http404("Image not found")


# # =====================================
# # 5. Download PDF
# # =====================================
# def download_batch_pdf(request, batch_id):
#     terminology = request.GET.get("terminology")
#     csv_name = request.GET.get("csv", "all")

#     if not terminology:
#         terminology = None

#     images = get_csv_images(batch_id, csv_name, terminology)

#     if not images:
#         return HttpResponse("No images found", status=404)

#     response = HttpResponse(content_type="application/pdf")
#     response["Content-Disposition"] = f'attachment; filename="{batch_id}.pdf"'

#     pdf = canvas.Canvas(response)

#     for img in images:
#         f = get_image(img["file_id"])
#         img_bytes = io.BytesIO(f.read())
#         pil_img = Image.open(img_bytes).convert("RGB")

#         width_px, height_px = pil_img.size
#         width_pt = width_px * 0.75
#         height_pt = height_px * 0.75

#         pdf.setPageSize((width_pt, height_pt))
#         pdf.drawImage(ImageReader(pil_img), 0, 0, width=width_pt, height=height_pt)
#         pdf.showPage()

#     pdf.save()
#     return response
#########################################################################3
import os
import io
import json
import time
import threading
import numpy as np

from PIL import Image
from reportlab.pdfgen import canvas

from django.conf import settings
from django.http import (
    JsonResponse,
    HttpResponse,
    StreamingHttpResponse
)
from django.views.decorators.csrf import csrf_exempt
from Scripts_Models.Scripts.Beat_Search import run_ecg_batch
from .db_image_store import get_csv_results
from django.shortcuts import render
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# =====================================
# 1. Save reference beat (unchanged)
# =====================================
def index(request):
    return render(request, "Beat_Search/Beat_Search.html")

@csrf_exempt
def save_reference_pattern(request):
    if request.method != "POST":
        return JsonResponse({"error": "Invalid method"}, status=405)

    try:
        data = json.loads(request.body)

        request.session["reference_beat"] = data["reference_beat"]
        request.session["reference_range"] = data["reference_range"]
        request.session["reference_raw_full"] = data["reference_raw_full"]

        request.session["reference_r"] = data.get("reference_r", [])
        request.session["reference_q"] = data.get("reference_q", [])
        request.session["reference_s"] = data.get("reference_s", [])

        request.session.modified = True

        return JsonResponse({"status": "ok"})

    except Exception as e:
        return JsonResponse({"error": str(e)}, status=400)

def start_batch_search(request):
    if request.method != "POST":
        return JsonResponse({"error": "Invalid method"}, status=405)

    files = request.FILES.getlist("files")
    terminology = request.POST.get("terminology")

    if not files:
        return JsonResponse({"error": "No files"}, status=400)
    if not terminology:
        return JsonResponse({"error": "Missing terminology"}, status=400)

    batch_id = f"batch_{int(time.time())}"
    batch_dir = os.path.join(
        settings.MEDIA_ROOT,
        "Beat_Search",
        "ecg_results",
        batch_id
    )
    os.makedirs(batch_dir, exist_ok=True)

    for f in files:
        with open(os.path.join(batch_dir, f.name), "wb+") as dst:
            for chunk in f.chunks():
                dst.write(chunk)

    return JsonResponse({"batch_id": batch_id})



# =====================================
# 3. SSE STREAM (core of your goal)
# =====================================

from django.http import StreamingHttpResponse
import json
import threading
import queue

def stream_batch_results(request, batch_id):
    terminology = request.GET.get("terminology")

    reference_beat = np.array(request.session["reference_beat"], dtype=np.float32)
    reference_range = request.session["reference_range"]
    reference_raw_full = np.array(request.session["reference_raw_full"], dtype=np.float32)

    reference_r = np.array(request.session.get("reference_r", []), dtype=np.int32)
    reference_q = np.array(request.session.get("reference_q", []), dtype=np.int32)
    reference_s = np.array(request.session.get("reference_s", []), dtype=np.int32)

    batch_dir = os.path.join(
        settings.MEDIA_ROOT,
        "Beat_Search",
        "ecg_results",
        batch_id
    )

    q = queue.Queue()

    def worker():
        for item in run_ecg_batch(
            batch_dir,
            reference_beat,
            reference_range,
            reference_raw_full,
            reference_r,
            reference_q,
            reference_s,
            terminology
        ):
            q.put(item)

    threading.Thread(target=worker, daemon=True).start()

    def event_stream():
        while True:
            item = q.get()

            if "done" in item:
                yield f"data: {json.dumps({'done': True})}\n\n"
                break

            # Send full plot payload to frontend
            yield f"data: {json.dumps(item)}\n\n"

    return StreamingHttpResponse(
        event_stream(),
        content_type="text/event-stream"
    )

