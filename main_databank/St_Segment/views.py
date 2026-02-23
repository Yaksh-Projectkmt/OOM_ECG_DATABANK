from django.shortcuts import render
import os, shutil, time
from django.http import JsonResponse, FileResponse,Http404
from django.views.decorators.csrf import csrf_exempt
from django.conf import settings
from bson import ObjectId
from pymongo import MongoClient
from Scripts_Models.Scripts.St_Segment import run_ecg_st_pipeline
from Scripts_Models.Scripts.St_Segment import save_pdf_to_gridfs
import gridfs

def index(request):
    return render(request, "St_Segment/St_Segment.html")

def get_gridfs():
    mongo_uri = os.getenv("MONGO_HOST")
    client = MongoClient(mongo_uri)
    db = client["St_Segment"]
    return gridfs.GridFS(db)
@csrf_exempt
@csrf_exempt
def run_ecg_analysis(request):
    if request.method != "POST":
        return JsonResponse({"error": "POST required"}, status=400)

    files = request.FILES.getlist("files")
    lead_type = request.POST.get("lead_type")

    if not files:
        return JsonResponse({"error": "No CSV files"}, status=400)

    job_id = str(int(time.time()))
    job_dir = os.path.join(settings.MEDIA_ROOT, "ecg_jobs", job_id)
    os.makedirs(job_dir, exist_ok=True)

    try:
        # Save CSVs
        for f in files:
            with open(os.path.join(job_dir, f.name), "wb+") as dst:
                for chunk in f.chunks():
                    dst.write(chunk)

        # Run pipeline
        final_pdf = run_ecg_st_pipeline(
            input_folder=job_dir,
            output_folder=job_dir,
            is_lead=lead_type or None
        )

        # Save to GridFS
        fs = get_gridfs()
        with open(final_pdf, "rb") as f:
            file_id = fs.put(
                f,
                filename=os.path.basename(final_pdf),
                contentType="application/pdf",
                metadata={
                    "job_id": job_id,
                    "lead_type": lead_type or "auto"
                }
            )

        # Cleanup
        shutil.rmtree(job_dir, ignore_errors=True)

        return JsonResponse({
            "status": "success",
            "file_id": str(file_id)
        })

    except Exception as e:
        shutil.rmtree(job_dir, ignore_errors=True)
        return JsonResponse({"error": str(e)}, status=500)

def view_pdf(request, file_id):
    try:
        fs = get_gridfs()
        grid_file = fs.get(ObjectId(file_id))

        response = FileResponse(
            grid_file,
            content_type="application/pdf"
        )
        response["Content-Disposition"] = "inline; filename=ecg.pdf"
        return response

    except Exception as e:
        raise Http404("PDF not found")


def download_pdf(request, file_id):
    fs = get_gridfs()
    pdf = fs.get(ObjectId(file_id))

    response = FileResponse(pdf, content_type="application/pdf")
    response["Content-Disposition"] = (
        f'attachment; filename="ST_Segment_{file_id}.pdf"'
    )
    return response

