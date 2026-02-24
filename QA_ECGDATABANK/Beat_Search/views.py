import os
import json
import time
import threading
import numpy as np
from django.conf import settings
from django.http import (
    JsonResponse,
    StreamingHttpResponse,
    Http404,
    FileResponse
)
from django.views.decorators.csrf import csrf_exempt
from Scripts_Models.Scripts.Beat_Search import run_ecg_batch
from .db_image_store import get_label_collection
from django.shortcuts import render
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import tempfile
import queue
from matplotlib.backends.backend_pdf import PdfPages
# Time grid (seconds)
X_MAJOR = 0.2     # 5 large boxes per second
X_MINOR = 0.04    # small ECG boxes

# Voltage grid (scaled units)
Y_MAJOR = 0.5
Y_MINOR = 0.1

# Styling
MAJOR_GRID_LW = 0.9
MINOR_GRID_LW = 0.35
SIGNAL_LW = 1.25

FIG_WIDTH_IN = 16.5   # wide, readable
FIG_HEIGHT_IN = 3.6   # taller so waveform breathes
DPI = 100             # print-quality
MAX_POINTS = 2000
MAX_SECONDS = 10

# ===== PDF PERFORMANCE SETTINGS =====
FAST_PDF_MODE = True       # Set True for 3-5x faster PDF
DOWNSAMPLE_TARGET = 1500   # max points per page (faster plotting)
TIMING_LOGS = True         # enable timing prints
def fast_decimate(signal, target_points=4000):
    n = len(signal)
    if n <= target_points:
        return signal, 1

    block = n // target_points
    trimmed = signal[:block * target_points]
    reshaped = trimmed.reshape(target_points, block)

    # Preserve waveform shape (important for ECG)
    mins = reshaped.min(axis=1)
    maxs = reshaped.max(axis=1)

    decimated = np.empty(target_points * 2)
    decimated[0::2] = mins
    decimated[1::2] = maxs

    return decimated, block // 2

def downsample_signal(signal, target_points):
    """Fast downsampling to reduce matplotlib load"""
    n = len(signal)
    if n <= target_points:
        return signal, 1  # no downsampling

    step = max(1, n // target_points)
    return signal[::step], step

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

def minmax_scale(arr, min_val=0, max_val=4):
    arr = np.asarray(arr, dtype=float)
    amin, amax = arr.min(), arr.max()
    if amin == amax:
        return np.full_like(arr, (min_val + max_val) / 2)
    return (arr - amin) / (amax - amin) * (max_val - min_val) + min_val

def draw_ecg_page(ax, result, start_idx=0, end_idx=None):
    signal = np.asarray(result["signal"], dtype=float)
    fs = result["fs"]

    if end_idx is None:
        end_idx = len(signal)

    # Slice signal only (markers must already be LOCAL indices)
    signal = signal[start_idx:end_idx]

    if len(signal) == 0:
        return

    x = np.arange(len(signal)) / fs
    y = minmax_scale(signal)

    # ---- ECG trace (faster settings) ----
    ax.plot(
        x,
        y,
        color="black",
        linewidth=SIGNAL_LW,
        zorder=3,
        solid_capstyle="round",
        solid_joinstyle="round"
    )

    # ---- Markers (NO index shifting here) ----
    def plot_markers(idxs, color, size):
        if not idxs:
            return
        # indices are already local to chunk
        idxs = [i for i in idxs if 0 <= i < len(signal)]
        if idxs:
            ax.scatter(
                x[idxs],
                y[idxs],
                s=size,
                color=color,
                zorder=4
            )

    plot_markers(result.get("r_peaks"), "#ef4444", 28)
    plot_markers(result.get("q_points"), "#3b82f6", 22)
    plot_markers(result.get("s_points"), "#10b981", 22)

    # ---- Beat windows (FIXED & LOCAL) ----
    for m in result.get("matches", []):
        start = int(m.get("start", 0))
        end = int(m.get("end", 0))

        if end <= 0 or start >= len(signal):
            continue

        start = max(0, start)
        end = min(len(signal), end)

        ax.axvspan(
            start / fs,
            end / fs,
            color="red",
            alpha=0.16,
            zorder=1
        )

    # ---- Limits ----
    ax.set_xlim(0, x[-1])
    ax.set_ylim(0, 4)

    # ================= ECG GRID (OPTIMIZED) =================
    duration = x[-1]

    # Standard ECG grid:
    # Major: 0.2 sec (large box)
    # Minor: 0.04 sec (small box)
    X_MAJOR = 0.2
    X_MINOR = 0.04
    Y_MAJOR = 0.5
    Y_MINOR = 0.1

    # Reduce grid density automatically for long pages (HUGE speed boost)
    if duration > 20:
        X_MINOR = 0.2   # fewer vertical lines
        Y_MINOR = 0.5

    ax.set_xticks(np.arange(0, duration + X_MAJOR, X_MAJOR))
    ax.set_xticks(np.arange(0, duration + X_MINOR, X_MINOR), minor=True)

    ax.set_yticks(np.arange(0, 4.1, Y_MAJOR))
    ax.set_yticks(np.arange(0, 4.1, Y_MINOR), minor=True)

    # Minor grid (light pink)
    ax.grid(
        which="minor",
        color="#facdcd",
        linewidth=0.4,
        zorder=0
    )

    # Major grid (bold red ECG style)
    ax.grid(
        which="major",
        color="#f76262",
        linewidth=0.8,
        zorder=1
    )

    # ---- Clean frame ----
    ax.tick_params(axis="both", labelsize=9)
    for spine in ax.spines.values():
        spine.set_linewidth(0.8)

    ax.set_title(
        f'{result["csv"]} ‚Äî Lead {result["lead"]}',
        fontsize=12,
        pad=10
    )

def render_batch_pdf(docs_cursor, pdf_path):
    import time

    # Ì†ΩÌ¥¥ CRITICAL FIX: convert cursor to list so we can loop twice
    docs = list(docs_cursor)

    t0_total = time.time()

    total_pages = 0
    total_docs = 0
    reference_plotted = False  # Only once for entire PDF

    with PdfPages(pdf_path) as pdf:

        # =====================================================
        # ‚≠ê PAGE 1: GLOBAL REFERENCE BEAT (ONLY ONCE)
        # =====================================================
        for doc in docs:
            if reference_plotted:
                break

            for lead in doc["leads"]:
                ref = lead.get("reference", None)
                if not ref:
                    continue

                signal = np.asarray(lead["signal"], dtype=float)
                fs = lead["fs"]
                signal_len = len(signal)

                ref_start = int(ref["start"])
                ref_end = int(ref["end"])

                if ref_end <= ref_start or ref_start >= signal_len:
                    continue

                # Downsample full signal (for speed if enabled)
                if FAST_PDF_MODE:
                    plot_signal, step = downsample_signal(
                        signal, DOWNSAMPLE_TARGET
                    )
                    ref_start_ds = int(ref_start / step)
                    ref_end_ds = int(ref_end / step)
                else:
                    plot_signal = signal
                    ref_start_ds = ref_start
                    ref_end_ds = ref_end

                fig, ax = plt.subplots(
                    figsize=(FIG_WIDTH_IN, FIG_HEIGHT_IN),
                    dpi=80 if FAST_PDF_MODE else DPI
                )

                # Full CSV signal plot (like your uploaded image)
                x = np.arange(len(plot_signal))
                ax.plot(x, plot_signal, linewidth=1.0)

                # Highlight selected reference beat
                ax.axvspan(
                    ref_start_ds,
                    ref_end_ds,
                    color="orange",
                    alpha=0.35,
                    label="Reference Beat"
                )

                ax.set_title(
                    f'{doc["csv_name"]} ‚Äî Reference Beat Overview '
                    f'(Lead {lead["lead"]}) [{ref_start}:{ref_end}]',
                    fontsize=13,
                    pad=12,
                    color="darkred"
                )

                ax.set_xlabel("Sample Index")
                ax.set_ylabel("Amplitude")
                ax.legend(loc="upper right")

                fig.subplots_adjust(
                    left=0.05,
                    right=0.995,
                    top=0.9,
                    bottom=0.12
                )

                pdf.savefig(fig)
                plt.close(fig)

                total_pages += 1
                reference_plotted = True
                break  # stop after first valid reference

        # =====================================================
        # Ì†ΩÌ≥Ñ ALL RESULT PAGES (NOW WILL WORK)
        # =====================================================
        for doc_idx, doc in enumerate(docs):
            t0_doc = time.time()
            total_docs += 1

            for lead_idx, lead in enumerate(doc["leads"]):
                t0_lead = time.time()

                signal = np.asarray(lead["signal"], dtype=float)
                fs = lead["fs"]
                signal_len = len(signal)

                max_points = min(
                    MAX_POINTS,
                    int(MAX_SECONDS * fs)
                )

                for start in range(0, signal_len, max_points):
                    t0_page = time.time()
                    end = min(start + max_points, signal_len)
                    chunk = signal[start:end]

                    # Downsample
                    if FAST_PDF_MODE:
                        chunk, step = downsample_signal(
                            chunk, DOWNSAMPLE_TARGET
                        )
                    else:
                        step = 1

                    # Remap indices
                    def remap_indices(idxs):
                        if not idxs:
                            return []
                        local = [i for i in idxs if start <= i < end]
                        local = [(i - start) for i in local]
                        if step > 1:
                            local = [
                                int(i / step)
                                for i in local
                                if int(i / step) < len(chunk)
                            ]
                        return local

                    r_peaks = remap_indices(lead.get("r_peaks", []))
                    q_points = remap_indices(lead.get("q_points", []))
                    s_points = remap_indices(lead.get("s_points", []))

                    matches = []
                    for m in lead.get("matches", []):
                        if m["end"] < start or m["start"] > end:
                            continue
                        matches.append({
                            "start": max(m["start"] - start, 0) / step,
                            "end": min(m["end"] - start, end - start) / step
                        })

                    fast_lead = {
                        "signal": chunk,
                        "fs": fs / step,
                        "r_peaks": r_peaks,
                        "q_points": q_points,
                        "s_points": s_points,
                        "matches": matches,
                        "lead": lead["lead"],
                        "csv": doc["csv_name"]
                    }

                    fig, ax = plt.subplots(
                        figsize=(FIG_WIDTH_IN, FIG_HEIGHT_IN),
                        dpi=80 if FAST_PDF_MODE else DPI
                    )

                    draw_ecg_page(ax, fast_lead, 0, len(chunk))

                    ax.set_title(
                        f'{doc["csv_name"]} ‚Äî Lead {lead["lead"]} '
                        f'[{start}:{end}]',
                        fontsize=10,
                        pad=8
                    )

                    fig.subplots_adjust(
                        left=0.04,
                        right=0.995,
                        top=0.88,
                        bottom=0.14
                    )

                    pdf.savefig(fig)
                    plt.close(fig)

                    total_pages += 1

def download_lead_pdf(request, batch_id):
    import time
    t0 = time.time()
    terminology = request.GET.get("terminology")
    if not terminology:
        raise Http404("Missing terminology")

    pdf_dir = os.path.join(
        settings.MEDIA_ROOT,
        "Beat_Search",
        "pdf_cache"
    )
    os.makedirs(pdf_dir, exist_ok=True)

    pdf_path = os.path.join(pdf_dir, f"{batch_id}.pdf")

    # FAST PATH (cached)
    if os.path.exists(pdf_path):
        return FileResponse(
            open(pdf_path, "rb"),
            as_attachment=True,
            filename=f"{batch_id}.pdf",
            content_type="application/pdf"
        )

    t0_db = time.time()
    collection = get_label_collection(terminology)

    # CURSOR (NO RAM EXPLOSION)
    docs_cursor = collection.find(
        {"batch_id": batch_id},
        {"_id": 0, "csv_name": 1, "leads": 1},
        no_cursor_timeout=True
    )

    # Quick existence check without loading all
    if collection.count_documents({"batch_id": batch_id}) == 0:
        raise Http404("Batch not found")

    # Generate PDF (heavy)
    t0_render = time.time()
    render_batch_pdf(docs_cursor, pdf_path)
    return FileResponse(
        open(pdf_path, "rb"),
        as_attachment=True,
        filename=f"{batch_id}.pdf",
        content_type="application/pdf"
    )
