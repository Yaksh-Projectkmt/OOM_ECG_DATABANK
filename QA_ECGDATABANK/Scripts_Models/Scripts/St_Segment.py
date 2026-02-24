import os
import glob
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import tensorflow as tf
from scipy import signal
from scipy.signal import find_peaks, argrelextrema, savgol_filter
from scipy.stats import mode
from scipy.interpolate import interp1d
import warnings
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from PyPDF2 import PdfMerger
import random
from concurrent.futures import ThreadPoolExecutor, as_completed, ProcessPoolExecutor
import multiprocessing
from pymongo import MongoClient
import gridfs
from django.conf import settings
from sklearn.preprocessing import MinMaxScaler

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
from PyPDF2 import PdfMerger

DESIRED_CPU_THREADS = 4  # adjust based on your CPU

warnings.filterwarnings('ignore')
results_lock = threading.RLock()
# ---------------------- Server-level tuning (CPU / threading / TF / env) ----------------------
# Set these before heavy libs use BLAS/OMP/MKL threads
DESIRED_CPU_THREADS = 24
os.environ['OMP_NUM_THREADS'] = str(DESIRED_CPU_THREADS)
os.environ['OPENBLAS_NUM_THREADS'] = str(DESIRED_CPU_THREADS)
os.environ['MKL_NUM_THREADS'] = str(DESIRED_CPU_THREADS)
os.environ['NUMEXPR_NUM_THREADS'] = str(DESIRED_CPU_THREADS)
os.environ['VECLIB_MAXIMUM_THREADS'] = str(DESIRED_CPU_THREADS)
# TensorFlow GPU / thread config
try:
    physical_gpus = tf.config.list_physical_devices('GPU')
    if physical_gpus:
        # Allow memory growth so multiple processes/threads can share GPU more gracefully
        for g in physical_gpus:
            try:
                tf.config.experimental.set_memory_growth(g, True)
            except Exception:
                pass
    # Set TF threading parallelism
    tf.config.threading.set_intra_op_parallelism_threads(DESIRED_CPU_THREADS)
    tf.config.threading.set_inter_op_parallelism_threads(max(1, DESIRED_CPU_THREADS // 2))
except Exception as e:
    print("Warning: TensorFlow configuration failed:", e)
# Thread-local storage for per-thread interpreters
thread_local = threading.local()
# ---------------------- TFLite interpreter utilities ----------------------
def _load_gpu_delegate():
    """
    Try loading a TFLite GPU delegate. This is inherently platform-dependent.
    We try multiple common delegate names and fall back to None.
    """
    try:
        # TensorFlow's load_delegate helper
        load_delegate = tf.lite.experimental.load_delegate
    except Exception:
        load_delegate = None
    if load_delegate:
        candidates = [
            'libtensorflowlite_gpu_delegate.so', # linux
            'libtensorflowlite_gpu_delegate.dylib', # mac
            'tensorflowlite_gpu_delegate.dll', # windows (rare)
        ]
        for cand in candidates:
            try:
                delegate = load_delegate(cand)
                print(f"Loaded GPU delegate: {cand}")
                return delegate
            except Exception:
                continue
    # If we reach here, GPU delegate wasn't loaded
    return None
GPU_DELEGATE = _load_gpu_delegate()


def get_tflite_interpreter_for_thread(model_path: str, use_gpu_delegate=True):
   
    if not hasattr(thread_local, "interpreters"):
        thread_local.interpreters = {}
    key = f"{model_path}_gpu" if use_gpu_delegate and GPU_DELEGATE else model_path
    if key in thread_local.interpreters:
        return thread_local.interpreters[key]
    # Create interpreter
    try:
        if use_gpu_delegate and GPU_DELEGATE:
            interpreter = tf.lite.Interpreter(model_path=model_path, experimental_delegates=[GPU_DELEGATE])
            print(f"[Thread {threading.get_ident()}] Created GPU interpreter for {os.path.basename(model_path)}")
        else:
            interpreter = tf.lite.Interpreter(model_path=model_path)
            print(f"[Thread {threading.get_ident()}] Created CPU interpreter for {os.path.basename(model_path)}")
    except Exception as e:
        # Fallback to CPU interpreter if GPU delegate fails
        print(f"Interpreter creation failed for {model_path} with GPU delegate: {e}. Falling back to CPU.")
        interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    thread_local.interpreters[key] = (interpreter, input_details, output_details)
    return thread_local.interpreters[key]

def predict_tflite_model(model_path: str, input_data, use_gpu_delegate=True):
   
    # Acquire a lock around interpreter invocation to be safe for device resources, but interpreters are per-thread so contention is low.
    interpreter, input_details, output_details = get_tflite_interpreter_for_thread(model_path, use_gpu_delegate=use_gpu_delegate)
    with results_lock:
        input_data = input_data.astype(np.float32)
        interpreter.set_tensor(input_details[0]['index'], input_data)
        interpreter.invoke()
        output_data = interpreter.get_tensor(output_details[0]['index'])
    return output_data.squeeze()
# ---------------------- Your existing functions (kept mostly unchanged) ----------------------
def lowpass(file, cutoff=0.3):
    b, a = signal.butter(3, cutoff, btype='lowpass', analog=False)
    low_passed = signal.filtfilt(b, a, file)
    return low_passed

def baseline_construction_200(ecg_signal, kernel_size=191):
    s_corrected = signal.detrend(ecg_signal)
    baseline_corrected = s_corrected - signal.medfilt(s_corrected, kernel_size)
    return baseline_corrected

def normalize(signal):
    return (signal - np.mean(signal)) / np.std(signal)

def refined_non_max_suppression(ecg_signal, valid_indices, suppression_radius=40):
    if len(valid_indices) == 0:
        return []
    sorted_indices = sorted(valid_indices, reverse=True)
    selected = []
    occupied = np.zeros(len(ecg_signal), dtype=bool)
    for idx in sorted_indices:
        if not occupied[idx]:
            left = max(0, idx - suppression_radius)
            right = min(len(ecg_signal), idx + suppression_radius + 1)
            # Mark region as occupied
            occupied[left:right] = True
            for i in sorted_indices:
                maximum_idx = idx
                if occupied[i] in occupied[left:right]:
                    if occupied[i] > occupied[maximum_idx]:
                        maximum = i
                selected.append(maximum_idx)
    return sorted(selected)

def check_model_r(ecg_data, r_model_path, use_gpu_delegate=True):
    totaldata = len(ecg_data)
    i = 0
    if totaldata < 1000:
        step = totaldata
    else:
        step = 1000
    r_peaks = []
    all_preds = np.zeros((totaldata, 2))
    counts = np.zeros((totaldata, 1))
    temp_list= []
    df_ecg_signal = ecg_data.tolist()
    while i < totaldata:
        if i != 0 and totaldata > 1000:
            i = i-200
        ecg_signal = ecg_data[i:i + step]
        signal_len = len(ecg_signal)
        pad_len = 1000 - signal_len
        padded_signal = np.pad(ecg_signal, (0, pad_len), mode='constant', constant_values=0)
        raw_array = np.expand_dims(padded_signal, axis=0).astype(np.float32)[..., np.newaxis]
        preds = predict_tflite_model(r_model_path, raw_array, use_gpu_delegate=use_gpu_delegate)
        preds = preds[:signal_len]
        r_peak_prob = preds[:, 1]

        peak_indices, _ = find_peaks(r_peak_prob, height=0.2, distance=20)

        for j in range(len(peak_indices)):
            if ecg_signal[peak_indices[j]] in df_ecg_signal:
                temp_list.append(df_ecg_signal.index(ecg_signal[peak_indices[j]]))

        i += step
    rpeak = sorted(set(temp_list))
    r_peaks = refined_non_max_suppression(df_ecg_signal, rpeak)
    r_peaks = sorted(set(r_peaks))
    return r_peaks

def r_peak_detection(all_lead_data, is_lead, r_model_path, use_gpu_delegate=True):
    r_peaks = []
    result_dic = {}
    for lead in all_lead_data.keys():
        ecg_signal = all_lead_data[lead].values.flatten()
        baseline_signal = baseline_construction_200(ecg_signal, kernel_size=131)
        lowpass_signal = lowpass(baseline_signal)
        signal_normalized = normalize(lowpass_signal)
        r_peaks = check_model_r(signal_normalized, r_model_path, use_gpu_delegate=use_gpu_delegate)
        result_dic[lead] = r_peaks
    if is_lead == '2_lead':
        r_peaks = result_dic['II']
    return r_peaks, result_dic

# --- P/T detection functions (unchanged except calling predict_tflite_model with model path) ---
def resample_ecg(ecg_signal, target_length=520):
    x_old = np.linspace(0, 1, len(ecg_signal))
    x_new = np.linspace(0, 1, target_length)
    f_ecg = interp1d(x_old, ecg_signal, kind='linear')
    ecg_resampled = f_ecg(x_new)
    return ecg_resampled

def restore_org_ecg_mask(ecg_signal, mask, target_length=520):
    x_old = np.linspace(0, 1, len(ecg_signal))
    x_new = np.linspace(0, 1, target_length)
    f_ecg = interp1d(x_old, ecg_signal, kind='linear')
    ecg_resampled = f_ecg(x_new)
    f_mask = interp1d(x_old, mask, kind='nearest')
    mask_resampled = f_mask(x_new)
    return ecg_resampled, mask_resampled.astype(int)

# (find_p_t_peaks remains the same)
def find_p_t_peaks(ecg, mask, boundary_margin=3, merge_distance=15):
    ecg = np.array(ecg)
    mask = np.array(mask)
    def fix_1_2_confusions(mask):
        mask = mask.copy()
        i = 1
        while i < len(mask) - 1:
            if mask[i] in [1, 2] and mask[i - 1] == mask[i + 1] and mask[i] != mask[i - 1]:
                val_to_fill = mask[i - 1]
                start = i
                while i < len(mask) - 1 and mask[i] != val_to_fill and mask[i] in [1, 2]:
                    i += 1
                mask[start:i] = val_to_fill
            else:
                i += 1
        return mask
    
    def selective_majority_filter(mask, window_size=7):
        padded = np.pad(mask, (window_size // 2,), mode='edge')
        filtered = mask.copy()
        for i in range(len(mask)):
            window = padded[i:i + window_size]
            center = mask[i]
            window_mode = mode(window, keepdims=True)[0][0]
            if center == 0 and window_mode in [1, 2]:
                filtered[i] = window_mode
        return filtered
    
    def suppress_short_regions(mask, min_length=2):
        mask = mask.copy()
        current_val = mask[0]
        start_idx = 0
        for i in range(1, len(mask)):
            if mask[i] != current_val:
                if current_val in [1, 2] and (i - start_idx) < min_length:
                    mask[start_idx:i] = 0
                start_idx = i
                current_val = mask[i]
        if current_val in [1, 2] and (len(mask) - start_idx) < min_length:
            mask[start_idx:] = 0
        return mask
    
    def get_peak_indices(mask_val, ecg, mask, max_one=False):
        indices = []
        regions = []
        in_region = False
        start = 0
        for i in range(len(mask)):
            if mask[i] == mask_val and not in_region:
                start = i
                in_region = True
            elif mask[i] != mask_val and in_region:
                end = i
                regions.append((start, end))
                in_region = False
        if in_region:
            regions.append((start, len(mask)))
        if max_one and regions:
            max_len = max(end - start for start, end in regions)
            longest_regions = [seg for seg in regions if (seg[1] - seg[0]) == max_len]
            if len(longest_regions) > 1:
                abs_vals = [np.max(np.abs(ecg[start:end])) for start, end in longest_regions]
                chosen_region = longest_regions[np.argmax(abs_vals)]
            else:
                chosen_region = longest_regions[0]
            regions = [chosen_region]
        for start, end in regions:
            segment = ecg[start:end]
            maxima = argrelextrema(segment, np.greater)[0]
            inverted = False
            if len(maxima) == 0:
                maxima = argrelextrema(-segment, np.greater)[0]
                inverted = True
            if len(maxima) > 0:
                candidate_values = segment[maxima] if not inverted else -segment[maxima]
                best_idx = np.argmax(candidate_values)
                peak_relative = maxima[best_idx]
            else:
                derivative = np.gradient(segment)
                curvature = np.abs(np.gradient(derivative))
                peak_relative = np.argmax(curvature)
            peak_idx = start + peak_relative
            if boundary_margin <= peak_idx < len(ecg) - boundary_margin:
                indices.append(peak_idx)
        return indices
    
    def merge_close_peaks(peaks, ecg, merge_distance):
        if not peaks:
            return []
        peaks = sorted(peaks)
        merged_peaks = [peaks[0]]
        for idx in peaks[1:]:
            last_idx = merged_peaks[-1]
            if abs(idx - last_idx) < merge_distance:
                if abs(ecg[idx]) > abs(ecg[last_idx]):
                    merged_peaks[-1] = idx
            else:
                merged_peaks.append(idx)
        return merged_peaks
    
    def remove_peaks_near_other(peaks_to_filter, reference_peaks, merge_distance):
        filtered = []
        for p_idx in peaks_to_filter:
            if all(abs(p_idx - t_idx) >= merge_distance for t_idx in reference_peaks):
                filtered.append(p_idx)
        return filtered
    
    def refine_peak_positions(ecg, peak_indices, window=10):
        refined = []
        for idx in peak_indices:
            temp_seg = ecg[max(idx - 2, 0):min(idx + 2, len(ecg))]
            temp_idx = idx - 2 + np.argmax(np.abs(temp_seg))
            temp_max = idx - 2 + np.argmax(temp_seg)
            temp_min = idx - 2 + np.argmin(temp_seg)
            if idx != temp_idx and (idx != temp_max and idx != temp_min):
                start = max(idx - window, 0)
                end = min(idx + window + 1, len(ecg))
                segment = np.abs(ecg[start:end])
                maxima = argrelextrema(segment, np.greater)[0]
                inverted = False
                if len(maxima) == 0:
                    maxima = argrelextrema(-segment, np.greater)[0]
                    inverted = True
                if len(maxima) > 0:
                    candidate_values = segment[maxima] if not inverted else -segment[maxima]
                    best_idx = np.argmax(candidate_values)
                    peak_relative = maxima[best_idx]
                else:
                    derivative = np.gradient(segment)
                    curvature = np.abs(np.gradient(derivative))
                    peak_relative = np.argmax(curvature)
                peak_idx = start + peak_relative
                refined.append(peak_idx)
            else:
                refined.append(idx)
        return refined
   
    mask = fix_1_2_confusions(mask)
    mask = selective_majority_filter(mask, window_size=16)
    mask = suppress_short_regions(mask, min_length=3)
    t_peaks = get_peak_indices(mask_val=1, ecg=ecg, mask=mask, max_one=True)
    t_peaks = refine_peak_positions(ecg, t_peaks, window=10)
    t_peaks = merge_close_peaks(t_peaks, ecg, merge_distance=merge_distance)
    p_peaks = get_peak_indices(mask_val=2, ecg=ecg, mask=mask, max_one=False)
    p_peaks = merge_close_peaks(p_peaks, ecg, merge_distance=45)
    p_peaks = refine_peak_positions(ecg, p_peaks, window=10)
    p_peaks = remove_peaks_near_other(p_peaks, t_peaks, merge_distance=merge_distance)
    return p_peaks, t_peaks

def find_onset_offset(signal, peak_idx, smooth=True, window_size=11, polyorder=3,
                      min_drop_ratio=0.2, search_window=200):
    signal = np.array(signal)
    signal_len = len(signal)
    if smooth:
        win = min(window_size, signal_len - (signal_len % 2 == 0))
        signal_smooth = savgol_filter(signal, window_length=win, polyorder=polyorder)
    else:
        signal_smooth = signal
    peak_val = signal_smooth[peak_idx]
    baseline_window = min(40, signal_len // 6)
    start = max(0, peak_idx - baseline_window)
    end = min(signal_len, peak_idx + baseline_window)
    local_baseline = np.median(signal_smooth[start:end])
    drop_thresh = peak_val - (peak_val - local_baseline) * min_drop_ratio
    onset_idx = peak_idx
    for i in range(peak_idx, max(1, peak_idx - search_window), -1):
        if signal_smooth[i] < drop_thresh:
            onset_idx = i
            break
        if i > 1 and signal_smooth[i-1] < signal_smooth[i-2] and signal_smooth[i-1] < signal_smooth[i]:
            onset_idx = i - 1
            break
    offset_idx = peak_idx
    for i in range(peak_idx, min(signal_len - 2, peak_idx + search_window)):
        if signal_smooth[i] < drop_thresh:
            offset_idx = i
            break
        if signal_smooth[i+1] < signal_smooth[i] and signal_smooth[i+1] < signal_smooth[i+2]:
            offset_idx = i + 1
            break
    return onset_idx, offset_idx

def get_pt_peaks(ecg, r_indices, pt_model_path, use_gpu_delegate=True):
    t_peaks_all, p_peaks_all, pt_peaks_all, onset, offset = [], [], [], [], []
    for i in range(len(r_indices) - 1):
        segment = ecg[r_indices[i]:r_indices[i+1]]
        if len(segment) < 10:
            continue
        segment_signal = np.array(segment)
        resampled_ecgs = resample_ecg(segment_signal, 520)
        ecg_signal = np.array(resampled_ecgs)
        ecg_signal = np.expand_dims(ecg_signal, axis=(0, -1))
        predictions = predict_tflite_model(pt_model_path, ecg_signal, use_gpu_delegate=use_gpu_delegate)
        predicted_labels = np.argmax(predictions, axis=-1)
        _, pred_mask = restore_org_ecg_mask(
            ecg_signal[0].squeeze(), predicted_labels.squeeze(), len(segment_signal)
        )
        p_peaks, t_peaks = find_p_t_peaks(segment_signal, pred_mask)
        p_peaks = np.atleast_1d(p_peaks) + r_indices[i]
        t_peaks = np.atleast_1d(t_peaks) + r_indices[i]
        pt_peaks = tuple(list(t_peaks) + list(p_peaks))
        p_peaks_all.extend(p_peaks)
        t_peaks_all.extend(t_peaks)
        pt_peaks_all.extend(pt_peaks)
    return t_peaks_all, p_peaks_all, pt_peaks_all


def pt_peak_detection(all_lead_data, is_lead, r_peaks, r_result_dic = None, pt_model_path=None, use_gpu_delegate=True):
    result_dic = {}
    for lead in all_lead_data.keys():
        r_peaks = r_result_dic.get(lead)
        ecg_signal = all_lead_data[lead].values.flatten()
        baseline_signal = baseline_construction_200(ecg_signal, kernel_size=131)
        lowpass_signal = lowpass(baseline_signal)
        signal_normalized = normalize(lowpass_signal)
        t_peaks, p_peaks, rr_invl_peaks = get_pt_peaks(signal_normalized, r_peaks, pt_model_path, use_gpu_delegate=use_gpu_delegate)
        result_dic[lead] = {"p": p_peaks, "t": t_peaks, "comb": rr_invl_peaks}
    if is_lead == '2_lead':
        p_peaks = result_dic['II'].get("p")
        t_peaks = result_dic['II'].get("t")
        rr_invl_peaks = result_dic['II'].get("comb")

    return t_peaks, p_peaks, rr_invl_peaks, result_dic

# ---------------------- Plotting and other post processing functions ----------------------
def hr_count(ecg_signal, r_index, fs=200):
    if len(r_index) != 0:
        rr_intervals = np.diff(r_index)
        interval_mm = [(rr / fs) * 1000 for rr in rr_intervals]
        if interval_mm:
            HR = int((len(interval_mm) * 60000) / sum(interval_mm))
            return HR
    return 0

def detect_q_s(signal, r_peaks, polarity, fs):
    # If no R-peaks, skip detection
    if r_peaks is None or len(r_peaks) == 0:
        return [None]*0, [None]*0

    q_points = []
    s_points = []

    for r, pol in zip(r_peaks, polarity):

        # Q search: 40 ms before R
        q_start = max(0, r - int(0.04 * fs))
        q_seg = signal[q_start:r]

        if len(q_seg) == 0:
            q_points.append(None)
        else:
            q_idx = np.argmin(q_seg) if pol > 0 else np.argmax(q_seg)
            q_points.append(q_start + q_idx)

        # S search: 60 ms after R
        s_end = min(len(signal), r + int(0.06 * fs))
        s_seg = signal[r:s_end]

        if len(s_seg) == 0:
            s_points.append(None)
        else:
            s_idx = np.argmin(s_seg) if pol > 0 else np.argmax(s_seg)
            s_points.append(r + s_idx)

    return q_points, s_points


def detect_t_wave_onset(signal, s_points, fs):
    # If no S-points, skip T-wave detection
    if s_points is None or len(s_points) == 0:
        return [None]*0

    t_onsets = []

    for s in s_points:
        if s is None:
            t_onsets.append(None)
            continue

        start = s + int(0.08 * fs)
        end = min(len(signal), s + int(0.4 * fs))

        if start >= end:
            t_onsets.append(None)
            continue

        seg = signal[start:end]
        slope = np.diff(seg)

        idx = np.argmax(np.abs(slope))
        t_onsets.append(start + idx)

    return t_onsets


def plot_st_segment(
    all_leads_data,
    st_records_dict,
    r_peaks_dict,
    q_points_dict,
    s_points_dict,
    t_points_dict,
    fs=200,
    save_path=".",
    fname_prefix="ecg",
    x_range_sec=10.0,
    mm_per_mV=10.0
):
    SMALL_BOX_SEC = 0.04
    SMALL_BOX_MV  = 0.1
    Y_MV_RANGE    = 4.0
    Y_SQUARES     = int(Y_MV_RANGE / SMALL_BOX_MV)

    os.makedirs(save_path, exist_ok=True)


    for lead, signal_mV in all_leads_data.items():
        signal_mV = np.asarray(signal_mV, float)
        N = len(signal_mV)
        
        sig_min = signal_mV.min()
        sig_max = signal_mV.max()
        signal_scaled_mV = (signal_mV - sig_min) / (sig_max - sig_min) * Y_MV_RANGE
        signal_sq = signal_scaled_mV / SMALL_BOX_MV
        r_peaks = np.asarray(r_peaks_dict.get(lead, []), int)
        q_pts   = q_points_dict.get(lead, [])
        s_pts   = s_points_dict.get(lead, [])
        t_pts   = t_points_dict.get(lead, [])

        st_records_dict[lead] = []

        samples_per_chunk = int(fs * x_range_sec)
        total_chunks = int(np.ceil(N / samples_per_chunk))

        for c in range(total_chunks):
            start = c * samples_per_chunk
            end   = min(start + samples_per_chunk, N)

            fig, ax = plt.subplots(figsize=(15, 4))
            x_max_sq = int(x_range_sec / SMALL_BOX_SEC)
            ax.set_xlim(0, x_max_sq)
            ax.set_ylim(0, Y_SQUARES)
            ax.set_aspect("equal")

            # --- GRID ---
            for x in range(x_max_sq + 1):
                ax.axvline(x, color="#f4d2d8", lw=0.4)
            for x in range(0, x_max_sq + 1, 5):
                ax.axvline(x, color="#f58181", lw=1.0)
            for y in range(Y_SQUARES + 1):
                ax.axhline(y, color="#f4d2d8", lw=0.4)
            for y in range(0, Y_SQUARES + 1, 5):
                ax.axhline(y, color="#f58181", lw=1.0)

            # --- ISOELECTRIC LINE (TP) ---
            iso_vals = []
            for i in range(len(r_peaks) - 1):
                if i >= len(t_pts): continue

                t_end = t_pts[i]
                nxt_r = r_peaks[i + 1]

                tp_start = t_end + int(0.02 * fs)
                tp_end   = min(tp_start + int(0.04 * fs), nxt_r)

                if tp_end > tp_start:
                    iso_vals.append(signal_mV[tp_start:tp_end].mean()) 

            iso_mV = np.median(iso_vals) if iso_vals else 0.0
            iso_sq = (iso_mV - sig_min) / (sig_max - sig_min) * Y_MV_RANGE / SMALL_BOX_MV
            ax.hlines(iso_sq, 0, x_max_sq, color="blue", lw=1.2, ls="--")

            # --- ECG TRACE ---
            xs = np.arange(start, end)
            x_plot = (xs - start) / fs / SMALL_BOX_SEC
            y_plot = signal_sq[start:end]
            ax.plot(x_plot, y_plot, color="black", lw=1.2)

            # --- Q/R/S/T MARKERS ---
            def mark(idx, color):
                if idx is None or idx < start or idx >= end:
                    return
                x = (idx - start) / fs / SMALL_BOX_SEC
                y = signal_sq[idx]
                ax.plot(x, y, "o", color=color, ms=4)

            for i, r in enumerate(r_peaks):
                mark(r, "red")
                if i < len(q_pts): mark(q_pts[i], "purple")
                if i < len(s_pts): mark(s_pts[i], "orange")
                if i < len(t_pts): mark(t_pts[i], "green")

            # --- ST SEGMENT SHADING ---
            for i, r in enumerate(r_peaks[:-1]):
                if i >= len(s_pts) or i >= len(t_pts):
                    continue

                s = s_pts[i]
                t_on = t_pts[i]
                j = s + int(0.02 * fs)  
                if j < start or t_on >= end:
                    continue

                y_st = signal_sq[j:t_on]
                st_sq = np.mean(y_st - iso_sq)
                st_mm = st_sq

                rect_width_sq = int(0.2 / SMALL_BOX_SEC)
                x_rect_start = (j - start) / fs / SMALL_BOX_SEC
                x_rect_end = x_rect_start + rect_width_sq

                # COLOR DECISION (correct)
                fill_color = "red" if abs(st_mm) > 2 else "green"

                ax.fill_between(
                    [x_rect_start, x_rect_end],
                    iso_sq,
                    iso_sq + st_sq,
                    color=fill_color,
                    alpha=0.35
                )

                y_label = iso_sq + st_sq + (1 if st_sq > 0 else -1)
                ax.text(
                    x_rect_start + rect_width_sq / 2,
                    y_label,
                    f"{st_mm:+.1f} mm",
                    ha="center",
                    fontsize=9,
                    fontweight="bold",
                    color=fill_color
                )

            # --- AXES ---
            ax.set_xticks(np.arange(0, x_max_sq + 1, 25))
            ax.set_xticklabels((np.arange(0, x_max_sq + 1, 25) * SMALL_BOX_SEC).round(2))
            ax.set_xlabel("Time (seconds)")

            ax.set_yticks(np.arange(0, Y_SQUARES + 1, 10))
            ax.set_yticklabels((np.arange(0, Y_SQUARES + 1, 10) * SMALL_BOX_MV).round(1))
            ax.set_ylabel("Amplitude (mV)")

            plt.tight_layout(pad=0.5)
            plt.savefig(
                os.path.join(save_path, f"{fname_prefix}_{lead}_chunk_{c+1}.pdf"),
                dpi=300,
                bbox_inches="tight"
            )
            plt.close(fig)

# ---------------------- I/O / file handling helpers ----------------------
def load_and_rename_data(fn, is_lead_for):
    lead_columns = {
        '2_lead': ['ECG', 'II', 'Value',"'MLII'",'MLII'],
        '7_lead': ['I', 'II', 'III', 'aVR', 'aVL', 'aVF', 'v5'],
        '12_lead': ['I', 'II', 'III', 'aVR', 'aVL', 'aVF', 'v1', 'v2', 'v3', 'v4', 'v5', 'v6','V1','V2','V3','V4','V5','V6','ECG']
    }

    lead_columns_index = {
        '2_lead': {0: 'II'},
        '7_lead': {0: 'I', 1: 'II', 2: 'III', 3: 'aVR', 4: 'aVL', 5: 'aVF', 6: 'v5'},
        '12_lead': {0: 'I', 1: 'II', 2: 'III', 3: 'aVR', 4: 'aVL', 5: 'aVF', 6: 'v1', 7: 'v2', 8: 'v3', 9: 'v4', 10: 'v5', 11: 'v6'}
    }

    all_lead_data = pd.read_csv(fn).fillna(0)
    columns = all_lead_data.columns.tolist()
    if any(str(val).isalpha() for val in all_lead_data.iloc[0, :].values):
        if all(col in lead_columns['7_lead'] for col in columns):
            is_lead_for = '7_lead'
        elif all(col in lead_columns['12_lead'] for col in columns):
            is_lead_for = '12_lead'
        else:
            is_lead_for = '2_lead'
    else:
        if len(columns) >= 12:
            is_lead_for = '12_lead'
        elif len(columns) >= 7:
            is_lead_for = '7_lead'
        else:
            is_lead_for = '2_lead'

    if is_lead_for == '2_lead':
        available_columns = [col for col in lead_columns['2_lead'] if col in columns]
        all_lead_data = attempt_column_load(fn, available_columns)
    elif is_lead_for == '7_lead':
        available_columns = [col for col in lead_columns['7_lead'] if col in columns]
        all_lead_data = attempt_column_load(fn, available_columns)
    elif is_lead_for == '12_lead':
        available_columns = [col for col in lead_columns['12_lead'] if col in columns]
        all_lead_data = attempt_column_load(fn, available_columns)

    if all_lead_data is not None:
        all_lead_data = all_lead_data.rename(columns=lead_columns_index[is_lead_for])

    if is_lead_for == '2_lead':
        all_lead_data.columns = ['II']

    return all_lead_data, is_lead_for

def attempt_column_load(fn, columns):
    try:
        data = pd.read_csv(fn, usecols=columns).fillna(0)
        return data
    except ValueError as e:
        print("value Error ",e)
        return None
    except Exception as e:
        print("Error in Loading",e)
        return None

def find_csv_files(root_folder):
    csv_files = []
    for root, _, files in os.walk(root_folder):
        for file in files:
            if file.lower().endswith('.csv'):
                csv_files.append(os.path.join(root, file))
    return csv_files

# ------------------------- MERGE LEAD CHUNKS -------------------------
def merge_lead_chunks_to_pdf(save_path, fname_prefix, lead):
    merger = PdfMerger()
    pdf_files = sorted([f for f in os.listdir(save_path)
                        if f.startswith(f"{fname_prefix}_{lead}_chunk_") and f.endswith(".pdf")])
    if not pdf_files:
        return None
    for f in pdf_files:
        merger.append(os.path.join(save_path, f))
    merged_pdf = os.path.join(save_path, f"{fname_prefix}_{lead}_MERGED.pdf")
    merger.write(merged_pdf)
    merger.close()
    return merged_pdf


def process_single_file(
    fn,
    save_path,
    is_lead,
    r_model_path,
    pt_model_path,
    use_gpu_delegate=True,
    fs=200,
    mm_per_mV=10.0,
    x_range_sec=10.0
):
    local_name = os.path.splitext(os.path.basename(fn))[0]
    csv_root = os.path.join(save_path, local_name)
    os.makedirs(csv_root, exist_ok=True)

    all_leads_data, is_lead = load_and_rename_data(fn, is_lead)
    if all_leads_data is None:
        return f"Failed: {local_name}"

    # --- R PEAK DETECTION ---
    _, r_result_dic = r_peak_detection(
        all_leads_data, is_lead, r_model_path, use_gpu_delegate
    )

    # --- P/T MODEL (kept, but T will be gated later) ---
    _, _, _, pt_result_dic = pt_peak_detection(
        all_leads_data, is_lead, None, r_result_dic, pt_model_path, use_gpu_delegate
    )
    
    baseline_corrected_data = pd.DataFrame({
        lead: baseline_construction_200(all_leads_data[lead].values)
        for lead in all_leads_data.columns
    })

    st_records_dict = {}
    q_points_dict = {}
    s_points_dict = {}
    t_points_dict = {}
    r_peaks_dict = {}

    # --- PER LEAD PROCESSING ---
    for lead in baseline_corrected_data.columns:
        detection_signal = all_leads_data[lead].values.astype(float)

        r_peaks = np.asarray(r_result_dic.get(lead, []), dtype=int)
        r_peaks_dict[lead] = r_peaks

        # DEFAULT: NOTHING COMPUTED
        q_points, s_points, t_points = [], [], []

        # HARD RULE: no R → no Q/S/T
        if len(r_peaks) >= 2:
            # Q & S depend ONLY on R
            q_points, s_points = detect_q_s(
                detection_signal,
                r_peaks,
                np.sign(detection_signal[r_peaks]),
                fs
            )

            # T onset depends on S
            if len(s_points) > 0:
                t_points = detect_t_wave_onset(
                    detection_signal,
                    s_points,
                    fs
                )

        # Store results (may be empty → plotting will skip)
        q_points_dict[lead] = q_points
        s_points_dict[lead] = s_points
        t_points_dict[lead] = t_points

    # --- PLOTTING + ST COMPUTATION (already guarded) ---
    plot_st_segment(
        all_leads_data,
        st_records_dict,
        r_peaks_dict,
        q_points_dict,
        s_points_dict,
        t_points_dict,
        fs=fs,
        save_path=csv_root,
        fname_prefix=local_name,
        x_range_sec=x_range_sec,
        mm_per_mV=mm_per_mV
    )

    # --- MERGE PDFs ---
    merged_pdfs = []
    for lead in all_leads_data.columns:
        merged_pdf = merge_lead_chunks_to_pdf(csv_root, local_name, lead)
        if merged_pdf:
            merged_pdfs.append(merged_pdf)

    if merged_pdfs:
        final_pdf = os.path.join(csv_root, f"{local_name}_MERGED_ALL_LEADS.pdf")
        merger = PdfMerger()
        for pdf in merged_pdfs:
            merger.append(pdf)
        merger.write(final_pdf)
        merger.close()

    # --- ST SUMMARY CSV ---
    st_summary_records = []
    for lead, recs in st_records_dict.items():
        for rec in recs:
            st_summary_records.append({
                "file": local_name,
                "lead": lead,
                "r_index": rec["r_idx"],
                "pr_baseline_mV": rec["pr_baseline_mV"],
                "st_mV": rec["st_mV"],
                "st_mm": rec["st_mm"]
            })

    if st_summary_records:
        summary_df = pd.DataFrame(st_summary_records)
        summary_df.to_csv(
            os.path.join(csv_root, f"{local_name}_ALL_LEADS_ST_SUMMARY.csv"),
            index=False
        )

    return f"Processed: {local_name}"

def merge_all_merged_pdfs(root_output_dir, final_pdf_path):
    merger = PdfMerger()
    found_any = False
    for root, _, files in os.walk(root_output_dir):
        for file in sorted(files):
            if file.endswith("_MERGED_ALL_LEADS.pdf"):
                pdf_path = os.path.join(root, file)
                merger.append(pdf_path)
                found_any = True
    if found_any:
        merger.write(final_pdf_path)
        merger.close()
        print(f"FINAL MERGED PDF CREATED: {final_pdf_path}")
    else:
        merger.close()
        print("No merged PDFs found to combine.")


def ecg_processing(
    path,
    save_path,
    is_lead,
    r_model_path,
    pt_model_path,
    max_workers=DESIRED_CPU_THREADS,
    use_gpu_delegate=True,
    use_multiprocessing=False
):
    csv_files = find_csv_files(path)
    if not csv_files:
        print("No CSV files found.")
        return

    max_workers = min(max_workers, max(1, os.cpu_count() or 1))
    mode = "multiprocessing" if use_multiprocessing else "threading"
    print(f"🔹 Processing {len(csv_files)} files with {max_workers} workers ({mode})")

    if use_multiprocessing:
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(process_single_file, fn, save_path, is_lead, r_model_path, pt_model_path, use_gpu_delegate): fn for fn in csv_files}
            for future in as_completed(futures):
                try:
                    print(future.result())
                except Exception as e:
                    print("Worker exception:", e)
    else:
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(process_single_file, fn, save_path, is_lead, r_model_path, pt_model_path, use_gpu_delegate): fn for fn in csv_files}
            for future in as_completed(futures):
                try:
                    print(future.result())
                except Exception as e:
                    print("Worker exception:", e)

    final_pdf_path = os.path.join(save_path, "FINAL_ALL_CSV_ALL_LEADS.pdf")
    merge_all_merged_pdfs(save_path, final_pdf_path)
    print("\nECG 2-Lead ST Segment Processing Completed Successfully")

 
def save_pdf_to_gridfs(pdf_path, metadata=None):
    mongo_uri = os.getenv("MONGO_HOST")
    mongo_client = MongoClient(mongo_uri)
    db = mongo_client["St_Segment"]

    fs = gridfs.GridFS(db)

    with open(pdf_path, "rb") as f:
        file_id = fs.put(
            f,
            filename=pdf_path.split("\\")[-1],
            contentType="application/pdf",
            metadata=metadata or {}
        )

    return str(file_id)

def run_ecg_st_pipeline(
    input_folder,
    output_folder,
    is_lead,
    max_workers=4,
    use_gpu_delegate=True,
    use_multiprocessing=True
):
    os.makedirs(output_folder, exist_ok=True)

    ecg_processing(
        path=input_folder,
        save_path=output_folder,
        is_lead=is_lead,
        r_model_path = os.getenv("rnn_model"),
        pt_model_path = os.getenv("ecg_pt_detection"),
        max_workers=max_workers,
        use_gpu_delegate=use_gpu_delegate,
        use_multiprocessing=use_multiprocessing
    )

    final_pdf_path = os.path.join(
        output_folder,
        "FINAL_ALL_CSV_ALL_LEADS.pdf"
    )

    merge_all_merged_pdfs(
        root_output_dir=output_folder,
        final_pdf_path=final_pdf_path
    )

    return final_pdf_path
