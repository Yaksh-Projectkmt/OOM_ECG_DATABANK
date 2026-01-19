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

def baseline_construction_200(ecg_signal, kernel_size=131):
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
            selected.append(idx)
    return sorted(selected)




def check_model_r(ecg_data, r_model_path, use_gpu_delegate=True):
    totaldata = len(ecg_data)
    i = 0
    step = totaldata if totaldata < 1000 else 1000
    r_peaks = []
    temp_list = []
    df_ecg_signal = ecg_data.tolist()
    while i < totaldata:
        if i != 0 and totaldata > 1000:
            i -= 200
        ecg_signal = ecg_data[i:i + step]
        signal_len = len(ecg_signal)
        pad_len = 1000 - signal_len
        padded_signal = np.pad(ecg_signal, (0, pad_len), mode='constant', constant_values=0)
        raw_array = np.expand_dims(padded_signal, axis=0).astype(np.float32)[..., np.newaxis]
        preds = predict_tflite_model(r_model_path, raw_array, use_gpu_delegate=use_gpu_delegate)
        preds = preds[:signal_len]
        r_peak_prob = preds[:, 1]
        peak_indices, _ = find_peaks(r_peak_prob, height=0.2, distance=20)
        for j in peak_indices:
            if 0 <= i+j < len(df_ecg_signal):
                temp_list.append(i + j)
        i += step
    rpeak = sorted(set(temp_list))
    r_peaks = refined_non_max_suppression(df_ecg_signal, rpeak)
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
def add_standard_ecg_grid(ax, duration_sec, y_min, y_max):
    
    # ---- X axis (Time) ----
    ax.set_xlim(0, duration_sec)
    ax.set_xticks(np.arange(0, duration_sec + 0.04, 0.04), minor=True)
    ax.set_xticks(np.arange(0, duration_sec + 0.2, 0.2))
    # ---- Y axis (Voltage) ----
    ax.set_ylim(y_min, y_max)
    ax.set_yticks(np.arange(y_min, y_max + 0.1, 0.1), minor=True)
    ax.set_yticks(np.arange(y_min, y_max + 0.5, 0.5))
    # ---- Aspect ratio ----
    ax.set_aspect(0.04 / 0.1) # 0.04 sec = 0.1 mV
    # ---- Grid styling ----
    ax.grid(which='minor', color='#e6e6e6', linewidth=0.6)
    ax.grid(which='major', color='#b00000', linewidth=1.2)
    # Draw background over full canvas
    ax.set_facecolor('white')

# ---------------------- Plotting and other post processing functions ----------------------
def hr_count(r_index, fs=250):
    if len(r_index) < 2:
        return 0
    rr_intervals = np.diff(r_index)
    if len(rr_intervals) == 0:
        return 0
    HR = int((len(rr_intervals) * 60000) / np.sum(rr_intervals / fs * 1000))
    return HR

def refine_r_peaks(signal, r_peaks, fs):

    refined = []
    polarity = []

    window = int(0.04 * fs)  # Â±40 ms around detected R

    for r in r_peaks:
        if r <= 0 or r >= len(signal):
            continue

        start = max(0, r - window)
        end = min(len(signal), r + window + 1)
        seg = signal[start:end]

        if len(seg) == 0:
            continue

        # pick the true apex of QRS (largest absolute deflection)
        idx = np.argmax(np.abs(seg))
        true_r = start + idx

        refined.append(true_r)
        polarity.append(np.sign(signal[true_r]) or 1)

    return np.array(refined, dtype=int), np.array(polarity, dtype=int)


def detect_q_s(signal, r_peaks, r_polarity, fs):
    
    q_points = []
    s_points = []
    window = int(0.1 * fs)  # 100 ms search window

    for r, pol in zip(r_peaks, r_polarity):
        # Q point: before R
        start = max(0, r - window)
        seg = signal[start:r+1]
        if pol > 0:
            q_idx = start + np.argmin(seg)  # Q = min before upright R
        else:
            q_idx = start + np.argmax(seg)  # Q = max before inverted R
        q_points.append(q_idx)

        # S point: after R
        end = min(len(signal), r + window + 1)
        seg = signal[r:end]
        if pol > 0:
            s_idx = r + np.argmin(seg)  # S = min after upright R
        else:
            s_idx = r + np.argmax(seg)  # S = max after inverted R
        s_points.append(s_idx)

    return np.array(q_points, dtype=int), np.array(s_points, dtype=int)



def detect_t_wave_onset(signal, s_points, fs):
    """
    Detect T-wave onset AFTER ST segment.
    NOT used for ST measurement.
    """

    signal = np.asarray(signal)
    t_onsets = []

    for s_idx in s_points:
        j_point = int(s_idx)

        search_start = j_point + int(0.08 * fs)
        search_end   = min(j_point + int(0.30 * fs), len(signal) - 2)

        if search_end <= search_start:
            t_onsets.append(search_start)
            continue

        slope = np.diff(signal)
        onset = search_end

        for i in range(search_start, search_end - 1):
            if slope[i] > 0 and slope[i+1] > 0:
                onset = i
                break

        t_onsets.append(onset)

    return np.array(t_onsets, dtype=int)

def measure_st_segment_full(signal, r_peaks, s_points, t_onsets, fs, calibration=10.0):
    import numpy as np

    st_results = []
    signal_len = len(signal)
    beats = min(len(r_peaks), len(s_points), len(t_onsets))

    for i in range(beats):
        r = int(r_peaks[i])
        s = int(s_points[i])
        t_idx = int(t_onsets[i])

        r = np.clip(r, 0, signal_len - 1)
        s = np.clip(s, 0, signal_len - 1)
        t_idx = np.clip(t_idx, 0, signal_len - 1)

        # J-point â‰ˆ 10 ms after S
        j_point = s + int(0.01 * fs)
        j_point = np.clip(j_point, 0, signal_len - 1)

        # PR baseline (âˆ’200 to âˆ’120 ms before R)
        pr_start = max(r - int(0.20 * fs), 0)
        pr_end   = max(r - int(0.12 * fs), pr_start + 1)
        pr_baseline = np.median(signal[pr_start:pr_end])

        st_end = min(t_idx, signal_len - 1)
        if st_end <= j_point:
            st_end = j_point  # prevent negative slicing

        st_mV = np.median(signal[j_point:st_end + 1]) - pr_baseline

        st_results.append({
            "j_idx": j_point,
            "st_end": st_end,
            "st_mV": st_mV,
            "pr_baseline": pr_baseline
        })

    return st_results


# def plotting(
#     baseline_corrected_data,
#     save_path,
#     local_name,
#     pt_result_dic,
#     r_result_dic,
#     fs=250,
#     calibration=10.0
# ):
#     import os
#     import numpy as np
#     import matplotlib.pyplot as plt
#     import matplotlib.patheffects as path_effects

#     os.makedirs(save_path, exist_ok=True)
#     mm_per_mV = calibration
#     summary = []

#     for lead in baseline_corrected_data.columns:

#         fig, ax = plt.subplots(figsize=(22, 9))

#         # ---------------- SIGNAL ----------------
#         signal = baseline_corrected_data[lead].values.astype(float)
#         signal_len = len(signal)
#         t = np.arange(signal_len) / fs

#         # ---------- FULL SIGNAL ----------
#         ax.plot(t, signal, color="black", lw=1.3, zorder=1)

#         # ---------------- R PEAKS ----------------
#         r_peaks = np.asarray(r_result_dic.get(lead, []), dtype=int)
#         if len(r_peaks) == 0:
#             plt.close(fig)
#             continue

#         r_peaks, r_polarity = refine_r_peaks(signal, r_peaks, fs)

#         # ---------------- Q & S ----------------
#         q_points, s_points = detect_q_s(signal, r_peaks, r_polarity, fs)

#         # ---------------- T ONSET ----------------
#         t_onsets = detect_t_wave_onset(signal, s_points, fs)

#         # ---------------- MARKERS ----------------
#         ax.scatter(t[np.clip(r_peaks, 0, signal_len - 1)], signal[np.clip(r_peaks, 0, signal_len - 1)],
#                    c="red", s=30, zorder=5, label="R")
#         ax.scatter(t[np.clip(q_points, 0, signal_len - 1)], signal[np.clip(q_points, 0, signal_len - 1)],
#                    c="purple", s=35, zorder=5, label="Q")
#         ax.scatter(t[np.clip(s_points, 0, signal_len - 1)], signal[np.clip(s_points, 0, signal_len - 1)],
#                    c="orange", s=35, zorder=5, label="S")
#         ax.scatter(t[np.clip(t_onsets, 0, signal_len - 1)], signal[np.clip(t_onsets, 0, signal_len - 1)],
#                    c="green", s=40, zorder=5, label="T-onset")

#         # ---------------- ST MEASUREMENT ----------------
#         st_records = measure_st_segment_full(signal, r_peaks, s_points, t_onsets, fs)
#         if not st_records:
#             plt.close(fig)
#             continue

#         # ---------------- PR BASELINE (GLOBAL) ----------------
#         pr_global = np.median([rec["pr_baseline"] for rec in st_records])
#         ax.axhline(pr_global, color="gray", ls="--", lw=1.2, label="PR baseline")

#         # ---------------- ST SHADING (STRIP STYLE) ----------------
#         for i, rec in enumerate(st_records, start=1):

#             j_idx = np.clip(rec["j_idx"], 0, signal_len - 1)
#             st_idx = np.clip(rec["st_end"], 0, signal_len - 1)
#             if j_idx == st_idx:
#                 st_idx = min(j_idx + 1, signal_len - 1)

#             # Horizontal ST segment
#             x_segment = t[j_idx:st_idx+1]
#             y_signal_segment = signal[j_idx:st_idx+1]

#             # PR baseline
#             y_base = rec["pr_baseline"]

#             # Measured ST deviation
#             y_top = y_base + rec["st_mV"]

#             # Clip the ECG signal to stay within ST deviation
#             y_shade = np.clip(y_signal_segment, min(y_base, y_top), max(y_base, y_top))

#             # Color by ST deviation
#             color = "green" if rec["st_mV"] < 2.0 else "red"
#             # color = "red" if rec["st_mV"] > 0 else "green"
#             ax.fill_between(x_segment, y_base, y_shade, color=color, alpha=0.3, zorder=5)

#             # Label above or below the shaded ST deviation
#             label_y = y_top + 0.05 if rec["st_mV"] > 0 else y_top - 0.05
#             va = "bottom" if rec["st_mV"] > 0 else "top"
#             ax.text(
#                 np.mean(x_segment),
#                 label_y,
#                 f"{rec['st_mV']*mm_per_mV:+.1f} mm",
#                 fontsize=9,
#                 fontweight="bold",
#                 ha="center",
#                 va=va,
#                 path_effects=[path_effects.withStroke(linewidth=2, foreground="white")]
#             )

#             summary.append({
#                 "Lead": lead,
#                 "Beat_No": i,
#                 "ST_mm": round(rec["st_mV"]*mm_per_mV, 2),
#                 "PR_baseline_mV": round(rec["pr_baseline"], 4),
#                 "Status": "Normal" if abs(rec["st_mV"]*mm_per_mV) < 2 else "Abnormal"
#             })

#         # ---------- ECG GRID ----------
#         pad = 0.5
#         ax.set_ylim(signal.min() - pad, signal.max() + pad)
#         ax.set_xlim(t[0], t[-1])
#         add_standard_ecg_grid(ax, t[-1], ax.get_ylim()[0], ax.get_ylim()[1])

#         # ---------- TITLE & AXES ----------
#         hr = hr_count(r_peaks, fs)
#         ax.set_title(f"{local_name} â€” Lead {lead} â€” HR {hr} bpm")
#         ax.set_xlabel("Time (s)")
#         ax.set_ylabel("Amplitude (mV)")
#         ax.legend(loc="upper right")

#         # ---------- SAVE FIG ----------
#         plt.savefig(
#             os.path.join(save_path, f"{local_name}_{lead}.pdf"),
#             dpi=300,
#             bbox_inches="tight"
#         )
#         plt.close(fig)

#     return pd.DataFrame(summary)

def plotting(
    baseline_corrected_data,
    save_path,
    local_name,
    pt_result_dic,
    r_result_dic,
    fs=250,
    calibration=10.0
):
    import os
    import numpy as np
    import matplotlib.pyplot as plt
    import matplotlib.patheffects as path_effects

    os.makedirs(save_path, exist_ok=True)
    mm_per_mV = calibration
    summary = []

    for lead in baseline_corrected_data.columns:

        fig, ax = plt.subplots(figsize=(22, 9))

        signal = baseline_corrected_data[lead].values.astype(float)
        signal_len = len(signal)
        t = np.arange(signal_len) / fs

        # ---------- FULL SIGNAL ----------
        ax.plot(t, signal, color="black", lw=1.3, zorder=1)

        # ---------- DETECTIONS ----------
        r_peaks = np.asarray(r_result_dic.get(lead, []), dtype=int)
        if len(r_peaks) == 0:
            plt.close(fig)
            continue

        r_peaks, r_polarity = refine_r_peaks(signal, r_peaks, fs)
        q_points, s_points = detect_q_s(signal, r_peaks, r_polarity, fs)
        t_onsets = detect_t_wave_onset(signal, s_points, fs)

        # ---------- MARKERS ----------
        ax.scatter(t[np.clip(r_peaks, 0, signal_len - 1)], signal[np.clip(r_peaks, 0, signal_len - 1)],
                   c="red", s=30, zorder=5, label="R")
        ax.scatter(t[np.clip(q_points, 0, signal_len - 1)], signal[np.clip(q_points, 0, signal_len - 1)],
                   c="purple", s=35, zorder=5, label="Q")
        ax.scatter(t[np.clip(s_points, 0, signal_len - 1)], signal[np.clip(s_points, 0, signal_len - 1)],
                   c="orange", s=35, zorder=5, label="S")
        ax.scatter(t[np.clip(t_onsets, 0, signal_len - 1)], signal[np.clip(t_onsets, 0, signal_len - 1)],
                   c="green", s=40, zorder=5, label="T-onset")

        # ---------- ST MEASUREMENT ----------
        st_records = measure_st_segment_full(signal, r_peaks, s_points, t_onsets, fs)
        if not st_records:
            plt.close(fig)
            continue

        # Global PR baseline for reference
        pr_global = np.median([rec["pr_baseline"] for rec in st_records])
        ax.axhline(pr_global, color="gray", ls="--", lw=1.2, label="PR baseline")

        # ---------- ST SHADING & LABELS ----------
        # ---------- ST SHADING & LABELS ----------
        for i, rec in enumerate(st_records, start=1):

            j_idx = np.clip(rec["j_idx"], 0, signal_len - 1)
            st_idx = np.clip(rec["st_end"], 0, signal_len - 1)
            if j_idx == st_idx:
                st_idx = min(j_idx + 1, signal_len - 1)

            # ST segment samples
            x_segment = t[j_idx:st_idx + 1]
            y_signal_segment = signal[j_idx:st_idx + 1]

            # PR baseline (mV)
            y_base = rec["pr_baseline"]

            # ST deviation
            st_mV = rec["st_mV"]
            st_mm = st_mV * mm_per_mV   # í ½í´´ EVERYTHING DECIDED IN mm

            y_top = y_base + st_mV

            # Decide color & status based on mm
            if abs(st_mm) > 2.0:
                color = "red"
                status = "Critical Abnormal"
            elif abs(st_mm) > 0.5:
                color = "green"
                status = "Abnormal"
            else:
                color = None
                status = "Normal"

            # Shade ONLY if > 0.5 mm
            if color is not None:
                y_shade = np.clip(
                    y_signal_segment,
                    min(y_base, y_top),
                    max(y_base, y_top)
                )

                ax.fill_between(
                    x_segment,
                    y_base,
                    y_shade,
                    color=color,
                    alpha=0.3,
                    zorder=5
                )

                # Label placement
                label_offset_mV = 0.05
                label_y = y_top + label_offset_mV if st_mm > 0 else y_top - label_offset_mV
                va = "bottom" if st_mm > 0 else "top"

                ax.text(
                    np.mean(x_segment),
                    label_y,
                    f"{st_mm:+.1f} mm",
                    fontsize=9,
                    fontweight="bold",
                    ha="center",
                    va=va,
                    path_effects=[path_effects.withStroke(linewidth=2, foreground="white")]
                )

            summary.append({
                "Lead": lead,
                "Beat_No": i,
                "ST_mm": round(st_mm, 2),
                "PR_baseline_mV": round(y_base, 4),
                "Status": status
            })


        # ---------- ECG GRID ----------
        pad = 0.5
        ax.set_ylim(signal.min() - pad, signal.max() + pad)
        ax.set_xlim(t[0], t[-1])
        add_standard_ecg_grid(ax, t[-1], ax.get_ylim()[0], ax.get_ylim()[1])

        # ---------- TITLE & AXES ----------
        hr = hr_count(r_peaks, fs)
        ax.set_title(f"{local_name} â€” Lead {lead} â€” HR {hr} bpm")
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Amplitude (mV)")
        ax.legend(loc="upper right")

        # ---------- SAVE FIG ----------
        plt.savefig(
            os.path.join(save_path, f"{local_name}_{lead}.pdf"),
            dpi=300,
            bbox_inches="tight"
        )
        plt.close(fig)

    return pd.DataFrame(summary)

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


def process_single_file(
    fn,
    save_path,
    is_lead,
    r_model_path,
    pt_model_path,
    use_gpu_delegate=True
):
    import os
    import pandas as pd

    local_name = os.path.splitext(os.path.basename(fn))[0]
    csv_root = os.path.join(save_path, local_name)
    os.makedirs(csv_root, exist_ok=True)

    all_leads_data, is_lead = load_and_rename_data(fn, is_lead)
    if all_leads_data is None:
        return f"Failed: {local_name}"

    # ---------- R PEAK DETECTION ----------
    _, r_result_dic = r_peak_detection(
        all_leads_data,
        is_lead,
        r_model_path,
        use_gpu_delegate
    )

    # ---------- P & T PEAK DETECTION ----------
    _, _, _, pt_result_dic = pt_peak_detection(
        all_leads_data,
        is_lead,
        None,
        r_result_dic,
        pt_model_path,
        use_gpu_delegate
    )

    # ---------- BASELINE CORRECTED ECG (ONCE) ----------
    baseline_corrected_data = pd.DataFrame({
        lead: baseline_construction_200(all_leads_data[lead].values)
        for lead in all_leads_data.columns
    })

    summary_df = plotting(
        baseline_corrected_data,
        csv_root,
        local_name,
        pt_result_dic,
        r_result_dic,
        fs=250,
        calibration=10.0
    )

    if not summary_df.empty:
        summary_df.to_csv(
            os.path.join(csv_root, f"{local_name}_ALL_LEADS_ST_SUMMARY.csv"),
            index=False
        )

    merge_pdfs_in_lead_order(
        csv_root,
        os.path.join(csv_root, f"{local_name}_MERGED_ALL_LEADS.pdf")
    )

    return f"Processed: {local_name}"



def merge_pdfs_in_lead_order(pdf_dir, output_pdf):
    from PyPDF2 import PdfMerger
    import os

    LEAD_ORDER = [
        "I", "II", "III",
        "aVR", "aVL", "aVF",
        "V1", "V2", "V3", "V4", "V5", "V6"
    ]

    merger = PdfMerger()

    for lead in LEAD_ORDER:
        for file in sorted(os.listdir(pdf_dir)):
            if file.endswith(".pdf") and f"_{lead}.pdf" in file:
                merger.append(os.path.join(pdf_dir, file))
                break

    merger.write(output_pdf)
    merger.close()
def merge_all_merged_pdfs(root_output_dir, final_pdf_path):
    """
    Merge all per-CSV merged ECG PDFs into one final PDF.

    Expects files like:
    root_output_dir/
        patient1/patient1_MERGED_ALL_LEADS.pdf
        patient2/patient2_MERGED_ALL_LEADS.pdf
        ...

    Creates:
        final_pdf_path
    """
    import os
    from PyPDF2 import PdfMerger

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
        print(f" FINAL MERGED PDF CREATED: {final_pdf_path}")
    else:
        merger.close()
        print(" No merged PDFs found to combine.")


def ecg_processing(path, save_path, is_lead, r_model_path, pt_model_path, max_workers=DESIRED_CPU_THREADS, use_gpu_delegate=True):
    csv_files = find_csv_files(path)
    if not csv_files:
        print("No CSV files found.")
        return
    # Use ThreadPoolExecutor so per-thread interpreters remain in same process/threads (GPU delegate may not be picklable)
    max_workers = min(max_workers, max(1, os.cpu_count() or 1))
    print(f"Processing {len(csv_files)} files with {max_workers} workers (use_gpu_delegate={use_gpu_delegate})")
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(process_single_file, fn, save_path, is_lead, r_model_path, pt_model_path, use_gpu_delegate): fn for fn in csv_files}
        for future in as_completed(futures):
            try:
                print(future.result())
            except Exception as e:
                print("Worker exception:", e)
    print("All files processed successfully.")
def find_pdf_files(root_folder):
    pdf_files = []
    for root, _, files in os.walk(root_folder):
        for file in files:
            if file.lower().endswith('.pdf'):
                pdf_files.append(os.path.join(root, file))
    return sorted(pdf_files)
def merge_pdfs(pdf_files, output_path):
    merger = PdfMerger()
    for pdf in pdf_files:
        try:
            merger.append(pdf)
            print(f"Merged: {pdf}")
        except Exception as e:
            print(f"Failed to merge {pdf}: {e}")
    merger.write(output_path)
    merger.close()
    print(f"\n All PDFs merged into: {output_path}")
def process_single_file_wrapper(args):
    """Wrapper for multiprocessing (since functions must be pickleable)."""
    return process_single_file(*args)
def ecg_processing(path, save_path, is_lead, r_model_path, pt_model_path,
                   max_workers=DESIRED_CPU_THREADS, use_gpu_delegate=True,
                   use_multiprocessing=False):
    csv_files = find_csv_files(path)
    if not csv_files:
        print("No CSV files found.")
        return
    max_workers = min(max_workers, max(1, os.cpu_count() or 1))
    mode = "multiprocessing" if use_multiprocessing else "threading"
    print(f"Processing {len(csv_files)} files with {max_workers} workers ({mode}, use_gpu_delegate={use_gpu_delegate})")
    if use_multiprocessing:
        # Each process loads its own interpreters/models
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(process_single_file_wrapper,
                                (fn, save_path, is_lead, r_model_path, pt_model_path, use_gpu_delegate)): fn
                for fn in csv_files
            }
            for future in as_completed(futures):
                try:
                    print(future.result())
                except Exception as e:
                    print("Worker exception:", e)
    else:
        # Default threading (per-thread cached interpreters, GPU delegate works better here)
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(process_single_file, fn, save_path, is_lead, r_model_path, pt_model_path, use_gpu_delegate): fn
                for fn in csv_files
            }
            for future in as_completed(futures):
                try:
                    print(future.result())
                except Exception as e:
                    print("Worker exception:", e)
    print("All files processed successfully.")

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
    use_multiprocessing=False
):
    os.makedirs(output_folder, exist_ok=True)

    ecg_processing(
        path=input_folder,
        save_path=output_folder,
        is_lead=is_lead,
        r_model_path = r"/home/system/ecgdatabank_copy/Scripts_Models/Model/rnn_model1_19_12_Unet.tflite",
        pt_model_path = r"/home/system/ecgdatabank_copy/Scripts_Models/Model/ecg_pt_detection_LSTMGRU_v32.tflite",
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
