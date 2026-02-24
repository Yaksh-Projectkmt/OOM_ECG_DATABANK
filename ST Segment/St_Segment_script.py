import os
import glob
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
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
from scipy.signal import medfilt
warnings.filterwarnings('ignore')

results_lock = threading.RLock()
thread_local = threading.local()

# ---------------------- TFLite interpreter utilities ----------------------
def get_tflite_interpreter_for_thread(model_path: str):
    if not hasattr(thread_local, "interpreters"):
        thread_local.interpreters = {}
    key = model_path
    if key in thread_local.interpreters:
        return thread_local.interpreters[key]
    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()
    input_details  = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    thread_local.interpreters[key] = (interpreter, input_details, output_details)
    return thread_local.interpreters[key]

def predict_tflite_model(model_path: str, input_data):
    interpreter, input_details, output_details = get_tflite_interpreter_for_thread(model_path)
    with results_lock:
        input_data = input_data.astype(np.float32)
        interpreter.set_tensor(input_details[0]['index'], input_data)
        interpreter.invoke()
        output_data = interpreter.get_tensor(output_details[0]['index'])
    return output_data.squeeze()

# ---------------------- Signal processing ----------------------
def lowpass(file, cutoff=0.3):
    b, a = signal.butter(3, cutoff, btype='lowpass', analog=False)
    return signal.filtfilt(b, a, file)

def baseline_construction_200(ecg_signal, kernel_size=131):
    s_corrected = signal.detrend(ecg_signal)
    return s_corrected - signal.medfilt(s_corrected, kernel_size)

def normalize(sig):
    return (sig - np.mean(sig)) / np.std(sig)

# ---------------------- R peak detection utilities ----------------------
def refined_non_max_suppression(ecg_signal, valid_indices, suppression_radius=40):
    if len(valid_indices) == 0:
        return []
    ecg_arr = np.array(ecg_signal)
    sorted_indices = sorted(valid_indices, reverse=True)
    selected = []
    occupied = np.zeros(len(ecg_arr), dtype=bool)
    for idx in sorted_indices:
        if not occupied[idx]:
            left  = max(0, idx - suppression_radius)
            right = min(len(ecg_arr), idx + suppression_radius + 1)
            occupied[left:right] = True
            window_candidates = [i for i in sorted_indices if left <= i < right]
            if window_candidates:
                maximum_idx = max(window_candidates, key=lambda i: abs(ecg_arr[i]))
            else:
                maximum_idx = idx
            selected.append(maximum_idx)
    return sorted(set(selected))

def check_model_r(ecg_data, r_model_path):
    totaldata = len(ecg_data)
    i = 0
    step = 1000 if totaldata >= 1000 else totaldata
    temp_list = []
    while i < totaldata:
        if i != 0 and totaldata > 1000:
            i -= 200
        seg_start   = i
        ecg_segment = ecg_data[i:i + step]
        signal_len  = len(ecg_segment)
        pad_len     = 1000 - signal_len
        padded      = np.pad(ecg_segment, (0, pad_len), mode='constant', constant_values=0)
        raw_array   = np.expand_dims(padded, axis=0).astype(np.float32)[..., np.newaxis]
        preds       = predict_tflite_model(r_model_path, raw_array)
        preds       = preds[:signal_len]
        r_peak_prob = preds[:, 1]
        peak_indices, _ = find_peaks(r_peak_prob, height=0.2, distance=20)
        for peak_rel in peak_indices:
            global_idx = seg_start + int(peak_rel)
            if 0 <= global_idx < totaldata:
                temp_list.append(global_idx)
        i += step
    rpeak   = sorted(set(temp_list))
    r_peaks = refined_non_max_suppression(ecg_data, rpeak)
    return sorted(set(r_peaks))

def r_peak_detection(all_lead_data, is_lead, r_model_path):
    result_dic = {}
    for lead in all_lead_data.keys():
        ecg_signal        = all_lead_data[lead].values.flatten()
        baseline_signal   = baseline_construction_200(ecg_signal, kernel_size=131)
        lowpass_signal    = lowpass(baseline_signal)
        signal_normalized = normalize(lowpass_signal)
        r_peaks           = check_model_r(signal_normalized, r_model_path)
        result_dic[lead]  = r_peaks
    r_peaks = result_dic.get('II', []) if is_lead == '2_lead' else []
    return r_peaks, result_dic

# ---------------------- P & T peak detection utilities ----------------------
def resample_ecg(ecg_signal, target_length=520):
    x_old = np.linspace(0, 1, len(ecg_signal))
    x_new = np.linspace(0, 1, target_length)
    return interp1d(x_old, ecg_signal, kind='linear')(x_new)

def restore_org_ecg_mask(ecg_signal, mask, target_length=520):
    x_old = np.linspace(0, 1, len(ecg_signal))
    x_new = np.linspace(0, 1, target_length)
    ecg_resampled  = interp1d(x_old, ecg_signal, kind='linear')(x_new)
    mask_resampled = interp1d(x_old, mask, kind='nearest')(x_new)
    return ecg_resampled, mask_resampled.astype(int)

def find_p_t_peaks(ecg, mask, boundary_margin=3, merge_distance=15):
    ecg  = np.array(ecg)
    mask = np.array(mask)

    def fix_1_2_confusions(mask):
        mask = mask.copy()
        i = 1
        while i < len(mask) - 1:
            if mask[i] in [1, 2] and mask[i-1] == mask[i+1] and mask[i] != mask[i-1]:
                val_to_fill = mask[i-1]
                start = i
                while i < len(mask) - 1 and mask[i] != val_to_fill and mask[i] in [1, 2]:
                    i += 1
                mask[start:i] = val_to_fill
            else:
                i += 1
        return mask

    def selective_majority_filter(mask, window_size=7):
        padded   = np.pad(mask, (window_size // 2,), mode='edge')
        filtered = mask.copy()
        for i in range(len(mask)):
            window      = padded[i:i + window_size]
            center      = mask[i]
            window_mode = mode(window, keepdims=True)[0][0]
            if center == 0 and window_mode in [1, 2]:
                filtered[i] = window_mode
        return filtered

    def suppress_short_regions(mask, min_length=2):
        mask        = mask.copy()
        current_val = mask[0]
        start_idx   = 0
        for i in range(1, len(mask)):
            if mask[i] != current_val:
                if current_val in [1, 2] and (i - start_idx) < min_length:
                    mask[start_idx:i] = 0
                start_idx   = i
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
                start     = i
                in_region = True
            elif mask[i] != mask_val and in_region:
                regions.append((start, i))
                in_region = False
        if in_region:
            regions.append((start, len(mask)))
        if max_one and regions:
            max_len         = max(end - start for start, end in regions)
            longest_regions = [seg for seg in regions if (seg[1] - seg[0]) == max_len]
            if len(longest_regions) > 1:
                abs_vals      = [np.max(np.abs(ecg[s:e])) for s, e in longest_regions]
                chosen_region = longest_regions[np.argmax(abs_vals)]
            else:
                chosen_region = longest_regions[0]
            regions = [chosen_region]
        for start, end in regions:
            segment  = ecg[start:end]
            maxima   = argrelextrema(segment, np.greater)[0]
            inverted = False
            if len(maxima) == 0:
                maxima   = argrelextrema(-segment, np.greater)[0]
                inverted = True
            if len(maxima) > 0:
                candidate_values = segment[maxima] if not inverted else -segment[maxima]
                best_idx         = np.argmax(candidate_values)
                peak_relative    = maxima[best_idx]
            else:
                derivative    = np.gradient(segment)
                curvature     = np.abs(np.gradient(derivative))
                peak_relative = np.argmax(curvature)
            peak_idx = start + peak_relative
            if boundary_margin <= peak_idx < len(ecg) - boundary_margin:
                indices.append(peak_idx)
        return indices

    def merge_close_peaks(peaks, ecg, merge_distance):
        if not peaks:
            return []
        peaks        = sorted(peaks)
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
        return [p for p in peaks_to_filter if all(abs(p - t) >= merge_distance for t in reference_peaks)]

    def refine_peak_positions(ecg, peak_indices, window=10):
        refined = []
        for idx in peak_indices:
            temp_seg = ecg[max(idx-2, 0):min(idx+2, len(ecg))]
            temp_idx = idx - 2 + np.argmax(np.abs(temp_seg))
            temp_max = idx - 2 + np.argmax(temp_seg)
            temp_min = idx - 2 + np.argmin(temp_seg)
            if idx != temp_idx and (idx != temp_max and idx != temp_min):
                start   = max(idx - window, 0)
                end     = min(idx + window + 1, len(ecg))
                segment = np.abs(ecg[start:end])
                maxima  = argrelextrema(segment, np.greater)[0]
                inverted = False
                if len(maxima) == 0:
                    maxima   = argrelextrema(-segment, np.greater)[0]
                    inverted = True
                if len(maxima) > 0:
                    candidate_values = segment[maxima] if not inverted else -segment[maxima]
                    peak_relative    = maxima[np.argmax(candidate_values)]
                else:
                    derivative    = np.gradient(segment)
                    curvature     = np.abs(np.gradient(derivative))
                    peak_relative = np.argmax(curvature)
                refined.append(start + peak_relative)
            else:
                refined.append(idx)
        return refined

    mask    = fix_1_2_confusions(mask)
    mask    = selective_majority_filter(mask, window_size=16)
    mask    = suppress_short_regions(mask, min_length=3)
    t_peaks = get_peak_indices(mask_val=1, ecg=ecg, mask=mask, max_one=True)
    t_peaks = refine_peak_positions(ecg, t_peaks, window=10)
    t_peaks = merge_close_peaks(t_peaks, ecg, merge_distance=merge_distance)
    p_peaks = get_peak_indices(mask_val=2, ecg=ecg, mask=mask, max_one=False)
    p_peaks = merge_close_peaks(p_peaks, ecg, merge_distance=45)
    p_peaks = refine_peak_positions(ecg, p_peaks, window=10)
    p_peaks = remove_peaks_near_other(p_peaks, t_peaks, merge_distance=merge_distance)
    return p_peaks, t_peaks

def find_onset_offset(sig, peak_idx, smooth=True, window_size=11, polyorder=3,
                      min_drop_ratio=0.2, search_window=200):
    sig        = np.array(sig)
    signal_len = len(sig)
    if smooth:
        win        = min(window_size, signal_len - (signal_len % 2 == 0))
        sig_smooth = savgol_filter(sig, window_length=win, polyorder=polyorder)
    else:
        sig_smooth = sig
    peak_val        = sig_smooth[peak_idx]
    baseline_window = min(40, signal_len // 6)
    start           = max(0, peak_idx - baseline_window)
    end             = min(signal_len, peak_idx + baseline_window)
    local_baseline  = np.median(sig_smooth[start:end])
    drop_thresh     = peak_val - (peak_val - local_baseline) * min_drop_ratio
    onset_idx = peak_idx
    for i in range(peak_idx, max(1, peak_idx - search_window), -1):
        if sig_smooth[i] < drop_thresh:
            onset_idx = i
            break
        if i > 1 and sig_smooth[i-1] < sig_smooth[i-2] and sig_smooth[i-1] < sig_smooth[i]:
            onset_idx = i - 1
            break
    offset_idx = peak_idx
    for i in range(peak_idx, min(signal_len - 2, peak_idx + search_window)):
        if sig_smooth[i] < drop_thresh:
            offset_idx = i
            break
        if sig_smooth[i+1] < sig_smooth[i] and sig_smooth[i+1] < sig_smooth[i+2]:
            offset_idx = i + 1
            break
    return onset_idx, offset_idx

def get_pt_peaks(ecg, r_indices, pt_model_path):
    t_peaks_all, p_peaks_all, pt_peaks_all = [], [], []
    if not r_indices or len(r_indices) < 2:
        return [], [], []
    for i in range(len(r_indices) - 1):
        segment = ecg[r_indices[i]:r_indices[i+1]]
        if len(segment) < 10:
            continue
        segment_signal = np.array(segment)
        resampled_ecgs = resample_ecg(segment_signal, 520)
        ecg_signal     = np.expand_dims(np.array(resampled_ecgs), axis=(0, -1))
        predictions      = predict_tflite_model(pt_model_path, ecg_signal)
        predicted_labels = np.argmax(predictions, axis=-1)
        _, pred_mask     = restore_org_ecg_mask(ecg_signal[0].squeeze(), predicted_labels.squeeze(), len(segment_signal))
        p_peaks, t_peaks = find_p_t_peaks(segment_signal, pred_mask)
        p_peaks  = np.atleast_1d(p_peaks) + r_indices[i]
        t_peaks  = np.atleast_1d(t_peaks) + r_indices[i]
        pt_peaks = tuple(list(t_peaks) + list(p_peaks))
        p_peaks_all.extend(p_peaks)
        t_peaks_all.extend(t_peaks)
        pt_peaks_all.extend(pt_peaks)
    return t_peaks_all, p_peaks_all, pt_peaks_all

def pt_peak_detection(all_lead_data, is_lead, r_peaks, r_result_dic=None, pt_model_path=None):
    result_dic = {}
    for lead in all_lead_data.keys():
        lead_r_peaks      = (r_result_dic.get(lead) if r_result_dic else None) or []
        ecg_signal        = all_lead_data[lead].values.flatten()
        baseline_signal   = baseline_construction_200(ecg_signal, kernel_size=131)
        lowpass_signal    = lowpass(baseline_signal)
        signal_normalized = normalize(lowpass_signal)
        t_peaks_l, p_peaks_l, rr_l = get_pt_peaks(signal_normalized, lead_r_peaks, pt_model_path)
        result_dic[lead] = {"p": p_peaks_l, "t": t_peaks_l, "comb": rr_l}
    if is_lead == "2_lead" and "II" in result_dic:
        p_out  = result_dic["II"].get("p", [])
        t_out  = result_dic["II"].get("t", [])
        rr_out = result_dic["II"].get("comb", [])
    else:
        p_out, t_out, rr_out = [], [], []
    return t_out, p_out, rr_out, result_dic

# ---------------------- Q & S point detection utilities ----------------------
def find_s_indexs(ecg, R_index, d=20):
    d = int(d) + 1
    s = []
    for i in R_index:
        if i == len(ecg):
            continue
        elif i + d <= len(ecg):
            s_array = ecg[i:i + d]
        else:
            s_array = ecg[i:]
        if ecg[i] > 0:
            s_index = i + np.nonzero(s_array == min(s_array))[0][0]
        else:
            s_index = i + np.nonzero(s_array == max(s_array))[0][0]
        s.append(s_index)
    return s

def find_q_indexs(ecg, R_index, d=15):
    d = int(d) + 1
    q = []
    for i in R_index:
        if i == 0:
            continue
        elif 0 <= i - d:
            q_array = ecg[i - d:i]
        else:
            q_array = ecg[:i]
        if ecg[i] > 0:
            q_index = i - (len(q_array) - np.nonzero(q_array == min(q_array))[0][0])
        else:
            q_index = i - (len(q_array) - np.nonzero(q_array == max(q_array))[0][0])
        q.append(q_index)
    return q

def find_t_onsets(ecg, t_peaks, search_window=200):
    t_onsets = []
    for t in t_peaks:
        if t is None or t <= 1:
            t_onsets.append(None)
            continue
        try:
            onset, _ = find_onset_offset(ecg, int(t), smooth=True, search_window=search_window)
            t_onsets.append(onset)
        except:
            t_onsets.append(None)
    return t_onsets

# ---------------------- Data loading and preprocessing utilities ----------------------
def load_and_rename_data(fn, is_lead_for=None):
    lead_columns = {
        '2_lead':  ['ECG', 'II', 'Value', "'MLII'", 'MLII'],
        '7_lead':  ['I', 'II', 'III', 'aVR', 'aVL', 'aVF', 'v5'],
        '12_lead': ['I', 'II', 'III', 'aVR', 'aVL', 'aVF',
                    'v1', 'v2', 'v3', 'v4', 'v5', 'v6',
                    'V1', 'V2', 'V3', 'V4', 'V5', 'V6', 'ECG']
    }
    lead_columns_index = {
        '2_lead':  {0: 'II'},
        '7_lead':  {0: 'I', 1: 'II', 2: 'III', 3: 'aVR', 4: 'aVL', 5: 'aVF', 6: 'v5'},
        '12_lead': {0: 'I', 1: 'II', 2: 'III', 3: 'aVR', 4: 'aVL', 5: 'aVF',
                    6: 'v1', 7: 'v2', 8: 'v3', 9: 'v4', 10: 'v5', 11: 'v6'}
    }
    try:
        all_lead_data = pd.read_csv(fn)
    except Exception as e:
        return None, is_lead_for
    first_col_name = all_lead_data.columns[0].lower()
    if 'date' in first_col_name or 'time' in first_col_name or not pd.api.types.is_numeric_dtype(all_lead_data.iloc[:, 0]):
        all_lead_data = all_lead_data.iloc[:, 1:]
    columns = all_lead_data.columns.tolist()
    if is_lead_for is None:
        if all(col in lead_columns['7_lead'] for col in columns):
            is_lead_for = '7_lead'
        elif all(col in lead_columns['12_lead'] for col in columns):
            is_lead_for = '12_lead'
        else:
            is_lead_for = '2_lead'
    available_columns = [col for col in lead_columns[is_lead_for] if col in columns]
    if not available_columns:
        return None, is_lead_for
    all_lead_data = all_lead_data[available_columns]
    all_lead_data = all_lead_data.rename(columns=lead_columns_index[is_lead_for])
    _upper_to_lower = {'V1':'v1','V2':'v2','V3':'v3','V4':'v4','V5':'v5','V6':'v6'}
    all_lead_data = all_lead_data.rename(columns=_upper_to_lower)
    if is_lead_for == '2_lead':
        all_lead_data.columns = ['II']
    return all_lead_data, is_lead_for

def find_csv_files(root_folder):
    csv_files = []
    for root, _, files in os.walk(root_folder):
        for file in files:
            if file.lower().endswith('.csv'):
                csv_files.append(os.path.join(root, file))
    return csv_files

def merge_multiformat_chunks_to_pdf(folder, fname_prefix, is_lead):
    if is_lead == '12_lead':
        pattern  = os.path.join(folder, f"{fname_prefix}_12lead_chunk_*.pdf")
        out_name = f"{fname_prefix}_12lead_MERGED.pdf"
    elif is_lead == '7_lead':
        pattern  = os.path.join(folder, f"{fname_prefix}_7lead_chunk_*.pdf")
        out_name = f"{fname_prefix}_7lead_MERGED.pdf"
    else:
        pattern  = os.path.join(folder, f"{fname_prefix}_lead_*_chunk_*.pdf")
        out_name = f"{fname_prefix}_2lead_MERGED.pdf"
    pdfs = sorted(glob.glob(pattern))
    if not pdfs:
        for fallback_tag in ("7lead", "12lead"):
            alt = sorted(glob.glob(os.path.join(folder, f"{fname_prefix}_{fallback_tag}_chunk_*.pdf")))
            if alt:
                pdfs     = alt
                out_name = f"{fname_prefix}_{fallback_tag}_MERGED.pdf"
                break
    if not pdfs:
        return None
    out    = os.path.join(folder, out_name)
    merger = PdfMerger()
    for p in pdfs:
        merger.append(p)
    merger.write(out)
    merger.close()
    return out

# ==============================================================
# ST DETECTION  — with shading boundary export
# ==============================================================
def _find_j_point(signal_sq, s_nadir_idx, iso_sq, fs,
                  max_search_ms=120, recovery_ratio=0.5):
    max_search = int(max_search_ms / 1000 * fs)
    s_val      = signal_sq[s_nadir_idx]
    gap        = iso_sq - s_val
    if gap <= 0:
        return s_nadir_idx
    recovery_target = s_val + recovery_ratio * gap
    end_search = min(s_nadir_idx + max_search, len(signal_sq) - 1)
    for k in range(s_nadir_idx, end_search + 1):
        if signal_sq[k] >= recovery_target:
            return k
    return end_search

def detect_st_segment(
    all_leads_data, r_peaks_dict, q_points_dict,
    s_points_dict, t_points_dict,
    fs=200, x_range_sec=10.0, fname_prefix="ecg"
):
    """
    Beat-by-beat ST detection.
    Every beat (R peak) in every chunk produces exactly ONE row in the CSV,
    regardless of whether ST deviation was found.

    Row structure
    -------------
    Identity          : file_name, lead, chunk, beat_index (0-based within chunk)
    PQRST indices     : p_index, q_index, r_index, s_index, t_onset_index
    ST / J-point      : j_point_index, st_center_index
    Shading boundaries: st_shade_onset_index / offset_index  (None if no ST)
                        st_shade_onset_x_boxes / offset_x_boxes / center_x_boxes
                        st_shade_onset_y_mV   / offset_y_mV   / center_y_mV
    Isoelectric line  : iso_x_start_boxes, iso_x_end_boxes (chunk-relative),
                        iso_y_sq, iso_y_mV
    ST magnitude      : st_mm, st_elevation_mm, st_depression_mm  (None / 0 if normal)
    ST flag           : st_detected  (True / False)
    """
    SMALL_BOX_SEC = 0.04
    SMALL_BOX_MV  = 0.1
    Y_MV_RANGE    = 4.0
    THRESHOLD     = 1.5

    summary_rows   = []
    signal_sq_dict = {}
    iso_sq_dict    = {}

    def idx_to_xbox(idx, chunk_s):
        return (idx - chunk_s) / fs / SMALL_BOX_SEC

    for lead, sig in all_leads_data.items():
        sig = np.asarray(sig).flatten()
        N   = len(sig)
        if N < fs:
            continue

        baseline         = medfilt(sig, 131)
        signal_corr      = sig - baseline
        sig_min, sig_max = signal_corr.min(), signal_corr.max()
        if sig_max - sig_min == 0:
            continue

        signal_scaled_mV     = (signal_corr - sig_min) / (sig_max - sig_min) * Y_MV_RANGE
        signal_sq            = signal_scaled_mV / SMALL_BOX_MV
        signal_sq_dict[lead] = signal_sq

        r_peaks  = np.atleast_1d(r_peaks_dict.get(lead, [])).astype(int)
        q_pts    = list(q_points_dict.get(lead, []) or [])
        s_pts    = list(s_points_dict.get(lead, []) or [])
        t_onsets = list(t_points_dict.get(lead, []) or [])

        # ── Isoelectric baseline (PR segment before Q) ─────────────────────
        iso_vals = []
        for i in range(len(r_peaks)):
            if i >= len(q_pts) or q_pts[i] is None:
                continue
            q = int(q_pts[i])
            a = q - int(0.08 * fs)
            b = q - int(0.02 * fs)
            if a < 0 or b <= a:
                continue
            iso_vals.append(np.median(signal_sq[a:b]))
        iso_sq            = np.median(iso_vals) if len(iso_vals) >= 3 else np.median(signal_sq)
        iso_sq_dict[lead] = iso_sq
        iso_y_sq          = float(iso_sq)
        iso_y_mV          = iso_y_sq * SMALL_BOX_MV

        samples_per_chunk = int(fs * x_range_sec)
        total_chunks      = int(np.ceil(N / samples_per_chunk))

        for c in range(total_chunks):
            chunk_start = c * samples_per_chunk
            chunk_end   = min(chunk_start + samples_per_chunk, N)

            # Isoelectric x-range is always the full chunk width
            chunk_iso_x_start = 0.0
            chunk_iso_x_end   = (chunk_end - chunk_start - 1) / fs / SMALL_BOX_SEC

            beat_num = 0   # beat counter within this chunk

            # ── Iterate every beat whose R peak falls in this chunk ─────────
            for i in range(len(r_peaks)):
                r_idx = int(r_peaks[i])
                if not (chunk_start <= r_idx < chunk_end):
                    continue

                # ── PQRST indices for this beat ─────────────────────────────
                q_idx = int(q_pts[i])    if i < len(q_pts)    and q_pts[i]    is not None else None
                s_idx = int(s_pts[i])    if i < len(s_pts)    and s_pts[i]    is not None else None
                t_on  = int(t_onsets[i]) if i < len(t_onsets) and t_onsets[i] is not None else None
                # p_index enriched later in process_single_file from p_peaks_dict

                # ── ST calculation for this beat ────────────────────────────
                j_idx      = None
                st_center  = None
                st_mm      = None
                st_elev    = 0.0
                st_depr    = 0.0
                st_flag    = False

                shade_L_idx    = None
                shade_R_idx    = None
                shade_onset_x  = None
                shade_offset_x = None
                st_center_x    = None
                shade_onset_y  = None
                shade_offset_y = None
                st_center_y    = None

                # Need valid S and T-onset to compute ST
                if s_idx is not None and t_on is not None and t_on > s_idx:
                    j_idx_raw = _find_j_point(signal_sq, s_idx, iso_sq, fs,
                                              max_search_ms=120, recovery_ratio=0.5)
                    j_idx = j_idx_raw

                    if j_idx < t_on - 2:
                        st_center_raw = min(j_idx + int(0.06 * fs), t_on - 1)
                        st_center = st_center_raw

                        # Single-sample deviation
                        st_mm_point = float(signal_sq[st_center]) - iso_sq

                        # Median of [J-point .. t_onset)
                        seg_s = max(j_idx, 0)
                        seg_e = min(t_on, N)
                        if seg_e > seg_s + 2:
                            st_mm_median = float(np.median(signal_sq[seg_s:seg_e])) - iso_sq
                        else:
                            st_mm_median = st_mm_point

                        # Direction agreement guard
                        if (st_mm_point > 0) == (st_mm_median > 0):
                            st_mm_candidate = (st_mm_point
                                               if abs(st_mm_point) <= abs(st_mm_median)
                                               else st_mm_median)

                            if abs(st_mm_candidate) >= THRESHOLD:
                                st_mm   = round(float(st_mm_candidate), 4)
                                st_elev = round(float(st_mm), 4) if st_mm > 0 else 0.0
                                st_depr = round(float(st_mm), 4) if st_mm < 0 else 0.0
                                st_flag = True

                                # Shading region (identical to _draw_panel logic)
                                w = int(0.02 * fs)
                                shade_L_idx = max(j_idx, st_center - w)
                                shade_R_idx = min(t_on,  st_center + w)

                                shade_onset_x  = round(idx_to_xbox(shade_L_idx, chunk_start), 4)
                                shade_offset_x = round(idx_to_xbox(shade_R_idx, chunk_start), 4)
                                st_center_x    = round(idx_to_xbox(st_center,   chunk_start), 4)

                                shade_onset_y  = round(float(signal_sq[shade_L_idx]) * SMALL_BOX_MV, 4)
                                shade_offset_y = round(float(signal_sq[shade_R_idx]) * SMALL_BOX_MV, 4)
                                st_center_y    = round(float(signal_sq[st_center])   * SMALL_BOX_MV, 4)

                # ── Emit one row for this beat ──────────────────────────────
                summary_rows.append({
                    # Identity
                    "file_name":  fname_prefix,
                    "lead":       lead,
                    "chunk":      c + 1,
                    "beat_index": beat_num,
                    # PQRST (p_index filled later)
                    "p_index":        None,
                    "q_index":        q_idx,
                    "r_index":        r_idx,
                    "s_index":        s_idx,
                    "t_onset_index":  t_on,
                    # J-point & ST measurement point
                    "j_point_index":  j_idx,
                    "st_center_index": st_center,
                    # Shading boundaries — sample indices
                    "st_shade_onset_index":    shade_L_idx,
                    "st_shade_offset_index":   shade_R_idx,
                    # Shading boundaries — x-axis (small-box units, chunk-relative)
                    "st_shade_onset_x_boxes":  shade_onset_x,
                    "st_shade_offset_x_boxes": shade_offset_x,
                    "st_center_x_boxes":       st_center_x,
                    # Shading boundaries — y-axis (mV)
                    "st_shade_onset_y_mV":     shade_onset_y,
                    "st_shade_offset_y_mV":    shade_offset_y,
                    "st_center_y_mV":          st_center_y,
                    # Isoelectric line (chunk-relative x, absolute y)
                    "iso_x_start_boxes": round(chunk_iso_x_start, 4),
                    "iso_x_end_boxes":   round(chunk_iso_x_end,   4),
                    "iso_y_sq":          round(iso_y_sq, 4),
                    "iso_y_mV":          round(iso_y_mV, 4),
                    # ST magnitude
                    "st_mm":            st_mm,
                    "st_elevation_mm":  st_elev,
                    "st_depression_mm": st_depr,
                    "st_detected":      st_flag,
                })

                beat_num += 1

    return summary_rows, signal_sq_dict, iso_sq_dict


# ==============================================================
# HELPER: Draw a single ECG sub-panel onto a given Axes
# ==============================================================
SMALL_BOX_SEC = 0.04
SMALL_BOX_MV  = 0.1
X_RANGE_SEC   = 5.0
Y_MV_RANGE    = 4.0

X_BOXES = int(X_RANGE_SEC / SMALL_BOX_SEC)
Y_BOXES = int(Y_MV_RANGE  / SMALL_BOX_MV)

BOX_IN = 0.048

PANEL_W_IN = X_BOXES * BOX_IN
PANEL_H_IN = Y_BOXES * BOX_IN

MARGIN_LEFT   = 0.60
MARGIN_RIGHT  = 0.20
MARGIN_TOP    = 0.55
MARGIN_BOTTOM = 0.35
COL_GAP       = 0.40
ROW_GAP       = 0.45

N_COLS = 2

LEAD_GRID_12 = [
    (0, 0, 'I'),    (0, 1, 'v1'),
    (1, 0, 'II'),   (1, 1, 'v2'),
    (2, 0, 'III'),  (2, 1, 'v3'),
    (3, 0, 'aVR'),  (3, 1, 'v4'),
    (4, 0, 'aVL'),  (4, 1, 'v5'),
    (5, 0, 'aVF'),  (5, 1, 'v6'),
]

LEAD_GRID_7 = [
    (0, 0, 'I'),    (0, 1, 'aVR'),
    (1, 0, 'II'),   (1, 1, 'aVL'),
    (2, 0, 'III'),  (2, 1, 'aVF'),
    (3, 0, 'v5'),   (3, 1, None),
]


def _figure_geometry(n_rows):
    fig_w = MARGIN_LEFT + N_COLS * PANEL_W_IN + (N_COLS - 1) * COL_GAP + MARGIN_RIGHT
    fig_h = MARGIN_TOP  + n_rows * PANEL_H_IN + (n_rows - 1) * ROW_GAP + MARGIN_BOTTOM
    rect_map = {}
    for r in range(n_rows):
        for c in range(N_COLS):
            x_in = MARGIN_LEFT + c * (PANEL_W_IN + COL_GAP)
            y_in = MARGIN_BOTTOM + (n_rows - 1 - r) * (PANEL_H_IN + ROW_GAP)
            rect_map[(r, c)] = [
                x_in       / fig_w,
                y_in       / fig_h,
                PANEL_W_IN / fig_w,
                PANEL_H_IN / fig_h,
            ]
    return fig_w, fig_h, rect_map


def _draw_panel(ax, lead_name, signal_sq, iso_sq,
                start, end, fs,
                r_peaks, q_pts, s_pts, t_onsets,
                st_rows_for_chunk):
    ax.set_xlim(0, X_BOXES)
    ax.set_ylim(0, Y_BOXES)
    ax.tick_params(which="both", length=0)

    for x in range(X_BOXES + 1):
        color = "#e89aa8" if x % 5 == 0 else "#f7dde0"
        lw    = 0.70      if x % 5 == 0 else 0.28
        ax.axvline(x, color=color, lw=lw, zorder=1)
    for y in range(Y_BOXES + 1):
        color = "#e89aa8" if y % 5 == 0 else "#f7dde0"
        lw    = 0.70      if y % 5 == 0 else 0.28
        ax.axhline(y, color=color, lw=lw, zorder=1)

    ax.axhline(iso_sq, color="#3355bb", lw=0.8, ls="--", zorder=2, alpha=0.8)

    actual_end = min(end, len(signal_sq))
    if actual_end > start:
        xs     = np.arange(start, actual_end)
        x_plot = (xs - start) / fs / SMALL_BOX_SEC
        ax.plot(x_plot, signal_sq[start:actual_end],
                color="black", lw=0.85, zorder=4)

    def _mark(idx, color, ms=3):
        if idx is None or not (start <= idx < actual_end):
            return
        x_pos = (idx - start) / fs / SMALL_BOX_SEC
        ax.plot(x_pos, signal_sq[idx], "o",
                color=color, ms=ms, zorder=5, markeredgewidth=0)

    for i, r in enumerate(r_peaks):
        _mark(r, "red")
        if i < len(q_pts):    _mark(q_pts[i],    "#9933cc")
        if i < len(s_pts):    _mark(s_pts[i],    "#ff8800")
        if i < len(t_onsets): _mark(t_onsets[i], "#00aa44")

    for row in st_rows_for_chunk:
        if not row.get("st_detected") or row.get("lead") != lead_name:
            continue
        st_mm     = row["st_mm"]
        st_center = row["st_center_index"]
        j_off     = row.get("j_point_index") or st_center
        t_on      = row.get("t_onset_index") or st_center

        if st_center is None or t_on is None:
            continue

        shade_col = "green" if abs(st_mm) <= 2 else "red"
        w   = int(0.02 * fs)
        L   = max(j_off, st_center - w)
        R   = min(t_on,  st_center + w)

        if R <= L or L >= actual_end or R <= start:
            continue

        seg_idx = np.arange(max(L, start), min(R, actual_end))
        x_seg   = (seg_idx - start) / fs / SMALL_BOX_SEC
        y_seg   = signal_sq[seg_idx]

        ax.fill_between(x_seg, iso_sq, y_seg,
                        where=(y_seg - iso_sq) * np.sign(st_mm) > 0,
                        color=shade_col, alpha=0.38, zorder=3)

        lbl_x = (st_center - start) / fs / SMALL_BOX_SEC
        lbl_y = iso_sq + st_mm + (0.8 if st_mm > 0 else -0.8)
        ax.text(lbl_x, lbl_y, f"{st_mm:+.1f}mm",
                ha="center", fontsize=5.5, fontweight="bold",
                color=shade_col, zorder=6)

    ax.set_title(lead_name, fontsize=7, fontweight="bold",
                 pad=2, loc="left", color="#111111")

    sec_ticks = np.arange(0, X_RANGE_SEC + 0.01, 1.0)
    ax.set_xticks(sec_ticks / SMALL_BOX_SEC)
    ax.set_xticklabels([f"{int(t)}s" for t in sec_ticks], fontsize=4.5)

    mv_ticks = np.arange(0, Y_BOXES + 1, 10)
    ax.set_yticks(mv_ticks)
    ax.set_yticklabels([f"{v * SMALL_BOX_MV:.1f}" for v in mv_ticks], fontsize=4.5)

    for spine in ax.spines.values():
        spine.set_visible(False)


def _draw_blank_panel(ax):
    ax.set_xlim(0, X_BOXES)
    ax.set_ylim(0, Y_BOXES)
    ax.set_facecolor("white")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)


def _build_pages(
    lead_grid, n_rows,
    all_leads_data,
    r_peaks_dict, q_points_dict, s_points_dict, t_points_dict,
    st_summary_rows,
    signal_sq_dict, iso_sq_dict,
    fs, save_path, fname_prefix, tag,
):
    os.makedirs(save_path, exist_ok=True)

    st_lookup = {}
    for row in st_summary_rows:
        key = (row["lead"], row["chunk"])
        st_lookup.setdefault(key, []).append(row)

    N_max = 0
    for lead_name in [lg[2] for lg in lead_grid if lg[2] is not None]:
        if lead_name in all_leads_data:
            N_max = max(N_max, len(np.asarray(all_leads_data[lead_name]).flatten()))
    if N_max < fs:
        return

    samples_per_chunk = int(fs * X_RANGE_SEC)
    total_chunks      = int(np.ceil(N_max / samples_per_chunk))

    fig_w, fig_h, rect_map = _figure_geometry(n_rows)

    for c in range(total_chunks):
        start = c * samples_per_chunk
        end   = min(start + samples_per_chunk, N_max)

        fig = plt.figure(figsize=(fig_w, fig_h))
        fig.patch.set_facecolor("white")

        chunk_st_rows = [row for row in st_summary_rows if row.get("chunk") == c + 1]
        elev_leads = [row["lead"] for row in chunk_st_rows
                      if row.get("st_mm") is not None and row["st_mm"] > 0]
        depr_leads = [row["lead"] for row in chunk_st_rows
                      if row.get("st_mm") is not None and row["st_mm"] < 0]
        st_parts = []
        if elev_leads:
            st_parts.append(f"ST↑ {', '.join(sorted(set(elev_leads)))}")
        if depr_leads:
            st_parts.append(f"ST↓ {', '.join(sorted(set(depr_leads)))}")
        st_lbl = "  │  ".join(st_parts) if st_parts else "ST: Normal"

        line1 = (f"{fname_prefix}  │  {tag.replace('lead', '-Lead')} ECG  │  "
                 f"{start/fs:.1f}s – {end/fs:.1f}s  │  Chunk {c+1}/{total_chunks}")
        line2 = f"{st_lbl}"
        fig.suptitle(f"{line1}\n{line2}", fontsize=9, fontweight="bold",
                     y=0.998, va="top", linespacing=1.55)

        for (row_idx, col_idx, lead_name) in lead_grid:
            rect = rect_map[(row_idx, col_idx)]
            ax   = fig.add_axes(rect)

            if lead_name is None:
                _draw_blank_panel(ax)
                continue
            if lead_name not in all_leads_data:
                _draw_blank_panel(ax)
                continue

            signal_sq = signal_sq_dict.get(lead_name)
            iso_sq    = iso_sq_dict.get(lead_name)
            if signal_sq is None or iso_sq is None:
                _draw_blank_panel(ax)
                continue

            r_peaks  = np.atleast_1d(r_peaks_dict.get(lead_name,   [])).astype(int)
            q_pts    = list(q_points_dict.get(lead_name, []) or [])
            s_pts    = list(s_points_dict.get(lead_name, []) or [])
            t_onsets = list(t_points_dict.get(lead_name, []) or [])
            chunk_st = st_lookup.get((lead_name, c + 1), [])

            _draw_panel(ax, lead_name, signal_sq, iso_sq,
                        start, end, fs,
                        r_peaks, q_pts, s_pts, t_onsets, chunk_st)

            if col_idx == 0:
                ax.set_ylabel("mV", fontsize=5.5, labelpad=2)
            else:
                ax.set_ylabel("")

        out_path = os.path.join(save_path, f"{fname_prefix}_{tag}_chunk_{c+1:03d}.pdf")
        plt.savefig(out_path, dpi=200, bbox_inches=None, facecolor="white")
        plt.close(fig)


_12LEAD_EXCLUSIVE = {'v1', 'v2', 'v3', 'v4', 'v6'}
_7LEAD_SET = {'I', 'II', 'III', 'aVR', 'aVL', 'aVF', 'v5'}


def _has_real_signal(all_leads_data, lead_name, min_samples=10):
    raw = all_leads_data.get(lead_name)
    if raw is None:
        return False
    arr = np.asarray(raw).flatten().astype(float)
    if len(arr) < min_samples:
        return False
    if np.all(np.isnan(arr)):
        return False
    if np.all(arr == 0):
        return False
    return True


def _resolve_lead_type(all_leads_data, is_lead):
    if is_lead not in ("12_lead", "7_lead"):
        return "2_lead"
    has_exclusive = any(
        _has_real_signal(all_leads_data, lead) for lead in _12LEAD_EXCLUSIVE
    )
    return "12_lead" if has_exclusive else "7_lead"


def plot_ecg_with_markers(
    all_leads_data,
    r_peaks_dict, q_points_dict, s_points_dict, t_points_dict,
    st_summary_rows, signal_sq_dict, iso_sq_dict,
    fs=200, save_path=".", fname_prefix="ecg",
    x_range_sec=10.0, is_lead="7_lead",
):
    resolved = _resolve_lead_type(all_leads_data, is_lead)
    if resolved == "12_lead":
        _build_pages(
            LEAD_GRID_12, 6,
            all_leads_data,
            r_peaks_dict, q_points_dict, s_points_dict, t_points_dict,
            st_summary_rows, signal_sq_dict, iso_sq_dict,
            fs, save_path, fname_prefix, "12lead",
        )
    else:
        _build_pages(
            LEAD_GRID_7, 4,
            all_leads_data,
            r_peaks_dict, q_points_dict, s_points_dict, t_points_dict,
            st_summary_rows, signal_sq_dict, iso_sq_dict,
            fs, save_path, fname_prefix, "7lead",
        )


# ==============================================================
# MERGE ALL OUTPUT PDFs
# ==============================================================
def merge_all_merged_pdfs(root_output_dir, final_pdf_path):
    merger    = PdfMerger()
    found_any = False
    for root, _, files in os.walk(root_output_dir):
        for file in sorted(files):
            if file.endswith("_MERGED.pdf") or file.endswith("_MERGED_ALL_LEADS.pdf"):
                merger.append(os.path.join(root, file))
                found_any = True
    if found_any:
        merger.write(final_pdf_path)
        merger.close()
    else:
        merger.close()


# ==============================================================
# PROCESS SINGLE FILE
# ==============================================================
def process_single_file(
    fn, save_path, is_lead, r_model_path, pt_model_path,
    fs=200, mm_per_mV=10.0, x_range_sec=10.0, r_threshold=1.5
):
    local_name = os.path.splitext(os.path.basename(fn))[0]
    csv_root   = os.path.join(save_path, local_name)
    os.makedirs(csv_root, exist_ok=True)

    # -------- LOAD ECG DATA ----------
    all_leads_data, is_lead = load_and_rename_data(fn, is_lead)
    if all_leads_data is None:
        return f"Failed: {local_name}"

    # -------- R PEAK DETECTION ----------
    _, r_result_dic = r_peak_detection(all_leads_data, is_lead, r_model_path)

    # -------- PT PEAK DETECTION ----------
    _, _, _, pt_result_dic = pt_peak_detection(
        all_leads_data, is_lead, None, r_result_dic, pt_model_path
    )

    q_points_dict = {}
    s_points_dict = {}
    t_points_dict = {}
    r_peaks_dict  = {}

    # Also store per-lead p-peaks for CSV enrichment
    p_peaks_dict  = {}

    for lead in all_leads_data.columns:
        ecg     = all_leads_data[lead].values.astype(float).flatten()
        r_peaks = np.atleast_1d(r_result_dic.get(lead, [])).astype(int)
        r_peaks_dict[lead] = r_peaks

        t_peaks  = np.atleast_1d(pt_result_dic.get(lead, {}).get("t", [])).astype(int)
        p_peaks  = np.atleast_1d(pt_result_dic.get(lead, {}).get("p", [])).astype(int)
        p_peaks_dict[lead] = list(p_peaks)

        q_points, s_points, t_points = [], [], []

        if len(r_peaks) >= 2 and len(t_peaks) >= 1:
            q_points = find_q_indexs(ecg, r_peaks, d=15)
            s_points = find_s_indexs(ecg, r_peaks, d=20)

            min_len  = min(len(r_peaks), len(q_points), len(s_points))
            r_peaks  = r_peaks[:min_len]
            q_points = q_points[:min_len]
            s_points = s_points[:min_len]

            t_onsets = find_t_onsets(ecg, t_peaks, search_window=int(0.3 * fs))
            t_points = []
            for s in s_points:
                cand = [t for t in t_onsets if t is not None and t > s]
                t_points.append(cand[0] if cand else None)

        r_peaks_dict[lead]  = r_peaks
        q_points_dict[lead] = q_points
        s_points_dict[lead] = s_points
        t_points_dict[lead] = t_points

    # -------- ST DETECTION ----------
    st_x_range = 5.0
    st_rows, signal_sq_dict, iso_sq_dict = detect_st_segment(
        all_leads_data, r_peaks_dict, q_points_dict,
        s_points_dict, t_points_dict,
        fs=fs, x_range_sec=st_x_range, fname_prefix=local_name
    )

    # -------- ENRICH st_rows with P-peak index ----------------------
    # For each row find the nearest P peak (before the corresponding R)
    SMALL_BOX_SEC_local = 0.04
    for row in st_rows:
        lead    = row["lead"]
        r_idx   = row.get("r_index")
        p_list  = p_peaks_dict.get(lead, [])
        if r_idx is not None and p_list:
            # P peak is expected just before R peak
            candidates = [p for p in p_list if p < r_idx]
            if candidates:
                row["p_index"] = int(candidates[-1])   # nearest P before R

    # -------- PLOT ----------
    plot_ecg_with_markers(
        all_leads_data, r_peaks_dict, q_points_dict,
        s_points_dict, t_points_dict,
        st_summary_rows=st_rows,
        signal_sq_dict=signal_sq_dict,
        iso_sq_dict=iso_sq_dict,
        fs=fs,
        save_path=csv_root,
        fname_prefix=local_name,
        x_range_sec=st_x_range,
        is_lead=is_lead
    )

    # -------- MERGE CHUNK PDFs ----------
    resolved_lead_for_merge = _resolve_lead_type(all_leads_data, is_lead)
    merge_multiformat_chunks_to_pdf(csv_root, local_name, resolved_lead_for_merge)

    # -------- SAVE COMPREHENSIVE CSV (one file per ECG, all chunks + leads) ----
    # Every beat produces a row so st_rows is always populated when data exists
    if st_rows:
        df_st = pd.DataFrame(st_rows)

        # Reorder columns for readability
        col_order = [
            "file_name", "lead", "chunk", "beat_index",
            # PQRST indices (one row per beat)
            "p_index", "q_index", "r_index", "s_index", "t_onset_index",
            # J-point & ST measurement point
            "j_point_index", "st_center_index",
            # Shading boundaries — sample indices (None when no ST)
            "st_shade_onset_index", "st_shade_offset_index",
            # Shading boundaries — x-axis small-box units, chunk-relative
            "st_shade_onset_x_boxes", "st_shade_offset_x_boxes", "st_center_x_boxes",
            # Shading boundaries — y-axis mV
            "st_shade_onset_y_mV", "st_shade_offset_y_mV", "st_center_y_mV",
            # Isoelectric line
            "iso_x_start_boxes", "iso_x_end_boxes", "iso_y_sq", "iso_y_mV",
            # ST magnitude & flag
            "st_detected", "st_mm", "st_elevation_mm", "st_depression_mm",
        ]
        # Keep only columns that exist (safety)
        col_order = [c for c in col_order if c in df_st.columns]
        df_st = df_st[col_order]

        out_csv = os.path.join(csv_root, f"{local_name}_ST_analysis.csv")
        df_st.to_csv(out_csv, index=False)
        print(f"[{local_name}] ST analysis CSV saved → {out_csv}")

    return f"Processed: {local_name}"


# ==============================================================
# MAIN ECG PROCESSING ENTRY POINT
# ==============================================================
def ecg_processing(path, save_path, is_lead, r_model_path, pt_model_path):
    csv_files = find_csv_files(path)
    if not csv_files:
        return
    for fn in csv_files:
        try:
            result = process_single_file(fn, save_path, is_lead, r_model_path, pt_model_path)
            print(result)
        except Exception as e:
            print(f"❌ Error processing file {fn}: {e}")

    final_pdf_path = os.path.join(save_path, "FINAL_ALL_CSV_ALL_LEADS.pdf")
    merge_all_merged_pdfs(save_path, final_pdf_path)
    print("\n✅ ECG Processing Completed Successfully")


# ---------------------- Main entry ----------------------
if __name__ == "__main__":

    # -------------------- INPUT / OUTPUT PATHS --------------------
    path      = r"C:\Users\Admin\Downloads\livepatients\2_lead\13_02"
    save_path = r"C:\Users\Admin\Downloads\st_results\23_02_2lead"
    os.makedirs(save_path, exist_ok=True)

    # -------------------- MODEL FILE PATHS --------------------
    r_index_model_path  = r"C:\Users\Admin\Downloads\rnn_model1_16_02_26_final.tflite"
    pt_index_model_path = r"C:\Users\Admin\Downloads\ecg_pt_detection_LSTMGRU_v32.tflite"

    # -------------------- CONFIG --------------------
    is_lead = "2_lead"

    # -------------------- ECG PROCESSING --------------------
    ecg_processing(
        path=path,
        save_path=save_path,
        is_lead=is_lead,
        r_model_path=r_index_model_path,
        pt_model_path=pt_index_model_path,
    )

    # -------------------- MERGE ALL PDFs --------------------
    final_pdf_path = os.path.join(save_path, "FINAL_ALL_CSV_ALL_LEADS.pdf")
    merge_all_merged_pdfs(root_output_dir=save_path, final_pdf_path=final_pdf_path)

    print("\n✅ ECG 2-Lead ST Segment Processing Completed Successfully")