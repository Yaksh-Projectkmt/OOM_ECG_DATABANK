import random
import pandas as pd
import numpy as np
import tensorflow as tf
from scipy import signal
from scipy import sparse
from scipy.sparse.linalg import spsolve
from scipy.signal import (find_peaks, firwin, medfilt, argrelextrema, savgol_filter, butter, filtfilt, welch, resample)
import pywt
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from matplotlib import colormaps
from scipy.ndimage import gaussian_filter1d
import glob
import tools as st
import utils
from analysis_tool import tools as st
from analysis_tool  import utils
import neurokit2 as nk
import cv2
import warnings
import threading
import scipy
import os
from dotenv import load_dotenv
from PIL import Image
from scipy.interpolate import interp1d
from collections import Counter
from scipy.stats import mode
import re
import shutil
import uuid
from fpdf import FPDF
from matplotlib.gridspec import GridSpec
import matplotlib.ticker as ticker
from dotenv import load_dotenv
from PyPDF2 import PdfMerger
import gridfs
from pymongo import MongoClient
from datetime import datetime
from io import BytesIO
from collections import defaultdict
from bson.objectid import ObjectId
load_dotenv()   # loads .env
warnings.filterwarnings('ignore')

results_lock = threading.RLock()
mongo_client = MongoClient(os.getenv("MONGO_HOST"))

media_db = mongo_client["Download_files"]
admin_db=mongo_client["admin"]
download_fs = gridfs.GridFS(media_db, collection="downloads")

logs_collection = admin_db["pdf_logs"]

interpreter = tf.lite.Interpreter(
    model_path= os.getenv("PVC"))
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

def lowpass(file, cutoff=0.3):
    b, a = signal.butter(3, cutoff, btype='lowpass', analog=False)
    low_passed = signal.filtfilt(b, a, file)
    return low_passed

def baseline_construction_200(ecg_signal, kernel_size=101):
    s_corrected = signal.detrend(ecg_signal)
    baseline_corrected = s_corrected - signal.medfilt(s_corrected, kernel_size)
    return baseline_corrected

def load_tflite_model(model_path):
    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    return interpreter, input_details, output_details

with tf.device('/CPU:0'):

    pac_model = load_tflite_model(os.getenv("PAC"))
    afib_model = load_tflite_model(os.getenv("AFIB"))
    vfib_model = load_tflite_model(os.getenv("VIFIB"))
    block_model = load_tflite_model(os.getenv("BLOCK"))
    mi_model = load_tflite_model(os.getenv("MI"))

    r_index_model = load_tflite_model(os.getenv("rnn_model"))
    pt_index_model = load_tflite_model(os.getenv("ecg_pt_detection"))

def remove_closely_spaced_peaks(ecg_signal, peaks, min_distance):
    """
    Remove closely spaced peaks to avoid false positives.
    """
    sorted_peaks = np.sort(peaks)
    filtered_peaks = []
    for peak in sorted_peaks:
        if not filtered_peaks or (peak - filtered_peaks[-1]) >= min_distance:
            filtered_peaks.append(peak)
        else:
            # Retain the peak with the highest amplitude
            prev_peak = filtered_peaks[-1]
            if abs(ecg_signal[peak]) > abs(ecg_signal[prev_peak]):
                filtered_peaks[-1] = peak  # Replace previous peak with current peak
    return filtered_peaks

def extract_number(filename):
    match = re.search(r'(\d+)', os.path.basename(filename))
    return int(match.group(1)) if match else float('inf')

def detect_rpeaks_pvc(ecg, rate, ransac_window_size=3.35, lowfreq=5.0, highfreq=9.0):
    # Convert window size to samples
    rate = 200
    ransac_window_size = int(ransac_window_size * rate)
    
    # Bandpass filtering
    lowpass = signal.butter(1, highfreq / (rate / 2.0), 'low')
    highpass = signal.butter(1, lowfreq / (rate / 2.0), 'high')
    ecg_low = signal.filtfilt(*lowpass, x=ecg)
    ecg_band = signal.filtfilt(*highpass, x=ecg_low)
    
    # Derivative and squared power
    decg = np.diff(ecg_band)
    decg_power = decg ** 2
    
    # Robust thresholding using MAD
    thresholds, max_powers = [], []
    for i in range(int(len(decg_power) / ransac_window_size)):
        sample = slice(i * ransac_window_size, (i + 1) * ransac_window_size)
        d = decg_power[sample]
        thresholds.append(1.4826 * np.median(np.abs(d - np.median(d))))  # MAD
        max_powers.append(np.max(d))
    
    threshold = np.median(thresholds)
    max_power = np.median(max_powers)
    decg_power[decg_power < threshold] = 0
    decg_power /= max_power
    decg_power[decg_power > 1.0] = 1.0
    square_decg_power = decg_power ** 4
    
    # Shannon energy
    shannon_energy = -square_decg_power * np.log(square_decg_power)
    shannon_energy[~np.isfinite(shannon_energy)] = 0.0
    
    # Smoothing
    mean_window_len = int(rate * 0.125 + 1)
    lp_energy = np.convolve(shannon_energy, [1.0 / mean_window_len] * mean_window_len, mode='same')
    lp_energy = gaussian_filter1d(lp_energy, rate / 14.0)
    lp_energy_diff = np.diff(lp_energy)
    
    # Zero-crossing detection
    zero_crossings = (lp_energy_diff[:-1] > 0) & (lp_energy_diff[1:] < 0)
    zero_crossings = np.flatnonzero(zero_crossings)
    zero_crossings -= 1
    
    rpeaks = []
    for idx in zero_crossings:
        # Adaptive search window
        search_window = slice(max(0, idx - int(rate * 0.05)), min(len(ecg), idx + int(rate * 0.05)))
        local_signal = ecg[search_window]
        max_amplitude = np.max(local_signal)
        min_amplitude = np.min(local_signal)
        
        if abs(max_amplitude) > abs(min_amplitude):  # Normal beat
            rpeak = np.argmax(local_signal) + search_window.start
        elif abs(max_amplitude + 0.11) < abs(min_amplitude):  # Inverted beat
            rpeak = np.argmin(local_signal) + search_window.start
        else:  # Ambiguous case
            if max_amplitude >= 0:
                rpeak = np.argmax(local_signal) + search_window.start
            else:
                rpeak = np.argmin(local_signal) + search_window.start
               
        rpeaks.append(rpeak)
    rpeaks = remove_closely_spaced_peaks(ecg,rpeaks, min_distance=30) 
    return np.array(rpeaks)

def detect_beats(ecg, rate, ransac_window_size=3.35, lowfreq=5.0, highfreq=15.0):
    ransac_window_size = int(ransac_window_size * rate)
    lowpass = scipy.signal.butter(1, highfreq / (rate / 2.0), 'low')
    highpass = scipy.signal.butter(1, lowfreq / (rate / 2.0), 'high')
    ecg_low = scipy.signal.filtfilt(*lowpass, x=ecg)
    ecg_band = scipy.signal.filtfilt(*highpass, x=ecg_low)
    decg = np.diff(ecg_band)
    decg_power = decg ** 2
    thresholds, max_powers = [], []
    for i in range(int(len(decg_power) / ransac_window_size)):
        sample = slice(i * ransac_window_size, (i + 1) * ransac_window_size)
        d = decg_power[sample]
        thresholds.append(0.5 * np.std(d))
        max_powers.append(np.max(d))
    threshold = np.median(thresholds)
    max_power = np.median(max_powers)
    decg_power[decg_power < threshold] = 0
    decg_power /= max_power
    decg_power[decg_power > 1.0] = 1.0
    square_decg_power = decg_power ** 4
    shannon_energy = -square_decg_power * np.log(square_decg_power)
    shannon_energy[~np.isfinite(shannon_energy)] = 0.0
    mean_window_len = int(rate * 0.125 + 1)
    lp_energy = np.convolve(shannon_energy, [1.0 / mean_window_len] * mean_window_len, mode='same')
    lp_energy = gaussian_filter1d(lp_energy, rate / 14.0)
    lp_energy_diff = np.diff(lp_energy)
    zero_crossings = (lp_energy_diff[:-1] > 0) & (lp_energy_diff[1:] < 0)
    zero_crossings = np.flatnonzero(zero_crossings)
    zero_crossings -= 1

    rpeaks = []
    for idx in zero_crossings:
        search_window = slice(max(0, idx - int(rate * 0.2)), min(len(ecg), idx + int(rate * 0.1)))
        local_signal = ecg[search_window]
        max_amplitude = np.max(local_signal)
        min_amplitude = np.min(local_signal)

        if abs(max_amplitude) > abs(min_amplitude):  
            rpeak = np.argmax(local_signal) + search_window.start
        elif abs(max_amplitude+0.11) < abs(min_amplitude):  
            rpeak = np.argmin(local_signal) + search_window.start
        else:  
            if max_amplitude >= 0:
                rpeak = np.argmax(local_signal) + search_window.start
            else:
                rpeak = np.argmin(local_signal) + search_window.start
 
        rpeaks.append(rpeak)
    return np.array(rpeaks)

def predict_tflite_model(model: tuple, input_data: tuple):
    with results_lock:
        interpreter, input_details, output_details = model
        for i in range(len(input_data)):
            interpreter.set_tensor(input_details[i]['index'], input_data[i])
        interpreter.invoke()
        output = interpreter.get_tensor(output_details[0]['index'])

    return output

def find_normalize(signal):
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
            occupied[left:right] = True
            for i in sorted_indices:
                maximum_idx = idx
                if occupied[i] in occupied[left:right]:
                    if occupied[i] > occupied[maximum_idx]:
                        maximum = i
                selected.append(maximum_idx)

    return sorted(selected)

def predict_r_tflite_model(model:tuple, input_data):
    with results_lock:
        interpreter, input_details, output_details = model
        input_data = input_data.astype(np.float32)
        interpreter.set_tensor(input_details[0]['index'], input_data)
        interpreter.invoke()
        output_data = interpreter.get_tensor(output_details[0]['index'])
    return output_data.squeeze() 

# R index detection model
def check_model_r(ecg_data):
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
        preds = predict_r_tflite_model(r_index_model, raw_array)
        preds = preds[:signal_len]
        r_peak_prob = preds[:, 1]

        peak_indices, _ = find_peaks(r_peak_prob, height=0.2, distance=20)
        peak_indices = [valid_index for valid_index in peak_indices if 100 < valid_index < 900 or i == 0 or i]
        
        for j in range(len(peak_indices)): 
            if ecg_signal[peak_indices[j]] in df_ecg_signal:
                temp_list.append(df_ecg_signal.index(ecg_signal[peak_indices[j]]))
        
        i += step
    rpeak = sorted(set(temp_list))
    r_peaks = refined_non_max_suppression(df_ecg_signal, rpeak, suppression_radius=25) #30
    r_peaks = sorted(set(r_peaks))
    return r_peaks

def check_r_index(all_leads_data, version, frequency):
    median_r_list = []
    combine_r_index = {}
    leads_for_version = {
        "2": [],
        "7": [],
        "12": []}
    
    for lead in all_leads_data.keys():
        if lead in ["I",'II', 'III', "v1", "v5"]:
            ecg_signal = all_leads_data[lead].values
            # if use_for == "pvc":
            #     baseline_signal = baseline_construction_200(ecg_signal, 131)
            #     lowpass_signal = lowpass(baseline_signal,0.2)
            #     rpeaks = detect_rpeaks_pvc(lowpass_signal, frequency)
            # else:
            baseline_signal = baseline_construction_200(ecg_signal, 131)
            lowpass_signal = lowpass(baseline_signal, cutoff=0.3) 
            signal_normalized = find_normalize(lowpass_signal)
            rpeaks = check_model_r(signal_normalized)
            if len(rpeaks) <= 3:
              rpeaks = detect_beats(lowpass_signal, frequency).tolist()
            combine_r_index[lead] = rpeaks
            if rpeaks:
                leads_for_version[version].append(lead)
    if version == "2":
        median_r_list = combine_r_index['II']
    elif version == "7" or version == "12":
        rpeak_lists = [combine_r_index[lead] for lead in leads_for_version[version]]
        if rpeak_lists:
            min_length = min(len(peaks) for peaks in rpeak_lists)
            median_r_list = [int(np.median([combine_r_index[lead][i] for lead in leads_for_version[version]])) for i in range(min_length)]
          
    return median_r_list, combine_r_index

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

def find_onset_offset(signal, peak_idx, smooth=True, window_size=11, polyorder=3, min_drop_ratio=0.2, search_window=200):
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

def get_pt_peaks(ecg, r_indices):
    t_peaks_all, p_peaks_all, pt_peaks_all, onset, offset = [], [], [], [], []

    for i in range(len(r_indices) - 1):
        segment = ecg[r_indices[i]:r_indices[i+1]]
        if len(segment) < 10:
            continue

        segment_signal = np.array(segment)
        
        resampled_ecgs = resample_ecg(segment_signal, 520)
        ecg_signal = np.array(resampled_ecgs)
        ecg_signal = np.expand_dims(ecg_signal, axis=(0, -1))

        predictions = predict_r_tflite_model(pt_index_model, ecg_signal)
        predicted_labels = np.argmax(predictions, axis=-1)

        _, pred_mask = restore_org_ecg_mask(
            ecg_signal[0].squeeze(), predicted_labels.squeeze(), len(segment_signal)
        )
        p_peaks, t_peaks = find_p_t_peaks(segment_signal, pred_mask)
        

        if len(t_peaks)>0:
            t_onset, _ = find_onset_offset(segment_signal, t_peaks[0], min_drop_ratio=0.85)
            onset.append(t_onset + r_indices[i])
        for ppeak in p_peaks:
            _, p_offset = find_onset_offset(segment_signal, ppeak, min_drop_ratio=0.85)
            offset.append(p_offset + r_indices[i])

        p_peaks = np.atleast_1d(p_peaks) + r_indices[i]
        t_peaks = np.atleast_1d(t_peaks) + r_indices[i]
        pt_peaks = tuple(list(t_peaks) + list(p_peaks))

        p_peaks_all.extend(p_peaks)
        t_peaks_all.extend(t_peaks)
        pt_peaks_all.extend(pt_peaks)

    return t_peaks_all, p_peaks_all, pt_peaks_all , onset, offset

def check_pt_index(all_lead_data, version, r_peaks, r_index_dic):
    result_dic = {}
    t_peaks, p_peaks, rr_invl_peaks, T_onset, P_offset = [], [], [], [], []
    for lead in  all_lead_data.keys(): 
        if lead in r_index_dic.keys():
            r_peaks = r_index_dic[lead]
            ecg_signal = all_lead_data[lead].values.flatten()
            baseline_signal = baseline_construction_200(ecg_signal, kernel_size=131)
            lowpass_signal = lowpass(baseline_signal, cutoff=0.3)
            signal_normalized = find_normalize(lowpass_signal)
            t_peaks, p_peaks, rr_invl_peaks, T_onset, P_offset = get_pt_peaks(signal_normalized, r_peaks)

            result_dic[lead] = {"p": p_peaks, "t": t_peaks, "comb": rr_invl_peaks, "T_onset":T_onset, "P_offset":P_offset}
    if version == "2":
        p_peaks = result_dic['II']["p"]
        t_peaks = result_dic['II']["t"]
        rr_invl_peaks = result_dic['II']["comb"]
        T_onset = result_dic['II']["T_onset"]
        P_offset = result_dic['II']["P_offset"]
    elif version == "7":
        min_p_length = min(len(result_dic['I']['p']), len(result_dic['II']['p']), len(result_dic['III']['p']))
        median_p_list = [int(np.median([result_dic['I']['p'][i], result_dic['II']['p'][i], result_dic['III']['p'][i]])) for i in range(min_p_length)]
        p_peaks = median_p_list
        min_t_length = min(len(result_dic['I']['t']), len(result_dic['II']['t']), len(result_dic['III']['t']))
        median_t_list = [int(np.median([result_dic['I']['t'][i], result_dic['II']['t'][i], result_dic['III']['t'][i]])) for i in range(min_t_length)]
        t_peaks = median_t_list
        min_comb_length = min(len(result_dic['I']['comb']), len(result_dic['II']['comb']), len(result_dic['III']['comb']))
        median_comb_list = [int(np.median([result_dic['I']['comb'][i], result_dic['II']['comb'][i], result_dic['III']['comb'][i]])) for i in range(min_comb_length)]
        rr_invl_peaks = median_comb_list
        min_Tonset_length = min(len(result_dic['I']['T_onset']), len(result_dic['II']['T_onset']), len(result_dic['III']['T_onset']))
        median_Tonset_list = [int(np.median([result_dic['I']['T_onset'][i], result_dic['II']['T_onset'][i], result_dic['III']['T_onset'][i]])) for i in range(min_Tonset_length)]
        T_onset = median_Tonset_list
        min_Poffset_length = min(len(result_dic['I']['P_offset']), len(result_dic['II']['P_offset']), len(result_dic['III']['P_offset']))
        median_Poffset_list = [int(np.median([result_dic['I']['P_offset'][i], result_dic['II']['P_offset'][i], result_dic['III']['P_offset'][i]])) for i in range(min_Poffset_length)]
        P_offset = median_Poffset_list
    elif version == "7":
        min_p_length = min(len(result_dic['I']['p']), len(result_dic['II']['p']), len(result_dic['III']['p']), len(result_dic['v1']['p']), len(result_dic['v5']['p']))
        median_p_list = [int(np.median([result_dic['I']['p'][i], result_dic['II']['p'][i], result_dic['III']['p'][i], result_dic['v1']['p'][i], result_dic['v5']['p'][i]])) for i in range(min_p_length)]
        p_peaks = median_p_list
        min_t_length = min(len(result_dic['I']['t']), len(result_dic['II']['t']), len(result_dic['III']['t']), len(result_dic['v1']['t']), len(result_dic['v5']['t']))
        median_t_list = [int(np.median([result_dic['I']['t'][i], result_dic['II']['t'][i], result_dic['III']['t'][i], result_dic['v1']['t'][i], result_dic['v5']['t'][i]])) for i in range(min_t_length)]
        t_peaks = median_t_list
        min_comb_length = min(len(result_dic['I']['comb']), len(result_dic['II']['comb']), len(result_dic['III']['comb']), len(result_dic['v1']['comb']), len(result_dic['v5']['comb']))
        median_comb_list = [int(np.median([result_dic['I']['comb'][i], result_dic['II']['comb'][i], result_dic['III']['comb'][i], result_dic['v1']['comb'][i], result_dic['v5']['comb'][i]])) for i in range(min_comb_length)]
        rr_invl_peaks = median_comb_list
        min_Tonset_length = min(len(result_dic['I']['T_onset']), len(result_dic['II']['T_onset']), len(result_dic['III']['T_onset']), len(result_dic['v1']['T_onset']), len(result_dic['v5']['T_onset']))
        median_Tonset_list = [int(np.median([result_dic['I']['T_onset'][i], result_dic['II']['T_onset'][i], result_dic['III']['T_onset'][i], result_dic['v1']['T_onset'][i], result_dic['v5']['T_onset'][i]])) for i in range(min_Tonset_length)]
        T_onset = median_Tonset_list
        min_Poffset_length = min(len(result_dic['I']['P_offset']), len(result_dic['II']['P_offset']), len(result_dic['III']['P_offset']), len(result_dic['v1']['P_offset']), len(result_dic['v5']['P_offset']))
        median_Poffset_list = [int(np.median([result_dic['I']['P_offset'][i], result_dic['II']['P_offset'][i], result_dic['III']['P_offset'][i], result_dic['v1']['P_offset'][i], result_dic['v5']['P_offset'][i]])) for i in range(min_Poffset_length)]
        P_offset = median_Poffset_list

    return result_dic, t_peaks, p_peaks, rr_invl_peaks, T_onset, P_offset

def find_s_indexs(ecg, R_index, d):
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
            s_index = i + np.where(s_array == min(s_array))[0][0]
        else:
            s_index = i + np.where(s_array == max(s_array))[0][0]
        s.append(s_index)
    return s

def find_q_indexs(ecg, R_index, d):
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
            q_index = i - (len(q_array) - np.where(q_array == min(q_array))[0][0])
        else:
            q_index = i - (len(q_array) - np.where(q_array == max(q_array))[0][0])
        q.append(q_index)
    return q

def prediction_model_mi(input_arr):
    classes = ['Abnormal', 'stdep', 'stele', 't_abnormal', 't_invert']
    input_arr = tf.io.decode_jpeg(tf.io.read_file(input_arr), channels=3)
    input_arr = tf.image.resize(input_arr, size=(150,400), method=tf.image.ResizeMethod.BILINEAR)
    input_arr = (tf.expand_dims(input_arr, axis=0),)
    model_pred = predict_tflite_model(mi_model, input_arr )[0]
    # print(model_pred)
    idx = np.argmax(model_pred)
    return model_pred, classes[idx]

def mi_model_check(ecg_data, fs):
        # Create folder inside this script directory
    base_dir = os.path.dirname(os.path.abspath(__file__))
    stseg_dir = os.path.join(base_dir, "STsegimages")
    os.makedirs(stseg_dir, exist_ok=True)
    label = "Normal"
    i = 0
    if ecg_data.shape[0] <= 2500:
        steps = ecg_data.shape[0]
    else:
        steps = round(fs * 10)

    result_count = {'t_invert':0, 'stdep':0, 'stele':0, 't_abnormal':0, 'total_data':0}
    while i < ecg_data.shape[0]:
        ecg_signal = ecg_data[i : i+steps]
        baseline_signal = baseline_construction_200(ecg_signal, kernel_size=101)
        low_ecg_signal = lowpass(baseline_signal, cutoff=0.3)
        for st in glob.glob(stseg_dir + "/*.jpg"):
            os.remove(st)

        randome_number = random.randint(200000, 1000000)
        if len(low_ecg_signal)<700:
            plt.figure()
        else:
            plt.figure(layout="constrained")
        plt.plot(low_ecg_signal, color='blue', linewidth=1.5)
        plt.axis("off")
        plt.savefig(os.path.join(stseg_dir, f"{randome_number}.jpg"))
        temp_img = cv2.imread(os.path.join(stseg_dir, f"{randome_number}.jpg"))
        temp_img = cv2.resize(temp_img, (1200, 290))
        cv2.imwrite(os.path.join(stseg_dir, f"{randome_number}.jpg"), temp_img)
        # plt.show()
        plt.close()

        files = sorted(glob.glob(stseg_dir +"/*.jpg"), key=extract_number)
        for file in files:
            predictions, ids = prediction_model_mi(file)
            if str(ids) == "t_invert" and float(predictions[4]) > 0.70:
                result_count['t_invert'] += 1
            elif str(ids) == "stdep" and float(predictions[1]) > 0.70:
                result_count['stdep'] += 1
            elif str(ids) == "stele" and float(predictions[2]) > 0.70:
                result_count['stele'] += 1
            elif str(ids) == "t_abnormal" and float(predictions[3]) > 0.70:
                result_count['t_abnormal'] += 1
            result_count['total_data'] += 1

        i += steps
    if result_count['total_data'] != 0:
        t_inver_per = result_count['t_invert'] / result_count['total_data']
        stdep_per = result_count['stdep'] / result_count['total_data']
        stele_per = result_count['stele'] / result_count['total_data']
        t_abn_per = result_count['t_abnormal'] / result_count['total_data']
        if t_inver_per > 0.6:
            label = "INVERTED_T"
        elif stdep_per > 0.6:
            label = "STDEP"
        elif stele_per > 0.6:
            label = "STELE"
        elif t_abn_per >= 0.5:
            label = "T_ABNORMAL" 
        else:
            label = "Normal"
    return label

def mi_processing(all_lead_data, is_lead, fs):
    mi_result = 'Abnormal'
    result_dic = {}
    mi_labels = []
    analysis_leads = ["II"]
    if is_lead == "12":
        analysis_leads = ['II', 'III', 'aVF', 'I', 'aVL', 'v5', 'v1', 'v3']
    elif is_lead == "7":
        analysis_leads = ['II', 'III', 'aVF', 'I', 'aVL', 'v5']
    for lead in analysis_leads:
        ecg_signal = all_lead_data[lead].values
        mi_label = mi_model_check(ecg_signal, fs)
        result_dic[lead] = mi_label
        mi_labels.append(mi_label)
    
    if is_lead != '2':# ele[i, avl, v5, v6] dep[iii, avf]latral ------ ele[ii, iii, avf] dep[avl] inf
        if result_dic['II'] == 'STDEP' and result_dic['III'] == 'STDEP' and result_dic['aVF'] == 'STDEP':
            mi_result = 'Inferior STEMI'
        if result_dic['I'] == 'STDEP' and result_dic['aVL'] == 'STDEP' and result_dic['v5'] == 'STDEP':
            mi_result = 'Lateral STEMI'


        counts = Counter(mi_labels)
        repeated_element = [item for item, count in counts.items() if count >= 3]
        if len(repeated_element) > 1 and "Normal" in repeated_element:
            repeated_element.remove("Normal")
        mi_temp_labels = ' '.join(repeated_element)
        if mi_result != "Inferior STEMI" and mi_result != "Lateral STEMI":
            if 'T_ABNORMAL' in mi_temp_labels or 'INVERTED_T' in mi_temp_labels:
                mi_result = "T_wave_Abnormality"
            elif mi_result == "Abnormal":
                mi_result = mi_temp_labels
    else:
        mi_result = result_dic['II']
    return mi_result

def check_qs_index(all_leads_data, r_index ,frequency):
    s_index, q_index = [], []
    combine_indexs = {}
    lead_list = [c for c in all_leads_data.columns if c not in ['DateTime']]

    if lead_list == ['II'] or len(lead_list) == 1:
        is_lead_for = '2'
    elif set(['I', 'II', 'III']).issubset(lead_list):
        if set(['v1', 'v5']).issubset(lead_list):
            is_lead_for = '12'
        else:
            is_lead_for = '7'
    else:
        is_lead_for = 'unknown'
    for lead in all_leads_data.columns:
        if lead in ["I",'II', 'III', 'v1', 'v5']:
            ecg_signal = all_leads_data[lead].values
            baseline_signal = baseline_construction_200(ecg_signal, 101)
            lowpass_signal = lowpass(baseline_signal)
            s_index_list = find_s_indexs(baseline_signal, r_index, 20)
            q_index_list = find_q_indexs(baseline_signal, r_index, 15)
            combine_indexs[lead] = {
                's_idx': s_index_list,
                'q_idx': q_index_list
            }
    if is_lead_for == '7':
        min_s_length = min(len(combine_indexs['I']['s_idx']), len(combine_indexs['II']['s_idx']), len(combine_indexs['III']['s_idx']))
        median_s_list = [int(np.median([combine_indexs['I']['s_idx'][i], combine_indexs['II']['s_idx'][i], combine_indexs['III']['s_idx'][i]])) for i in range(min_s_length)]
        s_index = median_s_list
        min_q_length = min(len(combine_indexs['I']['q_idx']), len(combine_indexs['II']['q_idx']), len(combine_indexs['III']['q_idx']))
        median_q_list = [int(np.median([combine_indexs['I']['q_idx'][i], combine_indexs['II']['q_idx'][i], combine_indexs['III']['q_idx'][i]])) for i in range(min_q_length)]
        q_index = median_q_list
        
    elif is_lead_for == '12':
        min_s_length = min(len(combine_indexs['I']['s_idx']), len(combine_indexs['II']['s_idx']), len(combine_indexs['III']['s_idx']), len(combine_indexs['v1']['s_idx']), len(combine_indexs['v5']['s_idx']))
        median_s_list = [int(np.median([combine_indexs['I']['s_idx'][i], combine_indexs['II']['s_idx'][i], combine_indexs['III']['s_idx'][i], combine_indexs['v1']['s_idx'][i], combine_indexs['v5']['s_idx'][i]])) for i in range(min_s_length)]
        s_index = median_s_list
        min_q_length = min(len(combine_indexs['I']['q_idx']), len(combine_indexs['II']['q_idx']), len(combine_indexs['III']['q_idx']), len(combine_indexs['v1']['q_idx']), len(combine_indexs['v5']['q_idx']))
        median_q_list = [int(np.median([combine_indexs['I']['q_idx'][i], combine_indexs['II']['q_idx'][i], combine_indexs['III']['q_idx'][i], combine_indexs['v1']['q_idx'][i], combine_indexs['v5']['q_idx'][i]])) for i in range(min_q_length)]
        q_index = median_q_list
    else:
        s_index = combine_indexs['II']['s_idx']
        q_index = combine_indexs['II']['q_idx']
    return s_index, q_index

def prediction_model(image_path, target_shape=[224,224], class_name=True): #224, 224
    with results_lock:
        classes = ['LBBB', 'Noise', 'Normal', 'PVC', 'RBBB']
        image = tf.io.read_file(image_path)
        input_arr = tf.image.decode_jpeg(image, channels=3)
        input_arr = tf.image.resize(input_arr, size=target_shape, method=tf.image.ResizeMethod.BILINEAR)
        input_arr = tf.expand_dims(input_arr, axis=0)

        # Set the input tensor
        interpreter.set_tensor(input_details[0]['index'], input_arr)
        
        # Perform inference
        interpreter.invoke()
        # Get the output tensor
        output_data = interpreter.get_tensor(output_details[0]['index'])

    if class_name:
        idx = np.argmax(output_data[0])
        return output_data[0], classes[idx]
    else:
        return output_data[0]
    
def hr_count(ecg_signal, r_index):
    # cal_sec = round(ecg_signal.shape[0]/200)
    cal_sec = round(len(ecg_signal)/200)
    if cal_sec != 0:
        hr = round(len(r_index)*60/cal_sec)
        print(hr)
        return hr
    return 0

class NoiseDetection:
    def __init__(self, raw_data, frequency=200):
        self.frequency = frequency
        self.raw_data = raw_data

    def prediction_model(self, input_arr, noise_model):
        classes = ['Noise', 'Normal']
        input_arr = tf.cast(input_arr, dtype=tf.float32)
        input_arr = tf.image.resize(input_arr, size=(224, 224), method=tf.image.ResizeMethod.BILINEAR)
        input_arr = (tf.expand_dims(input_arr, axis=0),)
        model_pred = predict_tflite_model(noise_model, input_arr)[0]
        idx = np.argmax(model_pred)
        # print(model_pred, classes[idx])
        return classes[idx]

    def plot_to_imagearray(self, ecg_signal):
        # Ensure ecg_signal is a 1D array
        ecg_signal = np.asarray(ecg_signal).ravel()

        # Create the plot
        fig, ax = plt.subplots(num=1, clear=True)
        ax.plot(ecg_signal, color='black')  # Plot the flattened array
        ax.axis(False)  # Hide axes

        # Convert plot to image array
        fig.canvas.draw()
        data = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        
        plt.close(fig)
        return data[:, :, ::-1]
    
    def noise_model_check(self, noise_model):
        # Noise detection logic for individual lead
        steps_data = int(self.frequency * 2.5)
        total_data = self.raw_data.shape[0]
        start_data = 0
        normal_index, noise_index = [], []
        percentage = {'Normal': 0, 'Noise': 0, 'total_slice': 0}
        
        while start_data < total_data:
            end_data = start_data + steps_data
            
            if end_data - start_data == steps_data and end_data < total_data:
                img_data = pd.DataFrame(self.raw_data[start_data:end_data])
            else:
                img_data = pd.DataFrame(self.raw_data[-steps_data:total_data])
            end_data = total_data - 1
            
            # Assuming the noise detection model uses image input
            data1 = self.plot_to_imagearray(img_data)

            # Get noise model result for the image
            model_result = self.prediction_model(data1, noise_model)
            percentage['total_slice'] += 1
            if model_result == 'Normal':
                normal_index.append((start_data, end_data))
                # percentage['Normal'] += (end_data - start_data) / total_data
                percentage['Normal'] += 1
            else:
                noise_index.append((start_data, end_data))
                # percentage['Noise'] += (end_data - start_data) / total_data
                percentage['Noise'] += 1

            start_data += steps_data

        # If the percentage of noise is high, return 'ARTIFACTS'
        noise_label = 'Normal'
        # if int(percentage['Noise'] * 100) >= 60:
        #     noise_label = 'ARTIFACTS'
        if percentage['total_slice'] != 0:
            if percentage['Noise'] == percentage['total_slice']:
                noise_label = 'ARTIFACTS'
            elif (percentage['Noise']/percentage['total_slice'])  >= 0.6:
                noise_label = 'ARTIFACTS'
        return noise_label

def check_noise(all_leads_data, is_lead_for, noise_model):
    noise_result = []
    final_result = 'Normal'
    fs = 200
    noise_result_dic = {}

    for lead in all_leads_data.keys():
        ecg_signal = all_leads_data[lead]
        ecg_signal = np.asarray(ecg_signal).ravel()
        get_noise = NoiseDetection(ecg_signal, frequency=fs).noise_model_check(noise_model)
        noise_result.append(get_noise)
        noise_result_dic[lead] = get_noise
    noise_cou = noise_result.count('ARTIFACTS')
    # if noise_cou >= len(all_leads_data.keys()) / 2:
    #     final_result = 'ARTIFACTS'
    if is_lead_for == '2':
        if noise_result_dic['II'] == 'ARTIFACTS':
            final_result = 'ARTIFACTS'
    else:
        if noise_result_dic['I'] == 'ARTIFACTS' and noise_result_dic['II'] == 'ARTIFACTS':
            final_result = 'ARTIFACTS'
        elif noise_result_dic['I'] == 'ARTIFACTS' and noise_result_dic['v5'] == 'ARTIFACTS':
            final_result = 'ARTIFACTS'
    print(f"Final noise result: {final_result} (ARTIFACT count: {noise_cou})")
    
    return final_result


def remove_temp_folder(folder_path):
    def on_rm_error(func, path, exc_info):
        os.chmod(path, 0o777)  # Change permission
        func(path)  # Retry deleting

    if os.path.exists(folder_path):
        shutil.rmtree(folder_path, onerror=on_rm_error)
    else:
        print(f"Folder '{folder_path}' does not exist.")

def prediction_model_vfib_vfl(input_arr):
    classes = ['VFIB', 'asystole', 'noise', 'normal']
    input_arr = tf.io.decode_jpeg(tf.io.read_file(input_arr), channels=3)
    input_arr = tf.image.resize(input_arr, size=(224, 224), method=tf.image.ResizeMethod.BILINEAR)
    input_arr = (tf.expand_dims(input_arr, axis=0),)
    model_pred = predict_tflite_model(vfib_model, input_arr )[0]
    idx = np.argmax(model_pred)
    return model_pred, classes[idx]

def extract_number(filename):
    match = re.search(r'(\d+)', os.path.basename(filename))
    return int(match.group(1)) if match else float('inf')

def check_vfib_vfl_model(low_ecg_signal, fs, is_lead):
    label = 'Abnormal'
    temp_uuid = str(uuid.uuid1())
    temp_images=r'D:\try\Scripts_Models\temp_images'
    folder_path = os.path.join(temp_images,"vflutter_img",temp_uuid)
    os.makedirs(folder_path)


    total_data = len(low_ecg_signal)
    if is_lead in ["2", "7"]:
        total_data_len = 2300
    else:
        total_data_len = 3000

    if total_data <= total_data_len:
        step_size = total_data
    else:
        step_size = round(fs * 10)

    fi = 0
    while fi < total_data:
        temp_img = low_ecg_signal[fi: fi + step_size]
        fi += step_size
        plt.figure()
        plt.plot(temp_img)
        plt.axis("off")
        img_path = f"{folder_path}/p_{fi}.jpg"
        plt.savefig(img_path)
        plt.close()
        aq = cv2.imread(img_path)
        aq = cv2.resize(aq, (1080, 460))
        cv2.imwrite(img_path, aq)

    combine_result = []
    label = 'Abnormal'

    files = sorted(glob.glob(f"{folder_path}/*.jpg"), key=extract_number)
    for vfib_file in files:
        # with tf.device("CPU"):
        predictions, ids = prediction_model_vfib_vfl(vfib_file)
        label = "Abnormal" #"Normal"
        if str(ids) == "VFIB" and float(predictions[0]) > 0.75:
            label = "VFIB"
            combine_result.append(label)
        elif str(ids) == "asystole" and float(predictions[1]) > 0.75:
            label = "Asystole"
            combine_result.append(label)
        elif str(ids) == "noise" and float(predictions[2]) > 0.75:
            label = "Noise"
            combine_result.append(label)
        elif str(ids) == "normal" and float(predictions[3]) > 0.75:
            label = "Normal"
            combine_result.append(label)
        else:
            combine_result.append(label)

    for img_path in glob.glob(f'{folder_path}/*.jpg'):
        os.remove(img_path)
    
    remove_temp_folder(folder_path)

    temp_label = list(set(combine_result)) 
    if len(temp_label) > 1:
        label='Abnormal'
        if 'Asystole' in temp_label:
            label = 'Asystole'
        elif 'Noise' in temp_label:
            label = 'Noise'
        elif 'Abnormal' in temp_label:
            temp_label.remove('Abnormal')
            if temp_label:
                label = temp_label[0]
            else:
                label = 'Abnormal'
    else:
        label = temp_label[0]
    return label
   
def Vfib_asys_detection(all_leads_data, frequency, is_lead):
    vifib_asys_result = 'Abnormal'
    all_lead_det_data = {}
    vifib_results = []
    if is_lead == "12":
        rep_thresh = 2
        analysis_lead = ['I','II','III', 'v1', 'v5']
    elif is_lead == "7":
        rep_thresh = 1
        analysis_lead = ['I','II','III']
    else:
        analysis_lead = ['II']
    for lead in all_leads_data.columns:
        lead_data = {}
        if lead in analysis_lead:
            ecg_signal = all_leads_data[lead].values
            # baseline_signal = baseline_construction_200(ecg_signal, 101)
            # lowpass_signal = lowpass(baseline_signal, cutoff=0.2)
            mi_model_result = check_vfib_vfl_model(ecg_signal, frequency, is_lead)
            lead_data['vfib_result'] = mi_model_result
            all_lead_det_data[lead] = lead_data
            vifib_results.append(mi_model_result)
    if len(all_lead_det_data.keys()) > 1:
        flat_list = []
        for element in vifib_results:
            if isinstance(element, list):
                flat_list.extend(element)
            else:
                flat_list.append(element)
        counts = Counter(flat_list)
        repeated_elements = [item for item, count in counts.items() if count > rep_thresh and item != 'Abnormal']
        vfib_vfl_lab = ' '.join(repeated_elements)
        if vfib_vfl_lab:
            vifib_asys_result = vfib_vfl_lab
        else:
            vifib_asys_result = "Abnormal"
    else:
        vifib_asys_result =  all_lead_det_data['II']['vfib_result']
    return vifib_asys_result

# Low pass and baseline signal
class filter_signal:

    def __init__(self, ecg_signal, fs=200):
        self.ecg_signal = ecg_signal
        self.fs = fs
        self.baseline_signal = None

    def baseline_construction_200(self, kernel_size=131):
        """Removes the baseline from an ECG signal using a median filter
        of a specified kernel size.

        Args:
            ecg_signal (array): The ECG signal
            kernel_size (int, optional): The kernel_size parameter is the size of the median filter
        kernel used for baseline correction. Defaults to 101 (optional).

        Returns:
            array: The baseline-corrected ECG signal.
        """
        s_corrected = signal.detrend(self.ecg_signal)
        baseline_corrected = s_corrected - signal.medfilt(s_corrected, kernel_size)
        return baseline_corrected

    def baseline_als(self, file, lam, p, niter=10):
        L = len(file)
        D = sparse.csc_matrix(np.diff(np.eye(L), 2))
        w = np.ones(L)
        for i in range(niter):
            W = sparse.spdiags(w, 0, L, L)
            Z = W + lam * D.dot(D.transpose())
            z = spsolve(Z, w * file)
            w = p * (file > z) + (1 - p) * (file < z)
        return z

    def baseline_construction_250(self, kernel_size=131):
        als_baseline = self.baseline_als(self.ecg_signal, 16 ** 5, 0.01)
        s_als = self.ecg_signal - als_baseline
        s_corrected = signal.detrend(s_als)
        corrected_baseline = s_corrected - medfilt(s_corrected, kernel_size)
        return corrected_baseline

    def lowpass(self, cutoff=0.3):
        """A lowpass filter to a given file using the Butterworth filter.

        Args:
            signal (array): ECG Signal
            cutoff (float): 0.3 for PVC & 0.2 AFIB

        Returns:
            array: the low-pass filtered signal of the input file.
        """
        b, a = signal.butter(3, cutoff, btype='lowpass', analog=False)
        low_passed = signal.filtfilt(b, a, self.baseline_signal)
        return low_passed

    def get_data(self):
        lowpass_signal = None

        # if self.fs != 200:
        #     self.ecg_signal = MinMaxScaler(feature_range=(0, 4)).fit_transform(self.ecg_signal.reshape(-1, 1)).squeeze()
        if self.fs == 200:
            self.baseline_signal = self.baseline_construction_200(kernel_size=101)
            lowpass_signal = self.lowpass(cutoff=0.3)
        elif self.fs == 250:
            self.baseline_signal = self.baseline_construction_250(kernel_size=131)
            lowpass_signal = self.lowpass(cutoff=0.25)
        elif self.fs == 360:
            self.baseline_signal = self.baseline_construction_200(kernel_size=151)
            lowpass_signal = self.lowpass(cutoff=0.2)
        elif self.fs == 1000:
            self.baseline_signal = self.baseline_construction_200(kernel_size=399)
            lowpass_signal = self.lowpass(cutoff=0.05)
        elif self.fs == 128:
            self.baseline_signal = self.baseline_construction_200(kernel_size=101)
            lowpass_signal = self.lowpass(cutoff=0.5)
        else:
            self.baseline_signal = self.baseline_construction_200(kernel_size=101)
            lowpass_signal = self.lowpass(cutoff=0.5)
            # raise ValueError(f"Unsupported sampling frequency: {self.fs}")

        return self.baseline_signal, lowpass_signal

# Peak detection
class pqrst_detection:
    def __init__(self, ecg_signal, fs=200, thres=0.5, lp_thres=0.2, rr_thres=0.12, width=(5, 50), JR=False):
        self.ecg_signal = ecg_signal
        self.fs = fs
        self.thres = thres
        self.lp_thres = lp_thres
        self.rr_thres = rr_thres
        self.width = width
        self.JR = JR

    def hamilton_segmenter(self):

        if self.ecg_signal is None:
            print("Please specify an input signal.")

        sampling_rate = float(self.fs)
        length = len(self.ecg_signal)
        dur = length / sampling_rate

        # algorithm parameters
        v1s = int(1.0 * sampling_rate)
        v100ms = int(0.1 * sampling_rate)
        TH_elapsed = np.ceil(0.36 * sampling_rate)
        sm_size = int(0.08 * sampling_rate)
        init_ecg = 10  # seconds for initialization
        if dur < init_ecg:
            init_ecg = int(dur)

        # filtering
        filtered, _, _ = st.filter_signal(
            signal=self.ecg_signal,
            ftype="butter",
            band="lowpass",
            order=4,
            frequency=20.0,
            sampling_rate=sampling_rate,
        )
        filtered, _, _ = st.filter_signal(
            signal=filtered,
            ftype="butter",
            band="highpass",
            order=4,
            frequency=3.0,
            sampling_rate=sampling_rate,
        )

        # diff
        dx = np.abs(np.diff(filtered, 1) * sampling_rate)

        # smoothing
        dx, _ = st.smoother(signal=dx, kernel="hamming", size=sm_size, mirror=True)

        # buffers
        qrspeakbuffer = np.zeros(init_ecg)
        noisepeakbuffer = np.zeros(init_ecg)
        peak_idx_test = np.zeros(init_ecg)
        noise_idx = np.zeros(init_ecg)
        rrinterval = sampling_rate * np.ones(init_ecg)

        a, b = 0, v1s
        all_peaks, _ = st.find_extrema(signal=dx, mode="max")
        for i in range(init_ecg):
            peaks, values = st.find_extrema(signal=dx[a:b], mode="max")
            try:
                ind = np.argmax(values)
            except ValueError:
                pass
            else:
                # peak amplitude
                qrspeakbuffer[i] = values[ind]
                # peak location
                peak_idx_test[i] = peaks[ind] + a

            a += v1s
            b += v1s

        # thresholds
        ANP = np.median(noisepeakbuffer)
        AQRSP = np.median(qrspeakbuffer)
        TH = 0.475
        DT = ANP + TH * (AQRSP - ANP)
        DT_vec = []
        indexqrs = 0
        indexnoise = 0
        indexrr = 0
        npeaks = 0
        offset = 0

        beats = []

        # detection rules
        # 1 - ignore all peaks that precede or follow larger peaks by less than 200ms
        lim = int(np.ceil(0.15 * sampling_rate))
        diff_nr = int(np.ceil(0.045 * sampling_rate))
        bpsi, bpe = offset, 0

        for f in all_peaks:
            DT_vec += [DT]
            # 1 - Checking if f-peak is larger than any peak following or preceding it by less than 200 ms
            peak_cond = np.array(
                (all_peaks > f - lim) * (all_peaks < f + lim) * (all_peaks != f)
            )
            peaks_within = all_peaks[peak_cond]
            if peaks_within.any() and (max(dx[peaks_within]) > dx[f]):
                continue

            # 4 - If the peak is larger than the detection threshold call it a QRS complex, otherwise call it noise
            if dx[f] > DT:
                # 2 - look for both positive and negative slopes in raw signal
                if f < diff_nr:
                    diff_now = np.diff(self.ecg_signal[0: f + diff_nr])
                elif f + diff_nr >= len(self.ecg_signal):
                    diff_now = np.diff(self.ecg_signal[f - diff_nr: len(dx)])
                else:
                    diff_now = np.diff(self.ecg_signal[f - diff_nr: f + diff_nr])
                diff_signer = diff_now[diff_now > 0]
                if len(diff_signer) == 0 or len(diff_signer) == len(diff_now):
                    continue
                # RR INTERVALS
                if npeaks > 0:
                    # 3 - in here we check point 3 of the Hamilton paper
                    # that is, we check whether our current peak is a valid R-peak.
                    prev_rpeak = beats[npeaks - 1]

                    elapsed = f - prev_rpeak
                    # if the previous peak was within 360 ms interval
                    if elapsed < TH_elapsed:
                        # check current and previous slopes
                        if prev_rpeak < diff_nr:
                            diff_prev = np.diff(self.ecg_signal[0: prev_rpeak + diff_nr])
                        elif prev_rpeak + diff_nr >= len(self.ecg_signal):
                            diff_prev = np.diff(self.ecg_signal[prev_rpeak - diff_nr: len(dx)])
                        else:
                            diff_prev = np.diff(
                                self.ecg_signal[prev_rpeak - diff_nr: prev_rpeak + diff_nr]
                            )

                        slope_now = max(diff_now)
                        slope_prev = max(diff_prev)

                        if slope_now < 0.5 * slope_prev:
                            # if current slope is smaller than half the previous one, then it is a T-wave
                            continue
                    if dx[f] < 3.0 * np.median(qrspeakbuffer):  # avoid retarded noise peaks
                        beats += [int(f) + bpsi]
                    else:
                        continue

                    if bpe == 0:
                        rrinterval[indexrr] = beats[npeaks] - beats[npeaks - 1]
                        indexrr += 1
                        if indexrr == init_ecg:
                            indexrr = 0
                    else:
                        if beats[npeaks] > beats[bpe - 1] + v100ms:
                            rrinterval[indexrr] = beats[npeaks] - beats[npeaks - 1]
                            indexrr += 1
                            if indexrr == init_ecg:
                                indexrr = 0

                elif dx[f] < 3.0 * np.median(qrspeakbuffer):
                    beats += [int(f) + bpsi]
                else:
                    continue

                npeaks += 1
                qrspeakbuffer[indexqrs] = dx[f]
                peak_idx_test[indexqrs] = f
                indexqrs += 1
                if indexqrs == init_ecg:
                    indexqrs = 0
            if dx[f] <= DT:
                tf = f + bpsi
                # RR interval median
                RRM = np.median(rrinterval)  # initial values are good?

                if len(beats) >= 2:
                    elapsed = tf - beats[npeaks - 1]

                    if elapsed >= 1.5 * RRM and elapsed > TH_elapsed:
                        if dx[f] > 0.5 * DT:
                            beats += [int(f) + offset]
                            # RR INTERVALS
                            if npeaks > 0:
                                rrinterval[indexrr] = beats[npeaks] - beats[npeaks - 1]
                                indexrr += 1
                                if indexrr == init_ecg:
                                    indexrr = 0
                            npeaks += 1
                            qrspeakbuffer[indexqrs] = dx[f]
                            peak_idx_test[indexqrs] = f
                            indexqrs += 1
                            if indexqrs == init_ecg:
                                indexqrs = 0
                    else:
                        noisepeakbuffer[indexnoise] = dx[f]
                        noise_idx[indexnoise] = f
                        indexnoise += 1
                        if indexnoise == init_ecg:
                            indexnoise = 0
                else:
                    noisepeakbuffer[indexnoise] = dx[f]
                    noise_idx[indexnoise] = f
                    indexnoise += 1
                    if indexnoise == init_ecg:
                        indexnoise = 0

            # Update Detection Threshold
            ANP = np.median(noisepeakbuffer)
            AQRSP = np.median(qrspeakbuffer)
            DT = ANP + 0.475 * (AQRSP - ANP)

        beats = np.array(beats)

        r_beats = []
        thres_ch = 1
        adjacency = 0.01 * sampling_rate
        for i in beats:
            error = [False, False]
            if i - lim < 0:
                window = self.ecg_signal[0: i + lim]
                add = 0
            elif i + lim >= length:
                window = self.ecg_signal[i - lim: length]
                add = i - lim
            else:
                window = self.ecg_signal[i - lim: i + lim]
                add = i - lim
            # meanval = np.mean(window)
            w_peaks, _ = st.find_extrema(signal=window, mode="max")
            w_negpeaks, _ = st.find_extrema(signal=window, mode="min")
            zerdiffs = np.where(np.diff(window) == 0)[0]
            w_peaks = np.concatenate((w_peaks, zerdiffs))
            w_negpeaks = np.concatenate((w_negpeaks, zerdiffs))

            pospeaks = sorted(zip(window[w_peaks], w_peaks), reverse=True)
            negpeaks = sorted(zip(window[w_negpeaks], w_negpeaks))

            try:
                twopeaks = [pospeaks[0]]
            except IndexError:
                twopeaks = []
            try:
                twonegpeaks = [negpeaks[0]]
            except IndexError:
                twonegpeaks = []

            # getting positive peaks
            for i in range(len(pospeaks) - 1):
                if abs(pospeaks[0][1] - pospeaks[i + 1][1]) > adjacency:
                    twopeaks.append(pospeaks[i + 1])
                    break
            try:
                posdiv = abs(twopeaks[0][0] - twopeaks[1][0])
            except IndexError:
                error[0] = True

            # getting negative peaks
            for i in range(len(negpeaks) - 1):
                if abs(negpeaks[0][1] - negpeaks[i + 1][1]) > adjacency:
                    twonegpeaks.append(negpeaks[i + 1])
                    break
            try:
                negdiv = abs(twonegpeaks[0][0] - twonegpeaks[1][0])
            except IndexError:
                error[1] = True

            # choosing type of R-peak
            n_errors = sum(error)
            try:
                if not n_errors:
                    if posdiv > thres_ch * negdiv:
                        # pos noerr
                        r_beats.append(twopeaks[0][1] + add)
                    else:
                        # neg noerr
                        r_beats.append(twonegpeaks[0][1] + add)
                elif n_errors == 2:
                    if abs(twopeaks[0][1]) > abs(twonegpeaks[0][1]):
                        # pos allerr
                        r_beats.append(twopeaks[0][1] + add)
                    else:
                        # neg allerr
                        r_beats.append(twonegpeaks[0][1] + add)
                elif error[0]:
                    # pos poserr
                    r_beats.append(twopeaks[0][1] + add)
                else:
                    # neg negerr
                    r_beats.append(twonegpeaks[0][1] + add)
            except IndexError:
                continue

        rpeaks = sorted(list(set(r_beats)))
        rpeaks = np.array(rpeaks, dtype="int")

        return utils.ReturnTuple((rpeaks,), ("rpeaks",))

    def hr_count(self):
        cal_sec = round(self.ecg_signal.shape[0]/200)
        if cal_sec != 0:
            hr = round(self.r_index.shape[0]*60/cal_sec)
            return hr
        return 0

    def fir_lowpass_filter(self, data, cutoff, numtaps=21):
        """A finite impulse response (FIR) lowpass filter to a given data using a
        specified cutoff frequency and number of filter taps.

        Args:
            data (array): The input data to be filtered
            cutoff (float): The cutoff frequency of the lowpass filter, specified in the same units as the
        sampling frequency of the input data. It determines the frequency below which the filter allows
        signals to pass through and above which it attenuates them
            numtaps (int, optional): the number of coefficients (taps) in the FIR filter. Defaults to 21.

        Returns:
            array: The filtered signal 'y' after applying a lowpass filter with a specified cutoff frequency
        and number of filter taps to the input signal 'data'.
        """
        b = firwin(numtaps, cutoff)
        y = signal.convolve(data, b, mode="same")
        return y

    def find_j_index(self):
        """The index of the maximum value in a given range of a file and returns a list of
        those indices.

        Args:
            signal (array): ECG signal values
            s_index (list/array): _description_
            fs (int, optional): sampling rate of the ECG signal, defaults to 200 (optional)

        Returns:
            list: Indices (j) where the maximum value is found in a specific range of the input
        ecg_signal (signal) defined by the start indices (s_index).
        """
        j = []
        increment = int(self.fs * 0.05)
        for z in range(0, len(self.s_index)):
            data = []
            j_index = self.ecg_signal[self.s_index[z]:self.s_index[z] + increment]
            for k in range(0, len(j_index)):
                data.append(j_index[k])
            max_d = max(data)
            max_id = data.index(max_d)
            j.append(self.s_index[z] + max_id)
        return j

    def find_s_index(self, d):
        d = int(d) + 1
        s = []
        for i in self.r_index:
            if i == len(self.ecg_signal):
                s.append(i)
                continue
            elif i + d <= len(self.ecg_signal):
                s_array = self.ecg_signal[i:i + d]
            else:
                s_array = self.ecg_signal[i:]
            if self.ecg_signal[i] > 0:
                s_index = i + np.where(s_array == min(s_array))[0][0]
            else:
                s_index = i + np.where(s_array == max(s_array))[0][0]
                if abs(s_index - i) < d / 2:
                    s_index_ = i + np.where(s_array == min(s_array))[0][0]
                    if abs(s_index_ - i) > d / 2:
                        s_index = s_index_
            s.append(s_index)
        return np.sort(s)

    def find_new_q_index(self, d):
        q = []
        for i in self.r_index:
            q_ = []
            if i == 0:
                q.append(i)
                continue
            if self.ecg_signal[i] > 0:
                c = i
                while c > 0 and self.ecg_signal[c - 1] < self.ecg_signal[c]:
                    c -= 1
                if self.ecg_signal[i] * 0.01 > self.ecg_signal[c] or self.ecg_signal[c] < 0 or c == 0:
                    if abs(i - c) <= d:
                        q.append(c)
                        continue
                    else:
                        q_.append(c)
                while c > 0:
                    while c > 0 and self.ecg_signal[c - 1] > self.ecg_signal[c]:
                        c -= 1
                    # q_.append(c)
                    while c > 0 and self.ecg_signal[c - 1] < self.ecg_signal[c]:
                        c -= 1
                    if q_ and q_[-1] == c:
                        break
                    q_.append(c)
                    if self.ecg_signal[i] * 0.01 > self.ecg_signal[c] or self.ecg_signal[c] < 0 or c == 0:
                        break
            else:
                c = i
                while c > 0 and self.ecg_signal[c - 1] > self.ecg_signal[c]:
                    c -= 1
                if self.ecg_signal[i] * 0.01 < self.ecg_signal[c] or self.ecg_signal[c] > 0 or c == 0:
                    if abs(i - c) <= d:
                        q.append(c)
                        continue
                    else:
                        q_.append(c)
                while c > 0:
                    while c > 0 and self.ecg_signal[c - 1] < self.ecg_signal[c]:
                        c -= 1
                    # q_.append(c)
                    while c > 0 and self.ecg_signal[c - 1] > self.ecg_signal[c]:
                        c -= 1
                    if q_ and q_[-1] == c:
                        break
                    q_.append(c)
                    if self.ecg_signal[i] * 0.01 < self.ecg_signal[c] or self.ecg_signal[c] > 0 or c == 0:
                        break
            if q_:
                a = 0
                for _q in q_[::-1]:
                    if abs(i - _q) <= d:
                        a = 1
                        q.append(_q)
                        break
                if a == 0:
                    q.append(q_[0])
        return np.sort(q)

    def find_new_s_index(self, d):
        s = []
        end_index = len(self.ecg_signal)
        for i in self.r_index:
            s_ = []
            if i == len(self.ecg_signal):
                s.append(i)
                continue
            if self.ecg_signal[i] > 0:
                c = i
                while c + 1 < end_index and self.ecg_signal[c + 1] < self.ecg_signal[c]:
                    c += 1
                if self.ecg_signal[i] * 0.01 > self.ecg_signal[c] or self.ecg_signal[c] < 0 or c == end_index - 1:
                    if abs(i - c) <= d:
                        s.append(c)
                        continue
                    else:
                        s_.append(c)
                while c + 1 < end_index:
                    while c + 1 < end_index and self.ecg_signal[c + 1] > self.ecg_signal[c]:
                        c += 1
                    while c + 1 < end_index and self.ecg_signal[c + 1] < self.ecg_signal[c]:
                        c += 1
                    if s_ and s_[-1] == c:
                        break
                    s_.append(c)
                    if self.ecg_signal[i] * 0.01 > self.ecg_signal[c] or self.ecg_signal[c] < 0 or c == end_index - 1:
                        break
            else:
                c = i
                while c + 1 < end_index and self.ecg_signal[c + 1] > self.ecg_signal[c]:
                    c += 1
                if self.ecg_signal[i] * 0.01 < self.ecg_signal[c] or self.ecg_signal[c] > 0 or c == end_index - 1:
                    if abs(i - c) <= d:
                        s.append(c)
                        continue
                    else:
                        s_.append(c)
                while c < end_index:
                    while c + 1 < end_index and self.ecg_signal[c + 1] > self.ecg_signal[c]:
                        c += 1
                    while c + 1 < end_index and self.ecg_signal[c + 1] < self.ecg_signal[c]:
                        c += 1
                    if s_ and s_[-1] == c:
                        break
                    s_.append(c)
                    if self.ecg_signal[i] * 0.01 < self.ecg_signal[c] or self.ecg_signal[c] > 0 or c == end_index - 1:
                        break
            if s_:
                a = 0
                for _s in s_[::-1]:
                    if abs(i - _s) <= d:
                        a = 1
                        s.append(_s)
                        break
                if a == 0:
                    s.append(s_[0])
        return np.sort(s)
    
    def find_r_peaks(self):
        ecg_signal = self.ecg_signal
        baseline_signal = baseline_construction_200(ecg_signal, 131)
        lowpass_signal = lowpass(baseline_signal, cutoff=0.3) 
        signal_normalized = find_normalize(lowpass_signal)
        rpeaks = check_model_r(signal_normalized)
        return np.array(rpeaks)

    def pt_detection_1(self):
        """Detects peaks in a given signal within a specified range and returns the peak indices.

        Args:
            ecg_signal (array): ECG signal
            r_index (list/array): Indices representing the R-peaks in an ECG signal
            q_index (_type_): Indices representing the Q waves in an ECG signal
            s_index (_type_): Indices representing the S waves in an ECG signal
            width (_type_): In the find_peaks function to specify the minimum width of
        peaks to be detected. It is a positive integer value

        Returns:
            tuple: two lists: pt and p_t.
        """
        max_signal = max(self.ecg_signal) / 100
        pt = []
        p_t = []
        for i in range(0, len(self.r_index) - 1):
            aoi = self.ecg_signal[self.s_index[i]:self.q_index[i + 1]]
            max_signal = max(self.ecg_signal) / 100
            low = self.fir_lowpass_filter(aoi, self.lp_thres, 30)
            if self.ecg_signal[self.r_index[i]] < 0:
                max_signal = 0.05
            else:
                max_signal = max_signal
            if aoi.any():
                peaks, _ = find_peaks(low, height=max_signal, width=self.width)
                peaks1 = peaks + (self.s_index[i])
            else:
                peaks1 = [0]
            p_t.append(list(peaks1))
            pt.extend(list(peaks1))
            for i in range(len(p_t)):
                if not p_t[i]:
                    p_t[i] = [0]
        return pt, p_t

    def pt_detection_2(self):
        """Detects peaks in a given signal within a specified range and returns the peak indices.

        Args:
            ecg_signal (array): ECG signal
            r_index (list/array): Indices representing the R-peaks in an ECG signal
            q_index (_type_): Indices representing the Q waves in an ECG signal
            s_index (_type_): Indices representing the S waves in an ECG signal
            width (_type_): In the find_peaks function to specify the minimum width of
        peaks to be detected. It is a positive integer value

        Returns:
            tuple: two lists: pt and p_t.
        """
        pt = []
        p_t = []
        for i in range(0, len(self.r_index) - 1):
            aoi = self.ecg_signal[self.s_index[i]:self.q_index[i + 1]]
            if aoi.any():
                low = self.fir_lowpass_filter(aoi, self.lp_thres, 30)
                if self.ecg_signal[self.r_index[i]] < 0:
                    max_signal = 0.05
                else:
                    max_signal = max(low) * 0.2
                if aoi.any():
                    peaks, _ = find_peaks(low, height=max_signal, width=self.width)
                    peaks1 = peaks + (self.s_index[i])
                else:
                    peaks1 = [0]
                p_t.append(list(peaks1))
                pt.extend(list(peaks1))
                for i in range(len(p_t)):
                    if not p_t[i]:
                        p_t[i] = [0]
            else:
                p_t.append([0])
        return pt, p_t

    def pt_detection_3(self):
        """Detects peaks in a given signal within a specified range and returns the peak indices.

        Args:
            ecg_signal (array): ECG signal
            r_index (list/array): Indices representing the R-peaks in an ECG signal
            q_index (_type_): Indices representing the Q waves in an ECG signal
            s_index (_type_): Indices representing the S waves in an ECG signal
            width (_type_): In the find_peaks function to specify the minimum width of
        peaks to be detected. It is a positive integer value

        Returns:
            tuple: two lists: pt and p_t.
        """
        pt = []
        p_t = []
        for i in range(0, len(self.r_index) - 1):
            aoi = self.ecg_signal[self.s_index[i]:self.q_index[i + 1]]
            low = self.fir_lowpass_filter(aoi, self.lp_thres, 30)
            if aoi.any():
                peaks, _ = find_peaks(low, prominence=0.05, width=self.width)
                peaks1 = peaks + (self.s_index[i])
            else:
                peaks1 = [0]
            p_t.append(list(peaks1))
            pt.extend(list(peaks1))
            for i in range(len(p_t)):
                if not p_t[i]:
                    p_t[i] = [0]

        return pt, p_t

    def pt_detection_4(self):
        """Detects peaks in a given signal within a specified range and returns the peak indices.

        Args:
            b_signal (array): ECG signal
            r_index (list/array): Indices representing the R-peaks in an ECG signal
            q_index (_type_): Indices representing the Q waves in an ECG signal
            s_index (_type_): Indices representing the S waves in an ECG signal
            width (_type_): In the find_peaks function to specify the minimum width of
        peaks to be detected. It is a positive integer value

        Returns:
            tuple: two lists: pt and p_t.
        """

        def all_peaks_7(arr):
            """The indices of all peaks in the array, where a peak is
            defined as a point that is higher than its neighboring points.

            Args:
                arr (array): An input array of numbers

            Returns:
                array: The function `all_peaks_7` returns a sorted numpy array of indices where peaks occur in
            the input array `arr`.
            """
            sign_arr = np.sign(np.diff(arr))
            pos = np.where(np.diff(sign_arr) == -2)[0] + 1
            neg = np.where(np.diff(sign_arr) == 2)[0] + 1
            all_peaks = np.sort(np.concatenate((pos, neg)))
            al = all_peaks.tolist()
            diff = {}
            P, Pa, Pb = [], [], []
            if len(al) > 2:
                for p in pos:
                    index = al.index(p)
                    if index == 0:
                        m, n, o = arr[0], arr[al[index]], arr[al[index + 1]]
                    elif index == len(al) - 1:
                        m, n, o = arr[al[index - 1]], arr[al[index]], arr[-1]
                    else:
                        m, n, o = arr[al[index - 1]], arr[al[index]], arr[al[index + 1]]
                    diff[p] = [abs(n - m), abs(n - o)]
                th = np.mean([np.mean([v, m]) for v, m in diff.values()]) * .66
                for p, (a, b) in diff.items():
                    if a >= th and b >= th:
                        P.append(p)
                        continue
                    if a >= th and not Pa:
                        Pa.append(p)
                    elif a >= th and arr[p] > arr[Pa[-1]] and np.where(pos == Pa[-1])[0] + 1 == np.where(pos == p)[0]:
                        Pa[-1] = p
                    elif a >= th:
                        Pa.append(p)
                    if b >= th and not Pb:
                        Pb.append(p)
                    elif b >= th and arr[p] < arr[Pb[-1]] and np.where(pos == Pb[-1])[0] + 1 == np.where(pos == p)[0]:
                        Pb[-1] = p
                    elif b >= th:
                        Pb.append(p)
                if len(pos) > 1:
                    for i in range(1, len(pos)):
                        m, n = pos[i - 1], pos[i]
                        if m in Pa and n in Pb:
                            P.append(m) if arr[m] > arr[n] else P.append(n)
            else:
                P = pos
            return np.sort(P)

        pt, p_t = [], []
        for i in range(1, len(self.r_index)):
            q0, r0, s0 = self.q_index[i - 1], self.r_index[i - 1], self.s_index[i - 1]
            q1, r1, s1 = self.q_index[i], self.r_index[i], self.s_index[i]
            arr = self.ecg_signal[s0 + 7:q1 - 7]
            peaks = list(all_peaks_7(arr) + s0 + 7)
            if peaks:
                pt.extend(peaks)
                p_t.append(peaks)
            else:
                p_t.append([0])
        return pt, p_t

    def find_pt(self):
        _, p_t1 = self.pt_detection_1()
        _, p_t2 = self.pt_detection_2()
        _, p_t3 = self.pt_detection_3()
        _, p_t4 = self.pt_detection_4()
        pt = []
        p_t = []
        for i in range(len(p_t1)):
            _ = []
            for _pt in set(p_t1[i] + p_t2[i] + p_t3[i] + p_t4[i]):
                count = 0
                if any(val in p_t1[i] for val in range(_pt - 2, _pt + 3)):
                    count += 1
                if any(val in p_t2[i] for val in range(_pt - 2, _pt + 3)):
                    count += 1
                if any(val in p_t3[i] for val in range(_pt - 2, _pt + 3)):
                    count += 1
                if any(val in p_t4[i] for val in range(_pt - 2, _pt + 3)):
                    count += 1
                if count >= 3:
                    _.append(_pt)
                _.sort()
            if _:
                p_t.append(_)
            else:
                p_t.append([0])
        result = []
        for sublist in p_t:
            temp = [sublist[0]]
            for i in range(1, len(sublist)):
                if abs(sublist[i] - sublist[i - 1]) > 5:
                    temp.append(sublist[i])
                else:
                    temp[-1] = sublist[i]
            if temp:
                result.append(temp)
                pt.extend(temp)
            else:
                result.append([0])
        p_t = result
        return p_t, pt

    def segricate_p_t_pr_inerval(self):
        """
        threshold = 0.37 for JR and 0.5 for other diseases
        """
        diff_arr = ((np.diff(self.r_index) * self.thres) / self.fs).tolist()
        t_peaks_list, p_peaks_list, pr_interval, extra_peaks_list = [], [], [], []
        # threshold = (-0.0012 * len(r_index)) + 0.25
        for i in range(len(self.p_t)):
            p_dis = (self.r_index[i + 1] - self.p_t[i][-1]) / self.fs
            t_dis = (self.r_index[i + 1] - self.p_t[i][0]) / self.fs
            threshold = diff_arr[i]
            if t_dis > threshold and (self.p_t[i][0] > self.r_index[i]):
                t_peaks_list.append(self.p_t[i][0])
            else:
                t_peaks_list.append(0)
            if p_dis <= threshold:
                p_peaks_list.append(self.p_t[i][-1])
                pr_interval.append(p_dis * self.fs)
            else:
                p_peaks_list.append(0)
            if len(self.p_t[i]) > 0:
                if self.p_t[i][0] in t_peaks_list:
                    if self.p_t[i][-1] in p_peaks_list:
                        extra_peaks_list.extend(self.p_t[i][1:-1])
                    else:
                        extra_peaks_list.extend(self.p_t[i][1:])
                elif self.p_t[i][-1] in p_peaks_list:
                    extra_peaks_list.extend(self.p_t[i][:-1])
                else:
                    extra_peaks_list.extend(self.p_t[i])

        p_label, pr_label = "", ""
        if self.thres >= 0.5 and p_peaks_list and len(p_peaks_list) > 2:
            pp_intervals = np.diff(p_peaks_list)
            pp_std = np.std(pp_intervals)
            pp_mean = np.mean(pp_intervals)
            threshold = 0.12 * pp_mean
            if pp_std <= threshold:
                p_label = "Constanat"
            else:
                p_label = "Not Constant"

            count = 0
            for i in pr_interval:
                if round(np.mean(pr_interval) * 0.75) <= i <= round(np.mean(pr_interval) * 1.25):
                    count += 1
            if len(pr_interval) != 0:
                per = count / len(pr_interval)
                pr_label = 'Not Constant' if per <= 0.7 else 'Constant'
        data = {'T_Index': t_peaks_list,
                'P_Index': p_peaks_list,
                'PR_Interval': pr_interval,
                'P_Label': p_label,
                'PR_label': pr_label,
                'Extra_Peaks': extra_peaks_list}
        return data

    def find_inverted_t_peak(self):
        t_index = []
        for i in range(0, len(self.s_index) - 1):
            t = self.ecg_signal[self.s_index[i]: self.q_index[i + 1]]
            if t.any():
                check, _ = find_peaks(-t, height=(0.21, 1), distance=70)
                peaks = check + self.s_index[i]
            else:
                peaks = np.array([])
            if peaks.any():
                t_index.extend(list(peaks))
        # t_label =
        return t_index

    def get_data(self):
        self.r_index = self.find_r_peaks()
        rr_intervals = np.diff(self.r_index)
        rr_std = np.std(rr_intervals)
        rr_mean = np.mean(rr_intervals)
        threshold = self.rr_thres * rr_mean
        if rr_std <= threshold:
            self.r_label = "Regular"
        else:
            self.r_label = "Irregular"
        # detect_betas according HR count
        # lowpass_signal = lowpass(self.ecg_signal, 0.3)
        # new_r_index = detect_rpeaks_eq(lowpass_signal, self.fs)
        # self.hr_ = hr_count(new_r_index)
        self.hr_ = self.hr_count()
        sd, qd = int(self.fs * 0.115), int(self.fs * 0.08)
        self.s_index = self.find_s_index(sd)
        self.q_index = self.find_new_q_index(qd)
        self.j_index = self.find_j_index()
        self.p_t, self.pt = self.find_pt()
        self.data_ = self.segricate_p_t_pr_inerval()
        self.inv_t_index = self.find_inverted_t_peak()
        data = {'R_Label': self.r_label,
                'R_index': self.r_index,
                'Q_Index': self.q_index,
                'S_Index': self.s_index,
                'J_Index': self.j_index,
                'P_T List': self.p_t,
                'PT PLot': self.pt,
                'HR_Count': self.hr_,
                'T_Index': self.data_['T_Index'],
                'P_Index': self.data_['P_Index'],
                'Ex_Index': self.data_['Extra_Peaks'],
                'PR_Interval': self.data_['PR_Interval'],
                'P_Label': self.data_['P_Label'],
                'PR_label': self.data_['PR_label'],
                'inv_t_index': self.inv_t_index}
        return data

def extract_number(filename):
    match = re.search(r'(\d+)', os.path.basename(filename))
    return int(match.group(1)) if match else float('inf')

class PVCDetection:
    def __init__(self, get_signal, fs, r_index, is_lead = "2"):
        self.get_signal = get_signal
        self.fs = fs
        self.is_lead = is_lead
        self.r_index = r_index

    def pvc_count_finds(self, sequence, HR):
        triplet_pattern = [1, 1, 1]
        couplet_pattern = [1, 1]
        bigem_pattern = [1, 0, 1]
        trigeminy_pattern = [1, 0, 0, 1]
        quadrigeminy_pattern = [1, 0, 0, 0, 1]
    
        triplet_count = 0
        couplet_count = 0
        bigem_count = 0
        trigeminy_count = 0
        quadrigeminy_count = 0
        aivr_count = 0
        ivr_count = 0
        nsvt_count = 0
        vt_count = 0
        beat_indices = set()
    
        def matches(subsequence, pattern):
            return subsequence == pattern
    
        i = 0
        while i < len(sequence):
            if sequence[i] == 1:
                start = i
                while i < len(sequence) and sequence[i] == 1:
                    i += 1
                length = i - start
                if length >= 4 and int(HR)<=100:
                    ivr_count += 1
                    for j in range(start, i):
                        beat_indices.add(j)
                elif sequence.count(1)>=13 and length >= 5 and int(HR)>100:
                    vt_count += 1
                    for j in range(start, i):
                        beat_indices.add(j)
                elif sequence.count(1)<=12 and 5 <= length <= 12 and (int(HR)>60 and int(HR)<=300):
                    nsvt_count += 1
                    for j in range(start, i):
                        beat_indices.add(j)
            else:
                i += 1
    
        
        for i in range(len(sequence)):
            # Triplet
            if i + 3 <= len(sequence):
                subseq = sequence[i:i+3]
                if matches(subseq, triplet_pattern):
                    if all(j not in beat_indices for j in range(i, i+3)):
                        triplet_count += 1
                        for j in range(i, i+3):
                            if sequence[j] == 1:
                                beat_indices.add(j)
            # Couplet
            if i + 2 <= len(sequence):
                subseq = sequence[i:i+2]
                if matches(subseq, couplet_pattern):
                    if all(j not in beat_indices for j in range(i, i+2)):
                        couplet_count += 1
                        for j in range(i, i+2):
                            if sequence[j] == 1:
                                beat_indices.add(j)
            # Bigeminy
            if i + 3 <= len(sequence):
                subseq = sequence[i:i+3]
                if matches(subseq, bigem_pattern):
                    if all(j not in beat_indices for j in range(i, i+3)):
                        bigem_count += 1
                        for j in range(i, i+3):
                            if sequence[j] == 1:
                                beat_indices.add(j)
            # Trigeminy
            if i + 4 <= len(sequence):
                subseq = sequence[i:i+4]
                if matches(subseq, trigeminy_pattern):
                    if all(j not in beat_indices for j in range(i, i+4)):
                        trigeminy_count += 1
                        for j in range(i, i+4):
                            if sequence[j] == 1:
                                beat_indices.add(j)
            # Quadrigeminy
            if i + 5 <= len(sequence):
                subseq = sequence[i:i+5]
                if matches(subseq, quadrigeminy_pattern):
                    if all(j not in beat_indices for j in range(i, i+5)):
                        quadrigeminy_count += 1
                        for j in range(i, i+5):
                            if sequence[j] == 1:
                                beat_indices.add(j)
    
        # Isolated PVCs
        isolated_beats = 0
        for idx, val in enumerate(sequence):
            if val == 1 and idx not in beat_indices:
                isolated_beats += 1
                beat_indices.add(idx)
    
        total_beats = len(beat_indices)
    
        return {
            "Isolated": isolated_beats,
            "Bigeminy": bigem_count,
            "Trigeminy": trigeminy_count,
            "Quadrigeminy": quadrigeminy_count,
            "Couplet": couplet_count,
            "Triplet": triplet_count,
            "Aivr": aivr_count,
            "IVR": ivr_count,
            "NSVT": nsvt_count,
            "VT": vt_count,
            "Total Beats": total_beats
        }

    def get_pvc_data(self):
        base_dir = os.path.dirname(os.path.abspath(__file__))
        temp_pvc_img_path = os.path.join(base_dir, "temp_pvc_img")
        os.makedirs(temp_pvc_img_path, exist_ok=True)
        all_lead_pvc_data = {}
        newdatepvclist=[]
        pvc_label, lbbb_rbbb_label = "Abnormal", "Abnormal"
        all_lead_data = self.get_signal
        if self.is_lead == '12':
            analysis_lead = ['I','II','III', 'aVL', 'v1', 'v5','v6']
        elif self.is_lead == '7':
            analysis_lead = ['I','II','III', 'aVL','v5']
        else:
            analysis_lead = ['II']
        imageresource = temp_pvc_img_path
        for i in glob.glob(imageresource+"/*.jpg"):
            os.remove(i)
        for lead in all_lead_data.keys():
            if lead in analysis_lead: #['I', 'II', 'III', 'aVR', 'aVL', 'aVF', 'v1', 'v2', 'v3', 'v4', 'v5', 'v6']:
                lead_data = {}
                ecg_signal = all_lead_data[lead]
                lis= []
                
                base_ecg = baseline_construction_200(ecg_signal,131) # for stable 101, 
                pvc_data = lowpass(base_ecg, cutoff=0.2)
                    
                aboutdatas = pd.DataFrame(pvc_data)
                rpeaks = self.r_index 
                lead_data['rpeaks'] = rpeaks
                hr = hr_count(pvc_data, rpeaks)
                lead_data['hr'] = hr
                
                for i in rpeaks:
                    lis.append(i)
                    if int(lis[0]) - 50>0:
                        window_start = int(lis[0]) - 50
                    else:
                        window_start=0
                    window_end = int(lis[0]) + 80

                    aa = pd.DataFrame(aboutdatas.iloc[window_start:window_end])
                    plt.plot(aa,color='blue')
                    plt.axis("off")
                    os.makedirs(imageresource, exist_ok=True)
                    plt.savefig(f"{imageresource}/p_{lead}_{int(lis[0])}.jpg")
                    aq = cv2.imread(f"{imageresource}/p_{lead}_{int(lis[0])}.jpg")
                    aq = cv2.resize(aq, (360, 720))
                    cv2.imwrite(f"{imageresource}/p_{lead}_{int(lis[0])}.jpg", aq)
                    lis.clear()
                    plt.close()
                
                observer = []
                LBBB_list, RBBB_list = [], []

                files = sorted(glob.glob(imageresource+f"/p_{lead}_*.jpg"), key=extract_number)
                for pvcfilename in files:
                    predictions,ids = prediction_model(pvcfilename)
                    if self.is_lead == '7': 
                        for_pvc_leads = ['I', 'II', 'III']
                        for_lbbb_rbbb_leads = ['I', 'II', 'III'] #['I', 'aVL', 'V5']
                    elif self.is_lead == '12':
                        for_pvc_leads = ['I','II','III', 'v1', 'v5']
                        for_lbbb_rbbb_leads = ['I', 'II', 'III'] #['I', 'aVL', 'v1', 'v5','v6']
                    else:
                        for_pvc_leads = ['II']
                        for_lbbb_rbbb_leads = []
                    if (str(ids) == "PVC" and float(predictions[3])>0.92) and lead in for_pvc_leads:
                        observer.append(1)
                    else:
                        observer.append(0)

                    if str(ids) == "LBBB" and float(predictions[0]) > 0.78 and lead in for_lbbb_rbbb_leads:
                        LBBB_list.append(1)
                    else:
                        LBBB_list.append(0)

                    if str(ids) == "RBBB" and float(predictions[4]) > 0.78 and lead in for_lbbb_rbbb_leads:
                        RBBB_list.append(1)
                    else:
                        RBBB_list.append(0)
                counts_result = self.pvc_count_finds(observer, hr)
                r_index_plot = [rpeaks[i] for i in range(len(observer)) if observer[i] == 1]
                lbbb_index = [rpeaks[i] for i in range(len(LBBB_list)) if LBBB_list[i] == 1]
                rbbb_index = [rpeaks[i] for i in range(len(RBBB_list)) if RBBB_list[i] == 1]
                pvc_label_counts = {
                    'PVC-Isolated_counter': counts_result['Isolated'],
                    'PVC-Bigeminy_counter': counts_result['Bigeminy'],
                    'PVC-Trigeminy_counter': counts_result['Trigeminy'],
                    'PVC-Quadrigeminy_counter':counts_result['Quadrigeminy'],
                    'PVC-Couplet_counter':counts_result['Couplet'],
                    'PVC-Triplet_counter':counts_result['Triplet'],
                    'PVC-NSVT_counter':counts_result['NSVT'],
                    'PVC-Aivr_counter':counts_result['Aivr'],
                    'PVC-Ivr_counter':counts_result['IVR'],
                    'PVC-VT_counter':counts_result['VT'],
                    'pvc_r_index': r_index_plot,
                }
                pvc_label = '; '.join([key.split('_')[0] for key, val in pvc_label_counts.items() if 'counter' in key and val > 0])
                if len(pvc_label) == 0:
                    pvc_label = 'Normal'
                if rpeaks:
                    if len(lbbb_index)/ len(rpeaks)> 0.3:
                        lbbb_rbbb_label = "LBBB"
                    elif len(rbbb_index)/ len(rpeaks) > 0.3:
                        lbbb_rbbb_label = "RBBB"
                lead_data['pvc_index'] = pvc_label_counts['pvc_r_index']
                lead_data['pvc_label'] = pvc_label
                lead_data['lbbb_rbbb_label'] = lbbb_rbbb_label
                lead_data['lbbb_index'] = lbbb_index
                lead_data['rbbb_index'] = rbbb_index
                all_lead_pvc_data[lead]= lead_data
        if len(all_lead_pvc_data.keys())> 1:
            result_pvc_data = {}
            combined_labels = []
            for data in all_lead_pvc_data.values():
                temp_label = data['pvc_label'].split('; ')
                if len(temp_label) > 1:
                    combined_labels.extend(temp_label)
                else:
                    combined_labels.append(data['pvc_label'])
                combined_labels.append(data['lbbb_rbbb_label'])
            label_counts = Counter(combined_labels)
            if self.is_lead == '7':
                repeated_elements = [item for item, count in label_counts.items() if count > 1]
            else:
                repeated_elements = [item for item, count in label_counts.items() if count > 2]
            if 'NSVT' in label_counts and label_counts['NSVT'] != 3:
                repeated_elements.remove('PVC-NSVT')
            if 'Aivr' in label_counts and label_counts['Aivr'] != 3:
                repeated_elements.remove('PVC-Aivr')
            if 'Ivr' in label_counts and label_counts['Ivr'] != 3:
                repeated_elements.remove('PVC-Ivr')
            
            if len(repeated_elements) >1:
                if 'Abnormal' in repeated_elements:
                    repeated_elements.remove('Abnormal')
                if "Normal" in repeated_elements:
                    repeated_elements.remove("Normal")
            pvc_final_index = ' '.join(repeated_elements)
            
            if "LBBB" in pvc_final_index:
                pvc_final_index = pvc_final_index.strip('LBBB')
            if "RBBB" in pvc_final_index:
                pvc_final_index = pvc_final_index.strip("RBBB")
            result_pvc_data['pvc_label'] = pvc_final_index
            pvc_matching_keys, lbbb_rbbb_matching_keys = [], []
            pvc_matching_keys = [
                key for key, data in all_lead_pvc_data.items()
                if any(element in data['pvc_label'] for element in repeated_elements)
            ]
            if pvc_matching_keys:
                result_pvc_data['pvc_index'] = all_lead_pvc_data[pvc_matching_keys[0]]['pvc_index']
            else:
                result_pvc_data['pvc_index'] = []
            if self.is_lead == '7':
                l_r_thresh = 2
            elif self.is_lead == '12':
                l_r_thresh = 3
            if label_counts['RBBB'] >= l_r_thresh or label_counts['LBBB'] >= l_r_thresh:
                result_pvc_data['lbbb_rbbb_label'] = lbbb_rbbb_label
                lbbb_rbbb_matching_keys = [
                    key for key, data in all_lead_pvc_data.items()
                    if any(element in data['lbbb_rbbb_label'] for element in repeated_elements)
                ]
                if label_counts['RBBB'] >= l_r_thresh:
                    result_pvc_data['rbbb_index'] = all_lead_pvc_data[lbbb_rbbb_matching_keys[0]]['rbbb_index']
                if label_counts['LBBB'] >= l_r_thresh:
                    result_pvc_data['lbbb_index'] = all_lead_pvc_data[lbbb_rbbb_matching_keys[0]]['lbbb_index']
                
            else:
                result_pvc_data['lbbb_index'] = []
                result_pvc_data['rbbb_index'] = []
                result_pvc_data['lbbb_rbbb_label'] = 'Abnormal'
            pvc_observer = [1 if self.r_index[i] in result_pvc_data['pvc_index'] else 0 for i in range(len(self.r_index))]
            result_pvc_data['observer'] = pvc_observer
        else:
            pvc_observer = [1 if self.r_index[i] in all_lead_pvc_data['II']['pvc_index'] else 0 for i in range(len(self.r_index))]
            result_pvc_data = {
                'pvc_index': all_lead_pvc_data['II']['pvc_index'],
                'lbbb_index': all_lead_pvc_data['II']['lbbb_index'],
                'rbbb_index': all_lead_pvc_data['II']['rbbb_index'],
                'pvc_label': all_lead_pvc_data['II']['pvc_label'],
                'lbbb_rbbb_label': all_lead_pvc_data['II']['lbbb_rbbb_label'],
                'observer': pvc_observer
            }
        return result_pvc_data

def prediction_model_PAC(input_arr, target_shape=[224, 224]):
    classes = ['Abnormal', 'Junctional', 'Normal', 'PAC']
    input_arr = tf.io.decode_jpeg(tf.io.read_file(input_arr), channels=3)
    input_arr = tf.image.resize(input_arr, size=(224, 224), method=tf.image.ResizeMethod.BILINEAR)
    input_arr = (tf.expand_dims(input_arr, axis=0),)
    model_pred = predict_tflite_model(pac_model, input_arr )[0]
    idx = np.argmax(model_pred)
    return model_pred, classes[idx]

class PACDetection:
    def __init__(self,get_signal, r_index, fs, is_lead = "2"):
        self.get_signal = get_signal
        self.fs = fs
        self.is_lead = is_lead
        self.r_index = r_index

    def get_pac_data(self):
        base_dir = os.path.dirname(os.path.abspath(__file__))
        PAC_dir = os.path.join(base_dir, "temp_pac_img")
        os.makedirs(PAC_dir, exist_ok=True)
        all_lead_pac_data, results_pac = {}, {}
        
        all_lead_data = self.get_signal
        if self.is_lead == "12":
            rep_thresh = 2
            analysis_leads = ['I', 'II', 'III', 'v1', 'v2']
        elif self.is_lead == "7":
            rep_thresh = 1
            analysis_leads = ['I', 'II', 'III' ]
        else:
            analysis_leads = ['II']

        for img_path in glob.glob(f'{PAC_dir}/*.jpg'):
            os.remove(img_path)
        for lead in all_lead_data.keys():
            if lead in analysis_leads: #['I', 'II', 'III', 'aVR', 'aVL', 'aVF', 'v1', 'v2', 'v3', 'v4', 'v5', 'v6']:
                lead_data = {}
                ecg_signal = all_lead_data[lead]
                pac_label, junctional_label = 'Abnormal', 'Abnormal'
                
                base_signal = baseline_construction_200(ecg_signal,101)
                lowpass_signal = lowpass(base_signal, cutoff=0.2)

                apeds = []
                r_index = self.r_index
                hr = hr_count(lowpass_signal, r_index)
                rr_thres = 0.12
                rr_intervals = np.diff(r_index)
                rr_std = np.std(rr_intervals)
                rr_mean = np.mean(rr_intervals)
                threshold = rr_thres * rr_mean
                if rr_std <= threshold:
                    R_label = "Regular"
                else:
                    R_label = "Irregular"

                lead_data['r_index'] = r_index
                lead_data['hr'] = hr
                lead_data['R_label'] = R_label
                updated_union, pac_detect, junc_detect, junc_union =[], [],[], []
                for i in range(len(r_index)-1):
                    m=r_index[i+1]-r_index[i]
                    apeds.append(m*5/1000)

                variations=[]
                for i in range(len(apeds)-1):
                    variations.append(get_percentage_diff(apeds[i+1],apeds[i]))

                forPAC = Average(variations) 
                lead_data['forPAC'] = forPAC                  
                if Average(variations)<0.20:
                    updated_union=[0,0,0,0,0,0,0,0]
                    lead_data['updated_union'] = updated_union
                    lead_data['junc_union'] = junc_union
                    lead_data['junctional_label'] = junctional_label
                    lead_data['pac_label'] = pac_label
                    lead_data['pac_detect'] = pac_detect
                    lead_data['junc_detect'] = junc_detect
                    lead_data['pac_counts'] = {}
                    all_lead_pac_data[lead] = lead_data
                else:
                    try:
                        for i in range(len(r_index) - 1):
                            segment = lowpass_signal[r_index[i]-16:r_index[i + 1]+20]
                            plt.plot(segment,color='blue')
                            plt.axis("off")
                            plt.savefig(f"{PAC_dir}/p_{lead}_{int(r_index[i])}.jpg")
                            aq = cv2.imread(f"{PAC_dir}/p_{lead}_{int(r_index[i])}.jpg")
                            aq = cv2.resize(aq, (360, 720))
                            cv2.imwrite(f"{PAC_dir}/p_{lead}_{int(r_index[i])}.jpg", aq)
                            plt.close()
                            
                            predictions,ids = prediction_model_PAC(f"{PAC_dir}/p_{lead}_{int(r_index[i])}.jpg")
                            
                            if str(ids) == "PAC" and float(predictions[3])>0.90: # 0.91
                                updated_union.append(1)
                                junc_union.append(0)
                                pac_detect.append((int(r_index[i]), int(r_index[i+1])))
                            elif (str(ids) == "Junctional" and float(predictions[1])>0.80) and R_label == 'Regular' and lead in ['I','II', 'III']:
                                junc_union.append(1)
                                updated_union.append(0)
                                junc_detect.append((int(r_index[i]), int(r_index[i+1])))
                            else:
                                updated_union.append(0)
                                junc_union.append(0)
                        if len(r_index) != 0:
                            junc_count = junc_union.count(1)
                            if junc_count / len(r_index) >= 0.5 and hr <= 60:
                                junctional_label = "Junctional_Rhythm" if hr > 40 else "Junctional_Bradycardia"
                    except Exception as e:
                        print(e)
                        updated_union=[0,0,0,0,0,0,0,0]
                        junc_union = [0,0,0,0,0,0,0,0]
                    pac_data = self.pac_count_find(updated_union, hr)
                    pac_label = '; '.join([key.split('_')[0] for key, val in pac_data.items() if 'counter' in key and val > 0]) 
                    lead_data['updated_union'] = updated_union
                    lead_data['junc_union'] = junc_union
                    lead_data['junctional_label'] = junctional_label
                    lead_data['pac_label'] = pac_label
                    lead_data['pac_detect'] = pac_detect
                    lead_data['junc_detect'] = junc_detect
                    lead_data['pac_counts'] = pac_data
                    all_lead_pac_data[lead] = lead_data
        if len(all_lead_pac_data.keys()) > 1:
            results_pac = {}
            combined_labels = []
            pvc_final_label, jnc_label = 'Abnormal', 'Abnormal'
            for data in all_lead_pac_data.values():
                temp_label = data['pac_label'].split('; ')
                if len(temp_label) > 1:
                    combined_labels.extend(temp_label)
                else:
                    combined_labels.append(data['pac_label'])
                combined_labels.append(data['junctional_label'])
            label_counts = Counter(combined_labels)
            repeated_elements = [item for item, count in label_counts.items() if count > rep_thresh]
            if 'SVT' in label_counts and label_counts['SVT'] != 3:
                repeated_elements.remove('SVT')
            
            if 'Junctional_Rhythm' in label_counts and label_counts['Junctional_Rhythm'] > rep_thresh:
                jnc_label = 'Junctional_Rhythm'
                if 'Junctional_Rhythm' in repeated_elements:
                    repeated_elements.remove('Junctional_Rhythm')
            elif 'Junctional_Bradycardia' in label_counts and label_counts['Junctional_Bradycardia'] > rep_thresh:
                jnc_label = 'Junctional_Bradycardia'
                if 'Junctional_Bradycardia' in repeated_elements:
                    repeated_elements.remove('Junctional_Bradycardia')
            is_pac_present = any(map(lambda x: 'PAC' in x, repeated_elements))
            is_svt_present = any(map(lambda x: 'SVT' in x, repeated_elements))
            if (is_pac_present or is_svt_present) and 'Abnormal' in repeated_elements:
                repeated_elements.remove('Abnormal')
            if repeated_elements:
                pvc_final_label = ' '.join(repeated_elements)
            pac_matching_keys, junc_matching_keys = [], []
            pac_matching_keys = [
                key for key, data in all_lead_pac_data.items()
                if any(element in data['pac_label'] for element in repeated_elements)
            ]
            if pac_matching_keys:
                results_pac['pac_index'] = all_lead_pac_data[pac_matching_keys[0]]['pac_detect']
                results_pac['pac_union'] = all_lead_pac_data[pac_matching_keys[0]]['updated_union']
                results_pac['pac_counts']= all_lead_pac_data[pac_matching_keys[0]]['pac_counts']
                results_pac['juc_index'] = all_lead_pac_data[pac_matching_keys[0]]['junc_detect']
            else:
                results_pac['pac_index'] = []
                results_pac['pac_union'] = []
                results_pac['juc_index'] = []
                results_pac['pac_counts']= {}
            results_pac['pac_label'] = pvc_final_label
            results_pac['jnc_label'] = jnc_label
        else:
            results_pac['pac_index'] = all_lead_pac_data['II']['pac_detect']
            results_pac['pac_union'] = all_lead_pac_data['II']['updated_union']
            results_pac['pac_label'] =all_lead_pac_data['II']['pac_label']
            results_pac['jnc_label'] = all_lead_pac_data['II']['junctional_label']
            results_pac['juc_index'] = all_lead_pac_data['II']['junc_detect']
            results_pac['pac_counts']= all_lead_pac_data['II']['pac_counts']
        return results_pac
   
    def pac_count_find(self, PAC_R_Peaks, hr_counts):
        svt_counter = 0
        couplet_counter = 0
        triplet_counter = 0
        bigeminy_counter = 0
        trigeminy_counter = 0 
        quadrigeminy_counter = 0
        at = 0
        i = 0
        while i < len(PAC_R_Peaks):
            count = 0
            ones_count = 0
            while i < len(PAC_R_Peaks) and PAC_R_Peaks[i] == 1:
                count += 1
                ones_count += 1
                i += 1

            if count >= 4:
                svt_counter += 1
                at += ones_count
                count = 0
                ones_count = 0
            if count == 3:
                triplet_counter += 1
            elif count == 2:
                couplet_counter += 1
            i += 1
        j = 0
        while j < len(PAC_R_Peaks) - 1:
            if PAC_R_Peaks[j] == 1:
                k = j + 1
                spaces = 0
                while k < len(PAC_R_Peaks) and PAC_R_Peaks[k] == 0:
                    spaces += 1
                    k += 1

                if k < len(PAC_R_Peaks) and PAC_R_Peaks[k] == 1:
                    if spaces == 1:
                        bigeminy_counter += 1
                    elif spaces == 2:
                        trigeminy_counter += 1
                    elif spaces == 3:
                        quadrigeminy_counter += 1
                j = k
            else:
                j += 1

        total_one = (1 * at) + (couplet_counter * 2) + (triplet_counter * 3) + (bigeminy_counter * 2) + (
                trigeminy_counter * 2) + (quadrigeminy_counter * 2)
        total = svt_counter + couplet_counter + triplet_counter + bigeminy_counter + trigeminy_counter + quadrigeminy_counter
        ones = PAC_R_Peaks.count(1)

        if total == 0:
            Isolated = ones
        else:
            Common = total - 1
            Isolated = ones - (total_one - Common)
        if hr_counts > 100:
            if svt_counter != 0:
                triplet_counter = couplet_counter = quadrigeminy_counter = trigeminy_counter = bigeminy_counter = Isolated = 0
        if svt_counter >= 1 and hr_counts > 100:  # 190
            svt_counter = 1
        else:
            svt_counter = 0

        data = {"PAC-Isolated_counter": Isolated,
                "PAC-Bigem_counter": bigeminy_counter,
                "PAC-Trigem_counter": trigeminy_counter,
                "PAC-Quadrigem_counter": quadrigeminy_counter,
                "PAC-Couplet_counter": couplet_counter,
                "PAC-Triplet_counter": triplet_counter,
                "SVT_counter": svt_counter}  # svt_counter
        return data

class BlockDetected:
    def __init__(self, ecg_signal, fs):
        self.ecg_signal = ecg_signal
        self.fs = fs
        # self.block_processing()

    # def block_processing(self):
    #     self.baseline_signal, self.lowpass_signal = filter_signal(self.ecg_signal, self.fs).get_data()
    #     pqrst_data = pqrst_detection(self.baseline_signal, fs=self.fs).get_data()
    #     self.r_index = pqrst_data["R_index"]
    #     self.q_index = pqrst_data["Q_Index"]
    #     self.s_index = pqrst_data["S_Index"]
    #     self.p_index = pqrst_data["P_Index"]
    #     self.hr_counts = pqrst_data["HR_Count"]
    #     self.p_t = pqrst_data["P_T List"]
    #     self.pr = pqrst_data["PR_Interval"]

    # def third_degree_block_deetection(self):
    #     label = 'Abnormal'
    #     third_degree = []
    #     possible_mob_3rd = False
    #     if self.hr_counts <= 100 and len(self.p_t) != 0:  # 60 70
    #         constant_2 = all(map(lambda innerlist: len(innerlist) == 2, self.p_t))
    #         cons_2_1 = all(len(inner_list) in {1, 2} for inner_list in self.p_t)
    #         ampli_val = list(
    #             map(lambda inner_list: sum(self.baseline_signal[i] > 0.05 for i in inner_list) / len(inner_list),
    #                 self.p_t))
    #         count_above_threshold = sum(1 for value in ampli_val if value > 0.7)
    #         percentage_above_threshold = count_above_threshold / len(ampli_val)
    #         count = 0
    #         if percentage_above_threshold >= 0.7:
    #             inc_dec_count = 0
    #             for i in range(0, len(self.pr)):
    #                 if self.pr[i] > self.pr[i - 1]:
    #                     inc_dec_count += 1
    #             if len(self.pr) != 0:
    #                 if round(inc_dec_count / (len(self.pr)), 2) >= 0.50:  # if posibale to change more then 0.5
    #                     possible_mob_3rd = True
               
    #             for inner_list in self.p_t:
    #                 if len(inner_list) in [3, 4]:
    #                     ampli_val = [self.baseline_signal[i] for i in inner_list]
    #                     if ampli_val and (sum(value > 0.05 for value in ampli_val) / len(ampli_val)) > 0.7:
    #                         differences = np.diff(inner_list).tolist()
    #                         diff_list = [x for x in differences if x >= 70]
    #                         if len(diff_list) != 0:
    #                             third_degree.append(1)
    #                         else:
    #                             third_degree.append(0)
    #                 elif len(inner_list) in [3, 4] and possible_mob_3rd == True and constant_2 == False:
    #                     differences = np.diff(inner_list).tolist()
    #                     if all(diff > 70 for diff in differences):
    #                         third_degree.append(1)
    #                     else:
    #                         third_degree.append(0)
    #                 else:
    #                     third_degree.append(0)
    #     if len(third_degree) != 0:
    #         if third_degree.count(1) / len(third_degree) >= 0.4 or possible_mob_3rd:  # 0.5 0.4
    #             label = "3rd Degree block"
    #     return label

    # def second_degree_block_detection(self):
    #     label = 'Abnormal'
    #     constant_3_peak = []
    #     possible_mob_1 = False
    #     possible_mob_2 = False
    #     mob_count = 0
    #     if self.hr_counts <= 100:  # 80
    #         if len(self.p_t) != 0:
    #             constant_2 = all(map(lambda innerlist: len(innerlist) == 2, self.p_t))
    #             rhythm_flag = all(len(inner_list) in {1, 2, 3} for inner_list in self.p_t)
    #             ampli_val = list(
    #                 map(lambda inner_list: sum(self.baseline_signal[i] > 0.05 for i in inner_list) / len(inner_list),
    #                     self.p_t))
    #             count_above_threshold = sum(1 for value in ampli_val if value > 0.7)
    #             percentage_above_threshold = count_above_threshold / len(ampli_val)
    #             if percentage_above_threshold >= 0.7:
    #                 if rhythm_flag and constant_2 == False:
    #                     pr_interval = []
    #                     for i, r_element in enumerate(self.r_index[1:], start=1):
    #                         if i <= len(self.p_t):
    #                             inner_list = self.p_t[i - 1]
    #                             last_element = inner_list[-1]
    #                             result = r_element - last_element
    #                             pr_interval.append(result)

    #                     counts = {}
    #                     count_2 = 0
    #                     for i in range(0, len(pr_interval)):
    #                         counts[i] = 1
    #                         if i in counts:
    #                             counts[i] += 1
    #                         if pr_interval[i] > pr_interval[i - 1]:
    #                             count_2 += 1
    #                     most_frequent = max(counts.values())
    #                     if round(count_2 / (len(pr_interval)), 2) >= 0.50:
    #                         possible_mob_1 = True
    #                     elif round(most_frequent / len(pr_interval), 2) >= 0.4:
    #                         possible_mob_2 = True

    #                     for inner_list in self.p_t:
    #                         if len(inner_list) == 3:
    #                             differences = np.diff(inner_list).tolist()
    #                             if differences[0] <= 0.5 * differences[1] or differences[1] <= 0.5 * differences[0]:
    #                                 if possible_mob_1 or possible_mob_2:
    #                                     mob_count += 1
    #                                 else:
    #                                     constant_3_peak.append(1)
    #                         else:
    #                             constant_3_peak.append(0)
    #                 else:
    #                     for inner_list in self.p_t:
    #                         if len(inner_list) == 3:
    #                             differences = np.diff(inner_list).tolist()
    #                             if differences[0] <= 0.5 * differences[1] or differences[1] <= 0.5 * differences[0]:
    #                                 constant_3_peak.append(1)
    #                             else:
    #                                 constant_3_peak.append(0)
    #                         else:
    #                             constant_3_peak.append(0)
    #     if len(constant_3_peak) != 0 and constant_3_peak.count(1) != 0:
            
    #         if constant_3_peak.count(1) / len(constant_3_peak) >= 0.4:  # 0.4 0.5
    #             label = "Mobitz_II"
    #     elif possible_mob_1 and mob_count > 1:  # 0 1 4
    #         label = "Mobitz_I"
    #     elif possible_mob_2 and mob_count > 1:  # 0  4
    #         label = "Mobitz_II"
    #     return label

    # Block new trans model for added 
    def prediction_model_block(self, input_arr, block_model):
        classes = ['1st_deg', '2nd_deg', '3rd_deg', 'abnormal', 'normal']
        input_arr = tf.io.decode_jpeg(tf.io.read_file(input_arr), channels=3)
        input_arr = tf.image.resize(input_arr, size=(224, 224), method=tf.image.ResizeMethod.BILINEAR)
        input_arr = (tf.expand_dims(input_arr, axis=0),)
        model_pred = predict_tflite_model(block_model, input_arr )[0]
        idx = np.argmax(model_pred)
        return model_pred, classes[idx]
    
    def check_block_model(self,low_ecg_signal, block_model):
        label = 'Abnormal'
        # Create folder inside this script directory
        base_dir = os.path.dirname(os.path.abspath(__file__))
        block_dir = os.path.join(base_dir, "temp_block_img")
        os.makedirs(block_dir, exist_ok=True)
        for i in glob.glob(block_dir + "/*.jpg"):
            os.remove(i)
        
        randome_number = random.randint(200000, 1000000)
        temp_img = low_ecg_signal
        #plt.figure() # layout="constrained", dpi=300
        plt.plot(temp_img)
        plt.axis("off")
        plt.savefig(f"{block_dir}/p_{randome_number}.jpg")
        aq = cv2.imread(f"{block_dir}/p_{randome_number}.jpg")
        aq = cv2.resize(aq, (2400,360), interpolation=cv2.INTER_LANCZOS4) #1080, 460
        aq = Image.fromarray(cv2.cvtColor(aq, cv2.COLOR_BGR2RGB))
        aq.save(f"{block_dir}/p_{randome_number}.jpg", dpi=(2000,700))
        plt.close()
        ei_ti_label, predictions = [], []
        files = sorted(glob.glob(block_dir +"/*.jpg"), key=extract_number)
        for pvcfilename in files:
            predictions, ids = self.prediction_model_block(pvcfilename, block_model)
          
            label = "Abnormal" 
            if str(ids) == "3rd_deg" and float(predictions[2]) > 0.80:
                label = "3rd degree"
            if str(ids) == "2nd_deg" and float(predictions[1]) > 0.80:
                label = "2nd degree"
            if str(ids) == "1st_deg" and float(predictions[0]) > 0.80:
                label = "1st degree"

            if 0.40 < float(predictions[1]) < 0.70:
                ei_ti_label.append('2nd degree')
            if 0.40 < float(predictions[0]) < 0.70:
                ei_ti_label.append('1st degree')
            if 0.40 < float(predictions[2]) < 0.70:
                ei_ti_label.append('3rd degree')
        return label, ei_ti_label, predictions

def block_model_check(ecg_signal, frequency, abs_result, block_model):
    model_label = 'Abnormal'
    ei_ti_block = []
    lowpass_signal = lowpass(ecg_signal, cutoff=0.3)
    baseline_signal = baseline_construction_200(lowpass_signal, kernel_size=131)
    
    get_block = BlockDetected(ecg_signal, frequency)
    block_result, ei_ti_label, model_pre = get_block.check_block_model(baseline_signal, block_model)
    if block_result == '1st degree':
        model_label = 'I DEGREE'
    if block_result == '2nd degree':
        model_label = 'MOBITZ-II'
    if block_result == '3rd degree':
        model_label = 'III Degree'
    # if abs_result in ['1st deg. block', "3rd Degree block", 'Mobitz II', 'Mobitz I']:
    #     if block_result == '2nd degree':
    #         model_label = 'MOBITZ-I'
    #     elif block_result == '3rd degree':
    #         model_label = 'III Degree'
    if ei_ti_label:
        if '1st degree' in ei_ti_label:
            model_label = 'I DEGREE'
            ei_ti_block.append({"Arrhythmia":"I DEGREE","percentage":model_pre[0]*100})
        if '2nd degree' in ei_ti_label:
            model_label = 'MOBITZ-II'
            ei_ti_block.append({"Arrhythmia":"MOBITZ-II","percentage":model_pre[1]*100})
        if '3rd degree' in ei_ti_label:
            model_label = 'III Degree'
            ei_ti_block.append({"Arrhythmia":"III Degree","percentage":model_pre[2]*100})
    return model_label, ei_ti_block

def first_degree_detect(ecg_signal, fs=200):
    pqrst_data = pqrst_detection(ecg_signal, fs=fs, width=(3, 50)).get_data()
    r_index = pqrst_data['R_index']
    q_index = pqrst_data['Q_Index']
    s_index = pqrst_data['S_Index']
    r_Label = pqrst_data['R_Label']
    hr_ = pqrst_data['HR_Count']
    block = []
    label = 'Abnormal'

    # if r_Label == 'Regular' and hr_ <= 90:
    for i in range(len(r_index) - 1):
        aoi = ecg_signal[s_index[i]:q_index[i + 1]]
        if aoi.any():
            check, _ = find_peaks(aoi, width=(5, 80), height=(0.02, 0.70), distance=15)
            loc = check + s_index[i]

            if len(check) > 3:
                peaks1 = np.array([])
            else:
                if len(check) == 3:
                    sorted_indices = sorted(range(len(check)), key=lambda k: aoi[check[k]], reverse=True)
                    check = [check[sorted_indices[0]], check[sorted_indices[1]]]  # Keep only the top two peaks
                    loc = check + s_index[i]
                check1 = sorted(loc)
                if len(check) == 2:
                    dist_next_r_index = r_index[i + 1] - check1[1]
                    if dist_next_r_index >= 50:  # 0.3 sec
                        peaks1 = check + s_index[i]
                    else:
                        peaks1 = np.array([])
                else:
                    peaks1 = np.array([])
        else:
            peaks1 = np.array([])

        if peaks1.any():
            block.extend(list(peaks1))

    if len(r_index) != 0:
        block_per = round(((len(block) / 2) / len(r_index)) * 100)
    else:
        block_per = np.array([])
    if block_per > 50:
        label = "1st deg. block"
    else:
        label = 'Abnormal'
    return label, block


def get_percentage_diff(previous, current):
    try:
        percentage = abs(previous - current) / max(previous, current) * 100
    except ZeroDivisionError:
        percentage = float('inf')
    return percentage

def Average(lst):
    if lst:
        return sum(lst) / len(lst)
    else:
        return 0

def new_rr_check(r_index):
    variation = []
    r_label = "Regular"
    for i in range(len(r_index) - 1):
        variation.append(get_percentage_diff(r_index[i + 1], r_index[i]))
    if Average(variation) > 12:
        r_label = "Irregular"

    return r_label

def check_r_irregular(r_index):
    r_label = "Regular"
    mean_percentage_diff = irrgular_per_r = 0
    rpeak_diff = np.diff(r_index)
    if len(rpeak_diff) >= 3:
        percentage_diff = np.abs(np.diff(rpeak_diff) / rpeak_diff[:-1]) * 1003
        list_per_r = [value for value in percentage_diff if value > 14]
        irrgular_per_r = (len(list_per_r) / len(percentage_diff)) * 100
        mean_percentage_diff = np.mean(percentage_diff)

    if (mean_percentage_diff > 50) and (irrgular_per_r > 40):
        r_label = "Irregular"

    return r_label



def find_label_couter(labels_list):
    flat_list = []
    for element in labels_list:
        if isinstance(element, list):
            flat_list.extend(element)
        else:
            flat_list.append(element)
    return flat_list

def block_detection_processing(all_lead_data, is_lead,fs=200):
    block_label, ei_ti_label = 'Abnormal', 'Abnormal'
       
    all_lead_result_data= {}
    all_block_labels, all_ei_ti_laels = [], []
    if is_lead == '12':
        rep_thresh = 2
        analysis_lead = ['I','II','III', 'v1', 'v5']
    elif is_lead == '7':
        rep_thresh = 1
        analysis_lead = ['I','II','III']
    else:
        analysis_lead = ['II']
    if len(all_lead_data) != 0:
        for lead in all_lead_data.keys():
            lead_data = {}
            
            model_label = 'Abnormal'
            if lead in analysis_lead:
                ecg_signal = all_lead_data[lead]
                frequency = fs
                baseline_signal = baseline_construction_200(ecg_signal, 131)
                lowpass_signal = lowpass(baseline_signal)
                first_deg_block_label, first_deg_block_index = first_degree_detect(lowpass_signal, frequency)
                abs_result = first_deg_block_label
                # if abs_result == 'Abnormal':
                #     second_deg_block = BlockDetected(ecg_signal, frequency).second_degree_block_detection()
                #     if second_deg_block != 'Abnormal':
                #         abs_result = second_deg_block
                # if abs_result == 'Abnormal':
                #     third_deg_block = BlockDetected(ecg_signal, frequency).third_degree_block_deetection()
                #     if third_deg_block != 'Abnormal':
                #         abs_result = third_deg_block
                model_label, ei_ti_block = block_model_check(ecg_signal, frequency, abs_result, block_model)
                if model_label != 'Abnormal':
                    abs_result = model_label
                    
                lead_data['block_label'] = model_label
                lead_data['ei_ti_block'] = ei_ti_block
                all_block_labels.append(model_label)
                all_lead_result_data[lead] = lead_data
        if len(all_lead_result_data) > 1:
            if all_block_labels:
                labels_result = find_label_couter(all_block_labels)
                counts = Counter(labels_result)
                repeated_elements = [item for item, count in counts.items() if count > rep_thresh]
                block_label = ' '.join(repeated_elements)
            if all_ei_ti_laels:
                ei_ti_labels_result = find_label_couter(all_ei_ti_laels)
                et_count = Counter(ei_ti_labels_result)
                find_repeated = [item for item, count in et_count.items() if count > rep_thresh]
                ei_ti_label = ' '.join(find_repeated)
        else:
            block_label = all_lead_result_data['II']['block_label']
            ei_ti_label = all_lead_result_data['II']['ei_ti_block']
    
    result_dic = {
        'block_label': block_label,
        'ei_ti_label': ei_ti_label
    }
    return result_dic

# Pacemaker detection
def pacemake_detect(ecg_signal, fs=200):
    pqrst_data = pqrst_detection(ecg_signal, fs=fs, width=(3, 50)).get_data()
    r_index = pqrst_data['R_index']
    q_index = pqrst_data['Q_Index']
    s_index = pqrst_data['S_Index']
    p_index = pqrst_data['P_Index']
    v_pacemaker = []
    a_pacemaker = []
    q_to_pace = []

    qd = int(fs * 0.08)
    percentage = 0
    for q in q_index:
        _q = q - qd
        aoi1 = ecg_signal[_q:q]
        if aoi1.any():
            peaks1 = np.where(np.min(aoi1) == aoi1)[0][0]
            peaks1 += _q
            q_peaks_distance = abs(q - peaks1)
            if q_peaks_distance < 11:
                q_to_pace.append(1)
            else:
                q_to_pace.append(0)

    if len(q_to_pace) != 0:
        percentage = (q_to_pace.count(1) / len(q_to_pace))

    for q in q_index:
        _q = q - qd
        aoi1 = ecg_signal[_q:q]
        if aoi1.any():
            peaks1 = np.where(np.min(aoi1) == aoi1)[0][0]
            peaks1 += _q
            if -0.6 <= ecg_signal[peaks1] <= -0.1 and ecg_signal[q] > ecg_signal[peaks1] and abs(
                    ecg_signal[q] - ecg_signal[peaks1]) >= 0.15 and percentage > 0.5:
                if np.min(np.abs(r_index - peaks1)) > 14:
                    v_pacemaker.append(peaks1)

    for i in range(0, len(r_index) - 1):
        aoi = ecg_signal[s_index[i]:q_index[i + 1]]
        if aoi.any():
            check, _ = find_peaks(aoi, prominence=(0.2, 0.3), distance=100, width=(1, 6))
            peaks1 = check + s_index[i]
        else:
            peaks1 = np.array([])
        if peaks1.any():
            a_pacemaker.extend(list(peaks1))

    # Remove a_pacemaker if it falls within 20 data points of a v_pacemaker or Atrial_&_Ventricular_pacemaker
    for v_peak in v_pacemaker:
        for k in range(len(a_pacemaker) - 1, -1, -1):
            if abs(a_pacemaker[k] - v_peak) <= 20:
                a_pacemaker.pop(k)

    atrial_per = venti_per = 0
    if len(r_index) != 0:
        atrial_per = round((len(a_pacemaker) / len(r_index)) * 100)
        venti_per = round((len(v_pacemaker) / len(r_index)) * 100)

    if atrial_per > 70 and venti_per > 70:
        pacemaker = np.concatenate((v_pacemaker, a_pacemaker)).astype('int64').tolist()
        pacmaker_per = round((len(a_pacemaker) / len(r_index)) * 100)
        label = "Atrial_&_Ventricular_pacemaker"
    elif atrial_per >= 80 and venti_per >= 80:
        if venti_per > atrial_per:
            label = "Ventricular_Pacemaker"
            pacemaker = v_pacemaker
        else:
            label = "Atrial_Pacemaker"
            pacemaker = a_pacemaker
    elif atrial_per >= 80:
        label = "Atrial_Pacemaker"
        pacemaker = a_pacemaker
    elif venti_per >= 80:
        label = "Ventricular_Pacemaker"
        pacemaker = v_pacemaker
    else:
        label = "False"
        pacemaker = np.array([])
    return label, pacemaker

def image_array_new(signal, scale=25):
    '''
    Other : scale=25, wavelet_name='gaus6'
    AFIB : scale=25, wavelet_name='morl'
    VFIB/VFlutter : scale=50, wavelet_name='mexh'
    '''
    scales = np.arange(1, scale, 1)
    coef, freqs = pywt.cwt(signal, scales, 'gaus6')
    # coef, freqs = pywt.cwt(signal, scales, wavelet_name)
    abs_coef = np.abs(coef)
    y_scale = abs_coef.shape[0] / 224
    x_scale = abs_coef.shape[1] / 224
    x_indices = np.arange(224) * x_scale
    y_indices = np.arange(224) * y_scale
    x, y = np.meshgrid(x_indices, y_indices, indexing='ij')
    x = x.astype(int)
    y = y.astype(int)
    rescaled_coef = abs_coef[y, x]
    min_val = np.min(rescaled_coef)
    max_val = np.max(rescaled_coef)
    normalized_coef = (rescaled_coef - min_val) / (max_val - min_val)
    cmap_indices = (normalized_coef * 256).astype(np.uint8)
    cmap = colormaps.get_cmap('viridis')
    rgb_values = cmap(cmap_indices)
    image = rgb_values.reshape((224, 224, 4))[:, :, :3]
    denormalized_image = (image * 254) + 1
    rotated_image = np.rot90(denormalized_image, k=1, axes=(1, 0))
    return rotated_image.astype(np.uint8)

# Afib & Flutter detection
class afib_flutter_detection:
    def __init__(self, ecg_signal, r_index, q_index, s_index, p_index, p_t, pr_interval, load_model):
        self.ecg_signal = ecg_signal
        self.r_index = r_index
        self.q_index = q_index
        self.s_index = s_index
        self.p_index = p_index
        self.p_t = p_t
        self.pr_inter = pr_interval
        self.load_model = load_model

    def image_array_new(self, signal, scale=25):
        scales = np.arange(1, scale, 1)
        coef, freqs = pywt.cwt(signal, scales, 'gaus6')
        # coef, freqs = pywt.cwt(signal, scales, wavelet_name)
        abs_coef = np.abs(coef)
        y_scale = abs_coef.shape[0] / 224
        x_scale = abs_coef.shape[1] / 224
        x_indices = np.arange(224) * x_scale
        y_indices = np.arange(224) * y_scale
        x, y = np.meshgrid(x_indices, y_indices, indexing='ij')
        x = x.astype(int)
        y = y.astype(int)
        rescaled_coef = abs_coef[y, x]
        min_val = np.min(rescaled_coef)
        max_val = np.max(rescaled_coef)
        normalized_coef = (rescaled_coef - min_val) / (max_val - min_val)
        cmap_indices = (normalized_coef * 256).astype(np.uint8)
        cmap = colormaps.get_cmap('viridis')
        rgb_values = cmap(cmap_indices)
        image = rgb_values.reshape((224, 224, 4))[:, :, :3]
        denormalized_image = (image * 254) + 1
        rotated_image = np.rot90(denormalized_image, k=1, axes=(1, 0))
        return rotated_image.astype(np.uint8)

    def abs_afib_flutter_check(self):
        check_afib_flutter = False
        list_per = 0
        rpeak_diff = np.diff(self.r_index)
        more_then_3_rhythm_per = len(list(filter(lambda x: len(x) >= 3, self.p_t))) / len(self.r_index)
        inner_list_less_2 = len(list(filter(lambda x: len(x) < 2, self.p_t))) / len(self.r_index)

        zeros_count = self.p_index.count(0)
        if self.p_index:
            list_per = zeros_count / len(self.p_index)
        pr_int = [round(num, 2) for num in self.pr_inter]

        constant_list = []
        if len(pr_int) > 1:
            for i in range(len(pr_int) - 1):
                diff = abs(pr_int[i] - pr_int[i + 1])
                if diff == 0 or diff == 1:
                    constant_list.append(pr_int[i])

            if abs(pr_int[-1] - pr_int[-2]) == 0 or abs(pr_int[-1] - pr_int[-2]) == 1:
                constant_list.append(pr_int[-1])

        if more_then_3_rhythm_per >= 0.6:
            check_afib_flutter = True
        elif list_per >= 0.5:
            check_afib_flutter = True
        elif len(constant_list) != 0:
            if (len(constant_list) / len(pr_int) < 0.7):
                check_afib_flutter = True
        else:
            p_peak_diff = np.diff(self.p_index)
            percentage_diff = np.abs(np.diff(p_peak_diff) / p_peak_diff[:-1]) * 100

            mean_p = np.mean(percentage_diff)
            if mean_p != mean_p or mean_p == float('inf') or mean_p == float('-inf'):
                check_afib_flutter = True
            if (mean_p > 15 and more_then_3_rhythm_per >= 0.4) or (mean_p > 70 and inner_list_less_2 > 0.3):
                check_afib_flutter = True
            elif mean_p > 100 and inner_list_less_2 > 0.3:
                check_afib_flutter = True
            elif (mean_p > 20 and more_then_3_rhythm_per >= 0.1):
                check_afib_flutter = True
        return check_afib_flutter

    def predict_tflite_model(self, model: tuple, input_data: tuple):
        with results_lock:
            interpreter, input_details, output_details = model
            for i in range(len(input_data)):
                interpreter.set_tensor(input_details[i]['index'], input_data[i])
            interpreter.invoke()
            output = interpreter.get_tensor(output_details[0]['index'])

        return output

    def check_model(self, q_new, s_new, ecg_signal, last_s, last_q):
        percent = {'ABNORMAL': 0, 'AFIB': 0, 'FLUTTER': 0, 'NOISE': 0, 'NORMAL': 0}
        total_data = len(self.s_index) - 1
        afib_data_index, flutter_data_index = [], []
        for q, s in zip(q_new, s_new):
            data = ecg_signal[q:s]
            if data.any():
                image_data = self.image_array_new(data)
                image_data = (tf.expand_dims(image_data.astype(np.float32), axis=0),)
                model_pred = self.predict_tflite_model(self.load_model, image_data)[0]

                model_idx = np.argmax(model_pred)

                if model_idx == 0:
                    if last_s and s > last_s[0]:
                        percent['ABNORMAL'] += last_s[1] / total_data
                    else:
                        percent['ABNORMAL'] += 4 / total_data
                elif model_idx == 1:
                    if last_s and s > last_s[0]:
                        percent['AFIB'] += last_s[1] / total_data
                        afib_data_index.append((last_q, s))
                    else:
                        percent['AFIB'] += 4 / total_data
                        afib_data_index.append((q, s))
                elif model_idx == 2:
                    if last_s and s > last_s[0]:
                        percent['FLUTTER'] += last_s[1] / total_data
                        flutter_data_index.append((last_q, s))
                    else:
                        percent['FLUTTER'] += 4 / total_data
                        flutter_data_index.append((q, s))
                elif model_idx == 3:
                    if last_s and s > last_s[0]:
                        percent['NOISE'] += last_s[1] / total_data
                    else:
                        percent['NOISE'] += 4 / total_data
                elif model_idx == 4:
                    if last_s and s > last_s[0]:
                        percent['NORMAL'] += last_s[1] / total_data
                    else:
                        percent['NORMAL'] += 4 / total_data
        return percent, afib_data_index, flutter_data_index

    def get_data(self):
        total_data = len(self.s_index) - 1
        last_s = None
        last_q = None
        check_2nd_lead = {'ABNORMAL': 0, 'AFIB': 0, 'FLUTTER': 0, 'NOISE': 0, 'NORMAL': 0}
        afib_data_index, flutter_data_index = [], []
        if len(self.q_index) > 4 and len(self.s_index) > 4:
            q_new = self.q_index[:-4:4] #.tolist()
            s_new = self.s_index[4::4] #.tolist()
            if s_new[-1] != self.s_index[-1]:
                temp_s = list(self.s_index).index(s_new[-1])
                fin_s = total_data - temp_s
                last_q = self.q_index[temp_s]
                last_s = (s_new[-1], fin_s)
                q_new.append(self.q_index[-5])
                s_new.append(self.s_index[-1])
            check_2nd_lead, afib_data_index, flutter_data_index = self.check_model(q_new, s_new, self.ecg_signal,
                                                                                   last_s, last_q)
        return check_2nd_lead, afib_data_index, flutter_data_index


# For RR regular or Irregular
# def get_percentage_diff(previous, current):
#     try:
#         percentage = abs(previous - current) / max(previous, current) * 100
#     except ZeroDivisionError:
#         percentage = float('inf')
#     return percentage

# def Average(lst):
#     return sum(lst) / len(lst)

# def new_rr_check(r_index):
#     variation = []
#     r_label = "Regular"
#     for i in range(len(r_index) - 1):
#         variation.append(get_percentage_diff(r_index[i + 1], r_index[i]))
#     if len(variation) != 0:
#         if Average(variation) > 12:
#             r_label = "Irregular"

#     return r_label

# def check_r_irregular(r_index):
#     r_label = "Regular"
#     mean_percentage_diff = irrgular_per_r = 0
#     rpeak_diff = np.diff(r_index)
#     if len(rpeak_diff) >= 3:
#         percentage_diff = np.abs(np.diff(rpeak_diff) / rpeak_diff[:-1]) * 1003
#         list_per_r = [value for value in percentage_diff if value > 14]
#         irrgular_per_r = (len(list_per_r) / len(percentage_diff)) * 100
#         mean_percentage_diff = np.mean(percentage_diff)

#     if (mean_percentage_diff > 75) and (irrgular_per_r > 80):
#         r_label = "Irregular"
#     return r_label

# Long  & Short Puse detection
def SACompare(list1, val):
    l = []
    for x in list1:
        if x >= val:
            l.append(1)
        else:
            l.append(0)
    if 1 in l:
        return True
    else:
        return False

def SACompareShort(list1, val1, val2):
    l = []
    for x in list1:
        if x >= val1 and x <= val2:
            l.append(1)
        else:
            l.append(0)
    if 1 in l:
        return True
    else:
        return False

def check_long_short_pause(r_index):
    SAf = []
    # r_interval = np.diff(r_index)
    pause_label = 'Abnormal'
    if len(r_index) > 1:
        for i in range(len(r_index) - 1):
            rr_peaks = abs(int(r_index[i]) * 5 - int(r_index[i + 1]) * 5)
            SAf.append(rr_peaks)

    if (SACompare(SAf, 4500)):
        l = []
        for x in SAf:
            if x >= 4500:
                l.append(1)
            else:
                l.append(0)
        if 1 in l:
            noofpause = l.count(1)
        else:
            noofpause = 0
        if noofpause != 0:
            pause_label = 'LONG_PAUSE'

        # "noOfPauseList":[a/1000 for a in SAf if a>3000]

    if SACompareShort(SAf, 3500, 4000):
        l = []
        for x in SAf:
            if x >= 3500 and x <= 4000:
                l.append(1)
            else:
                l.append(0)
        if 1 in l:
            noofpause = l.count(1)
        else:
            noofpause = 0
        if noofpause != 0:
            pause_label = 'SHORT_PAUSE'
    return pause_label

# long QT detection
def detection_long_qt(ecg_signal, rpeaks, fs=200):
    try:
        _, waves_peak = nk.ecg_delineate(ecg_signal, rpeaks, sampling_rate=fs, method="peak")
        signal_dwt, waves_dwt = nk.ecg_delineate(ecg_signal, rpeaks, sampling_rate=fs, method="dwt")

        Tpeaks = np.where(np.isnan(waves_peak['ECG_T_Peaks']), 0, waves_peak['ECG_T_Peaks']).astype('int64').tolist()
        Qpeaks = np.where(np.isnan(waves_peak['ECG_Q_Peaks']), 0, waves_peak['ECG_Q_Peaks']).astype('int64').tolist()
        QTint = []
        finallist = []

        for i in range(len(Qpeaks) - 1):
            try:
                if Qpeaks[i] == 0 or Tpeaks[i] == 0:
                    QTint.append(0)
                else:
                    QT = abs(int(Qpeaks[i]) - int(Tpeaks[i])) / 200
                    QTint.append(QT)
                    if QT > 0.5: finallist.append(QT)  # 0.2
            except:
                QTint.append(0)

        label = "Abnormal"
        if len(finallist) > 5:
            label = "Long_QT_Syndrome"
        return label
    except Exception as r:
        return "Abnormal"

def is_rhythm_pos_neg(baseline_signal, r_index, fs):
    pos_neg_ind = []
    rhy_label = 'Positive'
    for r_idx in r_index:
        st_idx = max(0, r_idx - int(0.1 * fs))
        ed_idx = min(len(baseline_signal), r_idx + int(0.1 * fs))
        qrs_complex = baseline_signal[st_idx: ed_idx]
        positive_sum = np.sum(qrs_complex[qrs_complex > 0])
        negative_sum = np.sum(qrs_complex[qrs_complex < 0])
        if positive_sum > abs(negative_sum):
            pos_neg_ind.append(1)
        else:
            pos_neg_ind.append(0)

    pos_count = pos_neg_ind.count(1)
    neg_count = pos_neg_ind.count(0)
    if len(pos_neg_ind) != 0:
        most_common_ele = max(set(pos_neg_ind), key=lambda x: pos_neg_ind.count(x))
        if pos_count == len(pos_neg_ind):
            rhy_label = 'Positive'
        elif neg_count == len(pos_neg_ind):
            rhy_label = 'Negative'
        elif pos_count == neg_count:
            rhy_label = 'Positive'
        elif most_common_ele == 1:
            rhy_label = 'Positive'
        elif most_common_ele == 0:
            rhy_label = 'Negative'
    return rhy_label

def check_axis_deviation_and_hypertrophy(all_leads_data, r_index, MI_results, fs):
    lead_rhythm = {}
    c = 0
    axis_devi_result = "No Axis Deviation Found"
    haypertrophy_result = "No Hypertrophy Found"
    
    for lead in ["I", "II", "III", "aVL", "aVF", "v1", "v3"]: #all_leads_data.columns:
        ecg_signal = all_leads_data[lead].values
        baseline_signal = baseline_construction_200(ecg_signal, kernel_size=101)
        is_rhythm = is_rhythm_pos_neg(baseline_signal, r_index, fs)
        lead_rhythm[lead] = is_rhythm
    
    if lead_rhythm:
        if (lead_rhythm['I'] == "Positive" and lead_rhythm["aVL"] == "Positive" and 
            lead_rhythm["II"] == "Negative" and lead_rhythm["aVF"] == "Negative"):
            axis_devi_result = "LAD"
        elif (lead_rhythm["I"] == "Negative" and lead_rhythm["aVL"] == "Negative" and
            lead_rhythm["II"] == "Positive" and lead_rhythm["aVF"] == "Positive" and 
            lead_rhythm["III"] == "Positive"):
            axis_devi_result = "RAD"
        elif lead_rhythm["I"] == "Positive" and lead_rhythm["aVL"] == "Positive":
            axis_devi_result = "Extreme Axis Deviation"
    # if MI_results:
    #     if (MI_results["I"]['mi_result'] == "STDEP" and MI_results["aVL"]['mi_result'] == "STDEP" and 
    #         MI_results["v1"]['mi_result'] == "STELE" and MI_results["v1"]['mi_result'] == "STELE" and 
    #         axis_devi_result == "LAD"):
    #         haypertrophy_result = "LVH"
    #     elif (MI_results["v1"]['mi_result'] == "STDEP" and MI_results["II"]['mi_result'] == "STDEP" and
    #         MI_results["III"]['mi_result'] == "STDEP" and MI_results["aVF"]['mi_result'] == "STDEP" and 
    #         axis_devi_result == "LAD"):
    #         haypertrophy_result = "RVH"

    return axis_devi_result, haypertrophy_result

def combine_ecg_detection(ecg_data, is_lead_for, frequency=200):
    pac_data, pvc_data = {}, {}
    c_label = ''
    afib_label, aflutter_label , jr_label = "Abnormal", "Abnormal", "Abnormal"
    wide_qrs_label, longqt_label, final_block_label = "Abnormal", "Abnormal", "Abnormal"
    check_pause, pac_class, pvc_class, lbbb_rbbb_label = "Abnormal", "Abnormal", "Abnormal", "Abnormal"
    data = {
        'Combine_Label': "",
        'MI_label': "",
        'R_Index': np.array([]),
        'Q_Index': [],
        'S_Index': [],
        'T_Index': [],
        'P_Index': [],
        'P_T': [],
        'HR_Count': 0,
    }

    mi_result, axis_devi_result = "Abnormal", "Abnormal"
    hr_counts = 0
    # noise_reult =  check_noise(ecg_data, is_lead_for, noise_model)
    vfib_result = Vfib_asys_detection(ecg_data, frequency,  is_lead_for)
    noise_reult = "Normal"
    if vfib_result not in ['VFIB', "Asystole", "Noise"] and noise_reult == "Normal":
        r_index, combine_r_index = check_r_index(ecg_data, is_lead_for, frequency)
        s_index, q_inedx = check_qs_index(ecg_data, r_index ,frequency)
        pt_dic, t_index, p_index, rr_invl_peaks, T_onset, P_offset = check_pt_index(ecg_data,  is_lead_for, r_index, combine_r_index)
        hr_counts = hr_count(ecg_data["II"], r_index)
        data["R_Index"] = r_index
        data["S_Index"] = s_index
        data["Q_Index"] = q_inedx
        data['HR_Count'] = hr_counts
        baseline_signal, lowpass_signal = filter_signal(ecg_data["II"], frequency).get_data()
        pace_label, pacemaker_index = pacemake_detect(baseline_signal, fs=frequency)
        pqrst_data = pqrst_detection(baseline_signal, fs=frequency).get_data()
        pr_interval = pqrst_data['PR_Interval']
        p_t = pqrst_data['P_T List']
        # p_index = pqrst_data['P_Index']
        r_label = "Regular"
        r_check_1 = new_rr_check(r_index)
        r_check_2 = check_r_irregular(r_index)
        if r_check_1 == 'Irregular' and r_check_2 == 'Irregular':
            r_label = "Irregular"
            
        pvc_detection_result = PVCDetection(ecg_data, frequency, r_index, is_lead_for).get_pvc_data()
        if pvc_detection_result['pvc_label']:
            pvc_class = pvc_detection_result['pvc_label']
        if pvc_detection_result['lbbb_rbbb_label']:
            lbbb_rbbb_label = pvc_detection_result['lbbb_rbbb_label']
        mi_labels = mi_processing(ecg_data, is_lead_for, frequency)
        if lbbb_rbbb_label:
            if lbbb_rbbb_label != "Abnormal":
                mi_result = lbbb_rbbb_label
            if mi_labels not in ['Normal', 'Abnormal'] and mi_result not in ['LBBB', 'RBBB']: # for temp only remove LBBB, RBBB
                if mi_result in ['LBBB', 'RBBB']:
                    mi_result += f", {mi_labels}" 
                else:
                    mi_result = mi_labels
        if is_lead_for == "12":
            axis_devi_result, haypertrophy_result = check_axis_deviation_and_hypertrophy(ecg_data, r_index, mi_result, frequency)
        if all(p not in ['VT', 'Ivr', 'NSVT', 'PVC-Triplet', 'PVC-Couplet'] for p in pvc_class):
            if hr_counts <= 60:
                check_pause = check_long_short_pause(r_index)
            if r_label == "Regular":
                pac_result = PACDetection(ecg_data, r_index, frequency, is_lead_for).get_pac_data()
                jr_label = pac_result['jnc_label']
                
                if hr_counts >= 55:
                    pac_class = pac_result['pac_label']
                

                if hr_counts <= 80:
                    block_result = block_detection_processing(ecg_data, is_lead_for, fs=frequency)
                    if block_result['block_label'] != "Abnormal":
                        final_block_label = block_result['block_label']

                if len(pac_class) == 0 and len(pvc_class) == 0:
                    lowpass_signal = lowpass(baseline_signal, 0.2)
                    longqt_label = detection_long_qt(lowpass_signal, r_index, frequency)
            else:
                afib_flutter_check = afib_flutter_detection(lowpass_signal, r_index, q_inedx, s_index, p_index, p_t,
                                                            pr_interval, afib_model)
                is_afib_flutter = afib_flutter_check.abs_afib_flutter_check()
                afib_model_per = flutter_model_per = 0
                if is_afib_flutter:
                    afib_flutter_per, afib_indexs, flutter_indexs = afib_flutter_check.get_data()
                    afib_model_per = int(afib_flutter_per['AFIB'] * 100)
                    flutter_model_per = int(afib_flutter_per['FLUTTER'] * 100)
                if afib_model_per >= 60: # >= 60
                    afib_label = 'AFIB'
                # elif afib_label != 'AFIB':
                #     if flutter_model_per >= 60:
                #         aflutter_label = 'AFL'
                
                if afib_label != 'AFIB':
                    if flutter_model_per >= 50:
                        aflutter_label = 'AFL'
                if afib_label != 'AFIB':
                    pac_result = PACDetection(ecg_data, r_index, frequency, is_lead_for).get_pac_data()
                    jr_label = pac_result['jnc_label']
                    pac_class = pac_result['pac_label']
                        
                    if hr_counts <= 80:
                        block_result = block_detection_processing(ecg_data, is_lead_for, fs=frequency)
                        if block_result['block_label'] != "Abnormal":
                            final_block_label = block_result['block_label']
                    
            
            pac_class = "Abnormal" if pac_class == '' else pac_class
            label = {'Afib_label': afib_label,
                     'Aflutter_label': aflutter_label,
                     'JR_label': jr_label,
                     'wide_qrs_label': wide_qrs_label,
                     'longqt_label': longqt_label,
                     'final_block_label': final_block_label,
                     'check_pause': check_pause,
                     'pac_class': pac_class,
                     'pvc_class': pvc_class}
            combine_labels = [l for l in label.values() if 'Abnormal' not in l]
            if combine_labels:
                if len(combine_labels) > 1:
                    c_label = "; ".join(combine_labels)
                else:
                    c_label = combine_labels[0]
            else:
                if len(c_label) ==  0:
                    if hr_counts > 100:
                        c_label = "TC"
                    elif 30 < hr_counts < 60:
                        c_label = "BR"
                    else:
                        c_label = "Normal"
        else:
            # c_label = "; ".join(pvc_class)
            c_label = pvc_class
        data['Combine_Label'] = c_label
        data['MI_label'] = mi_result
        data["axis_devi_label"] = axis_devi_result
    else:
        if vfib_result != "Abnormal":
            data['Combine_Label'] = vfib_result
        else:
            data['Combine_Label'] = noise_reult
        data['MI_label'] = mi_result
        data["axis_devi_label"] = axis_devi_result
    return data

class PDF(FPDF):
    def __init__(self):
        super().__init__(
            orientation="L",       # Landscape
            unit="mm",
            format=(355.6, 353.2)  # MATCH ECG PAGE SIZE
        )

        # Reduce default margins
        self.set_margins(left=10, top=10, right=10)
        self.set_auto_page_break(auto=True, margin=15)
    def header(self):
        self.set_font("Arial", "B", 12)
        self.cell(0, 10, "Arrhythmia Report", align="C", ln=True)

    def footer(self):
        self.set_y(-15)
        self.set_font("Arial", "I", 8)
        self.cell(0, 10, f"Page {self.page_no()}", align="C")

    def add_summary(self, label_counts, hr_counts):
        self.add_page()

        # Title
        self.set_font("Arial", "B", 14)
        self.cell(0, 10, "Summary", ln=True)
        self.ln(6)

        # Table layout
        page_width = self.w - self.l_margin - self.r_margin
        col_widths = [
            page_width * 0.30,  # Filename
            page_width * 0.30,  # Label
            page_width * 0.20,  # Occurrence
            page_width * 0.20,  # HR
        ]

        # Header
        self.set_font("Arial", "B", 12)
        headers = ["Filename", "Label", "Occurrence", "HR"]
        for w, h in zip(col_widths, headers):
            self.cell(w, 10, h, 1, align="C")
        self.ln()

        self.set_font("Arial", "", 11)

        for filename, labels in label_counts.items():
            for label, count in labels.items():
                if label in ("hr", "hr_counts", "HR", "HR_Count"):
                    continue
                hr = labels.get("hr", "-")
                self.cell(col_widths[0], 10, filename, 1)
                self.cell(col_widths[1], 10, label.capitalize(), 1)
                self.cell(col_widths[2], 10, str(count), 1, align="C")
                self.cell(col_widths[3], 10, str(hr), 1, align="C")
                self.ln()
def preprocess_label_counts(label_counts):
    processed_counts = defaultdict(lambda: defaultdict(int))

    for full_filename, labels in label_counts.items():
        base_filename = full_filename.split("_")[0].lower()

        for label, count in labels.items():
            if label == "hr_list":
                processed_counts[base_filename].setdefault("hr_values", [])

                # Keep only valid numeric HRs
                valid_hr = [
                    hr for hr in count
                    if isinstance(hr, (int, float))
                ]
                processed_counts[base_filename]["hr_values"].extend(valid_hr)

            else:
                processed_counts[base_filename][label] += count

    # Compute average HR
    for fname in processed_counts:
        hr_values = processed_counts[fname].get("hr_values", [])

        if hr_values:
            avg_hr = round(sum(hr_values) / len(hr_values))
            processed_counts[fname]["hr"] = avg_hr
        else:
            processed_counts[fname]["hr"] = "-"

        processed_counts[fname].pop("hr_values", None)

    return processed_counts
def setup_ecg_subplot(ax):
    total_time = 10  # seconds
    ax.set_xlim(0, total_time)

    # major grid: X every 0.2s, Y every 0.5mV (assumed 1 unit = 0.1mV ? 5 units = 0.5mV)
    ax.set_xticks(np.arange(0, total_time, 0.2))
    ax.set_yticks(np.arange(-50, 91, 5))  # 5 units ~ 0.5mV (if scaled appropriately)

    # Minor grid: X every 0.04s, Y every 1 unit (~0.1mV)
    ax.set_xticks(np.arange(0, total_time + 0.01, 0.04), minor=True)
    ax.set_yticks(np.arange(-50, 91, 1), minor=True)

    # Major grid styling (bold red)
    ax.grid(True, which='major', color='red', linewidth=0.6, alpha=0.7)

    # Minor grid styling (light red)
    ax.grid(True, which='minor', color='pink', linewidth=0.3, alpha=0.3)

    # Turn off ticks and labels
    ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
    
def plotting(result_dic, ecg_data, local_name,is_lead_for, pdf, label_counts, hr_counts):
    if is_lead_for == '7':
        limb_leads = ['I', 'III', 'aVL', 'v5']
        chest_leads = ['II', 'aVR', 'aVF']
    else:
        limb_leads = ['I', 'II', 'III', 'aVR', 'aVL', 'aVF']  # Left side leads
        chest_leads = ['v1', 'v2', 'v3', 'v4', 'v5', 'v6']
    # Dynamic figure size based on lead configuration
    if is_lead_for == "2":
        fig_size=(8,4)  # Smaller size for 2 leads
    else:
        fig_size = (14, 10)  # Standard size for 7 leads
    voltage_gain = 10  # mm/mV for voltage gain
    sampling_rate = 500  # Sampling rate in Hz
    fig, ax = plt.subplots(figsize=fig_size, dpi=100)

    ax.set_ylim(-50, 90)
    lead_spacing = 20
    base_y_left = 70
    base_y_right = 70
    mid_x = 5 

    setup_ecg_subplot(ax)
    ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
    ax.set_xlabel("")
    ax.set_ylabel("")

    # ---------------------------
    # 3. Normalize All Signals
    # ---------------------------
    max_length = max([len(ecg_data[lead]) for lead in ecg_data])
    left_x_scale  = np.linspace(0, mid_x - 0.1, max_length)
    right_x_scale = np.linspace(mid_x + 0.1, 10, max_length)

    scaler = MinMaxScaler(feature_range=(0, 16))

    scaled_data = {}
    # for lead in ecg_data.keys():
    #     raw = np.array(ecg_data[lead])
    #     scaled = scaler.fit_transform(raw.reshape(-1, 1)).squeeze()
    #     scaled_data[lead] = scaled
    for lead in ecg_data.columns:
        # skip non-numeric columns safely
        if ecg_data[lead].dtype == object:
            print(f"Skipping non-numeric column: {lead}")
            continue

        raw = pd.to_numeric(ecg_data[lead], errors='coerce').fillna(0).values
        scaled = scaler.fit_transform(raw.reshape(-1, 1)).squeeze()
        scaled_data[lead] = scaled
    # ---------------------------
    # 4. Fetch R, Q, S Indexes
    # ---------------------------
    r_index = result_dic.get("R_Index", [])
    q_index = result_dic.get("Q_Index", [])
    s_index = result_dic.get("S_Index", [])

    # ---------------------------
    # 5. Plot Limb Leads (Left Side)
    # ---------------------------
    for idx, lead in enumerate(limb_leads):
        if lead not in scaled_data:
            continue
        sig = scaled_data[lead]
        y_offset = base_y_left - (idx * lead_spacing)
        ax.plot(left_x_scale, sig + y_offset, color="black", linewidth=1)
        ax.text(left_x_scale[0] + 0.1, y_offset + 1, lead, fontsize=12, fontweight="bold", color="blue")

        # Mark Indexes
        if len(r_index) > 0:
            ax.plot(left_x_scale[r_index], sig[r_index] + y_offset, 'bo', markersize=3)
        if len(q_index) > 0:
            ax.plot(left_x_scale[q_index], sig[q_index] + y_offset, 'go', markersize=3)
        if len(s_index) > 0:
            ax.plot(left_x_scale[s_index], sig[s_index] + y_offset, 'mo', markersize=3)

    # ---------------------------
    # 6. Plot Chest Leads (Right Side)
    # ---------------------------
    for idx, lead in enumerate(chest_leads):
        if lead not in scaled_data:
            continue
        sig = scaled_data[lead]
        y_offset = base_y_right - (idx * lead_spacing)
        ax.plot(right_x_scale, sig + y_offset, color="black", linewidth=1)
        ax.text(right_x_scale[0] + 0.1, y_offset + 1, lead, fontsize=12, fontweight="bold", color="blue")

        if len(r_index) > 0:
            ax.plot(right_x_scale[r_index], sig[r_index] + y_offset, 'bo', markersize=3)
        if len(q_index) > 0:
            ax.plot(right_x_scale[q_index], sig[q_index] + y_offset, 'go', markersize=3)
        if len(s_index) > 0:
            ax.plot(right_x_scale[s_index], sig[s_index] + y_offset, 'mo', markersize=3)

    # ---------------------------
    # 7. Title (combine labels)
    # ---------------------------
    # title = f"{local_name}"
    # if "Combine_Label" in result_dic:
    #     title += f" | Arrhythmia: {result_dic['Combine_Label']}"
    # if "MI_label" in result_dic and result_dic["MI_label"] not in ["Normal", "Abnormal"]:
    #     title += f" | MI: {result_dic['MI_label']}"
    # if "axis_devi_label" in result_dic and result_dic["axis_devi_label"] != "Abnormal":
    #     title += f" | Axis Deviation: {result_dic['axis_devi_label']}"
    footer_text = f"{result_dic.get('Combine_Label', '')}"
    print(result_dic.keys())
    plt.figtext(0.5, 0.01, footer_text, wrap=True, ha='center', fontsize=12)
    # plt.suptitle(title)
    # print(title)
    # ---------------------------
    # 8. Save PDF
    # ---------------------------
    plt.tight_layout(rect=[0, 0.03, 1, 0.97])
    plt.show()
    pdf_buffer = BytesIO()
    fig.savefig(pdf_buffer, format="pdf")
    pdf_buffer.seek(0)
    
    plt.close()
    # Update 
    hr_value = result_dic.get("HR_Count")
    label_counts.setdefault(local_name, {}).setdefault("hr_list", []).append(hr_value)
    
    labels = result_dic.get("Combine_Label", "").lower().split(";")
    for label in labels:
        label = label.strip()
        if label:
            label_counts[local_name][label] = label_counts[local_name].get(label, 0) + 1

    return pdf_buffer
def create_summary_pdf(label_counts, hr_counts):
    pdf = PDF()
    final_counts = preprocess_label_counts(label_counts)

    pdf.add_summary(final_counts, hr_counts)

    # FPDF FIX
    pdf_bytes = pdf.output(dest="S").encode("latin1")

    buffer = BytesIO(pdf_bytes)
    buffer.seek(0)
    return buffer

def merge_pdfs_in_memory(summary_buffer, ecg_buffers):
    merger = PdfMerger()

    # FIRST PAGE = SUMMARY
    merger.append(summary_buffer)

    # ECG pages after
    for buf in ecg_buffers:
        merger.append(buf)

    final_buffer = BytesIO()
    merger.write(final_buffer)
    merger.close()

    final_buffer.seek(0)
    return final_buffer
def store_final_pdf_in_db(patient_id,arrhythmia, pdf_buffer):
    file_id = download_fs.put(
        pdf_buffer,
        filename=f"{patient_id}.pdf",
        contentType="application/pdf",
        meta={     
            "patient_id": patient_id,
            "arrhythmia": arrhythmia,
            "created_at": datetime.utcnow()
        }
    )

    logs_collection.insert_one({
        "patient_id": patient_id,
        "file_id": file_id,
        "filename": f"{patient_id}.pdf",
        "created_at": datetime.utcnow()
    })

    return str(file_id)

def merge_patient_pdfs(patient_id, pdf_folder, output_path):
    merger = PdfMerger()

    pdf_files = sorted([
        f for f in os.listdir(pdf_folder)
        if f.endswith(".pdf") and (
            f.startswith(patient_id) or f == "Final_Combined_Report.pdf"
        )
    ])

    if not pdf_files:
        raise ValueError("No PDFs found for patient")

    for pdf in pdf_files:
        merger.append(os.path.join(pdf_folder, pdf))

    merger.write(output_path)
    merger.close()

    return output_path
def store_patient_pdf_in_db(
    patient_id,
    pdf_folder,
    arrhythmia,
    download_fs,
    logs_collection
):
    merger = PdfMerger()

    pdf_files = sorted([
        f for f in os.listdir(pdf_folder)
        if f.endswith(".pdf") and (
            f.startswith(patient_id) or f == "Final_Combined_Report.pdf"
        )
    ])

    if not pdf_files:
        raise ValueError("No PDFs found for patient")

    # Ensure summary is last
    if "Final_Combined_Report.pdf" in pdf_files:
        pdf_files.remove("Final_Combined_Report.pdf")
        pdf_files.append("Final_Combined_Report.pdf")

    merged_path = os.path.join(pdf_folder, f"{patient_id}.pdf")

    try:
        for pdf in pdf_files:
            merger.append(os.path.join(pdf_folder, pdf))

        merger.write(merged_path)
        merger.close()

        with open(merged_path, "rb") as f:
            file_id = download_fs.put(
                f,
                filename=f"{patient_id}.pdf",
                contentType="application/pdf",
                meta={ 
                    "patient_id": patient_id,
                    "arrhythmia": arrhythmia,
                    "created_at": datetime.utcnow()
                }
            )

        logs_collection.insert_one({
            "patient_id": patient_id,
            "file_id": file_id,
            "filename": f"{patient_id}.pdf",
            "arrhythmia": arrhythmia,
            "created_at": datetime.utcnow()
        })

    finally:
        for pdf in pdf_files:
            path = os.path.join(pdf_folder, pdf)
            if os.path.exists(path):
                os.remove(path)

        if os.path.exists(merged_path):
            os.remove(merged_path)

    return str(file_id)
def model_check_for_ecg_data(path, is_lead):

    is_lead_for = is_lead
    print(is_lead)
    frequency = 200 if is_lead in ["2", "7"] else 240
    label_counts = {}
    hr_counts={}
    ecg_pdf_buffers = []
    for fn in glob.glob(path):
        file_name = fn.split("\\")[-1].split(".csv")[0]
        print(file_name)
        patient_id = file_name.split("_")[0]
        
        # Try reading with header first
        df = pd.read_csv(fn)

        if all(str(col).isdigit() for col in df.columns):
            df = pd.read_csv(fn, header=None)
        all_lead_data = df.fillna(0)
        if any(str(_).isalpha() for _ in all_lead_data.iloc[0, :].values):
            if is_lead_for == '2':
                is_lead = 2
                all_lead_data = pd.read_csv(fn, usecols=['ECG']).fillna(0)
                all_lead_data = all_lead_data.rename(columns={'ECG': 'II'})
            elif is_lead_for == '7':
                is_lead = 7
                all_lead_data = pd.read_csv(fn, usecols=['I', 'II', 'III', 'aVR', 'aVL', 'aVF', 'v5']).fillna(0)
            elif is_lead_for == '12':
                all_lead_data = pd.read_csv(fn, usecols=['I', 'II', 'III', 'aVR', 'aVL', 'aVF', 'v1', 'v2','v3', 'v4', 'v5', 'v6']).fillna(0)
                # all_lead_data = pd.read_csv(fn, usecols=['I', 'II', 'III', 'aVR', 'aVL', 'aVF', 'V1', 'V2','V3', 'V4', 'V5', 'V6']).fillna(0)
        else:   
            if is_lead_for == '2':
                is_lead = 2
                all_lead_data = all_lead_data.rename(columns={'ECG':'II'})
            elif is_lead_for == '7':
                is_lead = 7
                all_lead_data = all_lead_data.rename(columns={0:'I', 1:'II', 2:'III', 3: 'aVR', 4: 'aVL', 5:'aVF',  6:'v5'})
            elif is_lead_for == '12':  
                all_lead_data = all_lead_data.rename(columns={0:'I', 1:'II', 2:'III', 3: 'aVR', 4: 'aVL', 5:'aVF', 6: 'v1', 7:'v2', 8:'v3', 9:'v4', 10:'v5', 11:'v6'})
            
        i = 0
        if all_lead_data.shape[0] <= 2500:
            steps = all_lead_data.shape[0]
        else:
            steps = round(frequency * 15)

        while i < all_lead_data.shape[0]:
            ecg_data = all_lead_data[i : i+steps]
            if ecg_data.shape[0] < frequency*2.5:
                
                break
            local_name = f"{file_name}_{i}"
         
            results_dic = combine_ecg_detection(ecg_data, is_lead_for, frequency=frequency)
            
            buf = plotting(results_dic, ecg_data,local_name, is_lead_for,None, label_counts,hr_counts)
            ecg_pdf_buffers.append(buf)
            i += steps
 
    # Summary FIRST
    summary_buffer = create_summary_pdf(label_counts, hr_counts)
    # Merge in memory
    final_pdf_buffer = merge_pdfs_in_memory(
        summary_buffer,
        ecg_pdf_buffers
    )

    # Store directly in DB
    file_id = store_final_pdf_in_db(
        patient_id=patient_id,
        arrhythmia="ALL_Arrhythymia",
        pdf_buffer=final_pdf_buffer
    )

    print("Final report stored in DB:", file_id)
    return file_id




