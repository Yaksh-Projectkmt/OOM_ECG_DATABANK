import os
import pandas as pd
import numpy as np
import tensorflow as tf
from scipy import signal
from scipy.signal import find_peaks, argrelextrema, savgol_filter, butter, filtfilt, resample
from scipy.interpolate import interp1d
from scipy.ndimage import gaussian_filter1d
import warnings
import threading
from scipy.stats import mode
from django.conf import settings

warnings.filterwarnings('ignore')

results_lock = threading.RLock()

def load_tflite_model(model_path):
    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    return interpreter, input_details, output_details 

# Load TFLite models
with tf.device('/CPU:0'):
    r_index_model = load_tflite_model("D:\\hitesh\\ecgdatabank_new\\oom_ecg_data\\model\\rnn_model2_04_09_Unet.tflite")
    pt_index_model = load_tflite_model("D:\\hitesh\\ecgdatabank_new\\oom_ecg_data\\model\\ecg_pt_detection_LSTMGRU_TCN_Transpose_v27.tflite")

def lowpass(file, cutoff=0.3):
    b, a = signal.butter(3, cutoff, btype='lowpass', analog=False)
    low_passed = signal.filtfilt(b, a, file)
    return low_passed

def baseline_construction_200(ecg_signal, kernel_size=101):
    s_corrected = signal.detrend(ecg_signal)
    baseline_corrected = s_corrected - signal.medfilt(s_corrected, kernel_size)
    return baseline_corrected

def detect_beats(ecg, rate, ransac_window_size=3.35, lowfreq=5.0, highfreq=15.0):
    ransac_window_size = int(ransac_window_size * rate)
    lowpass = signal.butter(1, highfreq / (rate / 2.0), 'low')
    highpass = signal.butter(1, lowfreq / (rate / 2.0), 'high')
    ecg_low = signal.filtfilt(*lowpass, x=ecg)
    ecg_band = signal.filtfilt(*highpass, x=ecg_low)
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
        elif abs(max_amplitude + 0.11) < abs(min_amplitude):  
            rpeak = np.argmin(local_signal) + search_window.start
        else:  
            if max_amplitude >= 0:
                rpeak = np.argmax(local_signal) + search_window.start
            else:
                rpeak = np.argmin(local_signal) + search_window.start
        rpeaks.append(rpeak)
    return np.array(rpeaks)

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
                if i in range(left, right) and abs(ecg_signal[i]) > abs(ecg_signal[maximum_idx]):
                    maximum_idx = i
            selected.append(maximum_idx)
    return sorted(set(selected))

def predict_r_tflite_model(model, input_data):
    with results_lock:
        interpreter, input_details, output_details = model
        input_data = input_data.astype(np.float32)
        interpreter.set_tensor(input_details[0]['index'], input_data)
        interpreter.invoke()
        output_data = interpreter.get_tensor(output_details[0]['index'])
    return output_data.squeeze()

def check_model_r(ecg_data):
    totaldata = len(ecg_data)
    i = 0
    if totaldata < 1000:
        step = totaldata
    else:
        step = 1000
    r_peaks = []
    temp_list = []
    df_ecg_signal = ecg_data.tolist()
    while i < totaldata:
        if i != 0 and totaldata > 1000:
            i = i - 200
        ecg_signal = ecg_data[i:i + step]
        signal_len = len(ecg_signal)
        pad_len = 1000 - signal_len
        padded_signal = np.pad(ecg_signal, (0, pad_len), mode='constant', constant_values=0)
        raw_array = np.expand_dims(padded_signal, axis=0).astype(np.float32)[..., np.newaxis]
        preds = predict_r_tflite_model(r_index_model, raw_array)
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

def check_r_index(all_leads_data, version, frequency, r_index_model):
    if 'II' not in all_leads_data.keys():
        raise ValueError("Lead II is required for R-peak detection")
    ecg_signal = all_leads_data['II'].values
    baseline_signal = baseline_construction_200(ecg_signal, 131)
    lowpass_signal = lowpass(baseline_signal, cutoff=0.3) 
    signal_normalized = find_normalize(lowpass_signal)
    rpeaks = check_model_r(signal_normalized)
    if len(rpeaks) <= 3:
        rpeaks = detect_beats(signal_normalized, frequency).tolist()
    return rpeaks

def find_s_indexs(ecg, R_index, d):
    d = int(d) + 1
    s = []
    for i in R_index:
        if i >= len(ecg):
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
            q_index = i - (len(q_array) - np.nonzero(q_array == min(q_array))[0][0])
        else:
            q_index = i - (len(q_array) - np.nonzero(q_array == max(q_array))[0][0])
        q.append(q_index)
    return q

def check_qs_index(all_leads_data, r_index, version):
    if 'II' not in all_leads_data.keys():
        raise ValueError("Lead II is required for Q and S peak detection")
    ecg_signal = all_leads_data['II'].values
    baseline_signal = baseline_construction_200(ecg_signal, 101)
    lowpass_signal = lowpass(baseline_signal)
    signal_normalized = find_normalize(lowpass_signal)
    s_index_list = find_s_indexs(signal_normalized, r_index, 20)
    q_index_list = find_q_indexs(signal_normalized, r_index, 15)
    return s_index_list, q_index_list

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

    mask = fix_1_2_confusions(mask)
    mask = selective_majority_filter(mask, window_size=16)
    mask = suppress_short_regions(mask, min_length=3)
    t_peaks = get_peak_indices(mask_val=1, ecg=ecg, mask=mask, max_one=True)
    p_peaks = get_peak_indices(mask_val=2, ecg=ecg, mask=mask, max_one=False)
    t_peaks = merge_close_peaks(t_peaks, ecg, merge_distance=merge_distance)
    p_peaks = remove_peaks_near_other(p_peaks, t_peaks, merge_distance=merge_distance)
    p_peaks = merge_close_peaks(p_peaks, ecg, merge_distance=merge_distance)
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
    t_peaks_all, p_peaks_all, pt_peaks_all, T_onset, P_offset = [], [], [], [], []
    for i in range(len(r_indices) - 1):
        segment = ecg[r_indices[i]:r_indices[i+1]]
        if len(segment) < 10:
            continue
        segment_signal = np.array(segment)
        norm_segment_signal = (segment_signal - np.min(segment_signal)) / (np.max(segment_signal) - np.min(segment_signal))
        resampled_ecgs = resample_ecg(norm_segment_signal, 520)
        ecg_signal = np.array(resampled_ecgs)
        ecg_signal = np.expand_dims(ecg_signal, axis=(0, -1))
        predictions = predict_r_tflite_model(pt_index_model, ecg_signal)
        predicted_labels = np.argmax(predictions, axis=-1)
        _, pred_mask = restore_org_ecg_mask(
            ecg_signal[0].squeeze(), predicted_labels.squeeze(), len(segment_signal)
        )
        p_peaks, t_peaks = find_p_t_peaks(segment_signal, pred_mask)
        if t_peaks:
            t_onset, _ = find_onset_offset(segment_signal, t_peaks[0], min_drop_ratio=0.85)
            T_onset.append(t_onset + r_indices[i])
        for ppeak in p_peaks:
            _, p_offset = find_onset_offset(segment_signal, ppeak, min_drop_ratio=0.85)
            P_offset.append(p_offset + r_indices[i])
        p_peaks = np.atleast_1d(p_peaks) + r_indices[i]
        t_peaks = np.atleast_1d(t_peaks) + r_indices[i]
        pt_peaks = tuple(list(t_peaks) + list(p_peaks))
        p_peaks_all.extend(p_peaks)
        t_peaks_all.extend(t_peaks)
        pt_peaks_all.append(pt_peaks)
    return t_peaks_all, p_peaks_all, pt_peaks_all, T_onset, P_offset

def check_pt_index(all_lead_data, version, r_peaks):
    if 'II' not in all_lead_data.keys():
        raise ValueError("Lead II is required for P and T peak detection")
    ecg_signal = all_lead_data['II'].values.flatten()
    baseline_signal = baseline_construction_200(ecg_signal, kernel_size=131)
    lowpass_signal = lowpass(baseline_signal, cutoff=0.3)
    signal_normalized = find_normalize(lowpass_signal)
    t_peaks, p_peaks, rr_invl_peaks, T_onset, P_offset = get_pt_peaks(signal_normalized, r_peaks)
    return t_peaks, p_peaks, rr_invl_peaks, T_onset, P_offset

def save_results_to_csv(results, output_path):
    max_length = max(len(results['r_index']), len(results['s_index']), len(results['q_index']),
                     len(results['t_index']), len(results['p_index']))
    
    def pad_list(lst, length):
        return lst + [None] * (length - len(lst))
    
    data = {
        'R_index': pad_list(results['r_index'], max_length),
        'S_index': pad_list(results['s_index'], max_length),
        'Q_index': pad_list(results['q_index'], max_length),
        'T_index': pad_list(results['t_index'], max_length),
        'P_index': pad_list(results['p_index'], max_length),
    }
    
    df = pd.DataFrame(data)
    df.to_csv(output_path, index=False)

def process_ecg_data(patient_id, is_lead):
    input_path = os.path.join(settings.MEDIA_ROOT, "temp", f"{patient_id}_temp.csv")
    output_path = os.path.join(settings.MEDIA_ROOT, "temp", f"{patient_id}.csv")
    
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Input CSV file not found at {input_path}")

    frequency = 200
    all_results = []
    
    all_lead_data = pd.read_csv(input_path, header=None).fillna(0)
    column_names = all_lead_data.columns.tolist()
    
    if any(str(col).isalpha() for col in all_lead_data.iloc[0, :].values):
        if is_lead == '2_lead':
            all_lead_data = pd.read_csv(input_path, usecols=['ECG']).fillna(0)
            all_lead_data = all_lead_data.rename(columns={'ECG': 'II'})
        elif is_lead == '7_lead':
            all_lead_data = pd.read_csv(input_path, usecols=['I', 'II', 'III', 'aVR', 'aVL', 'aVF', 'v5']).fillna(0)
        elif is_lead == '12_lead':
            all_lead_data = pd.read_csv(input_path, usecols=['I', 'II', 'III', 'aVR', 'aVL', 'aVF', 'v1', 'v2', 'v3', 'v4', 'v5', 'v6']).fillna(0)
    else:
        if is_lead == '2_lead':
            all_lead_data = all_lead_data.rename(columns={0: 'II'})
        elif is_lead == '7_lead':
            all_lead_data = all_lead_data.rename(columns={0: 'I', 1: 'II', 2: 'III', 3: 'aVR', 4: 'aVL', 5: 'aVF', 6: 'v5'})
        elif is_lead == '12_lead':
            all_lead_data = all_lead_data.rename(columns={0: 'I', 1: 'II', 2: 'III', 3: 'aVR', 4: 'aVL', 5: 'aVF', 6: 'v1', 7: 'v2', 8: 'v3', 9: 'v4', 10: 'v5', 11: 'v6'})
    
    i = 0
    if all_lead_data.shape[0] <= 2500:
        steps = all_lead_data.shape[0]
    else:
        steps = round(frequency * 10)

    while i < all_lead_data.shape[0]:
        ecg_data = all_lead_data[i: i + steps]
        if ecg_data.shape[0] < frequency * 2.5:
            print("<<<<<<<<<<<<<<<<<<<<<<<<< Less data for analysis >>>>>>>>>>>>>>>>>>>>>>>>>>>")
            break
        local_name = f"{patient_id}_{i}"
        
        r_index = check_r_index(ecg_data, is_lead, frequency, r_index_model)
        s_index, q_index = check_qs_index(ecg_data, r_index, is_lead)
        t_index, p_index, rr_invl_peaks, T_onset, P_offset = check_pt_index(ecg_data, is_lead, r_peaks=r_index)
        
        all_results.append({
            'slice': local_name,
            'r_index': r_index,
            's_index': s_index,
            'q_index': q_index,
            't_index': t_index,
            'p_index': p_index,
            'T_onset': T_onset,
            'P_offset': P_offset
        })
        
        i += steps

    if all_results:
        combined_results = {
            'r_index': [],
            's_index': [],
            'q_index': [],
            't_index': [],
            'p_index': [],
            'T_onset': [],
            'P_offset': []
        }
        for result in all_results:
            combined_results['r_index'].extend(result['r_index'])
            combined_results['s_index'].extend(result['s_index'])
            combined_results['q_index'].extend(result['q_index'])
            combined_results['t_index'].extend(result['t_index'])
            combined_results['p_index'].extend(result['p_index'])
            combined_results['T_onset'].extend(result['T_onset'])
            combined_results['P_offset'].extend(result['P_offset'])
        
        save_results_to_csv(combined_results, output_path)
        return output_path
    else:
        raise ValueError("No valid ECG data processed.")