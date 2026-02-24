# # # # # import os
# # # # # import glob
# # # # # import pandas as pd
# # # # # import numpy as np
# # # # # import matplotlib
# # # # # matplotlib.use('Agg')
# # # # # import matplotlib.pyplot as plt
# # # # # import tensorflow as tf
# # # # # from scipy import signal
# # # # # from scipy.signal import find_peaks, argrelextrema, savgol_filter
# # # # # from scipy.stats import mode
# # # # # from scipy.interpolate import interp1d
# # # # # import warnings
# # # # # import threading
# # # # # from concurrent.futures import ThreadPoolExecutor, as_completed
# # # # # from PyPDF2 import PdfMerger
# # # # # import random
# # # # # from concurrent.futures import ThreadPoolExecutor, as_completed, ProcessPoolExecutor
# # # # # import multiprocessing
# # # # # from pymongo import MongoClient
# # # # # import gridfs
# # # # # from django.conf import settings


# # # # # warnings.filterwarnings('ignore')
# # # # # results_lock = threading.RLock()

# # # # # # ---------------------- Server-level tuning (CPU / threading / TF / env) ----------------------
# # # # # # Set these before heavy libs use BLAS/OMP/MKL threads
# # # # # DESIRED_CPU_THREADS = 24
# # # # # os.environ['OMP_NUM_THREADS'] = str(DESIRED_CPU_THREADS)
# # # # # os.environ['OPENBLAS_NUM_THREADS'] = str(DESIRED_CPU_THREADS)
# # # # # os.environ['MKL_NUM_THREADS'] = str(DESIRED_CPU_THREADS)
# # # # # os.environ['NUMEXPR_NUM_THREADS'] = str(DESIRED_CPU_THREADS)
# # # # # os.environ['VECLIB_MAXIMUM_THREADS'] = str(DESIRED_CPU_THREADS)
# # # # # # TensorFlow GPU / thread config
# # # # # try:
# # # # #     physical_gpus = tf.config.list_physical_devices('GPU')
# # # # #     if physical_gpus:
# # # # #         # Allow memory growth so multiple processes/threads can share GPU more gracefully
# # # # #         for g in physical_gpus:
# # # # #             try:
# # # # #                 tf.config.experimental.set_memory_growth(g, True)
# # # # #             except Exception:
# # # # #                 pass
# # # # #     # Set TF threading parallelism
# # # # #     tf.config.threading.set_intra_op_parallelism_threads(DESIRED_CPU_THREADS)
# # # # #     tf.config.threading.set_inter_op_parallelism_threads(max(1, DESIRED_CPU_THREADS // 2))
# # # # # except Exception as e:
# # # # #     print("Warning: TensorFlow configuration failed:", e)
# # # # # # Thread-local storage for per-thread interpreters
# # # # # thread_local = threading.local()
# # # # # # ---------------------- TFLite interpreter utilities ----------------------
# # # # # def _load_gpu_delegate():
# # # # #     """
# # # # #     Try loading a TFLite GPU delegate. This is inherently platform-dependent.
# # # # #     We try multiple common delegate names and fall back to None.
# # # # #     """
# # # # #     try:
# # # # #         # TensorFlow's load_delegate helper
# # # # #         load_delegate = tf.lite.experimental.load_delegate
# # # # #     except Exception:
# # # # #         load_delegate = None
# # # # #     if load_delegate:
# # # # #         candidates = [
# # # # #             'libtensorflowlite_gpu_delegate.so', # linux
# # # # #             'libtensorflowlite_gpu_delegate.dylib', # mac
# # # # #             'tensorflowlite_gpu_delegate.dll', # windows (rare)
# # # # #         ]
# # # # #         for cand in candidates:
# # # # #             try:
# # # # #                 delegate = load_delegate(cand)
# # # # #                 print(f"Loaded GPU delegate: {cand}")
# # # # #                 return delegate
# # # # #             except Exception:
# # # # #                 continue
# # # # #     # If we reach here, GPU delegate wasn't loaded
# # # # #     return None
# # # # # GPU_DELEGATE = _load_gpu_delegate()


# # # # # def get_tflite_interpreter_for_thread(model_path: str, use_gpu_delegate=True):
   
# # # # #     if not hasattr(thread_local, "interpreters"):
# # # # #         thread_local.interpreters = {}
# # # # #     key = f"{model_path}_gpu" if use_gpu_delegate and GPU_DELEGATE else model_path
# # # # #     if key in thread_local.interpreters:
# # # # #         return thread_local.interpreters[key]
# # # # #     # Create interpreter
# # # # #     try:
# # # # #         if use_gpu_delegate and GPU_DELEGATE:
# # # # #             interpreter = tf.lite.Interpreter(model_path=model_path, experimental_delegates=[GPU_DELEGATE])
# # # # #             print(f"[Thread {threading.get_ident()}] Created GPU interpreter for {os.path.basename(model_path)}")
# # # # #         else:
# # # # #             interpreter = tf.lite.Interpreter(model_path=model_path)
# # # # #             print(f"[Thread {threading.get_ident()}] Created CPU interpreter for {os.path.basename(model_path)}")
# # # # #     except Exception as e:
# # # # #         # Fallback to CPU interpreter if GPU delegate fails
# # # # #         print(f"Interpreter creation failed for {model_path} with GPU delegate: {e}. Falling back to CPU.")
# # # # #         interpreter = tf.lite.Interpreter(model_path=model_path)
# # # # #     interpreter.allocate_tensors()
# # # # #     input_details = interpreter.get_input_details()
# # # # #     output_details = interpreter.get_output_details()
# # # # #     thread_local.interpreters[key] = (interpreter, input_details, output_details)
# # # # #     return thread_local.interpreters[key]

# # # # # def predict_tflite_model(model_path: str, input_data, use_gpu_delegate=True):
   
# # # # #     # Acquire a lock around interpreter invocation to be safe for device resources, but interpreters are per-thread so contention is low.
# # # # #     interpreter, input_details, output_details = get_tflite_interpreter_for_thread(model_path, use_gpu_delegate=use_gpu_delegate)
# # # # #     with results_lock:
# # # # #         input_data = input_data.astype(np.float32)
# # # # #         interpreter.set_tensor(input_details[0]['index'], input_data)
# # # # #         interpreter.invoke()
# # # # #         output_data = interpreter.get_tensor(output_details[0]['index'])
# # # # #     return output_data.squeeze()
# # # # # # ---------------------- Your existing functions (kept mostly unchanged) ----------------------
# # # # # def lowpass(file, cutoff=0.3):
# # # # #     b, a = signal.butter(3, cutoff, btype='lowpass', analog=False)
# # # # #     low_passed = signal.filtfilt(b, a, file)
# # # # #     return low_passed

# # # # # def baseline_construction_200(ecg_signal, kernel_size=131):
# # # # #     s_corrected = signal.detrend(ecg_signal)
# # # # #     baseline_corrected = s_corrected - signal.medfilt(s_corrected, kernel_size)
# # # # #     return baseline_corrected

# # # # # def normalize(signal):
# # # # #     return (signal - np.mean(signal)) / np.std(signal)

# # # # # def refined_non_max_suppression(ecg_signal, valid_indices, suppression_radius=40):
# # # # #     if len(valid_indices) == 0:
# # # # #         return []
# # # # #     sorted_indices = sorted(valid_indices, reverse=True)
# # # # #     selected = []
# # # # #     occupied = np.zeros(len(ecg_signal), dtype=bool)
# # # # #     for idx in sorted_indices:
# # # # #         if not occupied[idx]:
# # # # #             left = max(0, idx - suppression_radius)
# # # # #             right = min(len(ecg_signal), idx + suppression_radius + 1)
# # # # #             # Mark region as occupied
# # # # #             occupied[left:right] = True
# # # # #             selected.append(idx)
# # # # #     return sorted(selected)




# # # # # def check_model_r(ecg_data, r_model_path, use_gpu_delegate=True):
# # # # #     totaldata = len(ecg_data)
# # # # #     i = 0
# # # # #     step = totaldata if totaldata < 1000 else 1000
# # # # #     r_peaks = []
# # # # #     temp_list = []
# # # # #     df_ecg_signal = ecg_data.tolist()
# # # # #     while i < totaldata:
# # # # #         if i != 0 and totaldata > 1000:
# # # # #             i -= 200
# # # # #         ecg_signal = ecg_data[i:i + step]
# # # # #         signal_len = len(ecg_signal)
# # # # #         pad_len = 1000 - signal_len
# # # # #         padded_signal = np.pad(ecg_signal, (0, pad_len), mode='constant', constant_values=0)
# # # # #         raw_array = np.expand_dims(padded_signal, axis=0).astype(np.float32)[..., np.newaxis]
# # # # #         preds = predict_tflite_model(r_model_path, raw_array, use_gpu_delegate=use_gpu_delegate)
# # # # #         preds = preds[:signal_len]
# # # # #         r_peak_prob = preds[:, 1]
# # # # #         peak_indices, _ = find_peaks(r_peak_prob, height=0.2, distance=20)
# # # # #         for j in peak_indices:
# # # # #             if 0 <= i+j < len(df_ecg_signal):
# # # # #                 temp_list.append(i + j)
# # # # #         i += step
# # # # #     rpeak = sorted(set(temp_list))
# # # # #     r_peaks = refined_non_max_suppression(df_ecg_signal, rpeak)
# # # # #     return r_peaks

# # # # # def r_peak_detection(all_lead_data, is_lead, r_model_path, use_gpu_delegate=True):
# # # # #     r_peaks = []
# # # # #     result_dic = {}
# # # # #     for lead in all_lead_data.keys():
# # # # #         ecg_signal = all_lead_data[lead].values.flatten()
# # # # #         baseline_signal = baseline_construction_200(ecg_signal, kernel_size=131)
# # # # #         lowpass_signal = lowpass(baseline_signal)
# # # # #         signal_normalized = normalize(lowpass_signal)
# # # # #         r_peaks = check_model_r(signal_normalized, r_model_path, use_gpu_delegate=use_gpu_delegate)
# # # # #         result_dic[lead] = r_peaks
# # # # #     if is_lead == '2_lead':
# # # # #         r_peaks = result_dic['II']
# # # # #     return r_peaks, result_dic

# # # # # # --- P/T detection functions (unchanged except calling predict_tflite_model with model path) ---
# # # # # def resample_ecg(ecg_signal, target_length=520):
# # # # #     x_old = np.linspace(0, 1, len(ecg_signal))
# # # # #     x_new = np.linspace(0, 1, target_length)
# # # # #     f_ecg = interp1d(x_old, ecg_signal, kind='linear')
# # # # #     ecg_resampled = f_ecg(x_new)
# # # # #     return ecg_resampled

# # # # # def restore_org_ecg_mask(ecg_signal, mask, target_length=520):
# # # # #     x_old = np.linspace(0, 1, len(ecg_signal))
# # # # #     x_new = np.linspace(0, 1, target_length)
# # # # #     f_ecg = interp1d(x_old, ecg_signal, kind='linear')
# # # # #     ecg_resampled = f_ecg(x_new)
# # # # #     f_mask = interp1d(x_old, mask, kind='nearest')
# # # # #     mask_resampled = f_mask(x_new)
# # # # #     return ecg_resampled, mask_resampled.astype(int)

# # # # # # (find_p_t_peaks remains the same)
# # # # # def find_p_t_peaks(ecg, mask, boundary_margin=3, merge_distance=15):
# # # # #     ecg = np.array(ecg)
# # # # #     mask = np.array(mask)
# # # # #     def fix_1_2_confusions(mask):
# # # # #         mask = mask.copy()
# # # # #         i = 1
# # # # #         while i < len(mask) - 1:
# # # # #             if mask[i] in [1, 2] and mask[i - 1] == mask[i + 1] and mask[i] != mask[i - 1]:
# # # # #                 val_to_fill = mask[i - 1]
# # # # #                 start = i
# # # # #                 while i < len(mask) - 1 and mask[i] != val_to_fill and mask[i] in [1, 2]:
# # # # #                     i += 1
# # # # #                 mask[start:i] = val_to_fill
# # # # #             else:
# # # # #                 i += 1
# # # # #         return mask
    
# # # # #     def selective_majority_filter(mask, window_size=7):
# # # # #         padded = np.pad(mask, (window_size // 2,), mode='edge')
# # # # #         filtered = mask.copy()
# # # # #         for i in range(len(mask)):
# # # # #             window = padded[i:i + window_size]
# # # # #             center = mask[i]
# # # # #             window_mode = mode(window, keepdims=True)[0][0]
# # # # #             if center == 0 and window_mode in [1, 2]:
# # # # #                 filtered[i] = window_mode
# # # # #         return filtered
    
# # # # #     def suppress_short_regions(mask, min_length=2):
# # # # #         mask = mask.copy()
# # # # #         current_val = mask[0]
# # # # #         start_idx = 0
# # # # #         for i in range(1, len(mask)):
# # # # #             if mask[i] != current_val:
# # # # #                 if current_val in [1, 2] and (i - start_idx) < min_length:
# # # # #                     mask[start_idx:i] = 0
# # # # #                 start_idx = i
# # # # #                 current_val = mask[i]
# # # # #         if current_val in [1, 2] and (len(mask) - start_idx) < min_length:
# # # # #             mask[start_idx:] = 0
# # # # #         return mask
    
# # # # #     def get_peak_indices(mask_val, ecg, mask, max_one=False):
# # # # #         indices = []
# # # # #         regions = []
# # # # #         in_region = False
# # # # #         start = 0
# # # # #         for i in range(len(mask)):
# # # # #             if mask[i] == mask_val and not in_region:
# # # # #                 start = i
# # # # #                 in_region = True
# # # # #             elif mask[i] != mask_val and in_region:
# # # # #                 end = i
# # # # #                 regions.append((start, end))
# # # # #                 in_region = False
# # # # #         if in_region:
# # # # #             regions.append((start, len(mask)))
# # # # #         if max_one and regions:
# # # # #             max_len = max(end - start for start, end in regions)
# # # # #             longest_regions = [seg for seg in regions if (seg[1] - seg[0]) == max_len]
# # # # #             if len(longest_regions) > 1:
# # # # #                 abs_vals = [np.max(np.abs(ecg[start:end])) for start, end in longest_regions]
# # # # #                 chosen_region = longest_regions[np.argmax(abs_vals)]
# # # # #             else:
# # # # #                 chosen_region = longest_regions[0]
# # # # #             regions = [chosen_region]
# # # # #         for start, end in regions:
# # # # #             segment = ecg[start:end]
# # # # #             maxima = argrelextrema(segment, np.greater)[0]
# # # # #             inverted = False
# # # # #             if len(maxima) == 0:
# # # # #                 maxima = argrelextrema(-segment, np.greater)[0]
# # # # #                 inverted = True
# # # # #             if len(maxima) > 0:
# # # # #                 candidate_values = segment[maxima] if not inverted else -segment[maxima]
# # # # #                 best_idx = np.argmax(candidate_values)
# # # # #                 peak_relative = maxima[best_idx]
# # # # #             else:
# # # # #                 derivative = np.gradient(segment)
# # # # #                 curvature = np.abs(np.gradient(derivative))
# # # # #                 peak_relative = np.argmax(curvature)
# # # # #             peak_idx = start + peak_relative
# # # # #             if boundary_margin <= peak_idx < len(ecg) - boundary_margin:
# # # # #                 indices.append(peak_idx)
# # # # #         return indices
    
# # # # #     def merge_close_peaks(peaks, ecg, merge_distance):
# # # # #         if not peaks:
# # # # #             return []
# # # # #         peaks = sorted(peaks)
# # # # #         merged_peaks = [peaks[0]]
# # # # #         for idx in peaks[1:]:
# # # # #             last_idx = merged_peaks[-1]
# # # # #             if abs(idx - last_idx) < merge_distance:
# # # # #                 if abs(ecg[idx]) > abs(ecg[last_idx]):
# # # # #                     merged_peaks[-1] = idx
# # # # #             else:
# # # # #                 merged_peaks.append(idx)
# # # # #         return merged_peaks
    
# # # # #     def remove_peaks_near_other(peaks_to_filter, reference_peaks, merge_distance):
# # # # #         filtered = []
# # # # #         for p_idx in peaks_to_filter:
# # # # #             if all(abs(p_idx - t_idx) >= merge_distance for t_idx in reference_peaks):
# # # # #                 filtered.append(p_idx)
# # # # #         return filtered
    
# # # # #     def refine_peak_positions(ecg, peak_indices, window=10):
# # # # #         refined = []
# # # # #         for idx in peak_indices:
# # # # #             temp_seg = ecg[max(idx - 2, 0):min(idx + 2, len(ecg))]
# # # # #             temp_idx = idx - 2 + np.argmax(np.abs(temp_seg))
# # # # #             temp_max = idx - 2 + np.argmax(temp_seg)
# # # # #             temp_min = idx - 2 + np.argmin(temp_seg)
# # # # #             if idx != temp_idx and (idx != temp_max and idx != temp_min):
# # # # #                 start = max(idx - window, 0)
# # # # #                 end = min(idx + window + 1, len(ecg))
# # # # #                 segment = np.abs(ecg[start:end])
# # # # #                 maxima = argrelextrema(segment, np.greater)[0]
# # # # #                 inverted = False
# # # # #                 if len(maxima) == 0:
# # # # #                     maxima = argrelextrema(-segment, np.greater)[0]
# # # # #                     inverted = True
# # # # #                 if len(maxima) > 0:
# # # # #                     candidate_values = segment[maxima] if not inverted else -segment[maxima]
# # # # #                     best_idx = np.argmax(candidate_values)
# # # # #                     peak_relative = maxima[best_idx]
# # # # #                 else:
# # # # #                     derivative = np.gradient(segment)
# # # # #                     curvature = np.abs(np.gradient(derivative))
# # # # #                     peak_relative = np.argmax(curvature)
# # # # #                 peak_idx = start + peak_relative
# # # # #                 refined.append(peak_idx)
# # # # #             else:
# # # # #                 refined.append(idx)
# # # # #         return refined
   
# # # # #     mask = fix_1_2_confusions(mask)
# # # # #     mask = selective_majority_filter(mask, window_size=16)
# # # # #     mask = suppress_short_regions(mask, min_length=3)
# # # # #     t_peaks = get_peak_indices(mask_val=1, ecg=ecg, mask=mask, max_one=True)
# # # # #     t_peaks = refine_peak_positions(ecg, t_peaks, window=10)
# # # # #     t_peaks = merge_close_peaks(t_peaks, ecg, merge_distance=merge_distance)
# # # # #     p_peaks = get_peak_indices(mask_val=2, ecg=ecg, mask=mask, max_one=False)
# # # # #     p_peaks = merge_close_peaks(p_peaks, ecg, merge_distance=45)
# # # # #     p_peaks = refine_peak_positions(ecg, p_peaks, window=10)
# # # # #     p_peaks = remove_peaks_near_other(p_peaks, t_peaks, merge_distance=merge_distance)
# # # # #     return p_peaks, t_peaks

# # # # # def find_onset_offset(signal, peak_idx, smooth=True, window_size=11, polyorder=3,
# # # # #                       min_drop_ratio=0.2, search_window=200):
# # # # #     signal = np.array(signal)
# # # # #     signal_len = len(signal)
# # # # #     if smooth:
# # # # #         win = min(window_size, signal_len - (signal_len % 2 == 0))
# # # # #         signal_smooth = savgol_filter(signal, window_length=win, polyorder=polyorder)
# # # # #     else:
# # # # #         signal_smooth = signal
# # # # #     peak_val = signal_smooth[peak_idx]
# # # # #     baseline_window = min(40, signal_len // 6)
# # # # #     start = max(0, peak_idx - baseline_window)
# # # # #     end = min(signal_len, peak_idx + baseline_window)
# # # # #     local_baseline = np.median(signal_smooth[start:end])
# # # # #     drop_thresh = peak_val - (peak_val - local_baseline) * min_drop_ratio
# # # # #     onset_idx = peak_idx
# # # # #     for i in range(peak_idx, max(1, peak_idx - search_window), -1):
# # # # #         if signal_smooth[i] < drop_thresh:
# # # # #             onset_idx = i
# # # # #             break
# # # # #         if i > 1 and signal_smooth[i-1] < signal_smooth[i-2] and signal_smooth[i-1] < signal_smooth[i]:
# # # # #             onset_idx = i - 1
# # # # #             break
# # # # #     offset_idx = peak_idx
# # # # #     for i in range(peak_idx, min(signal_len - 2, peak_idx + search_window)):
# # # # #         if signal_smooth[i] < drop_thresh:
# # # # #             offset_idx = i
# # # # #             break
# # # # #         if signal_smooth[i+1] < signal_smooth[i] and signal_smooth[i+1] < signal_smooth[i+2]:
# # # # #             offset_idx = i + 1
# # # # #             break
# # # # #     return onset_idx, offset_idx

# # # # # def get_pt_peaks(ecg, r_indices, pt_model_path, use_gpu_delegate=True):
# # # # #     t_peaks_all, p_peaks_all, pt_peaks_all, onset, offset = [], [], [], [], []
# # # # #     for i in range(len(r_indices) - 1):
# # # # #         segment = ecg[r_indices[i]:r_indices[i+1]]
# # # # #         if len(segment) < 10:
# # # # #             continue
# # # # #         segment_signal = np.array(segment)
# # # # #         resampled_ecgs = resample_ecg(segment_signal, 520)
# # # # #         ecg_signal = np.array(resampled_ecgs)
# # # # #         ecg_signal = np.expand_dims(ecg_signal, axis=(0, -1))
# # # # #         predictions = predict_tflite_model(pt_model_path, ecg_signal, use_gpu_delegate=use_gpu_delegate)
# # # # #         predicted_labels = np.argmax(predictions, axis=-1)
# # # # #         _, pred_mask = restore_org_ecg_mask(
# # # # #             ecg_signal[0].squeeze(), predicted_labels.squeeze(), len(segment_signal)
# # # # #         )
# # # # #         p_peaks, t_peaks = find_p_t_peaks(segment_signal, pred_mask)
# # # # #         p_peaks = np.atleast_1d(p_peaks) + r_indices[i]
# # # # #         t_peaks = np.atleast_1d(t_peaks) + r_indices[i]
# # # # #         pt_peaks = tuple(list(t_peaks) + list(p_peaks))
# # # # #         p_peaks_all.extend(p_peaks)
# # # # #         t_peaks_all.extend(t_peaks)
# # # # #         pt_peaks_all.extend(pt_peaks)
# # # # #     return t_peaks_all, p_peaks_all, pt_peaks_all


# # # # # def pt_peak_detection(all_lead_data, is_lead, r_peaks, r_result_dic = None, pt_model_path=None, use_gpu_delegate=True):
# # # # #     result_dic = {}
# # # # #     for lead in all_lead_data.keys():
# # # # #         r_peaks = r_result_dic.get(lead)
# # # # #         ecg_signal = all_lead_data[lead].values.flatten()
# # # # #         baseline_signal = baseline_construction_200(ecg_signal, kernel_size=131)
# # # # #         lowpass_signal = lowpass(baseline_signal)
# # # # #         signal_normalized = normalize(lowpass_signal)
# # # # #         t_peaks, p_peaks, rr_invl_peaks = get_pt_peaks(signal_normalized, r_peaks, pt_model_path, use_gpu_delegate=use_gpu_delegate)
# # # # #         result_dic[lead] = {"p": p_peaks, "t": t_peaks, "comb": rr_invl_peaks}
# # # # #     if is_lead == '2_lead':
# # # # #         p_peaks = result_dic['II'].get("p")
# # # # #         t_peaks = result_dic['II'].get("t")
# # # # #         rr_invl_peaks = result_dic['II'].get("comb")

# # # # #     return t_peaks, p_peaks, rr_invl_peaks, result_dic
# # # # # # ---------------------- Plotting and other post processing functions ----------------------
# # # # # def add_standard_ecg_grid(ax, duration_sec, y_min, y_max):

# # # # #     ax.set_xlim(0, duration_sec)
# # # # #     ax.set_xticks(np.arange(0, duration_sec + 0.2, 0.2))
# # # # #     ax.set_xticks(np.arange(0, duration_sec + 0.04, 0.04), minor=True)
 
# # # # #     # Y axis (amplitude)
# # # # #     ax.set_ylim(y_min, y_max)
# # # # #     ax.set_yticks(np.arange(y_min, y_max + 0.5, 0.5))
# # # # #     ax.set_yticks(np.arange(y_min, y_max + 0.1, 0.1), minor=True)
 
# # # # #     # CRITICAL: lock aspect so squares stay squares
# # # # #     ax.set_aspect((0.04 / 0.1), adjustable="box")
 
# # # # #     # Grid style
# # # # #     ax.grid(which='major', color='#b00000', linewidth=1.1)
# # # # #     ax.grid(which='minor', color='#e6e6e6', linewidth=0.6)
 
# # # # #     ax.set_facecolor("white")

# # # # # # ---------------------- Plotting and other post processing functions ----------------------
# # # # # def hr_count(r_index, fs=250):
# # # # #     if len(r_index) < 2:
# # # # #         return 0
# # # # #     rr_intervals = np.diff(r_index)
# # # # #     if len(rr_intervals) == 0:
# # # # #         return 0
# # # # #     HR = int((len(rr_intervals) * 60000) / np.sum(rr_intervals / fs * 1000))
# # # # #     return HR

# # # # # def refine_r_peaks(signal, r_peaks, fs):

# # # # #     refined = []
# # # # #     polarity = []

# # # # #     window = int(0.04 * fs)  # ±40 ms around detected R

# # # # #     for r in r_peaks:
# # # # #         if r <= 0 or r >= len(signal):
# # # # #             continue

# # # # #         start = max(0, r - window)
# # # # #         end = min(len(signal), r + window + 1)
# # # # #         seg = signal[start:end]

# # # # #         if len(seg) == 0:
# # # # #             continue

# # # # #         # pick the true apex of QRS (largest absolute deflection)
# # # # #         idx = np.argmax(np.abs(seg))
# # # # #         true_r = start + idx

# # # # #         refined.append(true_r)
# # # # #         polarity.append(np.sign(signal[true_r]) or 1)

# # # # #     return np.array(refined, dtype=int), np.array(polarity, dtype=int)


# # # # # def detect_q_s(signal, r_peaks, r_polarity, fs):
    
# # # # #     q_points = []
# # # # #     s_points = []
# # # # #     window = int(0.1 * fs)  # 100 ms search window

# # # # #     for r, pol in zip(r_peaks, r_polarity):
# # # # #         # Q point: before R
# # # # #         start = max(0, r - window)
# # # # #         seg = signal[start:r+1]
# # # # #         if pol > 0:
# # # # #             q_idx = start + np.argmin(seg)  # Q = min before upright R
# # # # #         else:
# # # # #             q_idx = start + np.argmax(seg)  # Q = max before inverted R
# # # # #         q_points.append(q_idx)

# # # # #         # S point: after R
# # # # #         end = min(len(signal), r + window + 1)
# # # # #         seg = signal[r:end]
# # # # #         if pol > 0:
# # # # #             s_idx = r + np.argmin(seg)  # S = min after upright R
# # # # #         else:
# # # # #             s_idx = r + np.argmax(seg)  # S = max after inverted R
# # # # #         s_points.append(s_idx)

# # # # #     return np.array(q_points, dtype=int), np.array(s_points, dtype=int)



# # # # # def detect_t_wave_onset(signal, s_points, fs):
# # # # #     """
# # # # #     Detect T-wave onset AFTER ST segment.
# # # # #     NOT used for ST measurement.
# # # # #     """

# # # # #     signal = np.asarray(signal)
# # # # #     t_onsets = []

# # # # #     for s_idx in s_points:
# # # # #         j_point = int(s_idx)

# # # # #         search_start = j_point + int(0.08 * fs)
# # # # #         search_end   = min(j_point + int(0.30 * fs), len(signal) - 2)

# # # # #         if search_end <= search_start:
# # # # #             t_onsets.append(search_start)
# # # # #             continue

# # # # #         slope = np.diff(signal)
# # # # #         onset = search_end

# # # # #         for i in range(search_start, search_end - 1):
# # # # #             if slope[i] > 0 and slope[i+1] > 0:
# # # # #                 onset = i
# # # # #                 break

# # # # #         t_onsets.append(onset)

# # # # #     return np.array(t_onsets, dtype=int)

# # # # # def measure_st_segment_full(signal, r_peaks, s_points, t_onsets, fs, calibration=10.0):
# # # # #     """
# # # # #     Measure ST segment from J-point → T-onset.
# # # # #     Keeps ALL beats (no rejection).
# # # # #     Returns tuples:
# # # # #     (j_point, st_end, st_mV, pr_baseline)
# # # # #     """
# # # # #     import numpy as np

# # # # #     st_results = []
# # # # #     signal_len = len(signal)
# # # # #     beats = min(len(r_peaks), len(s_points), len(t_onsets))

# # # # #     for i in range(beats):
# # # # #         r = int(r_peaks[i])
# # # # #         s = int(s_points[i])
# # # # #         t_idx = int(t_onsets[i])

# # # # #         if r <= 0 or s <= 0 or r >= signal_len or s >= signal_len:
# # # # #             continue

# # # # #         # J-point ≈ 10 ms after S
# # # # #         j_point = s + int(0.06 * fs)
# # # # #         if j_point >= signal_len:
# # # # #             continue

# # # # #         # PR baseline (−200 to −120 ms before R)
# # # # #         pr_start = max(r - int(0.20 * fs), 0)
# # # # #         pr_end   = max(r - int(0.12 * fs), pr_start + 1)
# # # # #         pr_baseline = np.median(signal[pr_start:pr_end])

# # # # #         st_end = min(t_idx, signal_len - 1)
# # # # #         if st_end <= j_point:
# # # # #             continue

# # # # #         st_mV = np.median(signal[j_point:st_end + 1]) - pr_baseline

# # # # #         st_results.append((j_point, st_end, st_mV, pr_baseline))

# # # # #     return st_results



# # # # # def plotting(
# # # # #     all_leads_data,
# # # # #     save_path,
# # # # #     local_name,
# # # # #     pt_result_dic,
# # # # #     r_result_dic,
# # # # #     fs=250,
# # # # #     calibration=10.0,
# # # # #     show_plots_for_first_n=5
# # # # # ):
# # # # #     import os
# # # # #     import numpy as np
# # # # #     import pandas as pd
# # # # #     import matplotlib.pyplot as plt
# # # # #     import matplotlib.patheffects as path_effects

# # # # #     os.makedirs(save_path, exist_ok=True)

# # # # #     if not hasattr(plotting, "_counter"):
# # # # #         plotting._counter = 0

# # # # #     show_plots = plotting._counter < show_plots_for_first_n
# # # # #     summary_records = []
# # # # #     mm_per_mV = calibration  # 10 mm / mV

# # # # #     for lead in all_leads_data.columns:

# # # # #         fig, ax = plt.subplots(figsize=(20, 8))

# # # # #         # ---------------- SIGNAL ----------------
# # # # #         signal = all_leads_data[lead].values.astype(float)
# # # # #         signal_len = len(signal)
# # # # #         time_axis = np.arange(signal_len) / fs

# # # # #         # ---------------- FIX Y LIMITS FIRST ----------------
# # # # #         y_min = np.floor(signal.min() * mm_per_mV) / mm_per_mV
# # # # #         y_max = np.ceil(signal.max() * mm_per_mV) / mm_per_mV

# # # # #         # ---------------- ECG GRID (MUST COME FIRST) ----------------
# # # # #         add_standard_ecg_grid(
# # # # #             ax=ax,
# # # # #             duration_sec=time_axis[-1],
# # # # #             y_min=y_min,
# # # # #             y_max=y_max
# # # # #         )

# # # # #         # ---------------- DRAW SIGNAL ----------------
# # # # #         ax.plot(time_axis, signal, color="black", linewidth=1.2, zorder=3)

# # # # #         # ---------------- R PEAKS ----------------
# # # # #         r_peaks = np.asarray(r_result_dic.get(lead, []), dtype=int)
# # # # #         if len(r_peaks) == 0:
# # # # #             plt.close(fig)
# # # # #             continue

# # # # #         r_peaks, r_polarity = refine_r_peaks(signal, r_peaks, fs)

# # # # #         # ---------------- Q & S ----------------
# # # # #         q_points, s_points = detect_q_s(signal, r_peaks, r_polarity, fs)

# # # # #         # ---------------- T ONSET ----------------
# # # # #         t_onsets = detect_t_wave_onset(signal, s_points, fs)
# # # # #         t_onsets = np.clip(t_onsets, 0, signal_len - 1)

# # # # #         # ---------------- MARKERS ----------------
# # # # #         ax.plot(time_axis[r_peaks], signal[r_peaks], "ro", ms=4, label="R", zorder=4)
# # # # #         ax.plot(time_axis[q_points], signal[q_points], "o", color="purple", ms=4, label="Q", zorder=4)
# # # # #         ax.plot(time_axis[s_points], signal[s_points], "o", color="orange", ms=4, label="S", zorder=4)
# # # # #         ax.plot(time_axis[t_onsets], signal[t_onsets], "o", color="green", ms=5, label="T-onset", zorder=4)

# # # # #         # ---------------- ST MEASUREMENT ----------------
# # # # #         st_records = measure_st_segment_full(
# # # # #             signal, r_peaks, s_points, t_onsets, fs, calibration
# # # # #         )

# # # # #         if len(st_records) == 0:
# # # # #             plt.close(fig)
# # # # #             continue

# # # # #         # ---------------- PR BASELINE ----------------
# # # # #         pr_baseline = np.median([rec[3] for rec in st_records])
# # # # #         ax.hlines(
# # # # #             pr_baseline,
# # # # #             time_axis[0],
# # # # #             time_axis[-1],
# # # # #             colors="gray",
# # # # #             linestyles="--",
# # # # #             linewidth=1.2,
# # # # #             label="PR baseline",
# # # # #             zorder=2
# # # # #         )

# # # # #         # ---------------- ST SHADING ----------------
# # # # #         for idx, (j_point, st_end, st_mV, _) in enumerate(st_records, start=1):

# # # # #             st_mm = st_mV * mm_per_mV
# # # # #             if abs(st_mm) < 0.5:
# # # # #                 continue

# # # # #             color = "green" if abs(st_mm) < 2.0 else "red"
# # # # #             status = "Normal" if abs(st_mm) < 2.0 else "Abnormal"

# # # # #             start_idx = max(0, min(j_point, signal_len - 1))
# # # # #             end_idx   = max(start_idx + 1, min(st_end, signal_len - 1))

# # # # #             segment_x = time_axis[start_idx:end_idx + 1]
# # # # #             segment_y = signal[start_idx:end_idx + 1]

# # # # #             lower = np.minimum(segment_y, pr_baseline)
# # # # #             upper = np.maximum(segment_y, pr_baseline)

# # # # #             ax.fill_between(
# # # # #                 segment_x,
# # # # #                 lower,
# # # # #                 upper,
# # # # #                 where=(upper != lower),
# # # # #                 color=color,
# # # # #                 alpha=0.35,
# # # # #                 zorder=1
# # # # #             )

# # # # #             label_y = upper.max() + 0.03 if st_mV >= 0 else lower.min() - 0.03

# # # # #             ax.text(
# # # # #                 segment_x[-1],
# # # # #                 label_y,
# # # # #                 f"{st_mm:+.1f} mm",
# # # # #                 ha="center",
# # # # #                 va="bottom" if st_mV >= 0 else "top",
# # # # #                 fontsize=8,
# # # # #                 fontweight="bold",
# # # # #                 zorder=5,
# # # # #                 path_effects=[path_effects.withStroke(linewidth=2, foreground="white")]
# # # # #             )

# # # # #             summary_records.append({
# # # # #                 "Lead": lead,
# # # # #                 "Beat_No": idx,
# # # # #                 "ST_mm": round(st_mm, 2),
# # # # #                 "Isoelectric_mV": round(pr_baseline, 4),
# # # # #                 "Status": status
# # # # #             })

# # # # #         # ---------------- FINAL DECORATION ----------------
# # # # #         hr = hr_count(r_peaks, fs)
# # # # #         ax.set_title(f"{local_name} — Lead {lead} — HR {hr} bpm", fontsize=14)
# # # # #         ax.set_xlabel("Time (s)")
# # # # #         ax.set_ylabel("Amplitude (mV)")
# # # # #         ax.legend(loc="upper right")

# # # # #         # ❗ DO NOT USE bbox_inches="tight"
# # # # #         plt.subplots_adjust(left=0.06, right=0.99, top=0.92, bottom=0.10)

# # # # #         pdf_path = os.path.join(save_path, f"{local_name}_{lead}.pdf")
# # # # #         plt.savefig(pdf_path, dpi=300)

# # # # #         if show_plots:
# # # # #             plt.show()

# # # # #         plt.close(fig)

# # # # #     plotting._counter += 1
# # # # #     return pd.DataFrame(summary_records)
 


# # # # # # ---------------------- I/O / file handling helpers ----------------------
# # # # # def load_and_rename_data(fn, is_lead_for):
# # # # #     lead_columns = {
# # # # #         '2_lead': ['ECG', 'II', 'Value',"'MLII'",'MLII'],
# # # # #         '7_lead': ['I', 'II', 'III', 'aVR', 'aVL', 'aVF', 'v5'],
# # # # #         '12_lead': ['I', 'II', 'III', 'aVR', 'aVL', 'aVF', 'v1', 'v2', 'v3', 'v4', 'v5', 'v6','V1','V2','V3','V4','V5','V6','ECG']
# # # # #     }

# # # # #     lead_columns_index = {
# # # # #         '2_lead': {0: 'II'},
# # # # #         '7_lead': {0: 'I', 1: 'II', 2: 'III', 3: 'aVR', 4: 'aVL', 5: 'aVF', 6: 'v5'},
# # # # #         '12_lead': {0: 'I', 1: 'II', 2: 'III', 3: 'aVR', 4: 'aVL', 5: 'aVF', 6: 'v1', 7: 'v2', 8: 'v3', 9: 'v4', 10: 'v5', 11: 'v6'}
# # # # #     }

# # # # #     all_lead_data = pd.read_csv(fn).fillna(0)
# # # # #     columns = all_lead_data.columns.tolist()
# # # # #     if any(str(val).isalpha() for val in all_lead_data.iloc[0, :].values):
# # # # #         if all(col in lead_columns['7_lead'] for col in columns):
# # # # #             is_lead_for = '7_lead'
# # # # #         elif all(col in lead_columns['12_lead'] for col in columns):
# # # # #             is_lead_for = '12_lead'
# # # # #         else:
# # # # #             is_lead_for = '2_lead'
# # # # #     else:
# # # # #         if len(columns) >= 12:
# # # # #             is_lead_for = '12_lead'
# # # # #         elif len(columns) >= 7:
# # # # #             is_lead_for = '7_lead'
# # # # #         else:
# # # # #             is_lead_for = '2_lead'

# # # # #     if is_lead_for == '2_lead':
# # # # #         available_columns = [col for col in lead_columns['2_lead'] if col in columns]
# # # # #         all_lead_data = attempt_column_load(fn, available_columns)
# # # # #     elif is_lead_for == '7_lead':
# # # # #         available_columns = [col for col in lead_columns['7_lead'] if col in columns]
# # # # #         all_lead_data = attempt_column_load(fn, available_columns)
# # # # #     elif is_lead_for == '12_lead':
# # # # #         available_columns = [col for col in lead_columns['12_lead'] if col in columns]
# # # # #         all_lead_data = attempt_column_load(fn, available_columns)

# # # # #     if all_lead_data is not None:
# # # # #         all_lead_data = all_lead_data.rename(columns=lead_columns_index[is_lead_for])

# # # # #     if is_lead_for == '2_lead':
# # # # #         all_lead_data.columns = ['II']

# # # # #     return all_lead_data, is_lead_for

# # # # # def attempt_column_load(fn, columns):
# # # # #     try:
# # # # #         data = pd.read_csv(fn, usecols=columns).fillna(0)
# # # # #         return data
# # # # #     except ValueError as e:
# # # # #         print("value Error ",e)
# # # # #         return None
# # # # #     except Exception as e:
# # # # #         print("Error in Loading",e)
# # # # #         return None

# # # # # def find_csv_files(root_folder):
# # # # #     csv_files = []
# # # # #     for root, _, files in os.walk(root_folder):
# # # # #         for file in files:
# # # # #             if file.lower().endswith('.csv'):
# # # # #                 csv_files.append(os.path.join(root, file))
# # # # #     return csv_files


# # # # # def process_single_file(
# # # # #     fn,
# # # # #     save_path,
# # # # #     is_lead,
# # # # #     r_model_path,
# # # # #     pt_model_path,
# # # # #     use_gpu_delegate=True
# # # # # ):
# # # # #     import os
# # # # #     import pandas as pd

# # # # #     local_name = os.path.splitext(os.path.basename(fn))[0]
# # # # #     csv_root = os.path.join(save_path, local_name)
# # # # #     os.makedirs(csv_root, exist_ok=True)

# # # # #     all_leads_data, is_lead = load_and_rename_data(fn, is_lead)
# # # # #     if all_leads_data is None:
# # # # #         return f"Failed: {local_name}"

# # # # #     # ---------- R PEAK DETECTION ----------
# # # # #     _, r_result_dic = r_peak_detection(
# # # # #         all_leads_data,
# # # # #         is_lead,
# # # # #         r_model_path,
# # # # #         use_gpu_delegate
# # # # #     )

# # # # #     # ---------- P & T PEAK DETECTION ----------
# # # # #     _, _, _, pt_result_dic = pt_peak_detection(
# # # # #         all_leads_data,
# # # # #         is_lead,
# # # # #         None,
# # # # #         r_result_dic,
# # # # #         pt_model_path,
# # # # #         use_gpu_delegate
# # # # #     )

# # # # #     # ---------- BASELINE CORRECTED ECG (ONCE) ----------
# # # # #     baseline_corrected_data = pd.DataFrame({
# # # # #         lead: baseline_construction_200(all_leads_data[lead].values)
# # # # #         for lead in all_leads_data.columns
# # # # #     })

# # # # #     summary_df = plotting(
# # # # #         baseline_corrected_data,
# # # # #         csv_root,
# # # # #         local_name,
# # # # #         pt_result_dic,
# # # # #         r_result_dic,
# # # # #         fs=250,
# # # # #         calibration=10.0
# # # # #     )

# # # # #     if not summary_df.empty:
# # # # #         summary_df.to_csv(
# # # # #             os.path.join(csv_root, f"{local_name}_ALL_LEADS_ST_SUMMARY.csv"),
# # # # #             index=False
# # # # #         )

# # # # #     merge_pdfs_in_lead_order(
# # # # #         csv_root,
# # # # #         os.path.join(csv_root, f"{local_name}_MERGED_ALL_LEADS.pdf")
# # # # #     )

# # # # #     return f"Processed: {local_name}"



# # # # # def merge_pdfs_in_lead_order(pdf_dir, output_pdf):
# # # # #     from PyPDF2 import PdfMerger
# # # # #     import os

# # # # #     LEAD_ORDER = [
# # # # #         "I", "II", "III",
# # # # #         "aVR", "aVL", "aVF",
# # # # #         "V1", "V2", "V3", "V4", "V5", "V6"
# # # # #     ]

# # # # #     merger = PdfMerger()

# # # # #     for lead in LEAD_ORDER:
# # # # #         for file in sorted(os.listdir(pdf_dir)):
# # # # #             if file.endswith(".pdf") and f"_{lead}.pdf" in file:
# # # # #                 merger.append(os.path.join(pdf_dir, file))
# # # # #                 break

# # # # #     merger.write(output_pdf)
# # # # #     merger.close()
# # # # # def merge_all_merged_pdfs(root_output_dir, final_pdf_path):
# # # # #     """
# # # # #     Merge all per-CSV merged ECG PDFs into one final PDF.

# # # # #     Expects files like:
# # # # #     root_output_dir/
# # # # #         patient1/patient1_MERGED_ALL_LEADS.pdf
# # # # #         patient2/patient2_MERGED_ALL_LEADS.pdf
# # # # #         ...

# # # # #     Creates:
# # # # #         final_pdf_path
# # # # #     """
# # # # #     import os
# # # # #     from PyPDF2 import PdfMerger

# # # # #     merger = PdfMerger()

# # # # #     found_any = False

# # # # #     for root, _, files in os.walk(root_output_dir):
# # # # #         for file in sorted(files):
# # # # #             if file.endswith("_MERGED_ALL_LEADS.pdf"):
# # # # #                 pdf_path = os.path.join(root, file)
# # # # #                 merger.append(pdf_path)
# # # # #                 found_any = True

# # # # #     if found_any:
# # # # #         merger.write(final_pdf_path)
# # # # #         merger.close()
# # # # #         print(f"✅ FINAL MERGED PDF CREATED: {final_pdf_path}")
# # # # #     else:
# # # # #         merger.close()
# # # # #         print("⚠️ No merged PDFs found to combine.")


# # # # # def ecg_processing(path, save_path, is_lead, r_model_path, pt_model_path, max_workers=DESIRED_CPU_THREADS, use_gpu_delegate=True):
# # # # #     csv_files = find_csv_files(path)
# # # # #     if not csv_files:
# # # # #         print("No CSV files found.")
# # # # #         return
# # # # #     # Use ThreadPoolExecutor so per-thread interpreters remain in same process/threads (GPU delegate may not be picklable)
# # # # #     max_workers = min(max_workers, max(1, os.cpu_count() or 1))
# # # # #     print(f"Processing {len(csv_files)} files with {max_workers} workers (use_gpu_delegate={use_gpu_delegate})")
# # # # #     with ThreadPoolExecutor(max_workers=max_workers) as executor:
# # # # #         futures = {executor.submit(process_single_file, fn, save_path, is_lead, r_model_path, pt_model_path, use_gpu_delegate): fn for fn in csv_files}
# # # # #         for future in as_completed(futures):
# # # # #             try:
# # # # #                 print(future.result())
# # # # #             except Exception as e:
# # # # #                 print("Worker exception:", e)
# # # # #     print("All files processed successfully.")
# # # # # def find_pdf_files(root_folder):
# # # # #     pdf_files = []
# # # # #     for root, _, files in os.walk(root_folder):
# # # # #         for file in files:
# # # # #             if file.lower().endswith('.pdf'):
# # # # #                 pdf_files.append(os.path.join(root, file))
# # # # #     return sorted(pdf_files)
# # # # # def merge_pdfs(pdf_files, output_path):
# # # # #     merger = PdfMerger()
# # # # #     for pdf in pdf_files:
# # # # #         try:
# # # # #             merger.append(pdf)
# # # # #             print(f"Merged: {pdf}")
# # # # #         except Exception as e:
# # # # #             print(f"Failed to merge {pdf}: {e}")
# # # # #     merger.write(output_path)
# # # # #     merger.close()
# # # # #     print(f"\n✅ All PDFs merged into: {output_path}")
# # # # # def process_single_file_wrapper(args):
# # # # #     """Wrapper for multiprocessing (since functions must be pickleable)."""
# # # # #     return process_single_file(*args)
# # # # # def ecg_processing(path, save_path, is_lead, r_model_path, pt_model_path,
# # # # #                    max_workers=DESIRED_CPU_THREADS, use_gpu_delegate=True,
# # # # #                    use_multiprocessing=False):
# # # # #     csv_files = find_csv_files(path)
# # # # #     if not csv_files:
# # # # #         print("No CSV files found.")
# # # # #         return
# # # # #     max_workers = min(max_workers, max(1, os.cpu_count() or 1))
# # # # #     mode = "multiprocessing" if use_multiprocessing else "threading"
# # # # #     print(f"Processing {len(csv_files)} files with {max_workers} workers ({mode}, use_gpu_delegate={use_gpu_delegate})")
# # # # #     if use_multiprocessing:
# # # # #         # Each process loads its own interpreters/models
# # # # #         with ProcessPoolExecutor(max_workers=max_workers) as executor:
# # # # #             futures = {
# # # # #                 executor.submit(process_single_file_wrapper,
# # # # #                                 (fn, save_path, is_lead, r_model_path, pt_model_path, use_gpu_delegate)): fn
# # # # #                 for fn in csv_files
# # # # #             }
# # # # #             for future in as_completed(futures):
# # # # #                 try:
# # # # #                     print(future.result())
# # # # #                 except Exception as e:
# # # # #                     print("Worker exception:", e)
# # # # #     else:
# # # # #         # Default threading (per-thread cached interpreters, GPU delegate works better here)
# # # # #         with ThreadPoolExecutor(max_workers=max_workers) as executor:
# # # # #             futures = {
# # # # #                 executor.submit(process_single_file, fn, save_path, is_lead, r_model_path, pt_model_path, use_gpu_delegate): fn
# # # # #                 for fn in csv_files
# # # # #             }
# # # # #             for future in as_completed(futures):
# # # # #                 try:
# # # # #                     print(future.result())
# # # # #                 except Exception as e:
# # # # #                     print("Worker exception:", e)
# # # # #     print("All files processed successfully.")

# # # # # def save_pdf_to_gridfs(pdf_path, metadata=None):
# # # # #     mongo_uri = os.getenv("MONGO_HOST")
# # # # #     mongo_client = MongoClient(mongo_uri)
# # # # #     db = mongo_client["St_Segment"]

# # # # #     fs = gridfs.GridFS(db)

# # # # #     with open(pdf_path, "rb") as f:
# # # # #         file_id = fs.put(
# # # # #             f,
# # # # #             filename=pdf_path.split("\\")[-1],
# # # # #             contentType="application/pdf",
# # # # #             metadata=metadata or {}
# # # # #         )

# # # # #     return str(file_id)

# # # # # def run_ecg_st_pipeline(
# # # # #     input_folder,
# # # # #     output_folder,
# # # # #     is_lead,
# # # # #     max_workers=4,
# # # # #     use_gpu_delegate=True,
# # # # #     use_multiprocessing=False
# # # # # ):
# # # # #     os.makedirs(output_folder, exist_ok=True)

# # # # #     ecg_processing(
# # # # #         path=input_folder,
# # # # #         save_path=output_folder,
# # # # #         is_lead=is_lead,
# # # # #         r_model_path = r"D:\\try3\\Scripts_Models\\Model\\rnn_model1_19_12_Unet.tflite",
# # # # #         pt_model_path = r"D:\\try3\\Scripts_Models\\Model\\ecg_pt_detection_LSTMGRU_TCN_Transpose_v27.tflite",
# # # # #         max_workers=max_workers,
# # # # #         use_gpu_delegate=use_gpu_delegate,
# # # # #         use_multiprocessing=use_multiprocessing
# # # # #     )

# # # # #     final_pdf_path = os.path.join(
# # # # #         output_folder,
# # # # #         "FINAL_ALL_CSV_ALL_LEADS.pdf"
# # # # #     )

# # # # #     merge_all_merged_pdfs(
# # # # #         root_output_dir=output_folder,
# # # # #         final_pdf_path=final_pdf_path
# # # # #     )

# # # # #     return final_pdf_path
# # # # # # ---------------------- Main entry ----------------------
# # # # # # if __name__ == "__main__":

# # # # # #     import os

# # # # # #     # -------------------- INPUT / OUTPUT PATHS --------------------
# # # # # #     path = r"C:\Users\Admin\Downloads\data set for st segment\7_lead\7L_stdep"
# # # # # #     save_path = r"C:\Users\Admin\Downloads\processed_results_13-01\7_lead_new"
# # # # # #     os.makedirs(save_path, exist_ok=True)

# # # # # #     # -------------------- MODEL FILE PATHS --------------------
# # # # # #     r_index_model_path = r"D:\\try\\Scripts_Models\\Model\\rnn_model1_29_10_Unet.tflite"
# # # # # #     pt_index_model_path = r"D:\\try\\Scripts_Models\\Model\\ecg_pt_detection_LSTMGRU_TCN_Transpose_v27.tflite"

# # # # # #     # -------------------- CONFIG --------------------
# # # # # #     is_lead = "7_lead"
# # # # # #     max_workers = 4

# # # # # #     # GPU delegate fallback: will use CPU if GPU is not available
# # # # # #     use_gpu_delegate = True if 'GPU_DELEGATE' in globals() and GPU_DELEGATE else False
# # # # # #     use_multiprocessing = True

# # # # # #     # -------------------- ECG PROCESSING --------------------
# # # # # #     ecg_processing(
# # # # # #         path=path,
# # # # # #         save_path=save_path,
# # # # # #         is_lead=is_lead,
# # # # # #         r_model_path=r_index_model_path,
# # # # # #         pt_model_path=pt_index_model_path,
# # # # # #         max_workers=max_workers,
# # # # # #         use_gpu_delegate=use_gpu_delegate,
# # # # # #         use_multiprocessing=use_multiprocessing
# # # # # #     )

# # # # # #     # -------------------- OPTIONAL: MERGE ALL PDFs --------------------
# # # # # #     final_pdf_path = os.path.join(save_path, "FINAL_ALL_CSV_ALL_LEADS.pdf")
# # # # # #     merge_all_merged_pdfs(
# # # # # #         root_output_dir=save_path,
# # # # # #         final_pdf_path=final_pdf_path
# # # # # #     )

# # # # # #     print("\n✅ ECG 7-Lead ST Segment Processing Completed Successfully")

# # # # # ---------------------------------------new version---------------------

# # # # import os
# # # # import glob
# # # # import pandas as pd
# # # # import numpy as np
# # # # import matplotlib
# # # # matplotlib.use('Agg')
# # # # import matplotlib.pyplot as plt
# # # # import tensorflow as tf
# # # # from scipy import signal
# # # # from scipy.signal import find_peaks, argrelextrema, savgol_filter
# # # # from scipy.stats import mode
# # # # from scipy.interpolate import interp1d
# # # # import warnings
# # # # import threading
# # # # from concurrent.futures import ThreadPoolExecutor, as_completed
# # # # from PyPDF2 import PdfMerger
# # # # import random
# # # # from concurrent.futures import ThreadPoolExecutor, as_completed, ProcessPoolExecutor
# # # # import multiprocessing
# # # # from pymongo import MongoClient
# # # # import gridfs
# # # # from django.conf import settings
# # # # warnings.filterwarnings('ignore')
# # # # results_lock = threading.RLock()
# # # # # ---------------------- Server-level tuning (CPU / threading / TF / env) ----------------------
# # # # # Set these before heavy libs use BLAS/OMP/MKL threads
# # # # DESIRED_CPU_THREADS = 24
# # # # os.environ['OMP_NUM_THREADS'] = str(DESIRED_CPU_THREADS)
# # # # os.environ['OPENBLAS_NUM_THREADS'] = str(DESIRED_CPU_THREADS)
# # # # os.environ['MKL_NUM_THREADS'] = str(DESIRED_CPU_THREADS)
# # # # os.environ['NUMEXPR_NUM_THREADS'] = str(DESIRED_CPU_THREADS)
# # # # os.environ['VECLIB_MAXIMUM_THREADS'] = str(DESIRED_CPU_THREADS)
# # # # # TensorFlow GPU / thread config
# # # # try:
# # # #     physical_gpus = tf.config.list_physical_devices('GPU')
# # # #     if physical_gpus:
# # # #         # Allow memory growth so multiple processes/threads can share GPU more gracefully
# # # #         for g in physical_gpus:
# # # #             try:
# # # #                 tf.config.experimental.set_memory_growth(g, True)
# # # #             except Exception:
# # # #                 pass
# # # #     # Set TF threading parallelism
# # # #     tf.config.threading.set_intra_op_parallelism_threads(DESIRED_CPU_THREADS)
# # # #     tf.config.threading.set_inter_op_parallelism_threads(max(1, DESIRED_CPU_THREADS // 2))
# # # # except Exception as e:
# # # #     print("Warning: TensorFlow configuration failed:", e)
# # # # # Thread-local storage for per-thread interpreters
# # # # thread_local = threading.local()
# # # # # ---------------------- TFLite interpreter utilities ----------------------
# # # # def _load_gpu_delegate():
# # # #     """
# # # #     Try loading a TFLite GPU delegate. This is inherently platform-dependent.
# # # #     We try multiple common delegate names and fall back to None.
# # # #     """
# # # #     try:
# # # #         # TensorFlow's load_delegate helper
# # # #         load_delegate = tf.lite.experimental.load_delegate
# # # #     except Exception:
# # # #         load_delegate = None
# # # #     if load_delegate:
# # # #         candidates = [
# # # #             'libtensorflowlite_gpu_delegate.so', # linux
# # # #             'libtensorflowlite_gpu_delegate.dylib', # mac
# # # #             'tensorflowlite_gpu_delegate.dll', # windows (rare)
# # # #         ]
# # # #         for cand in candidates:
# # # #             try:
# # # #                 delegate = load_delegate(cand)
# # # #                 print(f"Loaded GPU delegate: {cand}")
# # # #                 return delegate
# # # #             except Exception:
# # # #                 continue
# # # #     # If we reach here, GPU delegate wasn't loaded
# # # #     return None
# # # # GPU_DELEGATE = _load_gpu_delegate()


# # # # def get_tflite_interpreter_for_thread(model_path: str, use_gpu_delegate=True):
   
# # # #     if not hasattr(thread_local, "interpreters"):
# # # #         thread_local.interpreters = {}
# # # #     key = f"{model_path}_gpu" if use_gpu_delegate and GPU_DELEGATE else model_path
# # # #     if key in thread_local.interpreters:
# # # #         return thread_local.interpreters[key]
# # # #     # Create interpreter
# # # #     try:
# # # #         if use_gpu_delegate and GPU_DELEGATE:
# # # #             interpreter = tf.lite.Interpreter(model_path=model_path, experimental_delegates=[GPU_DELEGATE])
# # # #             print(f"[Thread {threading.get_ident()}] Created GPU interpreter for {os.path.basename(model_path)}")
# # # #         else:
# # # #             interpreter = tf.lite.Interpreter(model_path=model_path)
# # # #             print(f"[Thread {threading.get_ident()}] Created CPU interpreter for {os.path.basename(model_path)}")
# # # #     except Exception as e:
# # # #         # Fallback to CPU interpreter if GPU delegate fails
# # # #         print(f"Interpreter creation failed for {model_path} with GPU delegate: {e}. Falling back to CPU.")
# # # #         interpreter = tf.lite.Interpreter(model_path=model_path)
# # # #     interpreter.allocate_tensors()
# # # #     input_details = interpreter.get_input_details()
# # # #     output_details = interpreter.get_output_details()
# # # #     thread_local.interpreters[key] = (interpreter, input_details, output_details)
# # # #     return thread_local.interpreters[key]

# # # # def predict_tflite_model(model_path: str, input_data, use_gpu_delegate=True):
   
# # # #     # Acquire a lock around interpreter invocation to be safe for device resources, but interpreters are per-thread so contention is low.
# # # #     interpreter, input_details, output_details = get_tflite_interpreter_for_thread(model_path, use_gpu_delegate=use_gpu_delegate)
# # # #     with results_lock:
# # # #         input_data = input_data.astype(np.float32)
# # # #         interpreter.set_tensor(input_details[0]['index'], input_data)
# # # #         interpreter.invoke()
# # # #         output_data = interpreter.get_tensor(output_details[0]['index'])
# # # #     return output_data.squeeze()
# # # # # ---------------------- Your existing functions (kept mostly unchanged) ----------------------
# # # # def lowpass(file, cutoff=0.3):
# # # #     b, a = signal.butter(3, cutoff, btype='lowpass', analog=False)
# # # #     low_passed = signal.filtfilt(b, a, file)
# # # #     return low_passed

# # # # def baseline_construction_200(ecg_signal, kernel_size=131):
# # # #     s_corrected = signal.detrend(ecg_signal)
# # # #     baseline_corrected = s_corrected - signal.medfilt(s_corrected, kernel_size)
# # # #     return baseline_corrected

# # # # def normalize(signal):
# # # #     return (signal - np.mean(signal)) / np.std(signal)

# # # # def refined_non_max_suppression(ecg_signal, valid_indices, suppression_radius=40):
# # # #     if len(valid_indices) == 0:
# # # #         return []
# # # #     sorted_indices = sorted(valid_indices, reverse=True)
# # # #     selected = []
# # # #     occupied = np.zeros(len(ecg_signal), dtype=bool)
# # # #     for idx in sorted_indices:
# # # #         if not occupied[idx]:
# # # #             left = max(0, idx - suppression_radius)
# # # #             right = min(len(ecg_signal), idx + suppression_radius + 1)
# # # #             # Mark region as occupied
# # # #             occupied[left:right] = True
# # # #             selected.append(idx)
# # # #     return sorted(selected)




# # # # def check_model_r(ecg_data, r_model_path, use_gpu_delegate=True):
# # # #     totaldata = len(ecg_data)
# # # #     i = 0
# # # #     step = totaldata if totaldata < 1000 else 1000
# # # #     r_peaks = []
# # # #     temp_list = []
# # # #     df_ecg_signal = ecg_data.tolist()
# # # #     while i < totaldata:
# # # #         if i != 0 and totaldata > 1000:
# # # #             i -= 200
# # # #         ecg_signal = ecg_data[i:i + step]
# # # #         signal_len = len(ecg_signal)
# # # #         pad_len = 1000 - signal_len
# # # #         padded_signal = np.pad(ecg_signal, (0, pad_len), mode='constant', constant_values=0)
# # # #         raw_array = np.expand_dims(padded_signal, axis=0).astype(np.float32)[..., np.newaxis]
# # # #         preds = predict_tflite_model(r_model_path, raw_array, use_gpu_delegate=use_gpu_delegate)
# # # #         preds = preds[:signal_len]
# # # #         r_peak_prob = preds[:, 1]
# # # #         peak_indices, _ = find_peaks(r_peak_prob, height=0.2, distance=20)
# # # #         for j in peak_indices:
# # # #             if 0 <= i+j < len(df_ecg_signal):
# # # #                 temp_list.append(i + j)
# # # #         i += step
# # # #     rpeak = sorted(set(temp_list))
# # # #     r_peaks = refined_non_max_suppression(df_ecg_signal, rpeak)
# # # #     return r_peaks

# # # # def r_peak_detection(all_lead_data, is_lead, r_model_path, use_gpu_delegate=True):
# # # #     r_peaks = []
# # # #     result_dic = {}
# # # #     for lead in all_lead_data.keys():
# # # #         ecg_signal = all_lead_data[lead].values.flatten()
# # # #         baseline_signal = baseline_construction_200(ecg_signal, kernel_size=131)
# # # #         lowpass_signal = lowpass(baseline_signal)
# # # #         signal_normalized = normalize(lowpass_signal)
# # # #         r_peaks = check_model_r(signal_normalized, r_model_path, use_gpu_delegate=use_gpu_delegate)
# # # #         result_dic[lead] = r_peaks
# # # #     if is_lead == '2_lead':
# # # #         r_peaks = result_dic['II']
# # # #     return r_peaks, result_dic

# # # # # --- P/T detection functions (unchanged except calling predict_tflite_model with model path) ---
# # # # def resample_ecg(ecg_signal, target_length=520):
# # # #     x_old = np.linspace(0, 1, len(ecg_signal))
# # # #     x_new = np.linspace(0, 1, target_length)
# # # #     f_ecg = interp1d(x_old, ecg_signal, kind='linear')
# # # #     ecg_resampled = f_ecg(x_new)
# # # #     return ecg_resampled

# # # # def restore_org_ecg_mask(ecg_signal, mask, target_length=520):
# # # #     x_old = np.linspace(0, 1, len(ecg_signal))
# # # #     x_new = np.linspace(0, 1, target_length)
# # # #     f_ecg = interp1d(x_old, ecg_signal, kind='linear')
# # # #     ecg_resampled = f_ecg(x_new)
# # # #     f_mask = interp1d(x_old, mask, kind='nearest')
# # # #     mask_resampled = f_mask(x_new)
# # # #     return ecg_resampled, mask_resampled.astype(int)

# # # # # (find_p_t_peaks remains the same)
# # # # def find_p_t_peaks(ecg, mask, boundary_margin=3, merge_distance=15):
# # # #     ecg = np.array(ecg)
# # # #     mask = np.array(mask)
# # # #     def fix_1_2_confusions(mask):
# # # #         mask = mask.copy()
# # # #         i = 1
# # # #         while i < len(mask) - 1:
# # # #             if mask[i] in [1, 2] and mask[i - 1] == mask[i + 1] and mask[i] != mask[i - 1]:
# # # #                 val_to_fill = mask[i - 1]
# # # #                 start = i
# # # #                 while i < len(mask) - 1 and mask[i] != val_to_fill and mask[i] in [1, 2]:
# # # #                     i += 1
# # # #                 mask[start:i] = val_to_fill
# # # #             else:
# # # #                 i += 1
# # # #         return mask
    
# # # #     def selective_majority_filter(mask, window_size=7):
# # # #         padded = np.pad(mask, (window_size // 2,), mode='edge')
# # # #         filtered = mask.copy()
# # # #         for i in range(len(mask)):
# # # #             window = padded[i:i + window_size]
# # # #             center = mask[i]
# # # #             window_mode = mode(window, keepdims=True)[0][0]
# # # #             if center == 0 and window_mode in [1, 2]:
# # # #                 filtered[i] = window_mode
# # # #         return filtered
    
# # # #     def suppress_short_regions(mask, min_length=2):
# # # #         mask = mask.copy()
# # # #         current_val = mask[0]
# # # #         start_idx = 0
# # # #         for i in range(1, len(mask)):
# # # #             if mask[i] != current_val:
# # # #                 if current_val in [1, 2] and (i - start_idx) < min_length:
# # # #                     mask[start_idx:i] = 0
# # # #                 start_idx = i
# # # #                 current_val = mask[i]
# # # #         if current_val in [1, 2] and (len(mask) - start_idx) < min_length:
# # # #             mask[start_idx:] = 0
# # # #         return mask
    
# # # #     def get_peak_indices(mask_val, ecg, mask, max_one=False):
# # # #         indices = []
# # # #         regions = []
# # # #         in_region = False
# # # #         start = 0
# # # #         for i in range(len(mask)):
# # # #             if mask[i] == mask_val and not in_region:
# # # #                 start = i
# # # #                 in_region = True
# # # #             elif mask[i] != mask_val and in_region:
# # # #                 end = i
# # # #                 regions.append((start, end))
# # # #                 in_region = False
# # # #         if in_region:
# # # #             regions.append((start, len(mask)))
# # # #         if max_one and regions:
# # # #             max_len = max(end - start for start, end in regions)
# # # #             longest_regions = [seg for seg in regions if (seg[1] - seg[0]) == max_len]
# # # #             if len(longest_regions) > 1:
# # # #                 abs_vals = [np.max(np.abs(ecg[start:end])) for start, end in longest_regions]
# # # #                 chosen_region = longest_regions[np.argmax(abs_vals)]
# # # #             else:
# # # #                 chosen_region = longest_regions[0]
# # # #             regions = [chosen_region]
# # # #         for start, end in regions:
# # # #             segment = ecg[start:end]
# # # #             maxima = argrelextrema(segment, np.greater)[0]
# # # #             inverted = False
# # # #             if len(maxima) == 0:
# # # #                 maxima = argrelextrema(-segment, np.greater)[0]
# # # #                 inverted = True
# # # #             if len(maxima) > 0:
# # # #                 candidate_values = segment[maxima] if not inverted else -segment[maxima]
# # # #                 best_idx = np.argmax(candidate_values)
# # # #                 peak_relative = maxima[best_idx]
# # # #             else:
# # # #                 derivative = np.gradient(segment)
# # # #                 curvature = np.abs(np.gradient(derivative))
# # # #                 peak_relative = np.argmax(curvature)
# # # #             peak_idx = start + peak_relative
# # # #             if boundary_margin <= peak_idx < len(ecg) - boundary_margin:
# # # #                 indices.append(peak_idx)
# # # #         return indices
    
# # # #     def merge_close_peaks(peaks, ecg, merge_distance):
# # # #         if not peaks:
# # # #             return []
# # # #         peaks = sorted(peaks)
# # # #         merged_peaks = [peaks[0]]
# # # #         for idx in peaks[1:]:
# # # #             last_idx = merged_peaks[-1]
# # # #             if abs(idx - last_idx) < merge_distance:
# # # #                 if abs(ecg[idx]) > abs(ecg[last_idx]):
# # # #                     merged_peaks[-1] = idx
# # # #             else:
# # # #                 merged_peaks.append(idx)
# # # #         return merged_peaks
    
# # # #     def remove_peaks_near_other(peaks_to_filter, reference_peaks, merge_distance):
# # # #         filtered = []
# # # #         for p_idx in peaks_to_filter:
# # # #             if all(abs(p_idx - t_idx) >= merge_distance for t_idx in reference_peaks):
# # # #                 filtered.append(p_idx)
# # # #         return filtered
    
# # # #     def refine_peak_positions(ecg, peak_indices, window=10):
# # # #         refined = []
# # # #         for idx in peak_indices:
# # # #             temp_seg = ecg[max(idx - 2, 0):min(idx + 2, len(ecg))]
# # # #             temp_idx = idx - 2 + np.argmax(np.abs(temp_seg))
# # # #             temp_max = idx - 2 + np.argmax(temp_seg)
# # # #             temp_min = idx - 2 + np.argmin(temp_seg)
# # # #             if idx != temp_idx and (idx != temp_max and idx != temp_min):
# # # #                 start = max(idx - window, 0)
# # # #                 end = min(idx + window + 1, len(ecg))
# # # #                 segment = np.abs(ecg[start:end])
# # # #                 maxima = argrelextrema(segment, np.greater)[0]
# # # #                 inverted = False
# # # #                 if len(maxima) == 0:
# # # #                     maxima = argrelextrema(-segment, np.greater)[0]
# # # #                     inverted = True
# # # #                 if len(maxima) > 0:
# # # #                     candidate_values = segment[maxima] if not inverted else -segment[maxima]
# # # #                     best_idx = np.argmax(candidate_values)
# # # #                     peak_relative = maxima[best_idx]
# # # #                 else:
# # # #                     derivative = np.gradient(segment)
# # # #                     curvature = np.abs(np.gradient(derivative))
# # # #                     peak_relative = np.argmax(curvature)
# # # #                 peak_idx = start + peak_relative
# # # #                 refined.append(peak_idx)
# # # #             else:
# # # #                 refined.append(idx)
# # # #         return refined
   
# # # #     mask = fix_1_2_confusions(mask)
# # # #     mask = selective_majority_filter(mask, window_size=16)
# # # #     mask = suppress_short_regions(mask, min_length=3)
# # # #     t_peaks = get_peak_indices(mask_val=1, ecg=ecg, mask=mask, max_one=True)
# # # #     t_peaks = refine_peak_positions(ecg, t_peaks, window=10)
# # # #     t_peaks = merge_close_peaks(t_peaks, ecg, merge_distance=merge_distance)
# # # #     p_peaks = get_peak_indices(mask_val=2, ecg=ecg, mask=mask, max_one=False)
# # # #     p_peaks = merge_close_peaks(p_peaks, ecg, merge_distance=45)
# # # #     p_peaks = refine_peak_positions(ecg, p_peaks, window=10)
# # # #     p_peaks = remove_peaks_near_other(p_peaks, t_peaks, merge_distance=merge_distance)
# # # #     return p_peaks, t_peaks

# # # # def find_onset_offset(signal, peak_idx, smooth=True, window_size=11, polyorder=3,
# # # #                       min_drop_ratio=0.2, search_window=200):
# # # #     signal = np.array(signal)
# # # #     signal_len = len(signal)
# # # #     if smooth:
# # # #         win = min(window_size, signal_len - (signal_len % 2 == 0))
# # # #         signal_smooth = savgol_filter(signal, window_length=win, polyorder=polyorder)
# # # #     else:
# # # #         signal_smooth = signal
# # # #     peak_val = signal_smooth[peak_idx]
# # # #     baseline_window = min(40, signal_len // 6)
# # # #     start = max(0, peak_idx - baseline_window)
# # # #     end = min(signal_len, peak_idx + baseline_window)
# # # #     local_baseline = np.median(signal_smooth[start:end])
# # # #     drop_thresh = peak_val - (peak_val - local_baseline) * min_drop_ratio
# # # #     onset_idx = peak_idx
# # # #     for i in range(peak_idx, max(1, peak_idx - search_window), -1):
# # # #         if signal_smooth[i] < drop_thresh:
# # # #             onset_idx = i
# # # #             break
# # # #         if i > 1 and signal_smooth[i-1] < signal_smooth[i-2] and signal_smooth[i-1] < signal_smooth[i]:
# # # #             onset_idx = i - 1
# # # #             break
# # # #     offset_idx = peak_idx
# # # #     for i in range(peak_idx, min(signal_len - 2, peak_idx + search_window)):
# # # #         if signal_smooth[i] < drop_thresh:
# # # #             offset_idx = i
# # # #             break
# # # #         if signal_smooth[i+1] < signal_smooth[i] and signal_smooth[i+1] < signal_smooth[i+2]:
# # # #             offset_idx = i + 1
# # # #             break
# # # #     return onset_idx, offset_idx

# # # # def get_pt_peaks(ecg, r_indices, pt_model_path, use_gpu_delegate=True):
# # # #     t_peaks_all, p_peaks_all, pt_peaks_all, onset, offset = [], [], [], [], []
# # # #     for i in range(len(r_indices) - 1):
# # # #         segment = ecg[r_indices[i]:r_indices[i+1]]
# # # #         if len(segment) < 10:
# # # #             continue
# # # #         segment_signal = np.array(segment)
# # # #         resampled_ecgs = resample_ecg(segment_signal, 520)
# # # #         ecg_signal = np.array(resampled_ecgs)
# # # #         ecg_signal = np.expand_dims(ecg_signal, axis=(0, -1))
# # # #         predictions = predict_tflite_model(pt_model_path, ecg_signal, use_gpu_delegate=use_gpu_delegate)
# # # #         predicted_labels = np.argmax(predictions, axis=-1)
# # # #         _, pred_mask = restore_org_ecg_mask(
# # # #             ecg_signal[0].squeeze(), predicted_labels.squeeze(), len(segment_signal)
# # # #         )
# # # #         p_peaks, t_peaks = find_p_t_peaks(segment_signal, pred_mask)
# # # #         p_peaks = np.atleast_1d(p_peaks) + r_indices[i]
# # # #         t_peaks = np.atleast_1d(t_peaks) + r_indices[i]
# # # #         pt_peaks = tuple(list(t_peaks) + list(p_peaks))
# # # #         p_peaks_all.extend(p_peaks)
# # # #         t_peaks_all.extend(t_peaks)
# # # #         pt_peaks_all.extend(pt_peaks)
# # # #     return t_peaks_all, p_peaks_all, pt_peaks_all


# # # # def pt_peak_detection(all_lead_data, is_lead, r_peaks, r_result_dic = None, pt_model_path=None, use_gpu_delegate=True):
# # # #     result_dic = {}
# # # #     for lead in all_lead_data.keys():
# # # #         r_peaks = r_result_dic.get(lead)
# # # #         ecg_signal = all_lead_data[lead].values.flatten()
# # # #         baseline_signal = baseline_construction_200(ecg_signal, kernel_size=131)
# # # #         lowpass_signal = lowpass(baseline_signal)
# # # #         signal_normalized = normalize(lowpass_signal)
# # # #         t_peaks, p_peaks, rr_invl_peaks = get_pt_peaks(signal_normalized, r_peaks, pt_model_path, use_gpu_delegate=use_gpu_delegate)
# # # #         result_dic[lead] = {"p": p_peaks, "t": t_peaks, "comb": rr_invl_peaks}
# # # #     if is_lead == '2_lead':
# # # #         p_peaks = result_dic['II'].get("p")
# # # #         t_peaks = result_dic['II'].get("t")
# # # #         rr_invl_peaks = result_dic['II'].get("comb")

# # # #     return t_peaks, p_peaks, rr_invl_peaks, result_dic

# # # # # ---------------------- Plotting and other post processing functions ----------------------
# # # # def add_standard_ecg_grid(ax, duration_sec, y_min, y_max):
# # # #     """
# # # #     True ECG paper grid:
# # # #     - 25 mm/s  → 0.04 s per small square
# # # #     - 10 mm/mV → 0.1 mV per small square
# # # #     Grid MUST be square.
# # # #     """

# # # #     # X axis (time)
# # # #     ax.set_xlim(0, duration_sec)
# # # #     ax.set_xticks(np.arange(0, duration_sec + 0.04, 0.04), minor=True)
# # # #     ax.set_xticks(np.arange(0, duration_sec + 0.2, 0.2))

# # # #     # Y axis (amplitude)
# # # #     ax.set_ylim(y_min, y_max)
# # # #     ax.set_yticks(np.arange(y_min, y_max + 0.1, 0.1), minor=True)
# # # #     ax.set_yticks(np.arange(y_min, y_max + 0.5, 0.5))
# # # #     # ---- Aspect ratio ----
# # # #     # ax.set_aspect(0.04 / 0.1) # 0.04 sec = 0.1 mV

# # # #     # ✅ CRITICAL: lock aspect so squares stay squares
# # # #     ax.set_aspect((0.04 / 0.1), adjustable="box")

# # # #     # Grid style
# # # #     ax.grid(which='major', color='#b00000', linewidth=1.1)
# # # #     ax.grid(which='minor', color='#e6e6e6', linewidth=0.6)

# # # #     ax.set_facecolor("white")



# # # # # ---------------------- Plotting and other post processing functions ----------------------
# # # # def hr_count(r_index, fs=250):
# # # #     if len(r_index) < 2:
# # # #         return 0
# # # #     rr_intervals = np.diff(r_index)
# # # #     if len(rr_intervals) == 0:
# # # #         return 0
# # # #     HR = int((len(rr_intervals) * 60000) / np.sum(rr_intervals / fs * 1000))
# # # #     return HR

# # # # def refine_r_peaks(signal, r_peaks, fs):

# # # #     refined = []
# # # #     polarity = []

# # # #     window = int(0.04 * fs)  # ±40 ms around detected R

# # # #     for r in r_peaks:
# # # #         if r <= 0 or r >= len(signal):
# # # #             continue

# # # #         start = max(0, r - window)
# # # #         end = min(len(signal), r + window + 1)
# # # #         seg = signal[start:end]

# # # #         if len(seg) == 0:
# # # #             continue

# # # #         # pick the true apex of QRS (largest absolute deflection)
# # # #         idx = np.argmax(np.abs(seg))
# # # #         true_r = start + idx

# # # #         refined.append(true_r)
# # # #         polarity.append(np.sign(signal[true_r]) or 1)

# # # #     return np.array(refined, dtype=int), np.array(polarity, dtype=int)


# # # # def detect_q_s(signal, r_peaks, r_polarity, fs):
    
# # # #     q_points = []
# # # #     s_points = []
# # # #     window = int(0.1 * fs)  # 100 ms search window

# # # #     for r, pol in zip(r_peaks, r_polarity):
# # # #         # Q point: before R
# # # #         start = max(0, r - window)
# # # #         seg = signal[start:r+1]
# # # #         if pol > 0:
# # # #             q_idx = start + np.argmin(seg)  # Q = min before upright R
# # # #         else:
# # # #             q_idx = start + np.argmax(seg)  # Q = max before inverted R
# # # #         q_points.append(q_idx)

# # # #         # S point: after R
# # # #         end = min(len(signal), r + window + 1)
# # # #         seg = signal[r:end]
# # # #         if pol > 0:
# # # #             s_idx = r + np.argmin(seg)  # S = min after upright R
# # # #         else:
# # # #             s_idx = r + np.argmax(seg)  # S = max after inverted R
# # # #         s_points.append(s_idx)

# # # #     return np.array(q_points, dtype=int), np.array(s_points, dtype=int)



# # # # def detect_t_wave_onset(signal, s_points, fs):
# # # #     """
# # # #     Detect T-wave onset AFTER ST segment.
# # # #     NOT used for ST measurement.
# # # #     """

# # # #     signal = np.asarray(signal)
# # # #     t_onsets = []

# # # #     for s_idx in s_points:
# # # #         j_point = int(s_idx)

# # # #         search_start = j_point + int(0.08 * fs)
# # # #         search_end   = min(j_point + int(0.30 * fs), len(signal) - 2)

# # # #         if search_end <= search_start:
# # # #             t_onsets.append(search_start)
# # # #             continue

# # # #         slope = np.diff(signal)
# # # #         onset = search_end

# # # #         for i in range(search_start, search_end - 1):
# # # #             if slope[i] > 0 and slope[i+1] > 0:
# # # #                 onset = i
# # # #                 break

# # # #         t_onsets.append(onset)

# # # #     return np.array(t_onsets, dtype=int)

# # # # def measure_st_segment_full(signal, r_peaks, s_points, t_onsets, fs, calibration=10.0):
# # # #     """
# # # #     Measure ST segment from J-point → T-onset.
# # # #     Keeps ALL beats (no rejection).
# # # #     Returns tuples:
# # # #     (j_point, st_end, st_mV, pr_baseline)
# # # #     """
# # # #     import numpy as np

# # # #     st_results = []
# # # #     signal_len = len(signal)
# # # #     beats = min(len(r_peaks), len(s_points), len(t_onsets))

# # # #     for i in range(beats):
# # # #         r = int(r_peaks[i])
# # # #         s = int(s_points[i])
# # # #         t_idx = int(t_onsets[i])

# # # #         if r <= 0 or s <= 0 or r >= signal_len or s >= signal_len:
# # # #             continue

# # # #         # J-point ≈ 10 ms after S
# # # #         j_point = s + int(0.06 * fs)
# # # #         if j_point >= signal_len:
# # # #             continue

# # # #         # PR baseline (−200 to −120 ms before R)
# # # #         pr_start = max(r - int(0.20 * fs), 0)
# # # #         pr_end   = max(r - int(0.12 * fs), pr_start + 1)
# # # #         pr_baseline = np.median(signal[pr_start:pr_end])

# # # #         st_end = min(t_idx, signal_len - 1)
# # # #         if st_end <= j_point:
# # # #             continue

# # # #         st_mV = np.median(signal[j_point:st_end + 1]) - pr_baseline

# # # #         st_results.append((j_point, st_end, st_mV, pr_baseline))

# # # #     return st_results



# # # # def plotting(
# # # #     all_leads_data,
# # # #     save_path,
# # # #     local_name,
# # # #     pt_result_dic,
# # # #     r_result_dic,
# # # #     fs=250,
# # # #     calibration=10.0,
# # # #     show_plots_for_first_n=5
# # # # ):
# # # #     import os
# # # #     import numpy as np
# # # #     import pandas as pd
# # # #     import matplotlib.pyplot as plt
# # # #     import matplotlib.patheffects as path_effects

# # # #     os.makedirs(save_path, exist_ok=True)

# # # #     if not hasattr(plotting, "_counter"):
# # # #         plotting._counter = 0

# # # #     show_plots = plotting._counter < show_plots_for_first_n
# # # #     summary_records = []
# # # #     mm_per_mV = calibration  # 10 mm / mV

# # # #     for lead in all_leads_data.columns:

# # # #         # === PERFECT LAYOUT (NO WHITE PADDING) ===
# # # #         fig = plt.figure(figsize=(14, 6))
# # # #         ax = fig.add_axes([0.02, 0.06, 0.96, 0.90])  # fill page

# # # #         # ---------------- SIGNAL ----------------
# # # #         signal = all_leads_data[lead].values.astype(float)
# # # #         signal_len = len(signal)
# # # #         time_axis = np.arange(signal_len) / fs

# # # #         # ---------------- Y LIMITS ----------------
# # # #         y_min = np.floor(signal.min() * mm_per_mV) / mm_per_mV
# # # #         y_max = np.ceil(signal.max() * mm_per_mV) / mm_per_mV

# # # #         # ---------------- GRID ----------------
# # # #         add_standard_ecg_grid(
# # # #             ax=ax,
# # # #             duration_sec=time_axis[-1],
# # # #             y_min=y_min,
# # # #             y_max=y_max
# # # #         )

# # # #         # ---------------- DRAW ECG ----------------
# # # #         ax.plot(time_axis, signal, color="black", linewidth=1.2, zorder=3)

# # # #         # ---------------- R PEAKS ----------------
# # # #         r_peaks = np.asarray(r_result_dic.get(lead, []), dtype=int)
# # # #         if len(r_peaks) == 0:
# # # #             plt.close(fig)
# # # #             continue

# # # #         r_peaks, r_polarity = refine_r_peaks(signal, r_peaks, fs)

# # # #         # ---------------- Q & S ----------------
# # # #         q_points, s_points = detect_q_s(signal, r_peaks, r_polarity, fs)

# # # #         # ---------------- T ONSET ----------------
# # # #         t_onsets = detect_t_wave_onset(signal, s_points, fs)
# # # #         t_onsets = np.clip(t_onsets, 0, signal_len - 1)

# # # #         # ---------------- MARKERS ----------------
# # # #         ax.plot(time_axis[r_peaks], signal[r_peaks], "ro", ms=4, label="R", zorder=4)
# # # #         ax.plot(time_axis[q_points], signal[q_points], "o", color="purple", ms=4, label="Q", zorder=4)
# # # #         ax.plot(time_axis[s_points], signal[s_points], "o", color="orange", ms=4, label="S", zorder=4)
# # # #         ax.plot(time_axis[t_onsets], signal[t_onsets], "o", color="green", ms=5, label="T-onset", zorder=4)

# # # #         # ---------------- ST MEASUREMENT ----------------
# # # #         st_records = measure_st_segment_full(
# # # #             signal, r_peaks, s_points, t_onsets, fs, calibration
# # # #         )

# # # #         if len(st_records) == 0:
# # # #             plt.close(fig)
# # # #             continue

# # # #         # ---------------- PR BASELINE ----------------
# # # #         pr_baseline = np.median([rec[3] for rec in st_records])
# # # #         ax.hlines(
# # # #             pr_baseline,
# # # #             time_axis[0],
# # # #             time_axis[-1],
# # # #             colors="gray",
# # # #             linestyles="--",
# # # #             linewidth=1.2,
# # # #             label="PR baseline",
# # # #             zorder=2
# # # #         )

# # # #         # ---------------- ST SHADING ----------------
# # # #         for idx, (j_point, st_end, st_mV, _) in enumerate(st_records, start=1):

# # # #             st_mm = st_mV * mm_per_mV
# # # #             if abs(st_mm) < 0.5:
# # # #                 continue

# # # #             color = "green" if abs(st_mm) < 2.0 else "red"
# # # #             status = "Normal" if abs(st_mm) < 2.0 else "Abnormal"

# # # #             start_idx = max(0, min(j_point, signal_len - 1))
# # # #             end_idx   = max(start_idx + 1, min(st_end, signal_len - 1))

# # # #             segment_x = time_axis[start_idx:end_idx + 1]
# # # #             segment_y = signal[start_idx:end_idx + 1]

# # # #             lower = np.minimum(segment_y, pr_baseline)
# # # #             upper = np.maximum(segment_y, pr_baseline)

# # # #             ax.fill_between(
# # # #                 segment_x,
# # # #                 lower,
# # # #                 upper,
# # # #                 where=(upper != lower),
# # # #                 color=color,
# # # #                 alpha=0.35,
# # # #                 zorder=1
# # # #             )

# # # #             label_y = upper.max() + 0.03 if st_mV >= 0 else lower.min() - 0.03

# # # #             ax.text(
# # # #                 segment_x[-1],
# # # #                 label_y,
# # # #                 f"{st_mm:+.1f} mm",
# # # #                 ha="center",
# # # #                 va="bottom" if st_mV >= 0 else "top",
# # # #                 fontsize=8,
# # # #                 fontweight="bold",
# # # #                 zorder=5,
# # # #                 path_effects=[path_effects.withStroke(linewidth=2, foreground="white")]
# # # #             )

# # # #             summary_records.append({
# # # #                 "Lead": lead,
# # # #                 "Beat_No": idx,
# # # #                 "ST_mm": round(st_mm, 2),
# # # #                 "Isoelectric_mV": round(pr_baseline, 4),
# # # #                 "Status": status
# # # #             })

# # # #         # ---------------- FINAL DECORATION ----------------
# # # #         hr = hr_count(r_peaks, fs)
# # # #         ax.set_title(f"{local_name} — Lead {lead} — HR {hr} bpm", fontsize=14)
# # # #         ax.set_xlabel("Time (s)")
# # # #         ax.set_ylabel("Amplitude (mV)")
# # # #         ax.legend(loc="upper right")

# # # #         # ---------------- CRITICAL: REMOVE ALL PADDING ----------------
# # # #         ax.margins(x=0, y=0)
# # # #         ax.set_anchor('C')

# # # #         pdf_path = os.path.join(save_path, f"{local_name}_{lead}.pdf")
# # # #         plt.savefig(pdf_path, dpi=300, pad_inches=0)

# # # #         if show_plots:
# # # #             plt.show()

# # # #         plt.close(fig)

# # # #     plotting._counter += 1
# # # #     return pd.DataFrame(summary_records)

# # # # # ---------------------- I/O / file handling helpers ----------------------
# # # # def load_and_rename_data(fn, is_lead_for):
# # # #     lead_columns = {
# # # #         '2_lead': ['ecg', 'ii', 'value', 'mlii'],
# # # #         '7_lead': ['i', 'ii', 'iii', 'avr', 'avl', 'avf', 'v5'],
# # # #         '12_lead': ['i', 'ii', 'iii', 'avr', 'avl', 'avf', 'v1', 'v2', 'v3', 'v4', 'v5', 'v6', 'ecg']
# # # #     }

# # # #     lead_columns_index = {
# # # #         '2_lead': {0: 'II'},
# # # #         '7_lead': {0: 'I', 1: 'II', 2: 'III', 3: 'aVR', 4: 'aVL', 5: 'aVF', 6: 'V5'},
# # # #         '12_lead': {0: 'I', 1: 'II', 2: 'III', 3: 'aVR', 4: 'aVL', 5: 'aVF',
# # # #                     6: 'V1', 7: 'V2', 8: 'V3', 9: 'V4', 10: 'V5', 11: 'V6'}
# # # #     }

# # # #     all_lead_data = pd.read_csv(fn).fillna(0)
# # # #     all_lead_data.columns = (
# # # #         all_lead_data.columns
# # # #         .str.strip()
# # # #         .str.lower()
# # # #     )
# # # #     columns = all_lead_data.columns.tolist()
# # # #     if any(str(val).isalpha() for val in all_lead_data.iloc[0, :].values):
# # # #         if all(col in lead_columns['7_lead'] for col in columns):
# # # #             is_lead_for = '7_lead'
# # # #         elif all(col in lead_columns['12_lead'] for col in columns):
# # # #             is_lead_for = '12_lead'
# # # #         else:
# # # #             is_lead_for = '2_lead'
# # # #     else:
# # # #         if len(columns) >= 12:
# # # #             is_lead_for = '12_lead'
# # # #         elif len(columns) >= 7:
# # # #             is_lead_for = '7_lead'
# # # #         else:
# # # #             is_lead_for = '2_lead'

# # # #     if is_lead_for == '2_lead':
# # # #         available_columns = [col for col in lead_columns['2_lead'] if col in columns]
# # # #         # all_lead_data = attempt_column_load(fn, available_columns)
# # # #         all_lead_data = all_lead_data[available_columns]

# # # #     elif is_lead_for == '7_lead':
# # # #         available_columns = [col for col in lead_columns['7_lead'] if col in columns]
# # # #         # all_lead_data = attempt_column_load(fn, available_columns)
# # # #         all_lead_data = all_lead_data[available_columns]

# # # #     elif is_lead_for == '12_lead':
# # # #         available_columns = [col for col in lead_columns['12_lead'] if col in columns]
# # # #         # all_lead_data = attempt_column_load(fn, available_columns)
# # # #         all_lead_data = all_lead_data[available_columns]

# # # #     if all_lead_data is not None:
# # # #         all_lead_data = all_lead_data.rename(columns=lead_columns_index[is_lead_for])

# # # #     if is_lead_for == '2_lead':
# # # #         all_lead_data.columns = ['II']

# # # #     return all_lead_data, is_lead_for

# # # # def attempt_column_load(fn, columns):
# # # #     try:
# # # #         data = pd.read_csv(fn, usecols=columns).fillna(0)
# # # #         return data
# # # #     except ValueError as e:
# # # #         print("value Error ",e)
# # # #         return None
# # # #     except Exception as e:
# # # #         print("Error in Loading",e)
# # # #         return None

# # # # def find_csv_files(root_folder):
# # # #     csv_files = []
# # # #     for root, _, files in os.walk(root_folder):
# # # #         for file in files:
# # # #             if file.lower().endswith('.csv'):
# # # #                 csv_files.append(os.path.join(root, file))
# # # #     return csv_files


# # # # def process_single_file(
# # # #     fn,
# # # #     save_path,
# # # #     is_lead,
# # # #     r_model_path,
# # # #     pt_model_path,
# # # #     use_gpu_delegate=True
# # # # ):
# # # #     import os
# # # #     import pandas as pd

# # # #     local_name = os.path.splitext(os.path.basename(fn))[0]
# # # #     csv_root = os.path.join(save_path, local_name)
# # # #     os.makedirs(csv_root, exist_ok=True)

# # # #     # ---------- LOAD ----------
# # # #     all_leads_data, is_lead = load_and_rename_data(fn, is_lead)
# # # #     if all_leads_data is None:
# # # #         return f"Failed: {local_name}"

# # # #     # ---------- R PEAK DETECTION ----------
# # # #     _, r_result_dic = r_peak_detection(
# # # #         all_leads_data,
# # # #         is_lead,
# # # #         r_model_path,
# # # #         use_gpu_delegate
# # # #     )

# # # #     if r_result_dic is None:
# # # #         return f"Failed: {local_name}"

# # # #     # ---------- P & T PEAK DETECTION ----------
# # # #     _, _, _, pt_result_dic = pt_peak_detection(
# # # #         all_leads_data,
# # # #         is_lead,
# # # #         None,
# # # #         r_result_dic,
# # # #         pt_model_path,
# # # #         use_gpu_delegate
# # # #     )

# # # #     if pt_result_dic is None:
# # # #         return f"Failed: {local_name}"

# # # #     # ---------- BASELINE CORRECTED ECG (UNCHANGED, USED ELSEWHERE) ----------
# # # #     baseline_corrected_data = pd.DataFrame({
# # # #         lead: baseline_construction_200(all_leads_data[lead].values)
# # # #         for lead in all_leads_data.columns
# # # #     })

# # # #     # ---------- PLOTTING ----------
# # # #     summary_df = plotting(
# # # #         all_leads_data,   # RAW ECG
# # # #         csv_root,
# # # #         local_name,
# # # #         pt_result_dic,
# # # #         r_result_dic,
# # # #         fs=250,
# # # #         calibration=10.0
# # # #     )

# # # #     # FIX: guard BEFORE using .empty
# # # #     if summary_df is not None and not summary_df.empty:
# # # #         summary_df.to_csv(
# # # #             os.path.join(csv_root, f"{local_name}_ALL_LEADS_ST_SUMMARY.csv"),
# # # #             index=False
# # # #         )

# # # #     # ---------- MERGE PDFs ----------
# # # #     merge_pdfs_in_lead_order(
# # # #         csv_root,
# # # #         os.path.join(csv_root, f"{local_name}_MERGED_ALL_LEADS.pdf")
# # # #     )

# # # #     return f"Processed: {local_name}"




# # # # def merge_pdfs_in_lead_order(pdf_dir, output_pdf):
# # # #     from PyPDF2 import PdfMerger
# # # #     import os

# # # #     LEAD_ORDER = [
# # # #         "I", "II", "III",
# # # #         "aVR", "aVL", "aVF",
# # # #         "V1", "V2", "V3", "V4", "V5", "V6"
# # # #     ]

# # # #     merger = PdfMerger()

# # # #     for lead in LEAD_ORDER:
# # # #         for file in sorted(os.listdir(pdf_dir)):
# # # #             if file.endswith(".pdf") and f"_{lead}.pdf" in file:
# # # #                 merger.append(os.path.join(pdf_dir, file))
# # # #                 break

# # # #     merger.write(output_pdf)
# # # #     merger.close()
# # # # def merge_all_merged_pdfs(root_output_dir, final_pdf_path):
# # # #     """
# # # #     Merge all per-CSV merged ECG PDFs into one final PDF.

# # # #     Expects files like:
# # # #     root_output_dir/
# # # #         patient1/patient1_MERGED_ALL_LEADS.pdf
# # # #         patient2/patient2_MERGED_ALL_LEADS.pdf
# # # #         ...

# # # #     Creates:
# # # #         final_pdf_path
# # # #     """
# # # #     import os
# # # #     from PyPDF2 import PdfMerger

# # # #     merger = PdfMerger()

# # # #     found_any = False

# # # #     for root, _, files in os.walk(root_output_dir):
# # # #         for file in sorted(files):
# # # #             if file.endswith("_MERGED_ALL_LEADS.pdf"):
# # # #                 pdf_path = os.path.join(root, file)
# # # #                 merger.append(pdf_path)
# # # #                 found_any = True

# # # #     if found_any:
# # # #         merger.write(final_pdf_path)
# # # #         merger.close()
# # # #         print(f"FINAL MERGED PDF CREATED: {final_pdf_path}")
# # # #     else:
# # # #         merger.close()
# # # #         print("No merged PDFs found to combine.")


# # # # def ecg_processing(path, save_path, is_lead, r_model_path, pt_model_path, max_workers=DESIRED_CPU_THREADS, use_gpu_delegate=True):
# # # #     csv_files = find_csv_files(path)
# # # #     if not csv_files:
# # # #         print("No CSV files found.")
# # # #         return
# # # #     # Use ThreadPoolExecutor so per-thread interpreters remain in same process/threads (GPU delegate may not be picklable)
# # # #     max_workers = min(max_workers, max(1, os.cpu_count() or 1))
# # # #     print(f"Processing {len(csv_files)} files with {max_workers} workers (use_gpu_delegate={use_gpu_delegate})")
# # # #     with ThreadPoolExecutor(max_workers=max_workers) as executor:
# # # #         futures = {executor.submit(process_single_file, fn, save_path, is_lead, r_model_path, pt_model_path, use_gpu_delegate): fn for fn in csv_files}
# # # #         for future in as_completed(futures):
# # # #             try:
# # # #                 print(future.result())
# # # #             except Exception as e:
# # # #                 print("Worker exception:", e)
# # # #     print("All files processed successfully.")
# # # # def find_pdf_files(root_folder):
# # # #     pdf_files = []
# # # #     for root, _, files in os.walk(root_folder):
# # # #         for file in files:
# # # #             if file.lower().endswith('.pdf'):
# # # #                 pdf_files.append(os.path.join(root, file))
# # # #     return sorted(pdf_files)
# # # # def merge_pdfs(pdf_files, output_path):
# # # #     merger = PdfMerger()
# # # #     for pdf in pdf_files:
# # # #         try:
# # # #             merger.append(pdf)
# # # #             print(f"Merged: {pdf}")
# # # #         except Exception as e:
# # # #             print(f"Failed to merge {pdf}: {e}")
# # # #     merger.write(output_path)
# # # #     merger.close()
# # # #     print(f"\nAll PDFs merged into: {output_path}")
# # # # def process_single_file_wrapper(args):
# # # #     """Wrapper for multiprocessing (since functions must be pickleable)."""
# # # #     return process_single_file(*args)
# # # # def ecg_processing(path, save_path, is_lead, r_model_path, pt_model_path,
# # # #                    max_workers=DESIRED_CPU_THREADS, use_gpu_delegate=True,
# # # #                    use_multiprocessing=False):
# # # #     csv_files = find_csv_files(path)
# # # #     if not csv_files:
# # # #         print("No CSV files found.")
# # # #         return
# # # #     max_workers = min(max_workers, max(1, os.cpu_count() or 1))
# # # #     mode = "multiprocessing" if use_multiprocessing else "threading"
# # # #     print(f"Processing {len(csv_files)} files with {max_workers} workers ({mode}, use_gpu_delegate={use_gpu_delegate})")
# # # #     if use_multiprocessing:
# # # #         # Each process loads its own interpreters/models
# # # #         with ProcessPoolExecutor(max_workers=max_workers) as executor:
# # # #             futures = {
# # # #                 executor.submit(process_single_file_wrapper,
# # # #                                 (fn, save_path, is_lead, r_model_path, pt_model_path, use_gpu_delegate)): fn
# # # #                 for fn in csv_files
# # # #             }
# # # #             for future in as_completed(futures):
# # # #                 try:
# # # #                     print(future.result())
# # # #                 except Exception as e:
# # # #                     print("Worker exception:", e)
# # # #     else:
# # # #         # Default threading (per-thread cached interpreters, GPU delegate works better here)
# # # #         with ThreadPoolExecutor(max_workers=max_workers) as executor:
# # # #             futures = {
# # # #                 executor.submit(process_single_file, fn, save_path, is_lead, r_model_path, pt_model_path, use_gpu_delegate): fn
# # # #                 for fn in csv_files
# # # #             }
# # # #             for future in as_completed(futures):
# # # #                 try:
# # # #                     print(future.result())
# # # #                 except Exception as e:
# # # #                     print("Worker exception:", e)
# # # #     print("All files processed successfully.")

# # # # def save_pdf_to_gridfs(pdf_path, metadata=None):
# # # #     mongo_uri = os.getenv("MONGO_HOST")
# # # #     mongo_client = MongoClient(mongo_uri)
# # # #     db = mongo_client["St_Segment"]

# # # #     fs = gridfs.GridFS(db)

# # # #     with open(pdf_path, "rb") as f:
# # # #         file_id = fs.put(
# # # #             f,
# # # #             filename=pdf_path.split("\\")[-1],
# # # #             contentType="application/pdf",
# # # #             metadata=metadata or {}
# # # #         )

# # # #     return str(file_id)

# # # # def run_ecg_st_pipeline(
# # # #     input_folder,
# # # #     output_folder,
# # # #     is_lead,
# # # #     max_workers=4,
# # # #     use_gpu_delegate=True,
# # # #     use_multiprocessing=False
# # # # ):
# # # #     os.makedirs(output_folder, exist_ok=True)

# # # #     ecg_processing(
# # # #         path=input_folder,
# # # #         save_path=output_folder,
# # # #         is_lead=is_lead,
# # # #         r_model_path = r"D:\\try3\\Scripts_Models\\Model\\rnn_model1_19_12_Unet.tflite",
# # # #         pt_model_path = r"D:\\try3\\Scripts_Models\\Model\\ecg_pt_detection_LSTMGRU_TCN_Transpose_v27.tflite",
# # # #         max_workers=max_workers,
# # # #         use_gpu_delegate=use_gpu_delegate,
# # # #         use_multiprocessing=use_multiprocessing
# # # #     )

# # # #     final_pdf_path = os.path.join(
# # # #         output_folder,
# # # #         "FINAL_ALL_CSV_ALL_LEADS.pdf"
# # # #     )

# # # #     merge_all_merged_pdfs(
# # # #         root_output_dir=output_folder,
# # # #         final_pdf_path=final_pdf_path
# # # #     )

# # # #     return final_pdf_path

# # # # -----------------------------testing code--------------------------------
# # # import os
# # # import glob
# # # import pandas as pd
# # # import numpy as np
# # # import matplotlib
# # # matplotlib.use('Agg')
# # # import matplotlib.pyplot as plt
# # # import tensorflow as tf
# # # from scipy import signal
# # # from scipy.signal import find_peaks, argrelextrema, savgol_filter
# # # from scipy.stats import mode
# # # from scipy.interpolate import interp1d
# # # import warnings
# # # import threading
# # # from concurrent.futures import ThreadPoolExecutor, as_completed
# # # from PyPDF2 import PdfMerger
# # # import random
# # # from concurrent.futures import ThreadPoolExecutor, as_completed, ProcessPoolExecutor
# # # import multiprocessing
# # # from pymongo import MongoClient
# # # import gridfs
# # # from django.conf import settings


# # # warnings.filterwarnings('ignore')
# # # results_lock = threading.RLock()

# # # # ---------------------- Server-level tuning (CPU / threading / TF / env) ----------------------
# # # # Set these before heavy libs use BLAS/OMP/MKL threads
# # # DESIRED_CPU_THREADS = 24
# # # os.environ['OMP_NUM_THREADS'] = str(DESIRED_CPU_THREADS)
# # # os.environ['OPENBLAS_NUM_THREADS'] = str(DESIRED_CPU_THREADS)
# # # os.environ['MKL_NUM_THREADS'] = str(DESIRED_CPU_THREADS)
# # # os.environ['NUMEXPR_NUM_THREADS'] = str(DESIRED_CPU_THREADS)
# # # os.environ['VECLIB_MAXIMUM_THREADS'] = str(DESIRED_CPU_THREADS)
# # # # TensorFlow GPU / thread config
# # # try:
# # #     physical_gpus = tf.config.list_physical_devices('GPU')
# # #     if physical_gpus:
# # #         # Allow memory growth so multiple processes/threads can share GPU more gracefully
# # #         for g in physical_gpus:
# # #             try:
# # #                 tf.config.experimental.set_memory_growth(g, True)
# # #             except Exception:
# # #                 pass
# # #     # Set TF threading parallelism
# # #     tf.config.threading.set_intra_op_parallelism_threads(DESIRED_CPU_THREADS)
# # #     tf.config.threading.set_inter_op_parallelism_threads(max(1, DESIRED_CPU_THREADS // 2))
# # # except Exception as e:
# # #     print("Warning: TensorFlow configuration failed:", e)
# # # # Thread-local storage for per-thread interpreters
# # # thread_local = threading.local()
# # # # ---------------------- TFLite interpreter utilities ----------------------
# # # def _load_gpu_delegate():
# # #     """
# # #     Try loading a TFLite GPU delegate. This is inherently platform-dependent.
# # #     We try multiple common delegate names and fall back to None.
# # #     """
# # #     try:
# # #         # TensorFlow's load_delegate helper
# # #         load_delegate = tf.lite.experimental.load_delegate
# # #     except Exception:
# # #         load_delegate = None
# # #     if load_delegate:
# # #         candidates = [
# # #             'libtensorflowlite_gpu_delegate.so', # linux
# # #             'libtensorflowlite_gpu_delegate.dylib', # mac
# # #             'tensorflowlite_gpu_delegate.dll', # windows (rare)
# # #         ]
# # #         for cand in candidates:
# # #             try:
# # #                 delegate = load_delegate(cand)
# # #                 print(f"Loaded GPU delegate: {cand}")
# # #                 return delegate
# # #             except Exception:
# # #                 continue
# # #     # If we reach here, GPU delegate wasn't loaded
# # #     return None
# # # GPU_DELEGATE = _load_gpu_delegate()


# # # def get_tflite_interpreter_for_thread(model_path: str, use_gpu_delegate=True):
   
# # #     if not hasattr(thread_local, "interpreters"):
# # #         thread_local.interpreters = {}
# # #     key = f"{model_path}_gpu" if use_gpu_delegate and GPU_DELEGATE else model_path
# # #     if key in thread_local.interpreters:
# # #         return thread_local.interpreters[key]
# # #     # Create interpreter
# # #     try:
# # #         if use_gpu_delegate and GPU_DELEGATE:
# # #             interpreter = tf.lite.Interpreter(model_path=model_path, experimental_delegates=[GPU_DELEGATE])
# # #             print(f"[Thread {threading.get_ident()}] Created GPU interpreter for {os.path.basename(model_path)}")
# # #         else:
# # #             interpreter = tf.lite.Interpreter(model_path=model_path)
# # #             print(f"[Thread {threading.get_ident()}] Created CPU interpreter for {os.path.basename(model_path)}")
# # #     except Exception as e:
# # #         # Fallback to CPU interpreter if GPU delegate fails
# # #         print(f"Interpreter creation failed for {model_path} with GPU delegate: {e}. Falling back to CPU.")
# # #         interpreter = tf.lite.Interpreter(model_path=model_path)
# # #     interpreter.allocate_tensors()
# # #     input_details = interpreter.get_input_details()
# # #     output_details = interpreter.get_output_details()
# # #     thread_local.interpreters[key] = (interpreter, input_details, output_details)
# # #     return thread_local.interpreters[key]

# # # def predict_tflite_model(model_path: str, input_data, use_gpu_delegate=True):
   
# # #     # Acquire a lock around interpreter invocation to be safe for device resources, but interpreters are per-thread so contention is low.
# # #     interpreter, input_details, output_details = get_tflite_interpreter_for_thread(model_path, use_gpu_delegate=use_gpu_delegate)
# # #     with results_lock:
# # #         input_data = input_data.astype(np.float32)
# # #         interpreter.set_tensor(input_details[0]['index'], input_data)
# # #         interpreter.invoke()
# # #         output_data = interpreter.get_tensor(output_details[0]['index'])
# # #     return output_data.squeeze()
# # # # ---------------------- Your existing functions (kept mostly unchanged) ----------------------
# # # def lowpass(file, cutoff=0.3):
# # #     b, a = signal.butter(3, cutoff, btype='lowpass', analog=False)
# # #     low_passed = signal.filtfilt(b, a, file)
# # #     return low_passed

# # # def baseline_construction_200(ecg_signal, kernel_size=131):
# # #     s_corrected = signal.detrend(ecg_signal)
# # #     baseline_corrected = s_corrected - signal.medfilt(s_corrected, kernel_size)
# # #     return baseline_corrected

# # # def normalize(signal):
# # #     return (signal - np.mean(signal)) / np.std(signal)

# # # def refined_non_max_suppression(ecg_signal, valid_indices, suppression_radius=40):
# # #     if len(valid_indices) == 0:
# # #         return []
# # #     sorted_indices = sorted(valid_indices, reverse=True)
# # #     selected = []
# # #     occupied = np.zeros(len(ecg_signal), dtype=bool)
# # #     for idx in sorted_indices:
# # #         if not occupied[idx]:
# # #             left = max(0, idx - suppression_radius)
# # #             right = min(len(ecg_signal), idx + suppression_radius + 1)
# # #             # Mark region as occupied
# # #             occupied[left:right] = True
# # #             selected.append(idx)
# # #     return sorted(selected)




# # # def check_model_r(ecg_data, r_model_path, use_gpu_delegate=True):
# # #     totaldata = len(ecg_data)
# # #     i = 0
# # #     step = totaldata if totaldata < 1000 else 1000
# # #     r_peaks = []
# # #     temp_list = []
# # #     df_ecg_signal = ecg_data.tolist()
# # #     while i < totaldata:
# # #         if i != 0 and totaldata > 1000:
# # #             i -= 200
# # #         ecg_signal = ecg_data[i:i + step]
# # #         signal_len = len(ecg_signal)
# # #         pad_len = 1000 - signal_len
# # #         padded_signal = np.pad(ecg_signal, (0, pad_len), mode='constant', constant_values=0)
# # #         raw_array = np.expand_dims(padded_signal, axis=0).astype(np.float32)[..., np.newaxis]
# # #         preds = predict_tflite_model(r_model_path, raw_array, use_gpu_delegate=use_gpu_delegate)
# # #         preds = preds[:signal_len]
# # #         r_peak_prob = preds[:, 1]
# # #         peak_indices, _ = find_peaks(r_peak_prob, height=0.2, distance=20)
# # #         for j in peak_indices:
# # #             if 0 <= i+j < len(df_ecg_signal):
# # #                 temp_list.append(i + j)
# # #         i += step
# # #     rpeak = sorted(set(temp_list))
# # #     r_peaks = refined_non_max_suppression(df_ecg_signal, rpeak)
# # #     return r_peaks

# # # def r_peak_detection(all_lead_data, is_lead, r_model_path, use_gpu_delegate=True):
# # #     r_peaks = []
# # #     result_dic = {}
# # #     for lead in all_lead_data.keys():
# # #         ecg_signal = all_lead_data[lead].values.flatten()
# # #         baseline_signal = baseline_construction_200(ecg_signal, kernel_size=131)
# # #         lowpass_signal = lowpass(baseline_signal)
# # #         signal_normalized = normalize(lowpass_signal)
# # #         r_peaks = check_model_r(signal_normalized, r_model_path, use_gpu_delegate=use_gpu_delegate)
# # #         result_dic[lead] = r_peaks
# # #     if is_lead == '2_lead':
# # #         r_peaks = result_dic['II']
# # #     return r_peaks, result_dic

# # # # --- P/T detection functions (unchanged except calling predict_tflite_model with model path) ---
# # # def resample_ecg(ecg_signal, target_length=520):
# # #     x_old = np.linspace(0, 1, len(ecg_signal))
# # #     x_new = np.linspace(0, 1, target_length)
# # #     f_ecg = interp1d(x_old, ecg_signal, kind='linear')
# # #     ecg_resampled = f_ecg(x_new)
# # #     return ecg_resampled

# # # def restore_org_ecg_mask(ecg_signal, mask, target_length=520):
# # #     x_old = np.linspace(0, 1, len(ecg_signal))
# # #     x_new = np.linspace(0, 1, target_length)
# # #     f_ecg = interp1d(x_old, ecg_signal, kind='linear')
# # #     ecg_resampled = f_ecg(x_new)
# # #     f_mask = interp1d(x_old, mask, kind='nearest')
# # #     mask_resampled = f_mask(x_new)
# # #     return ecg_resampled, mask_resampled.astype(int)

# # # # (find_p_t_peaks remains the same)
# # # def find_p_t_peaks(ecg, mask, boundary_margin=3, merge_distance=15):
# # #     ecg = np.array(ecg)
# # #     mask = np.array(mask)
# # #     def fix_1_2_confusions(mask):
# # #         mask = mask.copy()
# # #         i = 1
# # #         while i < len(mask) - 1:
# # #             if mask[i] in [1, 2] and mask[i - 1] == mask[i + 1] and mask[i] != mask[i - 1]:
# # #                 val_to_fill = mask[i - 1]
# # #                 start = i
# # #                 while i < len(mask) - 1 and mask[i] != val_to_fill and mask[i] in [1, 2]:
# # #                     i += 1
# # #                 mask[start:i] = val_to_fill
# # #             else:
# # #                 i += 1
# # #         return mask
    
# # #     def selective_majority_filter(mask, window_size=7):
# # #         padded = np.pad(mask, (window_size // 2,), mode='edge')
# # #         filtered = mask.copy()
# # #         for i in range(len(mask)):
# # #             window = padded[i:i + window_size]
# # #             center = mask[i]
# # #             window_mode = mode(window, keepdims=True)[0][0]
# # #             if center == 0 and window_mode in [1, 2]:
# # #                 filtered[i] = window_mode
# # #         return filtered
    
# # #     def suppress_short_regions(mask, min_length=2):
# # #         mask = mask.copy()
# # #         current_val = mask[0]
# # #         start_idx = 0
# # #         for i in range(1, len(mask)):
# # #             if mask[i] != current_val:
# # #                 if current_val in [1, 2] and (i - start_idx) < min_length:
# # #                     mask[start_idx:i] = 0
# # #                 start_idx = i
# # #                 current_val = mask[i]
# # #         if current_val in [1, 2] and (len(mask) - start_idx) < min_length:
# # #             mask[start_idx:] = 0
# # #         return mask
    
# # #     def get_peak_indices(mask_val, ecg, mask, max_one=False):
# # #         indices = []
# # #         regions = []
# # #         in_region = False
# # #         start = 0
# # #         for i in range(len(mask)):
# # #             if mask[i] == mask_val and not in_region:
# # #                 start = i
# # #                 in_region = True
# # #             elif mask[i] != mask_val and in_region:
# # #                 end = i
# # #                 regions.append((start, end))
# # #                 in_region = False
# # #         if in_region:
# # #             regions.append((start, len(mask)))
# # #         if max_one and regions:
# # #             max_len = max(end - start for start, end in regions)
# # #             longest_regions = [seg for seg in regions if (seg[1] - seg[0]) == max_len]
# # #             if len(longest_regions) > 1:
# # #                 abs_vals = [np.max(np.abs(ecg[start:end])) for start, end in longest_regions]
# # #                 chosen_region = longest_regions[np.argmax(abs_vals)]
# # #             else:
# # #                 chosen_region = longest_regions[0]
# # #             regions = [chosen_region]
# # #         for start, end in regions:
# # #             segment = ecg[start:end]
# # #             maxima = argrelextrema(segment, np.greater)[0]
# # #             inverted = False
# # #             if len(maxima) == 0:
# # #                 maxima = argrelextrema(-segment, np.greater)[0]
# # #                 inverted = True
# # #             if len(maxima) > 0:
# # #                 candidate_values = segment[maxima] if not inverted else -segment[maxima]
# # #                 best_idx = np.argmax(candidate_values)
# # #                 peak_relative = maxima[best_idx]
# # #             else:
# # #                 derivative = np.gradient(segment)
# # #                 curvature = np.abs(np.gradient(derivative))
# # #                 peak_relative = np.argmax(curvature)
# # #             peak_idx = start + peak_relative
# # #             if boundary_margin <= peak_idx < len(ecg) - boundary_margin:
# # #                 indices.append(peak_idx)
# # #         return indices
    
# # #     def merge_close_peaks(peaks, ecg, merge_distance):
# # #         if not peaks:
# # #             return []
# # #         peaks = sorted(peaks)
# # #         merged_peaks = [peaks[0]]
# # #         for idx in peaks[1:]:
# # #             last_idx = merged_peaks[-1]
# # #             if abs(idx - last_idx) < merge_distance:
# # #                 if abs(ecg[idx]) > abs(ecg[last_idx]):
# # #                     merged_peaks[-1] = idx
# # #             else:
# # #                 merged_peaks.append(idx)
# # #         return merged_peaks
    
# # #     def remove_peaks_near_other(peaks_to_filter, reference_peaks, merge_distance):
# # #         filtered = []
# # #         for p_idx in peaks_to_filter:
# # #             if all(abs(p_idx - t_idx) >= merge_distance for t_idx in reference_peaks):
# # #                 filtered.append(p_idx)
# # #         return filtered
    
# # #     def refine_peak_positions(ecg, peak_indices, window=10):
# # #         refined = []
# # #         for idx in peak_indices:
# # #             temp_seg = ecg[max(idx - 2, 0):min(idx + 2, len(ecg))]
# # #             temp_idx = idx - 2 + np.argmax(np.abs(temp_seg))
# # #             temp_max = idx - 2 + np.argmax(temp_seg)
# # #             temp_min = idx - 2 + np.argmin(temp_seg)
# # #             if idx != temp_idx and (idx != temp_max and idx != temp_min):
# # #                 start = max(idx - window, 0)
# # #                 end = min(idx + window + 1, len(ecg))
# # #                 segment = np.abs(ecg[start:end])
# # #                 maxima = argrelextrema(segment, np.greater)[0]
# # #                 inverted = False
# # #                 if len(maxima) == 0:
# # #                     maxima = argrelextrema(-segment, np.greater)[0]
# # #                     inverted = True
# # #                 if len(maxima) > 0:
# # #                     candidate_values = segment[maxima] if not inverted else -segment[maxima]
# # #                     best_idx = np.argmax(candidate_values)
# # #                     peak_relative = maxima[best_idx]
# # #                 else:
# # #                     derivative = np.gradient(segment)
# # #                     curvature = np.abs(np.gradient(derivative))
# # #                     peak_relative = np.argmax(curvature)
# # #                 peak_idx = start + peak_relative
# # #                 refined.append(peak_idx)
# # #             else:
# # #                 refined.append(idx)
# # #         return refined
   
# # #     mask = fix_1_2_confusions(mask)
# # #     mask = selective_majority_filter(mask, window_size=16)
# # #     mask = suppress_short_regions(mask, min_length=3)
# # #     t_peaks = get_peak_indices(mask_val=1, ecg=ecg, mask=mask, max_one=True)
# # #     t_peaks = refine_peak_positions(ecg, t_peaks, window=10)
# # #     t_peaks = merge_close_peaks(t_peaks, ecg, merge_distance=merge_distance)
# # #     p_peaks = get_peak_indices(mask_val=2, ecg=ecg, mask=mask, max_one=False)
# # #     p_peaks = merge_close_peaks(p_peaks, ecg, merge_distance=45)
# # #     p_peaks = refine_peak_positions(ecg, p_peaks, window=10)
# # #     p_peaks = remove_peaks_near_other(p_peaks, t_peaks, merge_distance=merge_distance)
# # #     return p_peaks, t_peaks

# # # def find_onset_offset(signal, peak_idx, smooth=True, window_size=11, polyorder=3,
# # #                       min_drop_ratio=0.2, search_window=200):
# # #     signal = np.array(signal)
# # #     signal_len = len(signal)
# # #     if smooth:
# # #         win = min(window_size, signal_len - (signal_len % 2 == 0))
# # #         signal_smooth = savgol_filter(signal, window_length=win, polyorder=polyorder)
# # #     else:
# # #         signal_smooth = signal
# # #     peak_val = signal_smooth[peak_idx]
# # #     baseline_window = min(40, signal_len // 6)
# # #     start = max(0, peak_idx - baseline_window)
# # #     end = min(signal_len, peak_idx + baseline_window)
# # #     local_baseline = np.median(signal_smooth[start:end])
# # #     drop_thresh = peak_val - (peak_val - local_baseline) * min_drop_ratio
# # #     onset_idx = peak_idx
# # #     for i in range(peak_idx, max(1, peak_idx - search_window), -1):
# # #         if signal_smooth[i] < drop_thresh:
# # #             onset_idx = i
# # #             break
# # #         if i > 1 and signal_smooth[i-1] < signal_smooth[i-2] and signal_smooth[i-1] < signal_smooth[i]:
# # #             onset_idx = i - 1
# # #             break
# # #     offset_idx = peak_idx
# # #     for i in range(peak_idx, min(signal_len - 2, peak_idx + search_window)):
# # #         if signal_smooth[i] < drop_thresh:
# # #             offset_idx = i
# # #             break
# # #         if signal_smooth[i+1] < signal_smooth[i] and signal_smooth[i+1] < signal_smooth[i+2]:
# # #             offset_idx = i + 1
# # #             break
# # #     return onset_idx, offset_idx

# # # def get_pt_peaks(ecg, r_indices, pt_model_path, use_gpu_delegate=True):
# # #     t_peaks_all, p_peaks_all, pt_peaks_all, onset, offset = [], [], [], [], []
# # #     for i in range(len(r_indices) - 1):
# # #         segment = ecg[r_indices[i]:r_indices[i+1]]
# # #         if len(segment) < 10:
# # #             continue
# # #         segment_signal = np.array(segment)
# # #         resampled_ecgs = resample_ecg(segment_signal, 520)
# # #         ecg_signal = np.array(resampled_ecgs)
# # #         ecg_signal = np.expand_dims(ecg_signal, axis=(0, -1))
# # #         predictions = predict_tflite_model(pt_model_path, ecg_signal, use_gpu_delegate=use_gpu_delegate)
# # #         predicted_labels = np.argmax(predictions, axis=-1)
# # #         _, pred_mask = restore_org_ecg_mask(
# # #             ecg_signal[0].squeeze(), predicted_labels.squeeze(), len(segment_signal)
# # #         )
# # #         p_peaks, t_peaks = find_p_t_peaks(segment_signal, pred_mask)
# # #         p_peaks = np.atleast_1d(p_peaks) + r_indices[i]
# # #         t_peaks = np.atleast_1d(t_peaks) + r_indices[i]
# # #         pt_peaks = tuple(list(t_peaks) + list(p_peaks))
# # #         p_peaks_all.extend(p_peaks)
# # #         t_peaks_all.extend(t_peaks)
# # #         pt_peaks_all.extend(pt_peaks)
# # #     return t_peaks_all, p_peaks_all, pt_peaks_all


# # # def pt_peak_detection(all_lead_data, is_lead, r_peaks, r_result_dic = None, pt_model_path=None, use_gpu_delegate=True):
# # #     result_dic = {}
# # #     for lead in all_lead_data.keys():
# # #         r_peaks = r_result_dic.get(lead)
# # #         ecg_signal = all_lead_data[lead].values.flatten()
# # #         baseline_signal = baseline_construction_200(ecg_signal, kernel_size=131)
# # #         lowpass_signal = lowpass(baseline_signal)
# # #         signal_normalized = normalize(lowpass_signal)
# # #         t_peaks, p_peaks, rr_invl_peaks = get_pt_peaks(signal_normalized, r_peaks, pt_model_path, use_gpu_delegate=use_gpu_delegate)
# # #         result_dic[lead] = {"p": p_peaks, "t": t_peaks, "comb": rr_invl_peaks}
# # #     if is_lead == '2_lead':
# # #         p_peaks = result_dic['II'].get("p")
# # #         t_peaks = result_dic['II'].get("t")
# # #         rr_invl_peaks = result_dic['II'].get("comb")

# # #     return t_peaks, p_peaks, rr_invl_peaks, result_dic
# # # # ---------------------- Plotting and other post processing functions ----------------------
# # # # def add_standard_ecg_grid(ax, duration_sec, y_min, y_max):
    
# # # #     # ---- X axis (Time) ----
# # # #     ax.set_xlim(0, duration_sec)
# # # #     ax.set_xticks(np.arange(0, duration_sec + 0.04, 0.04), minor=True)
# # # #     ax.set_xticks(np.arange(0, duration_sec + 0.2, 0.2))
# # # #     # ---- Y axis (Voltage) ----
# # # #     ax.set_ylim(y_min, y_max)
# # # #     ax.set_yticks(np.arange(y_min, y_max + 0.1, 0.1), minor=True)
# # # #     ax.set_yticks(np.arange(y_min, y_max + 0.5, 0.5))
# # # #     # ---- Aspect ratio ----
# # # #     ax.set_aspect(0.04 / 0.1) # 0.04 sec = 0.1 mV
# # # #     # ---- Grid styling ----
# # # #     ax.grid(which='minor', color='#e6e6e6', linewidth=0.6)
# # # #     ax.grid(which='major', color='#b00000', linewidth=1.2)
# # # #     # Draw background over full canvas
# # # #     ax.set_facecolor('white')

# # # # ---------------------- Plotting and other post processing functions ----------------------
# # # def hr_count(r_index, fs=250):
# # #     if len(r_index) < 2:
# # #         return 0
# # #     rr_intervals = np.diff(r_index)
# # #     if len(rr_intervals) == 0:
# # #         return 0
# # #     HR = int((len(rr_intervals) * 60000) / np.sum(rr_intervals / fs * 1000))
# # #     return HR

# # # def refine_r_peaks(signal, r_peaks, fs):

# # #     refined = []
# # #     polarity = []

# # #     window = int(0.04 * fs)  # ±40 ms around detected R

# # #     for r in r_peaks:
# # #         if r <= 0 or r >= len(signal):
# # #             continue

# # #         start = max(0, r - window)
# # #         end = min(len(signal), r + window + 1)
# # #         seg = signal[start:end]

# # #         if len(seg) == 0:
# # #             continue

# # #         # pick the true apex of QRS (largest absolute deflection)
# # #         idx = np.argmax(np.abs(seg))
# # #         true_r = start + idx

# # #         refined.append(true_r)
# # #         polarity.append(np.sign(signal[true_r]) or 1)

# # #     return np.array(refined, dtype=int), np.array(polarity, dtype=int)


# # # def detect_q_s(signal, r_peaks, r_polarity, fs):
    
# # #     q_points = []
# # #     s_points = []
# # #     window = int(0.1 * fs)  # 100 ms search window

# # #     for r, pol in zip(r_peaks, r_polarity):
# # #         # Q point: before R
# # #         start = max(0, r - window)
# # #         seg = signal[start:r+1]
# # #         if pol > 0:
# # #             q_idx = start + np.argmin(seg)  # Q = min before upright R
# # #         else:
# # #             q_idx = start + np.argmax(seg)  # Q = max before inverted R
# # #         q_points.append(q_idx)

# # #         # S point: after R
# # #         end = min(len(signal), r + window + 1)
# # #         seg = signal[r:end]
# # #         if pol > 0:
# # #             s_idx = r + np.argmin(seg)  # S = min after upright R
# # #         else:
# # #             s_idx = r + np.argmax(seg)  # S = max after inverted R
# # #         s_points.append(s_idx)

# # #     return np.array(q_points, dtype=int), np.array(s_points, dtype=int)



# # # def detect_t_wave_onset(signal, s_points, fs):
# # #     """
# # #     Detect T-wave onset AFTER ST segment.
# # #     NOT used for ST measurement.
# # #     """

# # #     signal = np.asarray(signal)
# # #     t_onsets = []

# # #     for s_idx in s_points:
# # #         j_point = int(s_idx)

# # #         search_start = j_point + int(0.08 * fs)
# # #         search_end   = min(j_point + int(0.30 * fs), len(signal) - 2)

# # #         if search_end <= search_start:
# # #             t_onsets.append(search_start)
# # #             continue

# # #         slope = np.diff(signal)
# # #         onset = search_end

# # #         for i in range(search_start, search_end - 1):
# # #             if slope[i] > 0 and slope[i+1] > 0:
# # #                 onset = i
# # #                 break

# # #         t_onsets.append(onset)

# # #     return np.array(t_onsets, dtype=int)

# # # def measure_st_segment_full(signal, r_peaks, s_points, t_onsets, fs, calibration=10.0):
# # #     import numpy as np

# # #     st_results = []
# # #     signal_len = len(signal)
# # #     beats = min(len(r_peaks), len(s_points), len(t_onsets))

# # #     for i in range(beats):
# # #         r = int(r_peaks[i])
# # #         s = int(s_points[i])
# # #         t_idx = int(t_onsets[i])

# # #         r = np.clip(r, 0, signal_len - 1)
# # #         s = np.clip(s, 0, signal_len - 1)
# # #         t_idx = np.clip(t_idx, 0, signal_len - 1)

# # #         # J-point ≈ 10 ms after S
# # #         j_point = s + int(0.01 * fs)
# # #         j_point = np.clip(j_point, 0, signal_len - 1)

# # #         # PR baseline (−200 to −120 ms before R)
# # #         pr_start = max(r - int(0.20 * fs), 0)
# # #         pr_end   = max(r - int(0.12 * fs), pr_start + 1)
# # #         pr_baseline = np.median(signal[pr_start:pr_end])

# # #         st_end = min(t_idx, signal_len - 1)
# # #         if st_end <= j_point:
# # #             st_end = j_point  # prevent negative slicing

# # #         st_mV = np.median(signal[j_point:st_end + 1]) - pr_baseline

# # #         st_results.append({
# # #             "j_idx": j_point,
# # #             "st_end": st_end,
# # #             "st_mV": st_mV,
# # #             "pr_baseline": pr_baseline
# # #         })

# # #     return st_results

# # # def plotting(
# # #     baseline_corrected_data,
# # #     save_path,
# # #     local_name,
# # #     pt_result_dic,
# # #     r_result_dic,
# # #     fs=250,
# # #     calibration=10.0
# # # ):
# # #     import os
# # #     import numpy as np
# # #     import matplotlib.pyplot as plt
# # #     import matplotlib.patheffects as path_effects

# # #     os.makedirs(save_path, exist_ok=True)
# # #     mm_per_mV = calibration
# # #     summary = []

# # #     for lead in baseline_corrected_data.columns:

# # #         dpi = 300
# # #         fig_width = 15
# # #         fig_height = 3.5

# # #         fig, ax = plt.subplots(figsize=(fig_width, fig_height), dpi=dpi)

# # #         signal = baseline_corrected_data[lead].values.astype(float)
# # #         signal_len = len(signal)
# # #         t = np.arange(signal_len) / fs

# # #         # ---------- FULL SIGNAL ----------
# # #         ax.plot(t, signal, color="black", lw=1.3, zorder=1)

# # #         # ---------- DETECTIONS ----------
# # #         r_peaks = np.asarray(r_result_dic.get(lead, []), dtype=int)
# # #         if len(r_peaks) == 0:
# # #             plt.close(fig)
# # #             continue

# # #         r_peaks, r_polarity = refine_r_peaks(signal, r_peaks, fs)
# # #         q_points, s_points = detect_q_s(signal, r_peaks, r_polarity, fs)
# # #         t_onsets = detect_t_wave_onset(signal, s_points, fs)

# # #         # ---------- MARKERS ----------
# # #         ax.scatter(t[np.clip(r_peaks, 0, signal_len - 1)], signal[np.clip(r_peaks, 0, signal_len - 1)],
# # #                    c="red", s=30, zorder=5, label="R")
# # #         ax.scatter(t[np.clip(q_points, 0, signal_len - 1)], signal[np.clip(q_points, 0, signal_len - 1)],
# # #                    c="purple", s=35, zorder=5, label="Q")
# # #         ax.scatter(t[np.clip(s_points, 0, signal_len - 1)], signal[np.clip(s_points, 0, signal_len - 1)],
# # #                    c="orange", s=35, zorder=5, label="S")
# # #         ax.scatter(t[np.clip(t_onsets, 0, signal_len - 1)], signal[np.clip(t_onsets, 0, signal_len - 1)],
# # #                    c="green", s=40, zorder=5, label="T-onset")

# # #         # ---------- ST MEASUREMENT ----------
# # #         st_records = measure_st_segment_full(signal, r_peaks, s_points, t_onsets, fs)
# # #         if not st_records:
# # #             plt.close(fig)
# # #             continue

# # #         # Global PR baseline for reference
# # #         pr_global = np.median([rec["pr_baseline"] for rec in st_records])
# # #         ax.axhline(pr_global, color="gray", ls="--", lw=1.2, label="PR baseline")

# # #         # ---------- ST SHADING & LABELS ----------
# # #         # ---------- ST SHADING & LABELS ----------
# # #         for i, rec in enumerate(st_records, start=1):

# # #             j_idx = np.clip(rec["j_idx"], 0, signal_len - 1)
# # #             st_idx = np.clip(rec["st_end"], 0, signal_len - 1)
# # #             if j_idx == st_idx:
# # #                 st_idx = min(j_idx + 1, signal_len - 1)

# # #             # ST segment samples
# # #             x_segment = t[j_idx:st_idx + 1]
# # #             y_signal_segment = signal[j_idx:st_idx + 1]

# # #             # PR baseline (mV)
# # #             y_base = rec["pr_baseline"]

# # #             # ST deviation
# # #             st_mV = rec["st_mV"]
# # #             st_mm = st_mV * mm_per_mV   # EVERYTHING DECIDED IN mm

# # #             y_top = y_base + st_mV

# # #             # Decide color & status based on mm
# # #             if abs(st_mm) > 2.0:
# # #                 color = "red"
# # #                 status = "Critical Abnormal"
# # #             elif abs(st_mm) > 0.5:
# # #                 color = "green"
# # #                 status = "Abnormal"
# # #             else:
# # #                 color = None
# # #                 status = "Normal"

# # #             # Shade ONLY if > 0.5 mm
# # #             if color is not None:
# # #                 y_shade = np.clip(
# # #                     y_signal_segment,
# # #                     min(y_base, y_top),
# # #                     max(y_base, y_top)
# # #                 )

# # #                 ax.fill_between(
# # #                     x_segment,
# # #                     y_base,
# # #                     y_shade,
# # #                     color=color,
# # #                     alpha=0.3,
# # #                     zorder=5
# # #                 )

# # #                 # Label placement
# # #                 label_offset_mV = 0.05
# # #                 label_y = y_top + label_offset_mV if st_mm > 0 else y_top - label_offset_mV
# # #                 va = "bottom" if st_mm > 0 else "top"

# # #                 ax.text(
# # #                     np.mean(x_segment),
# # #                     label_y,
# # #                     f"{st_mm:+.1f} mm",
# # #                     fontsize=9,
# # #                     fontweight="bold",
# # #                     ha="center",
# # #                     va=va,
# # #                     path_effects=[path_effects.withStroke(linewidth=2, foreground="white")]
# # #                 )

# # #             summary.append({
# # #                 "Lead": lead,
# # #                 "Beat_No": i,
# # #                 "ST_mm": round(st_mm, 2),
# # #                 "PR_baseline_mV": round(y_base, 4),
# # #                 "Status": status
# # #             })


# # #         # ---------- ECG GRID ----------
# # #         pad = 0.5
# # #         ax.set_ylim(signal.min() - pad, signal.max() + pad)
# # #         ax.set_xlim(t[0], t[-1])
# # #         # add_standard_ecg_grid(ax, t[-1], ax.get_ylim()[0], ax.get_ylim()[1])
# # #         # ---------- ECG GRID (TRUE ECG STYLE) ----------
# # #         window_sec = 10.0
# # #         ax.set_xlim(t[0], t[0] + window_sec)

# # #         # ECG standard grid
# # #         major_x = 0.2     # 1 big box = 0.2s
# # #         minor_x = 0.04    # 1 small box = 0.04s

# # #         major_y = 0.5     # 0.5 mV big box
# # #         minor_y = 0.1     # 0.1 mV small box

# # #         ax.xaxis.set_major_locator(plt.MultipleLocator(major_x))
# # #         ax.xaxis.set_minor_locator(plt.MultipleLocator(minor_x))
# # #         ax.yaxis.set_major_locator(plt.MultipleLocator(major_y))
# # #         ax.yaxis.set_minor_locator(plt.MultipleLocator(minor_y))

# # #         ax.grid(which="major", color="#f08c8c", linewidth=0.8)
# # #         ax.grid(which="minor", color="#f4d2d8", linewidth=0.5)

# # #         # ---------- TITLE & AXES ----------
# # #         hr = hr_count(r_peaks, fs)
# # #         ax.set_title(f"{local_name} — Lead {lead} — HR {hr} bpm")
# # #         ax.set_xlabel("Time (s)")
# # #         ax.set_ylabel("Amplitude (mV)")
# # #         ax.legend(loc="upper right")
# # #         plt.tight_layout(pad=0.8)
# # #         # ---------- SAVE FIG ----------
# # #         plt.savefig(os.path.join(save_path, f"{local_name}_{lead}.pdf"))
# # #         plt.close(fig)

# # #     return pd.DataFrame(summary)

# # # # ---------------------- I/O / file handling helpers ----------------------
# # # def load_and_rename_data(fn, is_lead_for):
# # #     lead_columns = {
# # #         '2_lead': ['ECG', 'II', 'Value',"'MLII'",'MLII'],
# # #         '7_lead': ['I', 'II', 'III', 'aVR', 'aVL', 'aVF', 'v5'],
# # #         '12_lead': ['I', 'II', 'III', 'aVR', 'aVL', 'aVF', 'v1', 'v2', 'v3', 'v4', 'v5', 'v6','V1','V2','V3','V4','V5','V6','ECG']
# # #     }

# # #     lead_columns_index = {
# # #         '2_lead': {0: 'II'},
# # #         '7_lead': {0: 'I', 1: 'II', 2: 'III', 3: 'aVR', 4: 'aVL', 5: 'aVF', 6: 'v5'},
# # #         '12_lead': {0: 'I', 1: 'II', 2: 'III', 3: 'aVR', 4: 'aVL', 5: 'aVF', 6: 'v1', 7: 'v2', 8: 'v3', 9: 'v4', 10: 'v5', 11: 'v6'}
# # #     }

# # #     all_lead_data = pd.read_csv(fn).fillna(0)
# # #     columns = all_lead_data.columns.tolist()
# # #     if any(str(val).isalpha() for val in all_lead_data.iloc[0, :].values):
# # #         if all(col in lead_columns['7_lead'] for col in columns):
# # #             is_lead_for = '7_lead'
# # #         elif all(col in lead_columns['12_lead'] for col in columns):
# # #             is_lead_for = '12_lead'
# # #         else:
# # #             is_lead_for = '2_lead'
# # #     else:
# # #         if len(columns) >= 12:
# # #             is_lead_for = '12_lead'
# # #         elif len(columns) >= 7:
# # #             is_lead_for = '7_lead'
# # #         else:
# # #             is_lead_for = '2_lead'

# # #     if is_lead_for == '2_lead':
# # #         available_columns = [col for col in lead_columns['2_lead'] if col in columns]
# # #         all_lead_data = attempt_column_load(fn, available_columns)
# # #     elif is_lead_for == '7_lead':
# # #         available_columns = [col for col in lead_columns['7_lead'] if col in columns]
# # #         all_lead_data = attempt_column_load(fn, available_columns)
# # #     elif is_lead_for == '12_lead':
# # #         available_columns = [col for col in lead_columns['12_lead'] if col in columns]
# # #         all_lead_data = attempt_column_load(fn, available_columns)

# # #     if all_lead_data is not None:
# # #         all_lead_data = all_lead_data.rename(columns=lead_columns_index[is_lead_for])

# # #     if is_lead_for == '2_lead':
# # #         all_lead_data.columns = ['II']

# # #     return all_lead_data, is_lead_for

# # # def attempt_column_load(fn, columns):
# # #     try:
# # #         data = pd.read_csv(fn, usecols=columns).fillna(0)
# # #         return data
# # #     except ValueError as e:
# # #         print("value Error ",e)
# # #         return None
# # #     except Exception as e:
# # #         print("Error in Loading",e)
# # #         return None

# # # def find_csv_files(root_folder):
# # #     csv_files = []
# # #     for root, _, files in os.walk(root_folder):
# # #         for file in files:
# # #             if file.lower().endswith('.csv'):
# # #                 csv_files.append(os.path.join(root, file))
# # #     return csv_files


# # # def process_single_file(
# # #     fn,
# # #     save_path,
# # #     is_lead,
# # #     r_model_path,
# # #     pt_model_path,
# # #     use_gpu_delegate=True
# # # ):
# # #     import os
# # #     import pandas as pd

# # #     local_name = os.path.splitext(os.path.basename(fn))[0]
# # #     csv_root = os.path.join(save_path, local_name)
# # #     os.makedirs(csv_root, exist_ok=True)

# # #     all_leads_data, is_lead = load_and_rename_data(fn, is_lead)
# # #     if all_leads_data is None:
# # #         return f"Failed: {local_name}"

# # #     # ---------- R PEAK DETECTION ----------
# # #     _, r_result_dic = r_peak_detection(
# # #         all_leads_data,
# # #         is_lead,
# # #         r_model_path,
# # #         use_gpu_delegate
# # #     )

# # #     # ---------- P & T PEAK DETECTION ----------
# # #     _, _, _, pt_result_dic = pt_peak_detection(
# # #         all_leads_data,
# # #         is_lead,
# # #         None,
# # #         r_result_dic,
# # #         pt_model_path,
# # #         use_gpu_delegate
# # #     )

# # #     # ---------- BASELINE CORRECTED ECG (ONCE) ----------
# # #     baseline_corrected_data = pd.DataFrame({
# # #         lead: baseline_construction_200(all_leads_data[lead].values)
# # #         for lead in all_leads_data.columns
# # #     })

# # #     summary_df = plotting(
# # #         baseline_corrected_data,
# # #         csv_root,
# # #         local_name,
# # #         pt_result_dic,
# # #         r_result_dic,
# # #         fs=250,
# # #         calibration=10.0
# # #     )

# # #     if not summary_df.empty:
# # #         summary_df.to_csv(
# # #             os.path.join(csv_root, f"{local_name}_ALL_LEADS_ST_SUMMARY.csv"),
# # #             index=False
# # #         )

# # #     merge_pdfs_in_lead_order(
# # #         csv_root,
# # #         os.path.join(csv_root, f"{local_name}_MERGED_ALL_LEADS.pdf")
# # #     )

# # #     return f"Processed: {local_name}"



# # # def merge_pdfs_in_lead_order(pdf_dir, output_pdf):
# # #     from PyPDF2 import PdfMerger
# # #     import os

# # #     LEAD_ORDER = [
# # #         "I", "II", "III",
# # #         "aVR", "aVL", "aVF",
# # #         "V1", "V2", "V3", "V4", "V5", "V6"
# # #     ]

# # #     merger = PdfMerger()

# # #     for lead in LEAD_ORDER:
# # #         for file in sorted(os.listdir(pdf_dir)):
# # #             if file.endswith(".pdf") and f"_{lead}.pdf" in file:
# # #                 merger.append(os.path.join(pdf_dir, file))
# # #                 break

# # #     merger.write(output_pdf)
# # #     merger.close()
# # # def merge_all_merged_pdfs(root_output_dir, final_pdf_path):
# # #     """
# # #     Merge all per-CSV merged ECG PDFs into one final PDF.

# # #     Expects files like:
# # #     root_output_dir/
# # #         patient1/patient1_MERGED_ALL_LEADS.pdf
# # #         patient2/patient2_MERGED_ALL_LEADS.pdf
# # #         ...

# # #     Creates:
# # #         final_pdf_path
# # #     """
# # #     import os
# # #     from PyPDF2 import PdfMerger

# # #     merger = PdfMerger()

# # #     found_any = False

# # #     for root, _, files in os.walk(root_output_dir):
# # #         for file in sorted(files):
# # #             if file.endswith("_MERGED_ALL_LEADS.pdf"):
# # #                 pdf_path = os.path.join(root, file)
# # #                 merger.append(pdf_path)
# # #                 found_any = True

# # #     if found_any:
# # #         merger.write(final_pdf_path)
# # #         merger.close()
# # #         print(f" FINAL MERGED PDF CREATED: {final_pdf_path}")
# # #     else:
# # #         merger.close()
# # #         print(" No merged PDFs found to combine.")


# # # def ecg_processing(path, save_path, is_lead, r_model_path, pt_model_path, max_workers=DESIRED_CPU_THREADS, use_gpu_delegate=True):
# # #     csv_files = find_csv_files(path)
# # #     if not csv_files:
# # #         print("No CSV files found.")
# # #         return
# # #     # Use ThreadPoolExecutor so per-thread interpreters remain in same process/threads (GPU delegate may not be picklable)
# # #     max_workers = min(max_workers, max(1, os.cpu_count() or 1))
# # #     print(f"Processing {len(csv_files)} files with {max_workers} workers (use_gpu_delegate={use_gpu_delegate})")
# # #     with ThreadPoolExecutor(max_workers=max_workers) as executor:
# # #         futures = {executor.submit(process_single_file, fn, save_path, is_lead, r_model_path, pt_model_path, use_gpu_delegate): fn for fn in csv_files}
# # #         for future in as_completed(futures):
# # #             try:
# # #                 print(future.result())
# # #             except Exception as e:
# # #                 print("Worker exception:", e)
# # #     print("All files processed successfully.")
# # # def find_pdf_files(root_folder):
# # #     pdf_files = []
# # #     for root, _, files in os.walk(root_folder):
# # #         for file in files:
# # #             if file.lower().endswith('.pdf'):
# # #                 pdf_files.append(os.path.join(root, file))
# # #     return sorted(pdf_files)
# # # def merge_pdfs(pdf_files, output_path):
# # #     merger = PdfMerger()
# # #     for pdf in pdf_files:
# # #         try:
# # #             merger.append(pdf)
# # #             print(f"Merged: {pdf}")
# # #         except Exception as e:
# # #             print(f"Failed to merge {pdf}: {e}")
# # #     merger.write(output_path)
# # #     merger.close()
# # #     print(f"\n All PDFs merged into: {output_path}")
# # # def process_single_file_wrapper(args):
# # #     """Wrapper for multiprocessing (since functions must be pickleable)."""
# # #     return process_single_file(*args)
# # # def ecg_processing(path, save_path, is_lead, r_model_path, pt_model_path,
# # #                    max_workers=DESIRED_CPU_THREADS, use_gpu_delegate=True,
# # #                    use_multiprocessing=False):
# # #     csv_files = find_csv_files(path)
# # #     if not csv_files:
# # #         print("No CSV files found.")
# # #         return
# # #     max_workers = min(max_workers, max(1, os.cpu_count() or 1))
# # #     mode = "multiprocessing" if use_multiprocessing else "threading"
# # #     print(f"Processing {len(csv_files)} files with {max_workers} workers ({mode}, use_gpu_delegate={use_gpu_delegate})")
# # #     if use_multiprocessing:
# # #         # Each process loads its own interpreters/models
# # #         with ProcessPoolExecutor(max_workers=max_workers) as executor:
# # #             futures = {
# # #                 executor.submit(process_single_file_wrapper,
# # #                                 (fn, save_path, is_lead, r_model_path, pt_model_path, use_gpu_delegate)): fn
# # #                 for fn in csv_files
# # #             }
# # #             for future in as_completed(futures):
# # #                 try:
# # #                     print(future.result())
# # #                 except Exception as e:
# # #                     print("Worker exception:", e)
# # #     else:
# # #         # Default threading (per-thread cached interpreters, GPU delegate works better here)
# # #         with ThreadPoolExecutor(max_workers=max_workers) as executor:
# # #             futures = {
# # #                 executor.submit(process_single_file, fn, save_path, is_lead, r_model_path, pt_model_path, use_gpu_delegate): fn
# # #                 for fn in csv_files
# # #             }
# # #             for future in as_completed(futures):
# # #                 try:
# # #                     print(future.result())
# # #                 except Exception as e:
# # #                     print("Worker exception:", e)
# # #     print("All files processed successfully.")

# # # def save_pdf_to_gridfs(pdf_path, metadata=None):
# # #     mongo_uri = os.getenv("MONGO_HOST")
# # #     mongo_client = MongoClient(mongo_uri)
# # #     db = mongo_client["St_Segment"]

# # #     fs = gridfs.GridFS(db)

# # #     with open(pdf_path, "rb") as f:
# # #         file_id = fs.put(
# # #             f,
# # #             filename=pdf_path.split("\\")[-1],
# # #             contentType="application/pdf",
# # #             metadata=metadata or {}
# # #         )

# # #     return str(file_id)

# # # def run_ecg_st_pipeline(
# # #     input_folder,
# # #     output_folder,
# # #     is_lead,
# # #     max_workers=4,
# # #     use_gpu_delegate=True,
# # #     use_multiprocessing=True
# # # ):
# # #     os.makedirs(output_folder, exist_ok=True)

# # #     ecg_processing(
# # #         path=input_folder,
# # #         save_path=output_folder,
# # #         is_lead=is_lead,
# # #         r_model_path = r"D:\\try3\\Scripts_Models\\Model\\rnn_model1_19_12_Unet.tflite",
# # #         pt_model_path = r"D:\\try3\\Scripts_Models\\Model\\ecg_pt_detection_LSTMGRU_v32.tflite",
# # #         max_workers=max_workers,
# # #         use_gpu_delegate=use_gpu_delegate,
# # #         use_multiprocessing=use_multiprocessing
# # #     )

# # #     final_pdf_path = os.path.join(
# # #         output_folder,
# # #         "FINAL_ALL_CSV_ALL_LEADS.pdf"
# # #     )

# # #     merge_all_merged_pdfs(
# # #         root_output_dir=output_folder,
# # #         final_pdf_path=final_pdf_path
# # #     )

# # #     return final_pdf_path
# # # -----------------------------------------22-01-2026---------------------------------
# import os
# import glob
# import pandas as pd
# import numpy as np
# import matplotlib
# matplotlib.use('Agg')
# import matplotlib.pyplot as plt
# import tensorflow as tf
# from scipy import signal
# from scipy.signal import find_peaks, argrelextrema, savgol_filter
# from scipy.stats import mode
# from scipy.interpolate import interp1d
# import warnings
# import threading
# from concurrent.futures import ThreadPoolExecutor, as_completed
# from PyPDF2 import PdfMerger
# import random
# from concurrent.futures import ThreadPoolExecutor, as_completed, ProcessPoolExecutor
# import multiprocessing
# from pymongo import MongoClient
# import gridfs
# from django.conf import settings
# from sklearn.preprocessing import MinMaxScaler

# import os
# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt
# from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
# from PyPDF2 import PdfMerger

# DESIRED_CPU_THREADS = 4  # adjust based on your CPU

# warnings.filterwarnings('ignore')
# results_lock = threading.RLock()
# # ---------------------- Server-level tuning (CPU / threading / TF / env) ----------------------
# # Set these before heavy libs use BLAS/OMP/MKL threads
# DESIRED_CPU_THREADS = 24
# os.environ['OMP_NUM_THREADS'] = str(DESIRED_CPU_THREADS)
# os.environ['OPENBLAS_NUM_THREADS'] = str(DESIRED_CPU_THREADS)
# os.environ['MKL_NUM_THREADS'] = str(DESIRED_CPU_THREADS)
# os.environ['NUMEXPR_NUM_THREADS'] = str(DESIRED_CPU_THREADS)
# os.environ['VECLIB_MAXIMUM_THREADS'] = str(DESIRED_CPU_THREADS)
# # TensorFlow GPU / thread config
# try:
#     physical_gpus = tf.config.list_physical_devices('GPU')
#     if physical_gpus:
#         # Allow memory growth so multiple processes/threads can share GPU more gracefully
#         for g in physical_gpus:
#             try:
#                 tf.config.experimental.set_memory_growth(g, True)
#             except Exception:
#                 pass
#     # Set TF threading parallelism
#     tf.config.threading.set_intra_op_parallelism_threads(DESIRED_CPU_THREADS)
#     tf.config.threading.set_inter_op_parallelism_threads(max(1, DESIRED_CPU_THREADS // 2))
# except Exception as e:
#     print("Warning: TensorFlow configuration failed:", e)
# # Thread-local storage for per-thread interpreters
# thread_local = threading.local()
# # ---------------------- TFLite interpreter utilities ----------------------
# def _load_gpu_delegate():
#     """
#     Try loading a TFLite GPU delegate. This is inherently platform-dependent.
#     We try multiple common delegate names and fall back to None.
#     """
#     try:
#         # TensorFlow's load_delegate helper
#         load_delegate = tf.lite.experimental.load_delegate
#     except Exception:
#         load_delegate = None
#     if load_delegate:
#         candidates = [
#             'libtensorflowlite_gpu_delegate.so', # linux
#             'libtensorflowlite_gpu_delegate.dylib', # mac
#             'tensorflowlite_gpu_delegate.dll', # windows (rare)
#         ]
#         for cand in candidates:
#             try:
#                 delegate = load_delegate(cand)
#                 print(f"Loaded GPU delegate: {cand}")
#                 return delegate
#             except Exception:
#                 continue
#     # If we reach here, GPU delegate wasn't loaded
#     return None
# GPU_DELEGATE = _load_gpu_delegate()


# def get_tflite_interpreter_for_thread(model_path: str, use_gpu_delegate=True):
   
#     if not hasattr(thread_local, "interpreters"):
#         thread_local.interpreters = {}
#     key = f"{model_path}_gpu" if use_gpu_delegate and GPU_DELEGATE else model_path
#     if key in thread_local.interpreters:
#         return thread_local.interpreters[key]
#     # Create interpreter
#     try:
#         if use_gpu_delegate and GPU_DELEGATE:
#             interpreter = tf.lite.Interpreter(model_path=model_path, experimental_delegates=[GPU_DELEGATE])
#             print(f"[Thread {threading.get_ident()}] Created GPU interpreter for {os.path.basename(model_path)}")
#         else:
#             interpreter = tf.lite.Interpreter(model_path=model_path)
#             print(f"[Thread {threading.get_ident()}] Created CPU interpreter for {os.path.basename(model_path)}")
#     except Exception as e:
#         # Fallback to CPU interpreter if GPU delegate fails
#         print(f"Interpreter creation failed for {model_path} with GPU delegate: {e}. Falling back to CPU.")
#         interpreter = tf.lite.Interpreter(model_path=model_path)
#     interpreter.allocate_tensors()
#     input_details = interpreter.get_input_details()
#     output_details = interpreter.get_output_details()
#     thread_local.interpreters[key] = (interpreter, input_details, output_details)
#     return thread_local.interpreters[key]

# def predict_tflite_model(model_path: str, input_data, use_gpu_delegate=True):
   
#     # Acquire a lock around interpreter invocation to be safe for device resources, but interpreters are per-thread so contention is low.
#     interpreter, input_details, output_details = get_tflite_interpreter_for_thread(model_path, use_gpu_delegate=use_gpu_delegate)
#     with results_lock:
#         input_data = input_data.astype(np.float32)
#         interpreter.set_tensor(input_details[0]['index'], input_data)
#         interpreter.invoke()
#         output_data = interpreter.get_tensor(output_details[0]['index'])
#     return output_data.squeeze()
# # ---------------------- Your existing functions (kept mostly unchanged) ----------------------
# def lowpass(file, cutoff=0.3):
#     b, a = signal.butter(3, cutoff, btype='lowpass', analog=False)
#     low_passed = signal.filtfilt(b, a, file)
#     return low_passed

# def baseline_construction_200(ecg_signal, kernel_size=131):
#     s_corrected = signal.detrend(ecg_signal)
#     baseline_corrected = s_corrected - signal.medfilt(s_corrected, kernel_size)
#     return baseline_corrected

# def normalize(signal):
#     return (signal - np.mean(signal)) / np.std(signal)

# def refined_non_max_suppression(ecg_signal, valid_indices, suppression_radius=40):
#     if len(valid_indices) == 0:
#         return []
#     sorted_indices = sorted(valid_indices, reverse=True)
#     selected = []
#     occupied = np.zeros(len(ecg_signal), dtype=bool)
#     for idx in sorted_indices:
#         if not occupied[idx]:
#             left = max(0, idx - suppression_radius)
#             right = min(len(ecg_signal), idx + suppression_radius + 1)
#             # Mark region as occupied
#             occupied[left:right] = True
#             selected.append(idx)
#     return sorted(selected)

# def check_model_r(ecg_data, r_model_path, use_gpu_delegate=True):
#     totaldata = len(ecg_data)
#     i = 0
#     step = totaldata if totaldata < 1000 else 1000
#     r_peaks = []
#     temp_list = []
#     df_ecg_signal = ecg_data.tolist()
#     while i < totaldata:
#         if i != 0 and totaldata > 1000:
#             i -= 200
#         ecg_signal = ecg_data[i:i + step]
#         signal_len = len(ecg_signal)
#         pad_len = 1000 - signal_len
#         padded_signal = np.pad(ecg_signal, (0, pad_len), mode='constant', constant_values=0)
#         raw_array = np.expand_dims(padded_signal, axis=0).astype(np.float32)[..., np.newaxis]
#         preds = predict_tflite_model(r_model_path, raw_array, use_gpu_delegate=use_gpu_delegate)
#         preds = preds[:signal_len]
#         r_peak_prob = preds[:, 1]
#         peak_indices, _ = find_peaks(r_peak_prob, height=0.2, distance=20)
#         for j in peak_indices:
#             if 0 <= i+j < len(df_ecg_signal):
#                 temp_list.append(i + j)
#         i += step
#     rpeak = sorted(set(temp_list))
#     r_peaks = refined_non_max_suppression(df_ecg_signal, rpeak)
#     return r_peaks

# def r_peak_detection(all_lead_data, is_lead, r_model_path, use_gpu_delegate=True):
#     r_peaks = []
#     result_dic = {}
#     for lead in all_lead_data.keys():
#         ecg_signal = all_lead_data[lead].values.flatten()
#         baseline_signal = baseline_construction_200(ecg_signal, kernel_size=131)
#         lowpass_signal = lowpass(baseline_signal)
#         signal_normalized = normalize(lowpass_signal)
#         r_peaks = check_model_r(signal_normalized, r_model_path, use_gpu_delegate=use_gpu_delegate)
#         result_dic[lead] = r_peaks
#     if is_lead == '2_lead':
#         r_peaks = result_dic['II']
#     return r_peaks, result_dic

# # --- P/T detection functions (unchanged except calling predict_tflite_model with model path) ---
# def resample_ecg(ecg_signal, target_length=520):
#     x_old = np.linspace(0, 1, len(ecg_signal))
#     x_new = np.linspace(0, 1, target_length)
#     f_ecg = interp1d(x_old, ecg_signal, kind='linear')
#     ecg_resampled = f_ecg(x_new)
#     return ecg_resampled

# def restore_org_ecg_mask(ecg_signal, mask, target_length=520):
#     x_old = np.linspace(0, 1, len(ecg_signal))
#     x_new = np.linspace(0, 1, target_length)
#     f_ecg = interp1d(x_old, ecg_signal, kind='linear')
#     ecg_resampled = f_ecg(x_new)
#     f_mask = interp1d(x_old, mask, kind='nearest')
#     mask_resampled = f_mask(x_new)
#     return ecg_resampled, mask_resampled.astype(int)

# # (find_p_t_peaks remains the same)
# def find_p_t_peaks(ecg, mask, boundary_margin=3, merge_distance=15):
#     ecg = np.array(ecg)
#     mask = np.array(mask)
#     def fix_1_2_confusions(mask):
#         mask = mask.copy()
#         i = 1
#         while i < len(mask) - 1:
#             if mask[i] in [1, 2] and mask[i - 1] == mask[i + 1] and mask[i] != mask[i - 1]:
#                 val_to_fill = mask[i - 1]
#                 start = i
#                 while i < len(mask) - 1 and mask[i] != val_to_fill and mask[i] in [1, 2]:
#                     i += 1
#                 mask[start:i] = val_to_fill
#             else:
#                 i += 1
#         return mask
    
#     def selective_majority_filter(mask, window_size=7):
#         padded = np.pad(mask, (window_size // 2,), mode='edge')
#         filtered = mask.copy()
#         for i in range(len(mask)):
#             window = padded[i:i + window_size]
#             center = mask[i]
#             window_mode = mode(window, keepdims=True)[0][0]
#             if center == 0 and window_mode in [1, 2]:
#                 filtered[i] = window_mode
#         return filtered
    
#     def suppress_short_regions(mask, min_length=2):
#         mask = mask.copy()
#         current_val = mask[0]
#         start_idx = 0
#         for i in range(1, len(mask)):
#             if mask[i] != current_val:
#                 if current_val in [1, 2] and (i - start_idx) < min_length:
#                     mask[start_idx:i] = 0
#                 start_idx = i
#                 current_val = mask[i]
#         if current_val in [1, 2] and (len(mask) - start_idx) < min_length:
#             mask[start_idx:] = 0
#         return mask
    
#     def get_peak_indices(mask_val, ecg, mask, max_one=False):
#         indices = []
#         regions = []
#         in_region = False
#         start = 0
#         for i in range(len(mask)):
#             if mask[i] == mask_val and not in_region:
#                 start = i
#                 in_region = True
#             elif mask[i] != mask_val and in_region:
#                 end = i
#                 regions.append((start, end))
#                 in_region = False
#         if in_region:
#             regions.append((start, len(mask)))
#         if max_one and regions:
#             max_len = max(end - start for start, end in regions)
#             longest_regions = [seg for seg in regions if (seg[1] - seg[0]) == max_len]
#             if len(longest_regions) > 1:
#                 abs_vals = [np.max(np.abs(ecg[start:end])) for start, end in longest_regions]
#                 chosen_region = longest_regions[np.argmax(abs_vals)]
#             else:
#                 chosen_region = longest_regions[0]
#             regions = [chosen_region]
#         for start, end in regions:
#             segment = ecg[start:end]
#             maxima = argrelextrema(segment, np.greater)[0]
#             inverted = False
#             if len(maxima) == 0:
#                 maxima = argrelextrema(-segment, np.greater)[0]
#                 inverted = True
#             if len(maxima) > 0:
#                 candidate_values = segment[maxima] if not inverted else -segment[maxima]
#                 best_idx = np.argmax(candidate_values)
#                 peak_relative = maxima[best_idx]
#             else:
#                 derivative = np.gradient(segment)
#                 curvature = np.abs(np.gradient(derivative))
#                 peak_relative = np.argmax(curvature)
#             peak_idx = start + peak_relative
#             if boundary_margin <= peak_idx < len(ecg) - boundary_margin:
#                 indices.append(peak_idx)
#         return indices
    
#     def merge_close_peaks(peaks, ecg, merge_distance):
#         if not peaks:
#             return []
#         peaks = sorted(peaks)
#         merged_peaks = [peaks[0]]
#         for idx in peaks[1:]:
#             last_idx = merged_peaks[-1]
#             if abs(idx - last_idx) < merge_distance:
#                 if abs(ecg[idx]) > abs(ecg[last_idx]):
#                     merged_peaks[-1] = idx
#             else:
#                 merged_peaks.append(idx)
#         return merged_peaks
    
#     def remove_peaks_near_other(peaks_to_filter, reference_peaks, merge_distance):
#         filtered = []
#         for p_idx in peaks_to_filter:
#             if all(abs(p_idx - t_idx) >= merge_distance for t_idx in reference_peaks):
#                 filtered.append(p_idx)
#         return filtered
    
#     def refine_peak_positions(ecg, peak_indices, window=10):
#         refined = []
#         for idx in peak_indices:
#             temp_seg = ecg[max(idx - 2, 0):min(idx + 2, len(ecg))]
#             temp_idx = idx - 2 + np.argmax(np.abs(temp_seg))
#             temp_max = idx - 2 + np.argmax(temp_seg)
#             temp_min = idx - 2 + np.argmin(temp_seg)
#             if idx != temp_idx and (idx != temp_max and idx != temp_min):
#                 start = max(idx - window, 0)
#                 end = min(idx + window + 1, len(ecg))
#                 segment = np.abs(ecg[start:end])
#                 maxima = argrelextrema(segment, np.greater)[0]
#                 inverted = False
#                 if len(maxima) == 0:
#                     maxima = argrelextrema(-segment, np.greater)[0]
#                     inverted = True
#                 if len(maxima) > 0:
#                     candidate_values = segment[maxima] if not inverted else -segment[maxima]
#                     best_idx = np.argmax(candidate_values)
#                     peak_relative = maxima[best_idx]
#                 else:
#                     derivative = np.gradient(segment)
#                     curvature = np.abs(np.gradient(derivative))
#                     peak_relative = np.argmax(curvature)
#                 peak_idx = start + peak_relative
#                 refined.append(peak_idx)
#             else:
#                 refined.append(idx)
#         return refined
   
#     mask = fix_1_2_confusions(mask)
#     mask = selective_majority_filter(mask, window_size=16)
#     mask = suppress_short_regions(mask, min_length=3)
#     t_peaks = get_peak_indices(mask_val=1, ecg=ecg, mask=mask, max_one=True)
#     t_peaks = refine_peak_positions(ecg, t_peaks, window=10)
#     t_peaks = merge_close_peaks(t_peaks, ecg, merge_distance=merge_distance)
#     p_peaks = get_peak_indices(mask_val=2, ecg=ecg, mask=mask, max_one=False)
#     p_peaks = merge_close_peaks(p_peaks, ecg, merge_distance=45)
#     p_peaks = refine_peak_positions(ecg, p_peaks, window=10)
#     p_peaks = remove_peaks_near_other(p_peaks, t_peaks, merge_distance=merge_distance)
#     return p_peaks, t_peaks

# def find_onset_offset(signal, peak_idx, smooth=True, window_size=11, polyorder=3,
#                       min_drop_ratio=0.2, search_window=200):
#     signal = np.array(signal)
#     signal_len = len(signal)
#     if smooth:
#         win = min(window_size, signal_len - (signal_len % 2 == 0))
#         signal_smooth = savgol_filter(signal, window_length=win, polyorder=polyorder)
#     else:
#         signal_smooth = signal
#     peak_val = signal_smooth[peak_idx]
#     baseline_window = min(40, signal_len // 6)
#     start = max(0, peak_idx - baseline_window)
#     end = min(signal_len, peak_idx + baseline_window)
#     local_baseline = np.median(signal_smooth[start:end])
#     drop_thresh = peak_val - (peak_val - local_baseline) * min_drop_ratio
#     onset_idx = peak_idx
#     for i in range(peak_idx, max(1, peak_idx - search_window), -1):
#         if signal_smooth[i] < drop_thresh:
#             onset_idx = i
#             break
#         if i > 1 and signal_smooth[i-1] < signal_smooth[i-2] and signal_smooth[i-1] < signal_smooth[i]:
#             onset_idx = i - 1
#             break
#     offset_idx = peak_idx
#     for i in range(peak_idx, min(signal_len - 2, peak_idx + search_window)):
#         if signal_smooth[i] < drop_thresh:
#             offset_idx = i
#             break
#         if signal_smooth[i+1] < signal_smooth[i] and signal_smooth[i+1] < signal_smooth[i+2]:
#             offset_idx = i + 1
#             break
#     return onset_idx, offset_idx

# def get_pt_peaks(ecg, r_indices, pt_model_path, use_gpu_delegate=True):
#     t_peaks_all, p_peaks_all, pt_peaks_all, onset, offset = [], [], [], [], []
#     for i in range(len(r_indices) - 1):
#         segment = ecg[r_indices[i]:r_indices[i+1]]
#         if len(segment) < 10:
#             continue
#         segment_signal = np.array(segment)
#         resampled_ecgs = resample_ecg(segment_signal, 520)
#         ecg_signal = np.array(resampled_ecgs)
#         ecg_signal = np.expand_dims(ecg_signal, axis=(0, -1))
#         predictions = predict_tflite_model(pt_model_path, ecg_signal, use_gpu_delegate=use_gpu_delegate)
#         predicted_labels = np.argmax(predictions, axis=-1)
#         _, pred_mask = restore_org_ecg_mask(
#             ecg_signal[0].squeeze(), predicted_labels.squeeze(), len(segment_signal)
#         )
#         p_peaks, t_peaks = find_p_t_peaks(segment_signal, pred_mask)
#         p_peaks = np.atleast_1d(p_peaks) + r_indices[i]
#         t_peaks = np.atleast_1d(t_peaks) + r_indices[i]
#         pt_peaks = tuple(list(t_peaks) + list(p_peaks))
#         p_peaks_all.extend(p_peaks)
#         t_peaks_all.extend(t_peaks)
#         pt_peaks_all.extend(pt_peaks)
#     return t_peaks_all, p_peaks_all, pt_peaks_all


# def pt_peak_detection(all_lead_data, is_lead, r_peaks, r_result_dic = None, pt_model_path=None, use_gpu_delegate=True):
#     result_dic = {}
#     for lead in all_lead_data.keys():
#         r_peaks = r_result_dic.get(lead)
#         ecg_signal = all_lead_data[lead].values.flatten()
#         baseline_signal = baseline_construction_200(ecg_signal, kernel_size=131)
#         lowpass_signal = lowpass(baseline_signal)
#         signal_normalized = normalize(lowpass_signal)
#         t_peaks, p_peaks, rr_invl_peaks = get_pt_peaks(signal_normalized, r_peaks, pt_model_path, use_gpu_delegate=use_gpu_delegate)
#         result_dic[lead] = {"p": p_peaks, "t": t_peaks, "comb": rr_invl_peaks}
#     if is_lead == '2_lead':
#         p_peaks = result_dic['II'].get("p")
#         t_peaks = result_dic['II'].get("t")
#         rr_invl_peaks = result_dic['II'].get("comb")

#     return t_peaks, p_peaks, rr_invl_peaks, result_dic

# # ---------------------- Plotting and other post processing functions ----------------------
# def hr_count(r_index, fs=200):
#     if len(r_index) < 2:
#         return 0
#     rr_intervals = np.diff(r_index)
#     if len(rr_intervals) == 0:
#         return 0
#     HR = int((len(rr_intervals) * 60000) / np.sum(rr_intervals / fs * 1000))
#     return HR

# def refine_r_peaks(signal, r_peaks, fs):

#     refined = []
#     polarity = []

#     window = int(0.04 * fs)  # ±40 ms around detected R

#     for r in r_peaks:
#         if r <= 0 or r >= len(signal):
#             continue

#         start = max(0, r - window)
#         end = min(len(signal), r + window + 1)
#         seg = signal[start:end]

#         if len(seg) == 0:
#             continue

#         # pick the true apex of QRS (largest absolute deflection)
#         idx = np.argmax(np.abs(seg))
#         true_r = start + idx

#         refined.append(true_r)
#         polarity.append(np.sign(signal[true_r]) or 1)

#     return np.array(refined, dtype=int), np.array(polarity, dtype=int)


# def detect_q_s(signal, r_peaks, polarity, fs):
#     q_points = []
#     s_points = []

#     for r, pol in zip(r_peaks, polarity):

#         # Q search: 40 ms before R
#         q_start = max(0, r - int(0.04 * fs))
#         q_seg = signal[q_start:r]

#         if len(q_seg) == 0:
#             q_points.append(None)
#         else:
#             q_idx = np.argmin(q_seg) if pol > 0 else np.argmax(q_seg)
#             q_points.append(q_start + q_idx)

#         # S search: 60 ms after R
#         s_end = min(len(signal), r + int(0.06 * fs))
#         s_seg = signal[r:s_end]

#         if len(s_seg) == 0:
#             s_points.append(None)
#         else:
#             s_idx = np.argmin(s_seg) if pol > 0 else np.argmax(s_seg)
#             s_points.append(r + s_idx)

#     return q_points, s_points



# def detect_t_wave_onset(signal, s_points, fs):

#     t_onsets = []

#     for s in s_points:
#         if s is None:
#             t_onsets.append(None)
#             continue

#         start = s + int(0.08 * fs)
#         end = min(len(signal), s + int(0.4 * fs))

#         if start >= end:
#             t_onsets.append(None)
#             continue

#         seg = signal[start:end]
#         slope = np.diff(seg)

#         idx = np.argmax(np.abs(slope))
#         t_onsets.append(start + idx)

#     return t_onsets


# def plot_st_segment(
#     all_leads_data,
#     st_records_dict,
#     r_peaks_dict,
#     q_points_dict,
#     s_points_dict,
#     t_points_dict,
#     fs=200,
#     save_path=".",
#     fname_prefix="ecg",
#     x_range_sec=10.0,
#     mm_per_mV=10.0
# ):
#     SMALL_BOX_SEC = 0.04
#     SMALL_BOX_MV  = 0.1
#     Y_MV_RANGE    = 4.0
#     Y_SQUARES     = int(Y_MV_RANGE / SMALL_BOX_MV)

#     os.makedirs(save_path, exist_ok=True)


#     for lead, signal_mV in all_leads_data.items():
#         signal_mV = np.asarray(signal_mV, float)
#         N = len(signal_mV)

#         r_peaks = np.asarray(r_peaks_dict.get(lead, []), dtype=int)
#         q_pts   = q_points_dict.get(lead, [])
#         s_pts   = s_points_dict.get(lead, [])
#         t_pts   = t_points_dict.get(lead, [])

#         # HARD RULE: no R peaks → skip entire lead
#         if len(r_peaks) < 2:
#             continue

#         # R exists but dependent waves missing → skip lead
#         if len(s_pts) == 0 or len(t_pts) == 0:
#             continue

#         st_records_dict[lead] = []

#         sig_min = signal_mV.min()
#         sig_max = signal_mV.max()
#         signal_scaled_mV = (signal_mV - sig_min) / (sig_max - sig_min) * Y_MV_RANGE
#         signal_sq = signal_scaled_mV / SMALL_BOX_MV

#         samples_per_chunk = int(fs * x_range_sec)
#         total_chunks = int(np.ceil(N / samples_per_chunk))

#         for c in range(total_chunks):
#             start = c * samples_per_chunk
#             end   = min(start + samples_per_chunk, N)

#             fig, ax = plt.subplots(figsize=(15, 4))
#             x_max_sq = int(x_range_sec / SMALL_BOX_SEC)
#             ax.set_xlim(0, x_max_sq)
#             ax.set_ylim(0, Y_SQUARES)
#             ax.set_aspect("equal")

#             # --- GRID ---
#             for x in range(x_max_sq + 1):
#                 ax.axvline(x, color="#f4d2d8", lw=0.4)
#             for x in range(0, x_max_sq + 1, 5):
#                 ax.axvline(x, color="#f58181", lw=1.0)
#             for y in range(Y_SQUARES + 1):
#                 ax.axhline(y, color="#f4d2d8", lw=0.4)
#             for y in range(0, Y_SQUARES + 1, 5):
#                 ax.axhline(y, color="#f58181", lw=1.0)

#             # --- ISOELECTRIC LINE (TP SEGMENT) ---
#             iso_vals = []
#             for i in range(len(r_peaks) - 1):
#                 if i >= len(t_pts):
#                     continue

#                 t_end = t_pts[i]
#                 nxt_r = r_peaks[i + 1]

#                 tp_start = t_end + int(0.02 * fs)
#                 tp_end   = min(tp_start + int(0.04 * fs), nxt_r)

#                 if tp_end > tp_start:
#                     iso_vals.append(signal_mV[tp_start:tp_end].mean())

#             # No valid baseline → skip chunk
#             if not iso_vals:
#                 continue

#             iso_mV = np.median(iso_vals)
#             iso_sq = (iso_mV - sig_min) / (sig_max - sig_min) * Y_MV_RANGE / SMALL_BOX_MV
#             ax.hlines(iso_sq, 0, x_max_sq, color="blue", lw=1.2, ls="--")

#             # --- ECG TRACE ---
#             xs = np.arange(start, end)
#             x_plot = (xs - start) / fs / SMALL_BOX_SEC
#             ax.plot(x_plot, signal_sq[start:end], color="black", lw=1.2)

#             # --- Q/R/S/T MARKERS (only valid because R exists) ---
#             def mark(idx, color):
#                 if idx is None or idx < start or idx >= end:
#                     return
#                 x = (idx - start) / fs / SMALL_BOX_SEC
#                 y = signal_sq[idx]
#                 ax.plot(x, y, "o", color=color, ms=4)

#             for i, r in enumerate(r_peaks):
#                 mark(r, "red")
#                 if i < len(q_pts): mark(q_pts[i], "purple")
#                 if i < len(s_pts): mark(s_pts[i], "orange")
#                 if i < len(t_pts): mark(t_pts[i], "green")

#             # --- ST SEGMENT SHADING ---
#             for i, r in enumerate(r_peaks[:-1]):
#                 if i >= len(s_pts) or i >= len(t_pts):
#                     continue

#                 s = s_pts[i]
#                 t_on = t_pts[i]

#                 if s is None or t_on is None:
#                     continue

#                 j = s + int(0.02 * fs)  # J + 20 ms
#                 if j < start or t_on >= end:
#                     continue

#                 y_st = signal_sq[j:t_on]
#                 st_sq = np.mean(y_st - iso_sq)
#                 st_mm = st_sq

#                 fill_color = "red" if abs(st_mm) > 2 else "green"

#                 rect_width_sq = int(0.2 / SMALL_BOX_SEC)
#                 x_rect_start = (j - start) / fs / SMALL_BOX_SEC

#                 ax.fill_between(
#                     [x_rect_start, x_rect_start + rect_width_sq],
#                     iso_sq,
#                     iso_sq + st_sq,
#                     color=fill_color,
#                     alpha=0.35
#                 )

#                 y_label = iso_sq + st_sq + (1 if st_sq > 0 else -1)
#                 ax.text(
#                     x_rect_start + rect_width_sq / 2,
#                     y_label,
#                     f"{st_mm:+.1f} mm",
#                     ha="center",
#                     fontsize=9,
#                     fontweight="bold",
#                     color=fill_color
#                 )

#                 st_records_dict[lead].append({
#                     "r_idx": r,
#                     "j_idx": j,
#                     "pr_baseline_mV": iso_mV,
#                     "st_mV": st_sq * SMALL_BOX_MV,
#                     "st_mm": st_mm
#                 })

#             # --- AXES ---
#             ax.set_xticks(np.arange(0, x_max_sq + 1, 25))
#             ax.set_xticklabels((np.arange(0, x_max_sq + 1, 25) * SMALL_BOX_SEC).round(2))
#             ax.set_xlabel("Time (seconds)")

#             ax.set_yticks(np.arange(0, Y_SQUARES + 1, 10))
#             ax.set_yticklabels((np.arange(0, Y_SQUARES + 1, 10) * SMALL_BOX_MV).round(1))
#             ax.set_ylabel("Amplitude (mV)")

#             plt.tight_layout(pad=0.5)
#             plt.savefig(
#                 os.path.join(save_path, f"{fname_prefix}_{lead}_chunk_{c+1}.pdf"),
#                 dpi=300,
#                 bbox_inches="tight"
#             )
#             plt.close(fig)

# # ---------------------- I/O / file handling helpers ----------------------
# def load_and_rename_data(fn, is_lead_for):
#     lead_columns = {
#         '2_lead': ['ECG', 'II', 'Value',"'MLII'",'MLII'],
#         '7_lead': ['I', 'II', 'III', 'aVR', 'aVL', 'aVF', 'v5'],
#         '12_lead': ['I', 'II', 'III', 'aVR', 'aVL', 'aVF', 'v1', 'v2', 'v3', 'v4', 'v5', 'v6','V1','V2','V3','V4','V5','V6','ECG']
#     }

#     lead_columns_index = {
#         '2_lead': {0: 'II'},
#         '7_lead': {0: 'I', 1: 'II', 2: 'III', 3: 'aVR', 4: 'aVL', 5: 'aVF', 6: 'v5'},
#         '12_lead': {0: 'I', 1: 'II', 2: 'III', 3: 'aVR', 4: 'aVL', 5: 'aVF', 6: 'v1', 7: 'v2', 8: 'v3', 9: 'v4', 10: 'v5', 11: 'v6'}
#     }

#     all_lead_data = pd.read_csv(fn).fillna(0)
#     columns = all_lead_data.columns.tolist()
#     if any(str(val).isalpha() for val in all_lead_data.iloc[0, :].values):
#         if all(col in lead_columns['7_lead'] for col in columns):
#             is_lead_for = '7_lead'
#         elif all(col in lead_columns['12_lead'] for col in columns):
#             is_lead_for = '12_lead'
#         else:
#             is_lead_for = '2_lead'
#     else:
#         if len(columns) >= 12:
#             is_lead_for = '12_lead'
#         elif len(columns) >= 7:
#             is_lead_for = '7_lead'
#         else:
#             is_lead_for = '2_lead'

#     if is_lead_for == '2_lead':
#         available_columns = [col for col in lead_columns['2_lead'] if col in columns]
#         all_lead_data = attempt_column_load(fn, available_columns)
#     elif is_lead_for == '7_lead':
#         available_columns = [col for col in lead_columns['7_lead'] if col in columns]
#         all_lead_data = attempt_column_load(fn, available_columns)
#     elif is_lead_for == '12_lead':
#         available_columns = [col for col in lead_columns['12_lead'] if col in columns]
#         all_lead_data = attempt_column_load(fn, available_columns)

#     if all_lead_data is not None:
#         all_lead_data = all_lead_data.rename(columns=lead_columns_index[is_lead_for])

#     if is_lead_for == '2_lead':
#         all_lead_data.columns = ['II']

#     return all_lead_data, is_lead_for

# def attempt_column_load(fn, columns):
#     try:
#         data = pd.read_csv(fn, usecols=columns).fillna(0)
#         return data
#     except ValueError as e:
#         print("value Error ",e)
#         return None
#     except Exception as e:
#         print("Error in Loading",e)
#         return None

# def find_csv_files(root_folder):
#     csv_files = []
#     for root, _, files in os.walk(root_folder):
#         for file in files:
#             if file.lower().endswith('.csv'):
#                 csv_files.append(os.path.join(root, file))
#     return csv_files

# # ------------------------- MERGE LEAD CHUNKS -------------------------
# def merge_lead_chunks_to_pdf(save_path, fname_prefix, lead):
#     merger = PdfMerger()
#     pdf_files = sorted([f for f in os.listdir(save_path)
#                         if f.startswith(f"{fname_prefix}_{lead}_chunk_") and f.endswith(".pdf")])
#     if not pdf_files:
#         return None
#     for f in pdf_files:
#         merger.append(os.path.join(save_path, f))
#     merged_pdf = os.path.join(save_path, f"{fname_prefix}_{lead}_MERGED.pdf")
#     merger.write(merged_pdf)
#     merger.close()
#     return merged_pdf


# def process_single_file(
#     fn,
#     save_path,
#     is_lead,
#     r_model_path,
#     pt_model_path,
#     use_gpu_delegate=True,
#     fs=200,
#     mm_per_mV=10.0,
#     x_range_sec=10.0
# ):
#     local_name = os.path.splitext(os.path.basename(fn))[0]
#     csv_root = os.path.join(save_path, local_name)
#     os.makedirs(csv_root, exist_ok=True)

#     all_leads_data, is_lead = load_and_rename_data(fn, is_lead)
#     if all_leads_data is None:
#         return f"Failed: {local_name}"

#     # --- R PEAK DETECTION ---
#     _, r_result_dic = r_peak_detection(
#         all_leads_data, is_lead, r_model_path, use_gpu_delegate
#     )

#     # --- P/T MODEL (kept, but T will be gated later) ---
#     _, _, _, pt_result_dic = pt_peak_detection(
#         all_leads_data, is_lead, None, r_result_dic, pt_model_path, use_gpu_delegate
#     )

#     st_records_dict = {}
#     q_points_dict = {}
#     s_points_dict = {}
#     t_points_dict = {}
#     r_peaks_dict = {}

#     # --- PER LEAD PROCESSING ---
#     for lead in all_leads_data.columns:
#         detection_signal = all_leads_data[lead].values.astype(float)

#         r_peaks = np.asarray(r_result_dic.get(lead, []), dtype=int)
#         r_peaks_dict[lead] = r_peaks

#         # DEFAULT: NOTHING COMPUTED
#         q_points, s_points, t_points = [], [], []

#         # HARD RULE: no R → no Q/S/T
#         if len(r_peaks) >= 2:
#             # Q & S depend ONLY on R
#             q_points, s_points = detect_q_s(
#                 detection_signal,
#                 r_peaks,
#                 np.sign(detection_signal[r_peaks]),
#                 fs
#             )

#             # T onset depends on S
#             if len(s_points) > 0:
#                 t_points = detect_t_wave_onset(
#                     detection_signal,
#                     s_points,
#                     fs
#                 )

#         # Store results (may be empty → plotting will skip)
#         q_points_dict[lead] = q_points
#         s_points_dict[lead] = s_points
#         t_points_dict[lead] = t_points

#     # --- PLOTTING + ST COMPUTATION (already guarded) ---
#     plot_st_segment(
#         all_leads_data,
#         st_records_dict,
#         r_peaks_dict,
#         q_points_dict,
#         s_points_dict,
#         t_points_dict,
#         fs=fs,
#         save_path=csv_root,
#         fname_prefix=local_name,
#         x_range_sec=x_range_sec,
#         mm_per_mV=mm_per_mV
#     )

#     # --- MERGE PDFs ---
#     merged_pdfs = []
#     for lead in all_leads_data.columns:
#         merged_pdf = merge_lead_chunks_to_pdf(csv_root, local_name, lead)
#         if merged_pdf:
#             merged_pdfs.append(merged_pdf)

#     if merged_pdfs:
#         final_pdf = os.path.join(csv_root, f"{local_name}_MERGED_ALL_LEADS.pdf")
#         merger = PdfMerger()
#         for pdf in merged_pdfs:
#             merger.append(pdf)
#         merger.write(final_pdf)
#         merger.close()

#     # --- ST SUMMARY CSV ---
#     st_summary_records = []
#     for lead, recs in st_records_dict.items():
#         for rec in recs:
#             st_summary_records.append({
#                 "file": local_name,
#                 "lead": lead,
#                 "r_index": rec["r_idx"],
#                 "pr_baseline_mV": rec["pr_baseline_mV"],
#                 "st_mV": rec["st_mV"],
#                 "st_mm": rec["st_mm"]
#             })

#     if st_summary_records:
#         summary_df = pd.DataFrame(st_summary_records)
#         summary_df.to_csv(
#             os.path.join(csv_root, f"{local_name}_ALL_LEADS_ST_SUMMARY.csv"),
#             index=False
#         )

#     return f"Processed: {local_name}"

# def merge_all_merged_pdfs(root_output_dir, final_pdf_path):
#     merger = PdfMerger()
#     found_any = False
#     for root, _, files in os.walk(root_output_dir):
#         for file in sorted(files):
#             if file.endswith("_MERGED_ALL_LEADS.pdf"):
#                 pdf_path = os.path.join(root, file)
#                 merger.append(pdf_path)
#                 found_any = True
#     if found_any:
#         merger.write(final_pdf_path)
#         merger.close()
#         print(f"FINAL MERGED PDF CREATED: {final_pdf_path}")
#     else:
#         merger.close()
#         print("No merged PDFs found to combine.")


# def ecg_processing(
#     path,
#     save_path,
#     is_lead,
#     r_model_path,
#     pt_model_path,
#     max_workers=DESIRED_CPU_THREADS,
#     use_gpu_delegate=True,
#     use_multiprocessing=False
# ):
#     csv_files = find_csv_files(path)
#     if not csv_files:
#         print("No CSV files found.")
#         return

#     max_workers = min(max_workers, max(1, os.cpu_count() or 1))
#     mode = "multiprocessing" if use_multiprocessing else "threading"
#     print(f"🔹 Processing {len(csv_files)} files with {max_workers} workers ({mode})")

#     if use_multiprocessing:
#         with ProcessPoolExecutor(max_workers=max_workers) as executor:
#             futures = {executor.submit(process_single_file, fn, save_path, is_lead, r_model_path, pt_model_path, use_gpu_delegate): fn for fn in csv_files}
#             for future in as_completed(futures):
#                 try:
#                     print(future.result())
#                 except Exception as e:
#                     print("Worker exception:", e)
#     else:
#         with ThreadPoolExecutor(max_workers=max_workers) as executor:
#             futures = {executor.submit(process_single_file, fn, save_path, is_lead, r_model_path, pt_model_path, use_gpu_delegate): fn for fn in csv_files}
#             for future in as_completed(futures):
#                 try:
#                     print(future.result())
#                 except Exception as e:
#                     print("Worker exception:", e)

#     final_pdf_path = os.path.join(save_path, "FINAL_ALL_CSV_ALL_LEADS.pdf")
#     merge_all_merged_pdfs(save_path, final_pdf_path)
#     print("\nECG 2-Lead ST Segment Processing Completed Successfully")

 
# def save_pdf_to_gridfs(pdf_path, metadata=None):
#     mongo_uri = os.getenv("MONGO_HOST")
#     mongo_client = MongoClient(mongo_uri)
#     db = mongo_client["St_Segment"]

#     fs = gridfs.GridFS(db)

#     with open(pdf_path, "rb") as f:
#         file_id = fs.put(
#             f,
#             filename=pdf_path.split("\\")[-1],
#             contentType="application/pdf",
#             metadata=metadata or {}
#         )

#     return str(file_id)

# def run_ecg_st_pipeline(
#     input_folder,
#     output_folder,
#     is_lead,
#     max_workers=4,
#     use_gpu_delegate=True,
#     use_multiprocessing=True
# ):
#     os.makedirs(output_folder, exist_ok=True)

#     ecg_processing(
#         path=input_folder,
#         save_path=output_folder,
#         is_lead=is_lead,
#         r_model_path = r"D:\\try3\\Scripts_Models\\Model\\rnn_model1_19_12_Unet.tflite",
#         pt_model_path = r"D:\\try3\\Scripts_Models\\Model\\ecg_pt_detection_LSTMGRU_v32.tflite",
#         max_workers=max_workers,
#         use_gpu_delegate=use_gpu_delegate,
#         use_multiprocessing=use_multiprocessing
#     )

#     final_pdf_path = os.path.join(
#         output_folder,
#         "FINAL_ALL_CSV_ALL_LEADS.pdf"
#     )

#     merge_all_merged_pdfs(
#         root_output_dir=output_folder,
#         final_pdf_path=final_pdf_path
#     )

#     return final_pdf_path
# -------------------------------------------------------test=--------------------------------
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
        r_model_path = r"D:\\try3\\Scripts_Models\\Model\\rnn_model1_19_12_Unet.tflite",
        pt_model_path = r"D:\\try3\\Scripts_Models\\Model\\ecg_pt_detection_LSTMGRU_v32.tflite",
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
