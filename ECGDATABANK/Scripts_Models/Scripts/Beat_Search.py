import os
import gc
import threading
import warnings

import numpy as np
import pandas as pd

from scipy import signal
from scipy.signal import find_peaks, medfilt
import tensorflow as tf
import shutil
warnings.filterwarnings("ignore")
from Beat_Search.db_image_store import create_csv_document,save_lead_result
FS = 200
TFLITE_LOCK = threading.Lock()
DTW_THRESHOLD = 0.45      # start loose, then tune
CORR_THRESHOLD = 0.60
 
# ---------------- DL R-PEAK MODEL ----------------
try:
   R_PEAK_INTERPRETER = tf.lite.Interpreter(
       model_path=r"D:\try3\Scripts_Models\Model\rnn_model1_21_11_Unet.tflite"
   )
   R_PEAK_INTERPRETER.allocate_tensors()
   INPUT_DETAILS = R_PEAK_INTERPRETER.get_input_details()
   OUTPUT_DETAILS = R_PEAK_INTERPRETER.get_output_details()
except Exception as e:
   R_PEAK_INTERPRETER = None

def lowpass(ecg, cutoff=30.0, fs=FS):
   ecg = np.asarray(ecg)
   if len(ecg) < 13:
       return ecg
   nyq = 0.5 * fs
   b, a = signal.butter(3, cutoff / nyq, btype="low")
   try:
       return signal.filtfilt(b, a, ecg)
   except ValueError:
       return signal.lfilter(b, a, ecg)


def remove_baseline_wander(ecg, kernel=151):
   ecg = np.asarray(ecg)
   if len(ecg) < 13:
       return ecg
   if kernel > len(ecg):
       kernel = max(3, len(ecg) // 4 * 4 + 1)
   if kernel % 2 == 0:
       kernel += 1
   try:
       return ecg - medfilt(ecg, kernel_size=kernel)
   except:
       baseline = np.convolve(ecg, np.ones(kernel)/kernel, mode='same')
       return ecg - baseline


def normalize(ecg):
   return (ecg - np.mean(ecg)) / (np.std(ecg) + 1e-8)


# ---------------- R PEAK ----------------
def predict_r_peaks_dl(ecg_signal, window=1000, step=800, fs=FS):
    if R_PEAK_INTERPRETER is None:
       return []

    sig_len = len(ecg_signal)
    peaks = []
    i = 0

    while i < sig_len:
        seg = ecg_signal[i:i + window]
        if len(seg) < window:
           seg = np.pad(seg, (0, window - len(seg)), 'constant')

        input_data = np.expand_dims(np.array(seg, dtype=np.float32), axis=(0, 2))

        with TFLITE_LOCK:
            R_PEAK_INTERPRETER.set_tensor(INPUT_DETAILS[0]['index'], input_data)
            R_PEAK_INTERPRETER.invoke()
            output = R_PEAK_INTERPRETER.get_tensor(OUTPUT_DETAILS[0]['index'])[0]
        if output.ndim == 2:
            r_prob = output[:len(seg), 1]
        elif output.ndim == 1:
            r_prob = output[:len(seg)]
        else:
            raise ValueError(f"Unexpected model output shape: {output.shape}")       
        local_peaks, _ = find_peaks(r_prob, height=0.3, distance=int(0.4 * FS))
        peaks.extend((local_peaks + i).tolist())
        i += step

    return sorted(set(peaks))


def refine_r_peaks(peaks, ecg, min_dist=80):
    if not peaks:
        return []
    if len(peaks) == 1:
        p = int(peaks[0])
        l, r = max(0, p-10), min(len(ecg), p+11)
        best = l + int(np.argmax(np.abs(ecg[l:r])))
        return [int(best)]
    refined = []
    last = -min_dist
    for p in sorted(peaks):
        p = int(p)
        if p - last >= min_dist:
            l, r = max(0, p-10), min(len(ecg), p+11)
            best = l + np.argmax(np.abs(ecg[l:r]))
            refined.append(int(best))
            last = best
    return refined

def detect_q_s(ecg, r_peaks, ecg_threshold=0.05):
    q_points, s_points, qrs_durations = [], [], []

    for r in r_peaks:
       r = int(r)
       if r < 0 or r >= len(ecg):
           continue
       if abs(ecg[r]) < ecg_threshold:
           continue

       d = int(0.08 * FS)
       start = max(0, r - d)
       seg_q = ecg[start:r] if r > start else np.array([ecg[r]])
       q = start + int(np.argmin(seg_q)) if ecg[r] > 0 else start + int(np.argmax(seg_q))
       end = min(len(ecg), r + d)
       s_rel = int(np.argmin(ecg[r:end])) if ecg[r] > 0 else int(np.argmax(ecg[r:end]))
       s_idx = r + s_rel
       q_points.append(int(q))
       s_points.append(int(s_idx))
       qrs_durations.append((s_idx - q) * 1000.0 / FS)
    return q_points, s_points, qrs_durations

# def multi_shape_dist(p, q):
#     p = np.asarray(p)
#     q = np.asarray(q)
#     return np.linalg.norm(p - q)
def normalize_lead_columns(df):
    lead_aliases = {
        "ECG": "II",
        "ECG1": "II",
        "ECG2": "II",
        "MLII": "II",
        "LEADII": "II",
        "LEAD_II": "II",
        "CH1": "II",
        "CHANNEL1": "II",
        "SIGNAL": "II"
    }

    new_columns = {}

    for col in df.columns:
        col_clean = col.strip().upper()
        if col_clean in lead_aliases:
            new_columns[col] = lead_aliases[col_clean]

    if new_columns:
        df.rename(columns=new_columns, inplace=True)

    return df
# ---------------- ENGINE ----------------
class ECGBatchEngine:
    def __init__(self, folder, reference_beat, reference_idx, reference_raw_full,
                reference_r, reference_q, reference_s,terminology):

       self.folder = folder
       self.terminology = terminology
       self.reference_beat = normalize(np.array(reference_beat, dtype=np.float32))
       self.reference_raw_full = np.array(reference_raw_full, dtype=np.float32)

       self.reference_idx = reference_idx
       self.reference_r = list(reference_r) if reference_r else []

       # AUTO detect R if missing (matches Tkinter behavior)
       if not self.reference_r:
           r = np.argmax(np.abs(self.reference_beat))
           self.reference_r = [r]

#    def is_similar_beat(self, sig_f, start, beat_len):
#        end = start + beat_len

#        if start < 0 or end > len(sig_f):
#            return False

#        win = sig_f[start:end]
#        if len(win) != len(self.reference_beat):
#            return False

#        win = normalize(win)

#        corr = np.dot(self.reference_beat, win) / len(win)
#        return corr > 0.55
    def is_similar_beat(self, multi_sig_f, sig_f, start, beat_len):
        end = start + beat_len
        if end > len(sig_f):
            return False, None, None
 
        # ---------------- Reference alignment ----------------
        ref_r = self.reference_r[0]
        ref_offset = ref_r - self.reference_idx[0]
 
        # ---------------- Find R peak in window ----------------
        r_candidates = [r for r in self.current_r_peaks if start <= r <= end]
        if not r_candidates:
            return False, None, None
 
        r = min(r_candidates, key=lambda x: abs((x - start) - ref_offset))
 
        # ---------------- Polarity gate ----------------
        ref_pol = np.sign(self.reference_sig[ref_r])
        cand_pol = np.sign(sig_f[r])
        if ref_pol == 0 or cand_pol == 0 or ref_pol != cand_pol:
            return False, None, None
 
        # ---------------- Align window ----------------
        s = r - ref_offset
        e = s + beat_len
        if s < 0 or e > len(sig_f):
            return False, None, None
 
        ref = self.reference_beat.copy()
        win = sig_f[s:e].copy()
 
        # ---------------- Normalize ----------------
        ref = (ref - np.mean(ref)) / (np.std(ref) + 1e-8)
        win = (win - np.mean(win)) / (np.std(win) + 1e-8)
 
        # =====================================================
        # QRS WINDOWS — DERIVED FROM beat_len (NO fs NEEDED)
        # =====================================================
        r0 = ref_offset
 
        # core QRS ≈ 10% of beat window
        qrs_core = max(1, int(0.10 * beat_len))
        a = max(0, r0 - qrs_core)
        b = min(len(ref), r0 + qrs_core)
 
        ref_core = ref[a:b]
        win_core = win[a:b]
 
        core_corr = np.dot(ref_core, win_core) / len(ref_core)
 
        # small shift tolerance (±3% of beat)
        shift = max(1, int(0.03 * beat_len))
        for sh in range(-shift, shift + 1):
            aa = a + sh
            bb = b + sh
            if aa < 0 or bb > len(win):
                continue
            c = np.dot(ref_core, win[aa:bb]) / len(ref_core)
            core_corr = max(core_corr, c)
 
        # =====================================================
        # EXTENDED QRS (LOOSER)
        # =====================================================
        qrs_ext = max(1, int(0.20 * beat_len))
        a2 = max(0, r0 - qrs_ext)
        b2 = min(len(ref), r0 + qrs_ext)
 
        ref_ext = ref[a2:b2]
        win_ext = win[a2:b2]
 
        ext_corr = np.dot(ref_ext, win_ext) / len(ref_ext)
 
        # =====================================================
        # AMPLITUDE CONSISTENCY (FORGIVING)
        # =====================================================
        ref_amp = np.max(np.abs(ref_core)) + 1e-8
        win_amp = np.max(np.abs(win_core)) + 1e-8
        amp_ratio = win_amp / ref_amp
 
        amp_score = 1.0 - min(abs(amp_ratio - 1.0) / 1.5, 1.0)
 
        # =====================================================
        # FINAL SCORE — ~35% MORPHOLOGY LOSS OK
        # =====================================================
        score = (
            0.45 * core_corr +
            0.35 * ext_corr +
            0.20 * amp_score
        )
 
        if score < 0.55:
            return False, None, score
 
        return True, None, score
    
    def process_file(self, file_path):
        df = pd.read_csv(file_path)
        df.columns = df.columns.str.strip().str.upper()

        # Normalize ECG column names (ECG → II, MLII → II, etc.)
        df = normalize_lead_columns(df)

        base = os.path.basename(file_path).replace(".csv", "")

        # -------------------------------
        # Detect lead type
        # -------------------------------
        cols = set(df.columns)

        if len(cols.intersection({"I","II","III","AVR","AVL","AVF","V1","V2","V3","V4","V5","V6"})) >= 10:
            lead_type = 12

        elif len(cols.intersection({"I","II","V5"})) >= 2:
            lead_type = 7

        elif "II" in cols:
            lead_type = 2

        else:
            print(f"Skipping {base}: unsupported leads {cols}")
            return

        beat_len = len(self.reference_beat)

        # Generate hex ONCE per CSV (like old behavior)
        hex_payload = leads_to_hex_from_df(df, lead_type)

        # Save CSV document ONCE
        create_csv_document(
            batch_id=os.path.basename(self.folder),
            csv_name=base,
            terminology=self.terminology,
            hex_payload=hex_payload
        )

        # -------------------------------
        # Per-lead processing
        # -------------------------------
        for lead in df.columns:
            if not np.issubdtype(df[lead].dtype, np.number):
                continue

            sig_raw = df[lead].values.astype(np.float32)
            if len(sig_raw) < beat_len + 10:
                continue

            sig_f = normalize(lowpass(remove_baseline_wander(sig_raw)))

            r_peaks = refine_r_peaks(predict_r_peaks_dl(sig_f), sig_f)
            q_points, s_points, _ = detect_q_s(sig_f, r_peaks)

            if not r_peaks:
                continue

            match_starts = []
            pre = self.reference_r[0]

            for r in r_peaks:
                start = r - pre
                if start < 0 or start + beat_len > len(sig_f):
                    continue

                win = sig_f[start:start+beat_len]
                corr = np.mean(self.reference_beat * win)

                if corr > 0.55:
                    match_starts.append(start)

            if not match_starts:
                continue

            matches_ranges = [
                {"start": int(s), "end": int(s + beat_len)}
                for s in match_starts
            ]

            result_payload = {
                "csv": base,
                "lead": lead,
                "signal": sig_raw.tolist(),
                "fs": FS,
                "r_peaks": list(map(int, r_peaks)),
                "q_points": list(map(int, q_points)),
                "s_points": list(map(int, s_points)),
                "matches": matches_ranges,
                "reference": {
                    "start": int(self.reference_idx[0]),
                    "end": int(self.reference_idx[1])
                },
                "meta": {
                    "matches": len(match_starts)
                }
            }

            # THIS IS WHAT YOU WERE MISSING
            # db_payload = dict(result_payload)
            # db_payload.pop("signal", None)

            save_lead_result(
                batch_id=os.path.basename(self.folder),
                csv_name=base,
                lead=lead,
                terminology=self.terminology,
                payload=result_payload
            )

            yield result_payload


        gc.collect()
     
def encode_signal_to_hex(voltage_list):
    arr = np.array(voltage_list, dtype=np.float32)
    arr = np.nan_to_num(arr)
    return arr.tobytes().hex().upper()


def leads_to_hex_from_df(df, lead_type):
    df.columns = df.columns.str.strip().str.upper()

    if lead_type == 2:
        mapping = {"II": "data"}

    elif lead_type == 7:
        mapping = {
            "II": "data",
            "I": "data1",
            "V5": "data5"
        }

    elif lead_type == 12:
        mapping = {
            "II": "data",
            "I": "data1",
            "V5": "data5",
            "V1": "vOne",
            "V2": "vTwo",
            "V3": "vThree",
            "V4": "vFour",
            "V6": "vSix"
        }
    else:
        raise ValueError("Invalid lead type")

    result = {}
    for csv_col, db_key in mapping.items():
        if csv_col not in df.columns:
            raise ValueError(f"Missing column: {csv_col}")
        result[db_key] = encode_signal_to_hex(df[csv_col].tolist())

    return result

def run_ecg_batch(folder, reference_beat, reference_idx, reference_raw_full,
                 reference_r, reference_q, reference_s,terminology):

    engine = ECGBatchEngine(
       folder, reference_beat, reference_idx,
       reference_raw_full, reference_r, reference_q, reference_s,terminology
    )

    csvs = [os.path.join(folder, f) for f in os.listdir(folder) if f.endswith(".csv")]

    for c in csvs:
        for result in engine.process_file(c):
            yield result   # sends to frontend instantly

    # finished
    yield {"done": True}
    shutil.rmtree(folder, ignore_errors=True)
    