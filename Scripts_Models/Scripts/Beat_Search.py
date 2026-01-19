import os
import gc
import threading
import warnings

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from scipy import signal
from scipy.signal import find_peaks, medfilt
import tensorflow as tf

from Beat_Search.db_image_store import save_image_to_db

warnings.filterwarnings("ignore")

FS = 200
TFLITE_LOCK = threading.Lock()

# ---------------- DL R-PEAK MODEL ----------------
try:
   R_PEAK_INTERPRETER = tf.lite.Interpreter(
       model_path=r"/home/system/ecgdatabank_copy/Scripts_Models/Model/rnn_model1_21_11_Unet.tflite"
   )
   R_PEAK_INTERPRETER.allocate_tensors()
   INPUT_DETAILS = R_PEAK_INTERPRETER.get_input_details()
   OUTPUT_DETAILS = R_PEAK_INTERPRETER.get_output_details()
except Exception as e:
   R_PEAK_INTERPRETER = None


# ---------------- PREPROCESS ----------------
def lowpass(ecg, cutoff=30.0, fs=FS):
   ecg = np.asarray(ecg)
   if len(ecg) < 13:
       return ecg
   nyq = 0.5 * fs
   b, a = signal.butter(3, cutoff / nyq, btype="low")
   try:
       return signal.filtfilt(b, a, ecg)
   except:
       return signal.lfilter(b, a, ecg)


def remove_baseline_wander(ecg, kernel=151):
   ecg = np.asarray(ecg)
   if len(ecg) < 13:
       return ecg
   if kernel > len(ecg):
       kernel = max(3, len(ecg)//4*4+1)
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
def predict_r_peaks_dl(ecg_signal, window=1000, step=800):
   if R_PEAK_INTERPRETER is None:
       return []

   sig_len = len(ecg_signal)
   peaks = []
   i = 0

   while i < sig_len:
       seg = ecg_signal[i:i + window]
       if len(seg) < window:
           seg = np.pad(seg, (0, window - len(seg)))

       input_data = np.expand_dims(np.array(seg, dtype=np.float32), axis=(0, 2))

       with TFLITE_LOCK:
           R_PEAK_INTERPRETER.set_tensor(INPUT_DETAILS[0]['index'], input_data)
           R_PEAK_INTERPRETER.invoke()
           output = R_PEAK_INTERPRETER.get_tensor(OUTPUT_DETAILS[0]['index'])[0]

       r_prob = output[:len(seg), 1] if output.ndim == 2 else output[:len(seg)]
       local_peaks, _ = find_peaks(r_prob, height=0.3, distance=int(0.4 * FS))
       peaks.extend((local_peaks + i).tolist())
       i += step

   return sorted(set(peaks))


def refine_r_peaks(peaks, ecg, min_dist=80):
   refined = []
   last = -min_dist
   for p in sorted(peaks):
       if p - last >= min_dist:
           l, r = max(0, p-10), min(len(ecg), p+11)
           best = l + np.argmax(np.abs(ecg[l:r]))
           refined.append(int(best))
           last = best
   return refined

def detect_q_s(ecg, r_peaks, ecg_threshold=0.05):
   q_points, s_points = [], []

   for r in r_peaks:
       r = int(r)
       if r < 0 or r >= len(ecg):
           continue
       if abs(ecg[r]) < ecg_threshold:
           continue

       d = int(0.08 * FS)
       start = max(0, r - d)
       end = min(len(ecg), r + d)

       seg_q = ecg[start:r] if r > start else np.array([ecg[r]])
       q = start + np.argmin(seg_q) if ecg[r] > 0 else start + np.argmax(seg_q)

       s_rel = np.argmin(ecg[r:end]) if ecg[r] > 0 else np.argmax(ecg[r:end])
       s = r + s_rel

       q_points.append(int(q))
       s_points.append(int(s))

   return q_points, s_points

# ---------------- ENGINE ----------------
class ECGBatchEngine:

   def __init__(self, folder, reference_beat, reference_idx, reference_raw_full,
                reference_r, reference_q, reference_s):

       self.folder = folder
       self.reference_beat = normalize(np.array(reference_beat, dtype=np.float32))
       self.reference_raw_full = np.array(reference_raw_full, dtype=np.float32)

       self.reference_idx = reference_idx
       self.reference_r = list(reference_r) if reference_r else []

       # AUTO detect R if missing (matches Tkinter behavior)
       if not self.reference_r:
           r = np.argmax(np.abs(self.reference_beat))
           self.reference_r = [r]

   def is_similar_beat(self, sig_f, start, beat_len):
       end = start + beat_len

       if start < 0 or end > len(sig_f):
           return False

       win = sig_f[start:end]
       if len(win) != len(self.reference_beat):
           return False

       win = normalize(win)

       corr = np.dot(self.reference_beat, win) / len(win)
       return corr > 0.55

   def process_file(self, file_path):
       df = pd.read_csv(file_path)
       base = os.path.basename(file_path).replace(".csv", "")
       out_dir = os.path.join(self.folder, base)
       os.makedirs(out_dir, exist_ok=True)

       beat_len = len(self.reference_beat)
       rows = []

       for lead in df.columns:
           if not np.issubdtype(df[lead].dtype, np.number):
               continue

           sig_raw = df[lead].values.astype(np.float32)
           sig_f = normalize(lowpass(remove_baseline_wander(sig_raw)))

           r_peaks = refine_r_peaks(predict_r_peaks_dl(sig_f), sig_f)
           q_points, s_points = detect_q_s(sig_f, r_peaks)
           match_starts = []
           pre = self.reference_r[0]

           for r in r_peaks:
               start = r - pre
               if self.is_similar_beat(sig_f, start, beat_len):
                   match_starts.append(start)

           if not match_starts:
               continue

           # -------- PLOT --------
           fig, axes = plt.subplots(1, 2, figsize=(14, 4))

           # ----- LEFT: Reference ECG -----
           ref_full = self.reference_raw_full
           ref_start = self.reference_idx[0]
           ref_end = self.reference_idx[1]

           t_ref = np.arange(len(ref_full)) / FS
           axes[0].plot(t_ref, ref_full, lw=0.8)
           axes[0].axvspan(ref_start/FS, ref_end/FS, color="orange", alpha=0.25)

           axes[0].set_title(
               f"Reference ECG\n"
               f"Index: {ref_start}_{ref_end} "
               f"({ref_start/FS:.2f}s_{ref_end/FS:.2f}s)"
           )
           axes[0].set_xlabel("Time (s)")
           axes[0].set_ylabel("Amplitude")

           # ----- RIGHT: Full signal with matches -----
           t = np.arange(len(sig_raw)) / FS
           axes[1].plot(t, sig_raw, lw=0.8)
           # ---- PQRST markers ----
           if r_peaks:
               axes[1].scatter(np.array(r_peaks)/FS,
                               sig_raw[r_peaks],
                               c="red", s=12, label="R")

           if q_points:
               axes[1].scatter(np.array(q_points)/FS,
                               sig_raw[q_points],
                               c="blue", s=10, label="Q")

           if s_points:
               axes[1].scatter(np.array(s_points)/FS,
                               sig_raw[s_points],
                           c="green", s=10, label="S")
           for s in match_starts:
               axes[1].axvspan(
                   s/FS,
                   (s + beat_len)/FS,
                   color="#cfe9ff",
                   alpha=0.35
               )

           min_idx = min(match_starts)
           max_idx = max(s + beat_len for s in match_starts)

           axes[1].set_title(
               f"{base} | {lead} | {len(match_starts)} matches\n"
               f"Index: {min_idx}_{max_idx} "
               f"({min_idx/FS:.2f}s_{max_idx/FS:.2f}s)"
           )
           axes[1].set_xlabel("Time (s)")
           axes[1].set_ylabel("Amplitude")

           plt.tight_layout()

           output_png = os.path.join(out_dir, f"{base}_{lead}_raw_matches.png")
           fig.savefig(output_png, dpi=150)
           plt.close(fig)
           save_image_to_db(
               image_path=output_png,
               batch_id=os.path.basename(self.folder),
               csv_name=base,
               lead=lead,
               meta={"matches": len(match_starts)}
           )

           os.remove(output_png)

           for s in match_starts:
               rows.append({"Lead": lead, "Start": s, "End": s + beat_len})

       if rows:
           pd.DataFrame(rows).to_csv(os.path.join(out_dir, f"{base}_summary.csv"), index=False)

       gc.collect()


def run_ecg_batch(folder, reference_beat, reference_idx, reference_raw_full,
                 reference_r, reference_q, reference_s):

   engine = ECGBatchEngine(
       folder, reference_beat, reference_idx,
       reference_raw_full, reference_r, reference_q, reference_s
   )

   csvs = [os.path.join(folder, f) for f in os.listdir(folder) if f.endswith(".csv")]

   for c in csvs:
       engine.process_file(c)

   open(os.path.join(folder, "__DONE__"), "w").write("completed")

   for c in csvs:
       if os.path.exists(c):
           os.remove(c)


