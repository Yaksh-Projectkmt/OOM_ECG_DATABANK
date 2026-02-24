import os
import sys
import gc
import threading
import warnings

import numpy as np
import pandas as pd

import matplotlib

# Select backend BEFORE importing pyplot
if "--batch" in sys.argv:
    matplotlib.use("Agg")    # off-screen for batch processing
else:
    matplotlib.use("TkAgg")  # GUI mode with Tk window

import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk

from scipy import signal
from scipy.signal import find_peaks, medfilt
from fastdtw import fastdtw
from scipy.spatial.distance import euclidean
import tensorflow as tf
import tkinter as tk
from tkinter import ttk, filedialog, messagebox

warnings.filterwarnings("ignore")

# ---------------- GLOBAL SETTINGS ----------------
FS = 200
DTW_THRESHOLD = 0.45      # start loose, then tune
CORR_THRESHOLD = 0.60     # morphology filter

TFLITE_LOCK = threading.Lock()

# ---------------- DL R-PEAK MODEL ----------------
try:
    R_PEAK_INTERPRETER = tf.lite.Interpreter(
        model_path=r"C:\Users\Admin\OneDrive - KALI MEDTECH PRIVATE LIMITED\KMT\rnn_model1_16_02_26_final.tflite"
    )
    R_PEAK_INTERPRETER.allocate_tensors()
    INPUT_DETAILS = R_PEAK_INTERPRETER.get_input_details()
    OUTPUT_DETAILS = R_PEAK_INTERPRETER.get_output_details()
    print("DL R-peak model loaded!")
except Exception as e:
    print(f"Warning: DL model not loaded: {e}")
    R_PEAK_INTERPRETER = None

# ---------------- PREPROCESSING ----------------
def lowpass(ecg, cutoff=30.0, fs=FS):
    ecg = np.asarray(ecg)
    if len(ecg) < 13:  # Strict minimum for filtfilt (padlen=12)
        return ecg
    nyq = 0.5 * fs
    b, a = signal.butter(3, cutoff / nyq, btype="low")
    try:
        return signal.filtfilt(b, a, ecg)
    except ValueError:
        # Fallback: single-pass lfilter for very short signals
        return signal.lfilter(b, a, ecg)

def remove_baseline_wander(ecg, kernel=151):
    ecg = np.asarray(ecg)
    if len(ecg) < 13:
        return ecg
    if kernel > len(ecg):
        kernel = max(3, len(ecg) // 4 * 4 + 1)  # Ensure odd, smaller kernel
    if kernel % 2 == 0:
        kernel += 1
    try:
        return ecg - medfilt(ecg, kernel_size=kernel)
    except:
        kernel = min(kernel, len(ecg))
        if kernel % 2 == 0:
            kernel += 1
        baseline = np.convolve(ecg, np.ones(kernel)/kernel, mode='same')
        return ecg - baseline

def normalize(ecg):
    ecg = np.asarray(ecg)
    return (ecg - np.mean(ecg)) / (np.std(ecg) + 1e-8)

# ---------------- DL R-PEAK DETECTION ----------------
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
        local_peaks, _ = find_peaks(r_prob, height=0.3, distance=int(0.4 * fs))
        peaks.extend((local_peaks + i).tolist())
        i += step
    return sorted(list(set(peaks)))

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
            best = l + int(np.argmax(np.abs(ecg[l:r])))
            refined.append(int(best))
            last = best
    return refined

# ---------------- Q / S DETECTION ----------------
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

# ------------- shape-based multilead distance -------------
def multi_shape_dist(p, q):
    p = np.asarray(p)
    q = np.asarray(q)
    return np.linalg.norm(p - q)

class ECGPatternGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("ECG Pattern Search – Hybrid")
        self.root.geometry("1600x900")

        self.folder = None
        self.csvs = []
        self.selected_file = None
        self.selected_lead = None

        self.reference_sig = None
        self.reference_raw = None
        self.reference_q = []
        self.reference_r = []
        self.reference_s = []
        self.reference_idx = None
        self.reference_beat = None
        self.reference_raw_full = None
        self.reference_beat_multi = None
        self.reference_time = None

        # 🔴 REQUIRED — per-lead R peaks used by matcher
        self.current_r_peaks = []

        # batch / viewer state
        self.result_items = []
        self.result_index = 0
        self.stop_batch_flag = False   

        self._build_controls()
        self._build_plot()

    
    def _build_controls(self):
        panel = ttk.Frame(self.root)
        panel.pack(side=tk.LEFT, fill=tk.Y, padx=5, pady=5)

        ttk.Button(panel, text="Select Folder", command=self.select_folder).pack(fill=tk.X, pady=3)
        ttk.Button(panel, text="Select File", command=self.select_file).pack(fill=tk.X, pady=3)
        ttk.Button(panel, text="Load Reference", command=self.load_reference_from_file).pack(fill=tk.X, pady=3)
        ttk.Button(panel, text="Start Batch Search", command=self.start_batch).pack(fill=tk.X, pady=3)
        ttk.Button(panel, text="Stop Batch", command=self.stop_batch_run).pack(fill=tk.X, pady=3)

        ttk.Button(panel, text="Load Results", command=self.build_results_index).pack(fill=tk.X, pady=3)
        ttk.Button(panel, text="Previous Plot", command=self.prev_result).pack(fill=tk.X, pady=3)
        ttk.Button(panel, text="Next Plot", command=self.next_result).pack(fill=tk.X, pady=3)
        ttk.Button(panel, text="Reset Reference", command=self.reset_reference).pack(fill=tk.X, pady=3)

        ttk.Label(panel, text="Lead").pack(pady=(10, 0))
        self.lead_combo = ttk.Combobox(panel, state="readonly")
        self.lead_combo.pack(fill=tk.X)
        self.lead_combo.bind("<<ComboboxSelected>>", self.change_lead)

        self.info_label = ttk.Label(panel, text="")
        self.info_label.pack(fill=tk.X, pady=5)

    # -------- plot area --------
    def _build_plot(self):
        plot_frame = ttk.Frame(self.root)
        plot_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        self.fig, (self.ax_ref, self.ax_match) = plt.subplots(1, 2, figsize=(12, 5))
        
        self.canvas = FigureCanvasTkAgg(self.fig, plot_frame)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        self.toolbar = NavigationToolbar2Tk(self.canvas, plot_frame)
        self.toolbar.update()
        self.toolbar.pack(side=tk.TOP, fill=tk.X)

    # -------- folder / file / lead --------
    def select_folder(self):
        self.folder = filedialog.askdirectory()
        if not self.folder:
            return
        self.csvs = [f for f in os.listdir(self.folder) if f.lower().endswith(".csv")]
        if not self.csvs:
            messagebox.showerror("Error", "No CSV files found in folder")
            return
        messagebox.showinfo("Folder Loaded", f"{len(self.csvs)} CSV files found.\nNow select a file.")

        # clear previous results when a new folder is selected
        self.result_items = []
        self.result_index = 0
        self.info_label.config(text=f"{len(self.csvs)} CSV files found. Now select a file.")


    def select_file(self):
        if not self.csvs:
            messagebox.showerror("Error", "Select folder first")
            return
        file = filedialog.askopenfilename(
            initialdir=self.folder,
            filetypes=[("CSV files", "*.csv")],
            title="Select CSV file"
        )
        if not file:
            return
        self.selected_file = file
        self.load_leads_from_file(file)

    def load_reference_from_file(self):
        if not self.selected_file:
            messagebox.showerror("Error", "Select file first")
            return
        self.load_leads_from_file(self.selected_file)

    def load_leads_from_file(self, file_path):
        df = pd.read_csv(file_path)
        leads = [c for c in df.columns if np.issubdtype(df[c].dtype, np.number)]
        if not leads:
            messagebox.showerror("Error", "No numeric leads found in file")
            return
        self.lead_combo["values"] = leads
        self.lead_combo.current(0)
        self.selected_lead = leads[0]
        self.plot_lead(file_path, self.selected_lead)

    def change_lead(self, event):
        if not self.selected_file:
            return
        self.selected_lead = self.lead_combo.get()
        self.plot_lead(self.selected_file, self.selected_lead)

    # -------- plot selected lead, QRS, reference selection --------
    def plot_lead(self, csv_path, lead):
        try:
            df = pd.read_csv(csv_path)
            if lead not in df.columns:
                messagebox.showerror("Error", f"Lead '{lead}' not found in {os.path.basename(csv_path)}")
                return
                
            raw = df[lead].dropna().values.astype(np.float32)
            if len(raw) < 100:
                messagebox.showerror("Error", f"Lead '{lead}' too short: {len(raw)} samples")
                return
                
            sig = normalize(lowpass(remove_baseline_wander(raw)))

            self.reference_raw = raw
            self.reference_sig = sig

            # ----- REF plot -----
            self.ax_ref.clear()
            t = np.arange(len(sig)) / FS
            self.ax_ref.plot(t, sig, lw=0.8, color="black")

            # Detect R/Q/S for entire lead
            r_peaks = refine_r_peaks(predict_r_peaks_dl(sig), sig)
            q_pts, s_pts, _ = detect_q_s(sig, r_peaks)

            # Plot QRS points
            if r_peaks:
                self.ax_ref.scatter(np.array(r_peaks)/FS, sig[r_peaks], 
                                  c="r", s=10, label="R")
            if q_pts:
                self.ax_ref.scatter(np.array(q_pts)/FS, sig[q_pts], 
                                  c="b", s=8, label="Q")
            if s_pts:
                self.ax_ref.scatter(np.array(s_pts)/FS, sig[s_pts], 
                                  c="g", s=8, label="S")

            self.ax_ref.set_title("Drag mouse to select reference beat")
            self.ax_ref.set_xlabel("Time (s)")
            self.ax_ref.set_ylabel("Amplitude")
            self.ax_ref.legend()

            # ----- MATCH plot placeholder -----
            self.ax_match.clear()
            self.ax_match.text(0.5, 0.5, "Drag on LEFT to select reference beat", 
                             ha='center', va='center', fontsize=14, 
                             transform=self.ax_match.transAxes)
            self.ax_match.set_title("Matches / Live Results")

            self.fig.tight_layout()
            self.canvas.draw_idle()

            self.enable_reference_selection()
            
        except Exception as e:
            messagebox.showerror("Plot Error", f"Failed to plot {lead}: {str(e)}")


    def enable_reference_selection(self):
     self.ref_start = None
     self.ref_patch = None

     # disconnect old handlers if any
     if hasattr(self, "cid_press"):
        self.canvas.mpl_disconnect(self.cid_press)
     if hasattr(self, "cid_move"):
        self.canvas.mpl_disconnect(self.cid_move)
     if hasattr(self, "cid_release"):
        self.canvas.mpl_disconnect(self.cid_release)

     self.cid_press = self.canvas.mpl_connect(
        "button_press_event", self.on_press
     )
     self.cid_move = self.canvas.mpl_connect(
        "motion_notify_event", self.on_motion
     )
     self.cid_release = self.canvas.mpl_connect(
        "button_release_event", self.on_release
     )



    def on_press(self, event):
       if event.inaxes != self.ax_ref or event.xdata is None:
         return

       self.ref_start = int(event.xdata * FS)


    def on_motion(self, event):
     if self.ref_start is None:
        return
     if event.inaxes != self.ax_ref or event.xdata is None:
        return

     x0 = self.ref_start / FS
     x1 = event.xdata

     if self.ref_patch is not None:
        try:
            self.ref_patch.remove()
        except Exception:
            pass
        self.ref_patch = None

     self.ref_patch = self.ax_ref.axvspan(
        min(x0, x1),
        max(x0, x1),
        color="orange",
        alpha=0.35
     )
     self.canvas.draw_idle()


    def on_release(self, event):
     if self.ref_start is None:
        return

     if event.inaxes != self.ax_ref or event.xdata is None:
        self.ref_start = None
        return

     s = int(min(self.ref_start, event.xdata * FS))
     e = int(max(self.ref_start, event.xdata * FS))

     # minimum beat length
     if e - s < int(0.3 * FS):
        messagebox.showwarning(
            "Selection too small",
            "Select at least one full beat (≥300 ms)"
        )
        self.ref_start = None
        return

     # ---- find R-peaks inside selection ----
     r_peaks = refine_r_peaks(
        predict_r_peaks_dl(self.reference_sig),
        self.reference_sig
     )

     r_inside = [r for r in r_peaks if s <= r <= e]
     if not r_inside:
        messagebox.showerror(
            "Invalid selection",
            "Selected region must contain an R peak"
        )
        self.ref_start = None
        return

     # ---- store reference ----
     self.reference_idx = (s, e)
     self.reference_beat = self.reference_sig[s:e].copy()
     self.reference_raw_full = self.reference_raw.copy()

     # ---- Q / R / S inside reference ----
     self.reference_r = r_inside

     q_pts, s_pts, _ = detect_q_s(self.reference_sig, r_peaks)
     self.reference_q = [q for q in q_pts if s <= q <= e]
     self.reference_s = [s0 for s0 in s_pts if s <= s0 <= e]

     # ---- remove patch ----
     if self.ref_patch is not None:
        try:
            self.ref_patch.remove()
        except Exception:
            pass
        self.ref_patch = None

     self.info_label.config(
        text=f"Reference beat selected: {s/FS:.2f}s → {e/FS:.2f}s"
     )

     self.ref_start = None
     self.canvas.draw_idle()

     



    def reset_reference(self):
        self.reference_beat = None
        self.reference_idx = None
        self.reference_q = []
        self.reference_r = []
        self.reference_s = []
        self.reference_beat_multi = None
        if self.selected_file and self.selected_lead:
            self.plot_lead(self.selected_file, self.selected_lead)

    def stop_batch_run(self):
     """Signal the running batch thread to stop."""
     self.stop_batch_flag = True
     self.info_label.config(
        text="⏹ Stop requested — batch will stop after current file"
    )



    

    def is_similar_beat(self, multi_sig_f, sig_f, start, beat_len):
        end = start + beat_len

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

        # ======================================================
        # 🔴 FIX: ALLOW EDGE BEATS BY PADDING
        # ======================================================
        s = r - ref_offset
        e = s + beat_len

        pad_left = max(0, -s)
        pad_right = max(0, e - len(sig_f))

        s0 = max(0, s)
        e0 = min(len(sig_f), e)

        win = sig_f[s0:e0]

        if pad_left > 0 or pad_right > 0:
            win = np.pad(win, (pad_left, pad_right), mode="edge")

        if len(win) != beat_len:
            return False, None, None

        ref = self.reference_beat.copy()

        # ---------------- Normalize ----------------
        ref = (ref - np.mean(ref)) / (np.std(ref) + 1e-8)
        win = (win - np.mean(win)) / (np.std(win) + 1e-8)

        # =====================================================
        # QRS WINDOWS — DERIVED FROM beat_len
        # =====================================================
        r0 = ref_offset

        qrs_core = max(1, int(0.10 * beat_len))
        a = max(0, r0 - qrs_core)
        b = min(len(ref), r0 + qrs_core)

        ref_core = ref[a:b]
        win_core = win[a:b]

        core_corr = np.dot(ref_core, win_core) / len(ref_core)

        shift = max(1, int(0.03 * beat_len))
        for sh in range(-shift, shift + 1):
            aa = a + sh
            bb = b + sh
            if aa < 0 or bb > len(win):
                continue
            c = np.dot(ref_core, win[aa:bb]) / len(ref_core)
            core_corr = max(core_corr, c)

        # ---------------- Extended QRS ----------------
        qrs_ext = max(1, int(0.20 * beat_len))
        a2 = max(0, r0 - qrs_ext)
        b2 = min(len(ref), r0 + qrs_ext)

        ref_ext = ref[a2:b2]
        win_ext = win[a2:b2]

        ext_corr = np.dot(ref_ext, win_ext) / len(ref_ext)

        # ---------------- Amplitude consistency ----------------
        ref_amp = np.max(np.abs(ref_core)) + 1e-8
        win_amp = np.max(np.abs(win_core)) + 1e-8
        amp_ratio = win_amp / ref_amp
        amp_score = 1.0 - min(abs(amp_ratio - 1.0) / 1.5, 1.0)

        # ---------------- Final score ----------------
        score = (
            0.45 * core_corr +
            0.35 * ext_corr +
            0.20 * amp_score
        )

        if score < 0.55:
            return False, None, score

        return True, None, score


    def overlaps_existing_r(self, r_idx, matched_r_peaks, fs=FS, min_rr=0.25):
        """
        Prevent duplicate detection of the same beat.
        Reject if R peak is within min_rr seconds of an already matched R.
        """
        min_dist = int(min_rr * fs)
        for r in matched_r_peaks:
            if abs(r_idx - r) < min_dist:
                return True
        return False


    def process_file(self, file_path):
        df = pd.read_csv(file_path)
        fname = os.path.basename(file_path)
        base = fname.replace(".csv", "")

        # ---------- SINGLE OUTPUT FOLDER ----------
        output_dir = os.path.join(self.folder, "matched_images")
        os.makedirs(output_dir, exist_ok=True)

        # ---------- VALID LEADS ----------
        lead_cols = [c for c in df.columns if np.issubdtype(df[c].dtype, np.number)]
        lead_cols = [l for l in lead_cols if len(df[l].dropna()) >= 13]
        if not lead_cols:
            return

        # ---------- MULTI-LEAD PREPROCESS ----------
        multi = df[lead_cols].values.astype(np.float32)
        multi_f = np.zeros_like(multi)

        for i in range(multi.shape[1]):
            sig = normalize(lowpass(remove_baseline_wander(multi[:, i])))
            multi_f[:, i] = sig

        beat_len = self.reference_idx[1] - self.reference_idx[0]

        # ======================================================
        # LOOP OVER LEADS
        # ======================================================
        for lead in lead_cols:
            sig_raw = df[lead].values.astype(np.float32)
            if len(sig_raw) < int(0.5 * beat_len):
                continue

            sig_f = normalize(lowpass(remove_baseline_wander(sig_raw)))

            # ---------- R PEAKS ----------
            r_peaks = refine_r_peaks(predict_r_peaks_dl(sig_f), sig_f)
            self.current_r_peaks = r_peaks

            if not r_peaks:
                continue

            q_points, s_points, _ = detect_q_s(sig_f, r_peaks)

            match_starts = []
            matched_r_peaks = []

            # reference R offset
            pre = self.reference_r[0] - self.reference_idx[0]

            # ======================================================
            # 🔴 FIXED R-ANCHORED SEARCH (EDGE BEATS ALLOWED)
            # ======================================================
            for r in r_peaks:
                start = r - pre

                ok, _, score = self.is_similar_beat(
                    multi_f, sig_f, start, beat_len
                )

                if not ok:
                    continue

                if self.overlaps_existing_r(r, matched_r_peaks):
                    continue

                match_starts.append(start)
                matched_r_peaks.append(r)

            if not match_starts:
                continue

            # ======================================================
            # SAVE IMAGE
            # ======================================================
            fig, ax = plt.subplots(figsize=(14, 4))

            t = np.arange(len(sig_raw)) / FS
            ax.plot(t, sig_raw, lw=0.8, color="steelblue")

            # Shade matches
            for s_idx in match_starts:
                s_idx = max(0, int(s_idx))
                e_idx = s_idx + beat_len
                e_idx = min(e_idx, len(sig_raw))

                ax.axvspan(
                    s_idx / FS,
                    e_idx / FS,
                    color="#cfe9ff",
                    alpha=0.35
                )

            # QRS markers
            r_peaks = np.array(r_peaks, dtype=int)
            q_points = np.array(q_points, dtype=int)
            s_points = np.array(s_points, dtype=int)

            ax.scatter(r_peaks / FS, sig_raw[r_peaks], c='r', s=10, zorder=5)
            ax.scatter(q_points / FS, sig_raw[q_points], c='b', s=8, zorder=5)
            ax.scatter(s_points / FS, sig_raw[s_points], c='g', s=8, zorder=5)

            ax.set_title(f"{base} | {lead} | {len(match_starts)} matches")
            ax.set_xlabel("Time (s)")
            ax.set_ylabel("Amplitude")

            plt.tight_layout()

            out_name = f"{base}_{lead}.png"
            fig.savefig(os.path.join(output_dir, out_name), dpi=150)
            plt.close(fig)

        gc.collect()



    # -------- batch search over all CSVs --------
    def start_batch(self):
     """Batch processes ALL CSVs in folder, writes PNG + summary; stoppable with Stop Batch."""
     if self.reference_sig is None or self.reference_beat is None:
        messagebox.showerror("Error", "Select reference beat first")
        return
     if self.reference_beat is None:
        messagebox.showerror("Error", "Reference beat not selected")
        return

     if not self.folder:
        messagebox.showerror("Error", "Select folder first")
        return

     # reset stop flag
     self.stop_batch_flag = False
     self.info_label.config(text="Batch starting...")

     def worker():
        csv_files = [
            os.path.join(self.folder, f)
            for f in os.listdir(self.folder)
            if f.lower().endswith(".csv")
        ]
        print(f" Total files: {len(csv_files)}")

        processed = 0
        for i, fp in enumerate(csv_files):
            if self.stop_batch_flag:
                print("⏹ Batch stopped by user.")
                break

            try:
                self.process_file(fp)
                processed += 1

                # safe UI update
                self.root.after(
                    0,
                    lambda p=i+1, n=len(csv_files), fn=os.path.basename(fp):
                        self.info_label.config(text=f"Processed {p}/{n}: {fn}")
                )

            except Exception as e:
                print(f"Error processing {fp}: {e}")

        print(f"Batch finished! Processed {processed}/{len(csv_files)} files")

         #FIX: after() MUST receive ONE callable
        def on_batch_complete():
            self.info_label.config(text="✅ Batch complete! Click 'Load Results'")
            print(" Run 'Load Results' to display matches")

        self.root.after(0, on_batch_complete)

     threading.Thread(target=worker, daemon=True).start()


    def build_results_index(self):
        """Group by file+lead (like raw ECG plots) - shows ALL matches shaded"""
        if not self.folder:
            messagebox.showerror("Error", "Select folder first")
            return

        from collections import defaultdict
        lead_matches = defaultdict(list)
        print(f"Scanning folder: {self.folder}")

        summary_count = 0
        for root, dirs, files in os.walk(self.folder):
            for f in files:
                if f.endswith("_summary.csv"):
                    summary_count += 1
                    summary_path = os.path.join(root, f)
                    base_name = f.replace("_summary.csv", "")
                    parent_dir = os.path.dirname(root)
                    csv_path = os.path.join(parent_dir, base_name + ".csv")
                    
                    print(f"Found summary: {f} → CSV: {os.path.basename(csv_path)}")
                    
                    if not os.path.exists(csv_path):
                        print(f" Missing CSV: {csv_path}")
                        continue

                    try:
                        df_sum = pd.read_csv(summary_path)
                        print(f" {f}: {len(df_sum)} rows")
                        df_sum = df_sum.dropna(subset=["Lead", "Start_Index"])
                        
                        for _, row in df_sum.iterrows():
                            try:
                                lead = str(row["Lead"])
                                s_idx = int(row["Start_Index"])
                                lead_matches[(csv_path, lead)].append(s_idx)
                            except (ValueError, TypeError, KeyError):
                                continue
                    except Exception as e:
                        print(f" Error reading {summary_path}: {e}")
                        continue

        self.result_items = []
        for (csv_path, lead), starts in sorted(lead_matches.items()):
            starts = sorted(set(starts))
            if starts:
                self.result_items.append((csv_path, lead, starts))
                print(f"Group: {os.path.basename(csv_path)} [{lead}] → {len(starts)} matches")

        print(f" Loaded {len(self.result_items)} GROUPS (file+lead)")
        
        if not self.result_items:
            self.info_label.config(text=" No matches found")
            return

        self.result_index = 0
        self.info_label.config(text=f" Loaded {len(self.result_items)} groups! Showing first...")
        self.show_result(self.result_index)


    def show_result(self, idx):
        if not self.result_items:
            self.info_label.config(text="No results loaded. Run batch and Load Results.")
            return
        if idx < 0 or idx >= len(self.result_items):
            return

        csv_path, lead, starts = self.result_items[idx]
        if not os.path.exists(csv_path):
            self.info_label.config(text=f"CSV missing: {os.path.basename(csv_path)}")
            return

        df = pd.read_csv(csv_path)
        if lead not in df.columns:
            self.info_label.config(text=f"Lead {lead} not in {os.path.basename(csv_path)}")
            return

        sig_raw = df[lead].dropna().values.astype(np.float32)
        if len(sig_raw) == 0:
            self.info_label.config(text=f"No data for {lead} in {os.path.basename(csv_path)}")
            return

        t = np.arange(len(sig_raw)) / FS
        beat_len = self.reference_idx[1] - self.reference_idx[0]

        # ---------------- LEFT: reference ----------------
        self.ax_ref.clear()
        ref_full_raw = self.reference_raw_full
        ref_start, ref_end = self.reference_idx
        t_ref = np.arange(len(ref_full_raw)) / FS
        self.ax_ref.plot(t_ref, ref_full_raw, lw=0.8, color="steelblue")
        self.ax_ref.axvspan(ref_start/FS, ref_end/FS, color="orange", alpha=0.25)
        if self.reference_r:
            self.ax_ref.scatter(np.array(self.reference_r)/FS,
                              np.array(ref_full_raw)[self.reference_r], c='r', s=10, zorder=5)
        if self.reference_q:
            self.ax_ref.scatter(np.array(self.reference_q)/FS,
                              np.array(ref_full_raw)[self.reference_q], c='b', s=8, zorder=5)
        if self.reference_s:
            self.ax_ref.scatter(np.array(self.reference_s)/FS,
                              np.array(ref_full_raw)[self.reference_s], c='g', s=8, zorder=5)
        self.ax_ref.set_title("Reference (ECG)")
        self.ax_ref.set_xlabel("Time (s)")
        self.ax_ref.set_ylabel("Amplitude")

        # ---------------- RIGHT: raw with matches ----------------
        self.ax_match.clear()
        self.ax_match.plot(t, sig_raw, lw=0.8, color="steelblue")

        # Shade matches first (background)
        for s_idx in starts:
            s_idx = max(0, min(int(s_idx), len(sig_raw)-1))
            e_idx = s_idx + beat_len
            if e_idx > len(sig_raw):
                e_idx = len(sig_raw)
            self.ax_match.axvspan(s_idx/FS, e_idx/FS, color="#cfe9ff", alpha=0.35)

        # QRS detection on preprocessed signal
        sig_f = normalize(lowpass(remove_baseline_wander(sig_raw)))
        r_peaks = refine_r_peaks(predict_r_peaks_dl(sig_f), sig_f)
        q_points, s_points, _ = detect_q_s(sig_f, r_peaks)

        # Convert indices to numpy arrays for safe indexing
        r_peaks = np.array(r_peaks, dtype=int)
        q_points = np.array(q_points, dtype=int)
        s_points = np.array(s_points, dtype=int)

        if len(r_peaks) > 0:
            self.ax_match.scatter(r_peaks/FS, sig_raw[r_peaks], c='r', s=10, zorder=5)
        if len(q_points) > 0:
            self.ax_match.scatter(q_points/FS, sig_raw[q_points], c='b', s=8, zorder=5)
        if len(s_points) > 0:
            self.ax_match.scatter(s_points/FS, sig_raw[s_points], c='g', s=8, zorder=5)

        self.ax_match.set_title(f"{os.path.basename(csv_path)} ({lead}) – {len(starts)} matches")
        self.ax_match.set_xlabel("Time (s)")
        self.ax_match.set_ylabel("Amplitude")

        self.fig.tight_layout()
        self.canvas.draw_idle()

        self.info_label.config(
            text=f"Group {idx+1}/{len(self.result_items)} | {os.path.basename(csv_path)} | "
                 f"{lead} | {len(starts)} matches"
        )


    def next_result(self):
        if not self.result_items:
            return
        self.result_index += 1
        if self.result_index >= len(self.result_items):
            self.result_index = len(self.result_items) - 1
        self.show_result(self.result_index)


    def prev_result(self):
        if not self.result_items:
            return
        self.result_index -= 1
        if self.result_index < 0:
            self.result_index = 0
        self.show_result(self.result_index)


# ---------------- GUI WRAPPER ----------------
class ECGPatternGUIApp:
    def __init__(self):
        
        self.root = tk.Tk()
        self.gui = ECGPatternGUI(self.root)
        self.stop_batch = False
        self.root.mainloop()

# ---------------- MAIN ----------------
if __name__ == "__main__":
    ECGPatternGUIApp()
