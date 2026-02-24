import time
import pandas as pd
import numpy as np
import tensorflow as tf
from scipy import signal
from scipy import sparse
from scipy.sparse.linalg import spsolve
from scipy.signal import (find_peaks, firwin,medfilt, welch, argrelextrema)
import pywt
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from matplotlib import colormaps
import glob
from analysis_tool import tools as st
from analysis_tool  import utils
import warnings
import threading
import scipy
import os
from collections import Counter
from fpdf import FPDF
from matplotlib.gridspec import GridSpec
import matplotlib.ticker as ticker
from dotenv import load_dotenv
import gridfs
from pymongo import MongoClient
from datetime import datetime
from io import BytesIO
from collections import defaultdict
from PyPDF2 import PdfMerger
from bson.objectid import ObjectId
load_dotenv()   # loads .env
warnings.filterwarnings('ignore')

results_lock = threading.RLock()
mongo_client = MongoClient(os.getenv("MONGO_HOST"))

media_db = mongo_client["Download_files"]
admin_db=mongo_client["admin"]
download_fs = gridfs.GridFS(media_db, collection="downloads")

logs_collection = admin_db["pdf_logs"]

def time_wrapper(func):
    def wrapper_function(*args, **kwargs):
        start_time = time.time_ns()
        data = func(*args, **kwargs)
        print('Execute Time : ', (time.time_ns() - start_time)/1000000000, 'Seconds')
        return data
    return wrapper_function

def lowpass(ecg_signal, cutoff=0.3):
    b, a = signal.butter(3, cutoff, btype='lowpass', analog=False)
    low_passed = signal.filtfilt(b, a, ecg_signal)
    return low_passed

def load_tflite_model(model_path):
    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    return interpreter, input_details, output_details

def is_rhythm_pos_neg(baseline_signal, fs):
    det_r_index = detect_beats(baseline_signal, fs)
    pos_neg_ind = []
    rhy_label = 'Positibve'
    for r_idx in det_r_index:
        st_idx = max(0, r_idx- int(0.1 * fs))
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
    most_common_ele = max(set(pos_neg_ind), key=lambda x:pos_neg_ind.count(x))
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
    return rhy_label, det_r_index


class PQRSTDetection:
    def __init__(self, ecg_signal, fs=200, thres=0.5, lp_thres=0.2, rr_thres=0.12, width=(5, 50), JR=False, MI= False):
        self.ecg_signal = ecg_signal
        self.fs = fs
        self.thres = thres
        self.lp_thres = lp_thres
        self.rr_thres = rr_thres
        self.width = width
        self.JR = JR
        self.MI = MI

    def hamilton_segmenter(self):
        if self.ecg_signal is None:
            raise TypeError("Please specify an input signal.")

        sampling_rate = float(self.fs)
        length = len(self.ecg_signal)
        dur = length / sampling_rate

        v1s = int(1.0 * sampling_rate)
        v100ms = int(0.1 * sampling_rate)
        TH_elapsed = np.ceil(0.36 * sampling_rate)
        sm_size = int(0.08 * sampling_rate)
        init_ecg = 10
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
                    diff_now = np.diff(self.ecg_signal[0 : f + diff_nr])
                elif f + diff_nr >= len(self.ecg_signal):
                    diff_now = np.diff(self.ecg_signal[f - diff_nr : len(dx)])
                else:
                    diff_now = np.diff(self.ecg_signal[f - diff_nr : f + diff_nr])
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
                            diff_prev = np.diff(self.ecg_signal[0 : prev_rpeak + diff_nr])
                        elif prev_rpeak + diff_nr >= len(self.ecg_signal):
                            diff_prev = np.diff(self.ecg_signal[prev_rpeak - diff_nr : len(dx)])
                        else:
                            diff_prev = np.diff(
                                self.ecg_signal[prev_rpeak - diff_nr : prev_rpeak + diff_nr]
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
                # 4 - not valid
                # 5 - If no QRS has been detected within 1.5 R-to-R intervals,
                # there was a peak that was larger than half the detection threshold,
                # and the peak followed the preceding detection by at least 360 ms,
                # classify that peak as a QRS complex
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
                window = self.ecg_signal[0 : i + lim]
                add = 0
            elif i + lim >= length:
                window = self.ecg_signal[i - lim : length]
                add = i - lim
            else:
                window = self.ecg_signal[i - lim : i + lim]
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
        cal_sec = round(self.ecg_signal.shape[0]/self.fs)
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
        j = []
        increment = int(self.fs*0.05)
        for z in range (0,len(self.s_index)):
            data = []
            j_index = self.ecg_signal[self.s_index[z]:self.s_index[z]+increment]
            for k in range (0,len(j_index)):
                data.append(j_index[k])
            max_d = max(data)
            max_id = data.index(max_d)
            j.append(self.s_index[z]+max_id)
        return j

    def find_s_index(self, d):
            d = int(d)+1
            s = []
            for i in self.r_index:
                if i == len(self.ecg_signal):
                    s.append(i)
                    continue
                elif i+d<=len(self.ecg_signal):
                    s_array = self.ecg_signal[i:i+d]
                else:
                    s_array = self.ecg_signal[i:]
                if self.ecg_signal[i] > 0:
                    s_index = i+np.where(s_array == min(s_array))[0][0]
                else:
                    s_index = i+np.where(s_array == max(s_array))[0][0]
                    if abs(s_index - i) < d/2:
                        s_index_ = i+np.where(s_array == min(s_array))[0][0]
                        if abs(s_index_ - i) > d/2:
                            s_index = s_index_
                s.append(s_index)
            return np.sort(s)

    def find_q_index(self, d):
        d = int(d) + 1
        q = []
        for i in self.r_index:
            if i == 0:
                q.append(i)
                continue
            elif 0 <= i - d:
                q_array = self.ecg_signal[i - d:i]
            else:
                q_array = self.ecg_signal[:i]
            if self.ecg_signal[i] > 0:
                q_index = i - (len(q_array) - np.where(q_array == min(q_array))[0][0])
            else:
                q_index = i - (len(q_array) - np.where(q_array == max(q_array))[0][0])
            q.append(q_index)
        return np.sort(q)

    def find_new_q_index(self, d):
        q = []
        for i in self.r_index:
            q_ = []
            if i == 0:
                q.append(i)
                continue
            if self.ecg_signal[i] > 0:
                c = i
                while c > 0 and self.ecg_signal[c-1] < self.ecg_signal[c]:
                    c -= 1                  
                if self.ecg_signal[i] * 0.01 > self.ecg_signal[c] or self.ecg_signal[c] < 0 or c == 0:
                    if abs(i-c) <= d:
                        q.append(c)
                        continue
                    else:
                        q_.append(c)
                while c > 0:
                    while c > 0 and self.ecg_signal[c-1] > self.ecg_signal[c]:
                        c -= 1
                    # q_.append(c)
                    while c > 0 and self.ecg_signal[c-1] < self.ecg_signal[c]:
                        c -= 1
                    if q_ and q_[-1] == c:
                        break
                    q_.append(c)
                    if self.ecg_signal[i] * 0.01 > self.ecg_signal[c] or self.ecg_signal[c] < 0 or c == 0:
                        break
            else:
                c = i
                while c > 0 and self.ecg_signal[c-1] > self.ecg_signal[c]:
                    c -= 1
                if self.ecg_signal[i] * 0.01 < self.ecg_signal[c] or self.ecg_signal[c] > 0 or c == 0:
                    if abs(i-c) <= d:
                        q.append(c)
                        continue
                    else:
                        q_.append(c)
                while c > 0:
                    while c > 0 and self.ecg_signal[c-1] < self.ecg_signal[c]:
                        c -= 1
                    # q_.append(c)
                    while c > 0 and self.ecg_signal[c-1] > self.ecg_signal[c]:
                        c -= 1
                    if q_ and q_[-1] == c:
                        break
                    q_.append(c)
                    if self.ecg_signal[i] * 0.01 < self.ecg_signal[c] or self.ecg_signal[c] > 0 or c == 0:
                        break
            if q_:
                a = 0
                for _q in q_[::-1]:
                    if abs(i-_q) <= d:
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
                while c+1 < end_index and self.ecg_signal[c+1] < self.ecg_signal[c]:
                    c += 1
                if self.ecg_signal[i] * 0.01 > self.ecg_signal[c] or self.ecg_signal[c] < 0 or c == end_index-1:
                    if abs(i-c) <= d:
                        s.append(c)
                        continue
                    else:
                        s_.append(c)
                while c+1 < end_index:
                    while c+1 < end_index and self.ecg_signal[c+1] > self.ecg_signal[c]:
                        c += 1
                    while c+1 < end_index and self.ecg_signal[c+1] < self.ecg_signal[c]:
                        c += 1
                    if s_ and s_[-1] == c:
                        break
                    s_.append(c)
                    if self.ecg_signal[i] * 0.01 > self.ecg_signal[c] or self.ecg_signal[c] < 0 or c == end_index-1:
                        break
            else:
                c = i
                while c+1 < end_index and self.ecg_signal[c+1] > self.ecg_signal[c]:
                    c += 1
                if self.ecg_signal[i] * 0.01 < self.ecg_signal[c] or self.ecg_signal[c] > 0 or c == end_index-1:
                    if abs(i-c) <= d:
                        s.append(c)
                        continue
                    else:
                        s_.append(c)
                while c < end_index:
                    while c+1 < end_index and self.ecg_signal[c+1] > self.ecg_signal[c]:
                        c += 1
                    while c+1 < end_index and self.ecg_signal[c+1] < self.ecg_signal[c]:
                        c += 1
                    if s_ and s_[-1] == c:
                        break
                    s_.append(c)
                    if self.ecg_signal[i] * 0.01 < self.ecg_signal[c] or self.ecg_signal[c] > 0 or c == end_index-1:
                        break
            if s_:
                a = 0
                for _s in s_[::-1]:
                    if abs(i-_s) <= d:
                        a = 1
                        s.append(_s)
                        break
                if a == 0:
                    s.append(s_[0])
        return np.sort(s)

    def find_r_peaks(self):
        rhy_label = is_rhythm_pos_neg(self.ecg_signal, self.fs)
        if rhy_label == 'Positive' and self.MI:
            self.r_index = RPeakDetection(self.ecg_signal, self.fs).find_r_peak()
            return np.array(self.r_index)
        else: 
            r_ = []
            out = self.hamilton_segmenter()
            self.r_index = out["rpeaks"]
            heart_rate = self.hr_count()
            if self.JR: #---------------------
                diff_indexs = abs(round((self.fs * 0.4492537) + (heart_rate * -1.05518351) + 40.40601032654332))
            else: #---------------------
                diff_indexs = abs(round((self.fs * 0.4492537) + (heart_rate * -1.009) + 58.40601032654332))
        
            for r in self.r_index:
                if r - diff_indexs >= 0 and len(self.ecg_signal) >= r+diff_indexs:
                    data = self.ecg_signal[r-diff_indexs:r+diff_indexs]
                    abs_data = np.abs(data)
                    r_.append(np.where(abs_data == max(abs_data))[0][0] + r-diff_indexs)
                else:
                    r_.append(r)
                
            new_r = np.unique(r_) if r_ else self.r_index
            fs_diff = int((25*self.fs)/200)
            final_r = []
            if new_r.any(): final_r = [new_r[0]] + [new_r[j+1] for j, i in enumerate(np.diff(new_r)) if i >= fs_diff]
            return np.array(final_r)

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
        max_signal = max(self.ecg_signal)/100
        pt = []
        p_t = []
        for i in range (0, len(self.r_index)-1):
            aoi = self.ecg_signal[self.s_index[i]:self.q_index[i+1]]
            max_signal = max(self.ecg_signal)/100
            low = self.fir_lowpass_filter(aoi,self.lp_thres,30)
            if self.ecg_signal[self.r_index[i]]<0:
                max_signal=0.05
            else:
                max_signal=max_signal
            if aoi.any():
                peaks,_ = find_peaks(low,height=max_signal,width=self.width)
                peaks1=peaks+(self.s_index[i])
            else:
                peaks1 = [0]
            p_t.append(list(peaks1))
            pt.extend(list(peaks1))
            for i in range (len(p_t)):
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
        for i in range (0, len(self.r_index)-1):
            aoi = self.ecg_signal[self.s_index[i]:self.q_index[i+1]]
            if aoi.any():
                low = self.fir_lowpass_filter(aoi,self.lp_thres,30)
                if self.ecg_signal[self.r_index[i]]<0:
                    max_signal=0.05
                else:
                    max_signal= max(low)*0.2
                if aoi.any():
                    peaks,_ = find_peaks(low,height=max_signal,width=self.width)
                    peaks1=peaks+(self.s_index[i])
                else:
                    peaks1 = [0]
                p_t.append(list(peaks1))
                pt.extend(list(peaks1))
                for i in range (len(p_t)):
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
        for i in range (0, len(self.r_index)-1):
            aoi = self.ecg_signal[self.s_index[i]:self.q_index[i+1]]
            low = self.fir_lowpass_filter(aoi,self.lp_thres,30)
            if aoi.any():
                peaks,_ = find_peaks(low,prominence=0.05,width=self.width)
                peaks1=peaks+(self.s_index[i])
            else:
                peaks1 = [0]
            p_t.append(list(peaks1))
            pt.extend(list(peaks1))
            for i in range (len(p_t)):
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
                        m, n, o = arr[0], arr[al[index]], arr[al[index+1]]
                    elif index == len(al)-1:
                        m, n, o = arr[al[index-1]], arr[al[index]], arr[-1]
                    else:
                        m, n, o = arr[al[index-1]], arr[al[index]], arr[al[index+1]]
                    diff[p] = [abs(n-m), abs(n-o)]
                th = np.mean([np.mean([v, m]) for v, m in diff.values()])*.66
                for p, (a, b) in diff.items():
                    if a >= th and b >= th:
                        P.append(p)
                        continue
                    if a >= th and not Pa:
                        Pa.append(p)
                    elif a >= th and arr[p] > arr[Pa[-1]] and np.where(pos==Pa[-1])[0]+1 == np.where(pos==p)[0]:
                        Pa[-1] = p
                    elif a >= th:
                        Pa.append(p)
                    if b >= th and not Pb:
                        Pb.append(p)
                    elif b >= th and arr[p] < arr[Pb[-1]] and np.where(pos==Pb[-1])[0]+1 == np.where(pos==p)[0]:
                        Pb[-1] = p
                    elif b >= th:
                        Pb.append(p)
                if len(pos)>1:
                    for i in range(1, len(pos)):
                        m, n = pos[i-1], pos[i]
                        if m in Pa and n in Pb:
                            P.append(m) if arr[m] > arr[n] else P.append(n)
                # if Pa and Pa[-1] == pos[-1]:
                #     P.append(Pa[-1])
                # if Pb and Pb[0] == pos[0]:
                #     P.append(Pb[0])
            else:
                P = pos
            return np.sort(P)
        pt, p_t = [], []
        for i in range(1, len(self.r_index)):
            q0, r0, s0 = self.q_index[i - 1], self.r_index[i - 1], self.s_index[i - 1]
            q1, r1, s1 = self.q_index[i], self.r_index[i], self.s_index[i]
            arr = self.ecg_signal[s0+7:q1-7]
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
            for _pt in set(p_t1[i]+p_t2[i]+p_t3[i]+p_t4[i]):
                count = 0
                if any(val in p_t1[i] for val in range (_pt-2,_pt+3)):
                    count += 1
                if any(val in p_t2[i] for val in range (_pt-2,_pt+3)):
                    count += 1
                if any(val in p_t3[i] for val in range (_pt-2,_pt+3)):
                    count += 1
                if any(val in p_t4[i] for val in range (_pt-2,_pt+3)):
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
                if abs(sublist[i] - sublist[i-1]) > 5:
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
        diff_arr = ((np.diff(self.r_index)*self.thres)/self.fs).tolist()
        t_peaks_list, p_peaks_list, pr_interval, extra_peaks_list = [], [], [], []
        # threshold = (-0.0012 * len(r_index)) + 0.25
        for i in range(len(self.p_t)):
            p_dis = (self.r_index[i+1]-self.p_t[i][-1])/self.fs
            t_dis = (self.r_index[i+1]-self.p_t[i][0])/self.fs
            threshold = diff_arr[i]
            if t_dis > threshold and (self.p_t[i][0]>self.r_index[i]): 
                t_peaks_list.append(self.p_t[i][0])
            else:
                t_peaks_list.append(0)
            if p_dis <= threshold: 
                p_peaks_list.append(self.p_t[i][-1])
                pr_interval.append(p_dis*self.fs)
            else:
                p_peaks_list.append(0)
            if len(self.p_t[i])>0:
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
        if self.thres >= 0.5 and p_peaks_list and len(p_peaks_list)>2:
            pp_intervals = np.diff(p_peaks_list)
            pp_std = np.std(pp_intervals)
            pp_mean = np.mean(pp_intervals)
            threshold = 0.12 * pp_mean
            if pp_std <= threshold:
                p_label = "Constanat"
            else:
                p_label = "Not Constant"
            
            count=0
            for i in pr_interval:
                if round(np.mean(pr_interval)*0.75) <= i <= round(np.mean(pr_interval)*1.25):
                    count +=1
            if len(pr_interval) != 0: 
                per = count/len(pr_interval)
                pr_label = 'Not Constant' if per<=0.7 else 'Constant'
        data = {'T_Index':t_peaks_list, 
                'P_Index':p_peaks_list, 
                'PR_Interval':pr_interval, 
                'P_Label':p_label, 
                'PR_label':pr_label,
                'Extra_Peaks':extra_peaks_list}
        return data

    def find_inverted_t_peak(self):
        t_index = []
        for i in range(0, len(self.s_index)-1):
            t = self.ecg_signal[self.s_index[i]: self.q_index[i+1]]
            if t.any():
                check, _ = find_peaks(-t,  height=(0.21, 1), distance=70)
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
        # if self.rr_thres == 0.15:
        #     self.ecg_signal = lowpass(self.ecg_signal,0.2)
        self.hr_ = self.hr_count()
        sd, qd = int(self.fs * 0.115), int(self.fs * 0.08)
        self.s_index = self.find_s_index(sd)
        # q_index = find_q_index(ecg_signal, r_index, qd)
        # s_index = find_new_s_index(ecg_signal,r_index,sd)
        self.q_index = self.find_new_q_index(qd)
        self.j_index = self.find_j_index()
        self.p_t, self.pt = self.find_pt()
        self.data_ = self.segricate_p_t_pr_inerval()
        self.inv_t_index = self.find_inverted_t_peak()
        data = {'R_Label':self.r_label, 
                'R_index':self.r_index, 
                'Q_Index':self.q_index, 
                'S_Index':self.s_index, 
                'J_Index':self.j_index, 
                'P_T List':self.p_t, 
                'PT PLot':self.pt, 
                'HR_Count':self.hr_, 
                'T_Index':self.data_['T_Index'], 
                'P_Index':self.data_['P_Index'],
                'Ex_Index':self.data_['Extra_Peaks'], 
                'PR_Interval':self.data_['PR_Interval'], 
                'P_Label':self.data_['P_Label'], 
                'PR_label':self.data_['PR_label'],
                'inv_t_index': self.inv_t_index}
        return data

# Filter signal
class FilterSignal:
    
    def __init__(self, ecg_signal, fs = 200):
        self.ecg_signal = ecg_signal
        self.fs = fs

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
            z = spsolve(Z, w*file)
            w = p * (file > z) + (1-p) * (file < z)
        return z

    def baseline_construction_250(self, kernel_size=131):
        als_baseline = self.baseline_als(self.ecg_signal, 16**5, 0.01) 
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

        if self.fs != 200:
            self.ecg_signal = MinMaxScaler(feature_range=(0,4)).fit_transform(self.ecg_signal.reshape(-1,1)).squeeze()
                    
        if self.fs == 200:
            self.baseline_signal = self.baseline_construction_200(kernel_size = 101)
            lowpass_signal = self.lowpass(cutoff = 0.3)
        elif self.fs == 250:
            self.baseline_signal = self.baseline_construction_250(kernel_size = 131)
            lowpass_signal = self.lowpass(cutoff = 0.25)
        elif self.fs == 360:
            self.baseline_signal = self.baseline_construction_200(kernel_size = 151)
            lowpass_signal = self.lowpass(cutoff = 0.2)
        elif self.fs == 1000:
            self.baseline_signal = self.baseline_construction_200(kernel_size = 399)
            lowpass_signal = self.lowpass(cutoff = 0.05)
        elif self.fs == 128:
            self.baseline_signal = self.baseline_construction_200(kernel_size = 101)
            lowpass_signal = self.lowpass(cutoff = 0.5)
            
        return self.baseline_signal, lowpass_signal
    


def baseline_construction_200(ecg_signal, kernel_size=101):
    s_corrected = signal.detrend(ecg_signal)
    baseline_corrected = s_corrected - signal.medfilt(s_corrected, kernel_size)
    return baseline_corrected

def lowpass_1(file):
  b, a = signal.butter(3, 0.2, btype='lowpass', analog=False)
  low_passed = signal.filtfilt(b, a, file)
  return low_passed

def findFile(name , path):
    try:
        if name in os.listdir(path):
            return True
        else:
            return False
    except:
        return False
    
def detect_beats(
        baseline_signal,  # The raw ECG signal
        fs,  # Sampling fs in HZ
        # Window size in seconds to use for
        ransac_window_size=3.0, #5.0
        # Low frequency of the band pass filter
        lowfreq=5.0,
        # High frequency of the band pass filter
        highfreq=7.0,
):
    ransac_window_size = int(ransac_window_size * fs)

    lowpass = scipy.signal.butter(1, highfreq / (fs / 2.0), 'low')
    highpass = scipy.signal.butter(1, lowfreq / (fs / 2.0), 'high')
    # TODO: Could use an actual bandpass filter
    ecg_low = scipy.signal.filtfilt(*lowpass, x=baseline_signal)
    ecg_band = scipy.signal.filtfilt(*highpass, x=ecg_low)

    # Square (=signal power) of the first difference of the signal
    decg = np.diff(ecg_band)
    decg_power = decg ** 2

    # Robust threshold and normalizator estimation
    thresholds = []
    max_powers = []
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
    # square_decg_power = decg_power**4

    shannon_energy = -square_decg_power * np.log(square_decg_power)
    shannon_energy[~np.isfinite(shannon_energy)] = 0.0

    mean_window_len = int(fs * 0.125 + 1)
    lp_energy = np.convolve(shannon_energy, [1.0 / mean_window_len] * mean_window_len, mode='same')
    # lp_energy = scipy.signal.filtfilt(*lowpass2, x=shannon_energy)

    lp_energy = scipy.ndimage.gaussian_filter1d(lp_energy, fs / 14.0) # 20.0
    # lp_energy = scipy.ndimage.gaussian_filter1d(lp_energy, fs/8.0)
    lp_energy_diff = np.diff(lp_energy)

    zero_crossings = (lp_energy_diff[:-1] > 0) & (lp_energy_diff[1:] < 0)
    zero_crossings = np.flatnonzero(zero_crossings)
    zero_crossings -= 1

    return zero_crossings

def predict_tflite_model(model:tuple, input_data:tuple):
    with results_lock:
        interpreter, input_details, output_details = model
        for i in range(len(input_data)):
            interpreter.set_tensor(input_details[i]['index'], input_data[i])
        interpreter.invoke()
        output = interpreter.get_tensor(output_details[0]['index'])
    
    return output

def hr_count(ecg_signal, r_index):
    cal_sec = round(ecg_signal.shape[0]/200)
    if cal_sec != 0:
        hr = round(r_index.shape[0]*60/cal_sec)
        return hr
    return 0

class afib_flutter_detection:
    def __init__(self, ecg_signal, r_index, q_index, s_index, p_index, p_t,p_r_int, load_model):
        self.ecg_signal = ecg_signal
        self.r_index = r_index
        self.q_index = q_index
        self.s_index = s_index
        self.p_index = p_index
        self.p_t = p_t
        self.p_r_int = p_r_int
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
        more_then_3_rhythm_per = len(list(filter(lambda x: len(x) >= 3, self.p_t))) / len(self.r_index)
        inner_list_less_2 = len(list(filter(lambda x: len(x) < 2, self.p_t))) / len(self.r_index)
        zeros_count = self.p_index.count(0)
        list_per = zeros_count / len(self.p_index)

        print('pr:',self.p_r_int)

        constant_list = []
        if len(self.p_r_int) >1:
            for i in range(len(self.p_r_int) - 1):
                diff = abs(self.p_r_int[i] - self.p_r_int[i + 1])
                if diff == 0 or diff == 1:
                    constant_list.append(self.p_r_int[i])

            # Check the last element with its previous element
            if abs(self.p_r_int[-1] - self.p_r_int[-2]) == 0 or abs(self.p_r_int[-1] - self.p_r_int[-2]) == 1:
                constant_list.append(self.p_r_int[-1])

            print("constant list:",constant_list)
            print('per:',round(len(constant_list) / len(self.p_r_int), 2))

        if more_then_3_rhythm_per >= 0.6:
            check_afib_flutter = True
        elif list_per >= 0.5:
            check_afib_flutter = True

        elif len(constant_list) != 0:
            if (len(constant_list) / len(self.p_r_int) < 0.7):
                check_afib_flutter = True
        else:
            p_peak_diff = np.diff(self.p_index)
            percentage_diff = np.abs(np.diff(p_peak_diff) / p_peak_diff[:-1]) * 100
            # print(percentage_diff,": percentage_diff")

            mean_p = np.mean(percentage_diff)
            print('\033[91m' + f"mean: {mean_p}" + '\033[0m')
            print("means_p:",mean_p,'rhythm:',more_then_3_rhythm_per)
            if mean_p != mean_p or mean_p == float('inf') or mean_p == float('-inf'):
                check_afib_flutter = True
            if (mean_p > 15 and more_then_3_rhythm_per >= 0.4) or (mean_p > 70 and inner_list_less_2 > 0.3):
                check_afib_flutter = True
            elif mean_p > 100 and inner_list_less_2 > 0.3:
                check_afib_flutter = True
            elif (mean_p > 20 and more_then_3_rhythm_per >= 0.1):
                check_afib_flutter = True
        print('\033[93m' + f"check_afib_flutter :{check_afib_flutter}" + '\033[0m')
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
        epoch_afib_index = []

        # raw_date_time = self.data['DateTime']
        for q, s in zip(q_new, s_new):
            data = ecg_signal[q:s]
            if data.any():
                image_data = self.image_array_new(data)
                image_data = (tf.expand_dims(image_data.astype(np.float32), axis=0),)
                model_pred = self.predict_tflite_model(self.load_model, image_data)[0]
                # print(model_pred)

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

        return percent, afib_data_index, flutter_data_index,epoch_afib_index

    def get_data(self):
        total_data = len(self.s_index) - 1
        last_s = None
        last_q = None
        afib_data_index = None
        flutter_data_index = None
        epoch_afib_index = None
        check_2nd_lead = {'ABNORMAL': 0, 'AFIB': 0, 'FLUTTER': 0, 'NOISE': 0, 'NORMAL': 0}
        if len(self.q_index) > 4 and len(self.s_index) > 4:
            q_new = self.q_index[:-4:4].tolist()
            s_new = self.s_index[4::4].tolist()
            if s_new[-1] != self.s_index[-1]:

                temp_s = list(self.s_index).index(s_new[-1])
                fin_s = total_data - temp_s
                last_q = self.q_index[temp_s]
                last_s = (s_new[-1], fin_s)
                q_new.append(self.q_index[-5])
                s_new.append(self.s_index[-1])
            check_2nd_lead, afib_data_index, flutter_data_index,epoch_afib_index = self.check_model(q_new, s_new, self.ecg_signal,last_s, last_q)

        return check_2nd_lead, afib_data_index, flutter_data_index,epoch_afib_index

def get_percentage_diff(previous, current):
    try:
        percentage = abs(previous - current) / max(previous, current) * 100
    except ZeroDivisionError:
        percentage = float('inf')
    return percentage


def Average(lst):
    return sum(lst) / len(lst)


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


def afib_afl_abs_model_check(all_lead_data, afib_afl_moedel):
    afib_afl_result = 'Abnormal'
    is_error = False
    hr_counts = None
    try:
        all_lead_result_data= {}
        final_result_list = []
        for lead in all_lead_data.columns:
            lead_data = {}
            if lead in ['II', 'III', 'aVF']:
                ecg_signal = all_lead_data[lead]
                frequency = 200
                baseline_signal, lowpass_signal = FilterSignal(ecg_signal, frequency).get_data()

                pqrst_data = PQRSTDetection(baseline_signal, fs=frequency).get_data()
                r_index = pqrst_data['R_index']
                q_index = pqrst_data['Q_Index']
                s_index = pqrst_data['S_Index']
                t_index = pqrst_data['T_Index']
                p_index = pqrst_data['P_Index']
                ex_index = pqrst_data['Ex_Index']
                hr_counts = pqrst_data['HR_Count']
                p_t = pqrst_data['P_T List']
                p_r = pqrst_data['PR_Interval']
                p_r_int =  [round(num, 2) for num in p_r]
                # print('pr:',p_r_int)

                afib_per = flutter_per = Normal_per = abanormal_per = noise_per = 0
                afib_data_index = flutter_data_index = []
                r_check_1 = new_rr_check(r_index)
                r_check_2 = check_r_irregular(r_index)
                print("r_check_1 : ", r_check_1, "r_check_2 : ", r_check_2)
                R_label = "Regular"
                if r_check_1 == 'Irregular' and r_check_2 == 'Irregular':
                    R_label = "Irregular"
                
                final_label = 'Abnormal'

                if hr_counts >= 60:
                    afib_flutter = afib_flutter_detection(lowpass_signal, r_index, q_index, s_index, p_index, p_t,p_r_int,afib_afl_moedel)
                    abs_afib_flutter_check = afib_flutter.abs_afib_flutter_check()
                    if abs_afib_flutter_check:
                        afib_flutter_per, afib_data_index, flutter_data_index,epoch_afib_index = afib_flutter.get_data()
                        afib_per = int(afib_flutter_per['AFIB'] * 100)
                        flutter_per = int(afib_flutter_per['FLUTTER'] * 100)
                    if afib_per >= 55 and R_label == 'Irregular':
                        final_label = 'AFIB'
                    elif flutter_per >=50:
                        final_label = 'AFLUTTER'
                lead_data['afib_afl_result'] = final_label
                final_result_list.append(final_label)
                all_lead_result_data[lead] = lead_data
        if len(all_lead_data.keys()) > 1:
            flat_list = []
            for element in final_result_list:
                if isinstance(element, list):
                    flat_list.extend(element)
                else:
                    flat_list.append(element)
            counts = Counter(flat_list)
            repeated_elements = [item for item, count in counts.items() if count >= 2]
            afib_afl_result = ' '.join(repeated_elements)
        else:
            afib_afl_result = all_lead_result_data['II']['afib_afl_result']
    except Exception as e:
        is_error = True
        print("Error of model ckeck", e)
    result_dic = {
        'afib_afl_result': afib_afl_result,
        'hr': hr_counts, 
        'is_error': is_error
    }
    return result_dic
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
        self.set_font("Arial", "B", 14)
        self.cell(0, 10, "Summary", ln=True)
        self.ln(6)

        # Stretch table to full page width
        page_width = self.w - self.l_margin - self.r_margin
        col_widths = [
            page_width * 0.30,  # Filename
            page_width * 0.30,  # Label
            page_width * 0.20,  # Occurrence
            page_width * 0.20,  # HR
        ]

        # Table headers
        self.set_font("Arial", "B", 12)
        headers = ["Filename", "Label", "Occurrence", "HR"]
        for w, h in zip(col_widths, headers):
            self.cell(w, 10, h, 1, align="C")
        self.ln()

        # Table rows
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
    total_time = 10
    ax.set_xlim(0, total_time)
    ax.set_xticks(np.arange(0, total_time, 0.2))
    ax.set_yticks(np.arange(-50, 91, 5))
    ax.set_xticks(np.arange(0, total_time + 0.01, 0.04), minor=True)
    ax.set_yticks(np.arange(-50, 91, 1), minor=True)
    ax.grid(True, which='major', color='red', linewidth=0.6, alpha=0.7)
    ax.grid(True, which='minor', color='pink', linewidth=0.3, alpha=0.3)
    ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)

def plotting(arrhythmia_result ,all_leads_data,local_name,is_lead_for, pdf, label_counts,hr_counts):
    if is_lead_for == '7_Lead':
        limb_leads = ['I', 'III', 'aVL', 'v5']
        chest_leads = ['II', 'aVR', 'aVF']
    else:
        limb_leads = ['I', 'II', 'III', 'aVR', 'aVL', 'aVF']
        chest_leads = ['v1', 'v2', 'v3', 'v4', 'v5', 'v6']

    if is_lead_for == "2_Lead":
        fig_size=(8,4)
    else:
        fig_size = (14, 10)

    voltage_gain = 10 
    sampling_rate = 500 
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

    max_length = max([len(all_leads_data[lead]) for lead in all_leads_data])

    left_x_scale = np.linspace(0, mid_x-0.1, max_length)
    right_x_scale = np.linspace(mid_x+0.1, 10, max_length) 

    scaler = MinMaxScaler(feature_range=(0, 16))
    for lead in all_leads_data:
        ecg_signal = np.array(all_leads_data[lead]) * voltage_gain
        all_leads_data[lead] = scaler.fit_transform(ecg_signal.reshape(-1, 1)).squeeze()

    for idx, lead in enumerate(limb_leads):
        if lead in all_leads_data:
            ecg_signal = all_leads_data[lead]
            y_offset = base_y_left - (idx * lead_spacing)
            ax.plot(left_x_scale, ecg_signal + y_offset, label=lead, color='black', linewidth=1)
            ax.text(left_x_scale[0] + 0.1, y_offset + 1, lead, fontsize=12, fontweight='bold', color='blue')

    for idx, lead in enumerate(chest_leads):
        if lead in all_leads_data:
            ecg_signal = all_leads_data[lead]
            y_offset = base_y_right - (idx * lead_spacing)
            ax.plot(right_x_scale, ecg_signal + y_offset, label=lead, color='black', linewidth=1)
            ax.text(right_x_scale[0] + 0.1, y_offset + 1, lead, fontsize=12, fontweight='bold', color='blue')

    hr_value = arrhythmia_result.get("hr")
    print(arrhythmia_result.keys())
    if hr_value is not None:
        if local_name not in label_counts:
            label_counts[local_name] = {"hr_list": []}
        label_counts[local_name]["hr_list"].append(hr_value)
    
    footer_text = f"{arrhythmia_result['afib_afl_result']}"
    plt.figtext(0.5, 0.01, footer_text, wrap=True, ha='center', fontsize=12)
    plt.tight_layout(rect=[0, 0.03, 1, 0.97])
    plt.show()
    pdf_buffer = BytesIO()
    fig.savefig(pdf_buffer, format="pdf")
    pdf_buffer.seek(0)
    plt.close()

    # HR + label counts
    hr_value = arrhythmia_result.get("hr")
    label_counts.setdefault(local_name, {}).setdefault("hr_list", []).append(hr_value)

    labels = arrhythmia_result.get("afib_afl_result", "").lower().split(";")
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

def model_check_for_ecg_data(file_path, is_lead_for):
    afib_model = load_tflite_model(os.getenv("AFIB"))
    frequency = 200

    label_counts = {}
    hr_counts = {}
    ecg_pdf_buffers = []

    for fn in glob.glob(file_path):
        file_name = os.path.basename(fn).split(".csv")[0]
        patient_id = file_name.split("_")[0]

        all_lead_data = pd.read_csv(fn).fillna(0)
        steps = round(frequency * 10)
        i = 0

        while i < len(all_lead_data):
            ecg_data = all_lead_data[i:i+steps]
            if len(ecg_data) < frequency * 2.5:
                break

            local_name = f"{file_name}_{i}"
            result = afib_afl_abs_model_check(ecg_data, afib_model)

            buf = plotting(
                result,
                ecg_data,
                local_name,
                is_lead_for,
                None,
                label_counts,
                hr_counts
            )
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
        arrhythmia="AFIB/AFL",
        pdf_buffer=final_pdf_buffer
    )

    print("Final report stored in DB:", file_id)
    return file_id

