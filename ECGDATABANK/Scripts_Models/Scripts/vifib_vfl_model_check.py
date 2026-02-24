import random
import pandas as pd
import numpy as np
import tensorflow as tf
from scipy import signal
import matplotlib.pyplot as plt
import glob
import warnings
import threading
from collections import Counter
import re
import scipy
from scipy import signal
import os
import cv2
from analysis_tool import tools as st
from analysis_tool  import utils
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
from sklearn.preprocessing import MinMaxScaler
load_dotenv()   # loads .env
warnings.filterwarnings('ignore')

results_lock = threading.RLock()
mongo_client = MongoClient(os.getenv("MONGO_HOST"))

media_db = mongo_client["Download_files"]
admin_db=mongo_client["admin"]
download_fs = gridfs.GridFS(media_db, collection="downloads")

logs_collection = admin_db["pdf_logs"]

results_lock = threading.RLock()

def load_tflite_model(model_path):
    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    return interpreter, input_details, output_details
 
 
def predict_tflite_model(model: tuple, input_data: tuple):
    with results_lock:
        interpreter, input_details, output_details = model
        for i in range(len(input_data)):
            interpreter.set_tensor(input_details[i]['index'], input_data[i])
        interpreter.invoke()
        output = interpreter.get_tensor(output_details[0]['index'])
 
    return output

def lowpass(ecg_signal, cutoff=0.2):
    b, a = signal.butter(3, cutoff, btype='lowpass', analog=False)
    low_passed = signal.filtfilt(b, a, ecg_signal)
    return low_passed
 
 
def baseline_construction_200(ecg_signal, kernel_size=101):
    s_corrected = signal.detrend(ecg_signal)
    baseline_corrected = s_corrected - signal.medfilt(s_corrected, kernel_size)
    return baseline_corrected

def detect_beats(ecg, rate, ransac_window_size=3.35, lowfreq=5.0, highfreq=13.0):
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
    lp_energy = scipy.ndimage.gaussian_filter1d(lp_energy, rate / 14.0)
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

# Rhythm is positive or negative check
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

        # check inputs
        if self.ecg_signal is None:
            raise TypeError("Please specify an input signal.")

        sampling_rate = float(self.fs)
        length = len(self.ecg_signal)
        dur = length / sampling_rate

        # algorithm parameters
        v1s = int(1.0 * sampling_rate)
        v100ms = int(0.1 * sampling_rate)
        TH_elapsed = np.ceil(0.36 * sampling_rate)
        sm_size = int(0.08 * sampling_rate)
        init_ecg = 10 # seconds for initialization
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
    
def hr_count(ecg_signal, r_index):
    cal_sec = round(ecg_signal.shape[0]/200)
    if cal_sec != 0:
        hr = round(r_index.shape[0]*60/cal_sec)
        return hr
    return 0

def prediction_model_vfib_vfl(input_arr, let_inf_moedel):
    classes = ['VFIB', 'asystole', 'noise', 'normal']
    input_arr = tf.io.decode_jpeg(tf.io.read_file(input_arr), channels=3)
    input_arr = tf.image.resize(input_arr, size=(224, 224), method=tf.image.ResizeMethod.BILINEAR)
    input_arr = (tf.expand_dims(input_arr, axis=0),)
    model_pred = predict_tflite_model(let_inf_moedel, input_arr )[0]
    idx = np.argmax(model_pred)
    return model_pred, classes[idx]
 
def extract_number(filename):
    match = re.search(r'(\d+)', os.path.basename(filename))
    return int(match.group(1)) if match else float('inf')
 
def check_vfib_vfl_model(low_ecg_signal, let_inf_moedel):
    label = 'Abnormal'
    img_path = r'D:\12-12\analysis_tool\Temp_image'
    temp_vifib_img_path = os.path.join( img_path, 'temp_pvc_img')
    os.makedirs(temp_vifib_img_path, exist_ok=True)
    # Remove existing images in the directory
    for i in glob.glob(os.path.join(temp_vifib_img_path, "*.jpg")):
        os.remove(i)

    total_data = len(low_ecg_signal)
    fi = 0
    while fi < total_data:
        if fi + 1000 < total_data:
            temp_img = low_ecg_signal[fi:fi + 1000]
        else:
            temp_img = low_ecg_signal[fi:]
        
        fi += 1000

        # Save and resize the plot
        plt.figure()
        plt.plot(temp_img)
        plt.axis("off")
        
        save_path = os.path.join(temp_vifib_img_path, f"p_{fi}.jpg")
        plt.savefig(save_path)
        
        aq = cv2.imread(save_path)
        aq = cv2.resize(aq, (640, 480))
        cv2.imwrite(save_path, aq)
        
        plt.close()
 
    combine_result = []
    label = 'Abnormal'
 
    # files = sorted(glob.glob("vflutter_img/*.jpg"), key=extract_number)
    files = sorted(glob.glob(os.path.join(temp_vifib_img_path, "*.jpg")), key=extract_number)
    for vfib_file in files:
        predictions, ids = prediction_model_vfib_vfl(vfib_file, let_inf_moedel)
        print(predictions, ids)
        label = "Abnormal" #"Normal"
        if str(ids) == "VFIB" and float(predictions[0]) > 0.75:
            label = "VFIB"
            combine_result.append(label)
        if str(ids) == "asystole" and float(predictions[1]) > 0.75:
            label = "Asystole"
            combine_result.append(label)
       
        if str(ids) == "noise" and float(predictions[2]) > 0.75:
            label = "Noise"
            combine_result.append(label)
       
        if str(ids) == "normal" and float(predictions[3]) > 0.75:
            label = "Normal"
            combine_result.append(label)
 
    temp_label = list(set(combine_result))
    print(temp_label,"======temp_label")
    if len(temp_label) > 1:
        label='Abnormal'
        if 'Asystole' in temp_label:
            label = 'Asystole'
        elif 'Noise' in temp_label:
            label = 'Noise'
    else:
        label = temp_label[0]
    return label
   
 
def check_vfib_model(all_leads_data, let_inf_moedel):
    fs=200
    vifib_result = 'Abnormal'
    all_lead_det_data = {}
    vifib_results = []
    hr_counts = None
    is_error = False
    try:
        for lead in all_leads_data.columns:
            lead_data = {}
            if lead in ['II', 'III', 'aVF', 'I', 'v5']:
                ecg_signal = all_leads_data[lead].values
                baseline_signal = baseline_construction_200(ecg_signal)
                lowpass_signal = lowpass(baseline_signal)
                mi_model_result = check_vfib_vfl_model(lowpass_signal, let_inf_moedel)
                lead_data['vifib_result'] = mi_model_result
                all_lead_det_data[lead] = lead_data
                rpeaks = detect_beats(lowpass_signal,fs)
                lead_data["r_peaks"]=rpeaks
                hr_counts = hr_count(lowpass_signal, rpeaks)
                # print("Heart Rate: ",hr_counts)
                lead_data['hr'] = hr_counts
                vifib_results.append(mi_model_result)
        if len(all_lead_det_data.keys()) > 1:
            flat_list = []
            for element in vifib_results:
                if isinstance(element, list):
                    flat_list.extend(element)
                else:
                    flat_list.append(element)
            counts = Counter(flat_list)
            repeated_elements = [item for item, count in counts.items() if count >= 2 and item != 'Abnormal']
            block_lab = ' '.join(repeated_elements)
            if block_lab:
                vifib_result = block_lab
            else:
                vifib_result = "Abnormal"
           
        else:
            vifib_result=  all_lead_det_data['II']['vifib_result']
    except Exception as e:
        is_error = True
        print("Vfib_error:", e)
    result_dic = {
        'vifib_result': vifib_result,
        'hr': hr_counts,
        'is_error': is_error
    }
    return result_dic
 
# class PDF(FPDF):
#     def header(self):
#         self.set_font("Arial", "B", 12)
#         self.cell(0, 10, "Arrhythmia Report", align="C", ln=True)

#     def footer(self):
#         self.set_y(-15)
#         self.set_font("Arial", "I", 8)
#         self.cell(0, 10, f"Page {self.page_no()}", align="C")

#     def add_summary(self, label_counts,hr_counts):
#       self.add_page()
#       self.set_font("Arial", "", 12)
#       self.cell(0, 10, "Summary", ln=True)
#       self.ln(10)
  
#       # Define consistent column widths
#       col_widths = [60, 60, 30, 30]  # Adjusted widths to fit page nicely
  
#       # Table headers
#       self.set_font("Arial", "B", 12)
#       self.cell(col_widths[0], 10, "Filename", 1, align="C")
#       self.cell(col_widths[1], 10, "Label", 1, align="C")
#       self.cell(col_widths[2], 10, "Count", 1, align="C")
#       self.cell(col_widths[3], 10, "HR", 1, align="C")
#       self.ln()
  
#       # Table content
#       self.set_font("Arial", "", 12)
#       for filename, labels in label_counts.items():
#           for label, count in labels.items():
#               if label == "hr":
#                   continue  # skip HR from label rows
#               hr = labels.get("hr", "-")
#               self.cell(col_widths[0], 10, filename, 1)
#               self.cell(col_widths[1], 10, label.capitalize(), 1)
#               self.cell(col_widths[2], 10, str(count), 1, align="C")
#               self.cell(col_widths[3], 10, str(hr), 1, align="C")
#               self.ln()
              
# def preprocess_label_counts(label_counts):
#     from collections import defaultdict
#     processed_counts = defaultdict(lambda: defaultdict(int))

#     for full_filename, labels in label_counts.items():
#         base_filename = full_filename.split("_")[0].lower()
#         for label, count in labels.items():
#             if label == "hr_list":
#                 if "hr_values" not in processed_counts[base_filename]:
#                     processed_counts[base_filename]["hr_values"] = []
#                 processed_counts[base_filename]["hr_values"].extend(count)
#             else:
#                 processed_counts[base_filename][label] += count

#     # Compute average HR from hr_values
#     for fname in processed_counts:
#         if "hr_values" in processed_counts[fname]:
#             hr_list = processed_counts[fname]["hr_values"]
#             avg_hr = round(sum(hr_list) / len(hr_list))
#             processed_counts[fname]["hr"] = avg_hr
#             del processed_counts[fname]["hr_values"]
#         else:
#             processed_counts[fname]["hr"] = "-"

#     return processed_counts
    
# def setup_ecg_subplot(ax):
#     total_time = 10  # seconds
#     ax.set_xlim(0, total_time)

#     # major grid: X every 0.2s, Y every 0.5mV (assumed 1 unit = 0.1mV ? 5 units = 0.5mV)
#     ax.set_xticks(np.arange(0, total_time, 0.2))
#     ax.set_yticks(np.arange(-50, 91, 5))  # 5 units ~ 0.5mV (if scaled appropriately)

#     # Minor grid: X every 0.04s, Y every 1 unit (~0.1mV)
#     ax.set_xticks(np.arange(0, total_time + 0.01, 0.04), minor=True)
#     ax.set_yticks(np.arange(-50, 91, 1), minor=True)

#     # Major grid styling (bold red)
#     ax.grid(True, which='major', color='red', linewidth=0.6, alpha=0.7)

#     # Minor grid styling (light red)
#     ax.grid(True, which='minor', color='pink', linewidth=0.3, alpha=0.3)

#     # Turn off ticks and labels
#     ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)

# def plotting(arrhythmia_result ,all_leads_data, save_path, local_name,is_lead_for, pdf, label_counts,hr_counts):
#     if is_lead_for == '7_Lead':
#         limb_leads = ['I', 'III', 'aVL', 'v5']
#         chest_leads = ['II', 'aVR', 'aVF']
#     else:
#         limb_leads = ['I', 'II', 'III', 'aVR', 'aVL', 'aVF']  # Left side leads
#         chest_leads = ['v1', 'v2', 'v3', 'v4', 'v5', 'v6']
#     # Dynamic figure size based on lead configuration
#     if is_lead_for == "2_Lead":
#         fig_size=(8,4)  # Smaller size for 2 leads
#     else:
#         fig_size = (14, 10)  # Standard size for 7 leads

#     voltage_gain = 10  # mm/mV for voltage gain
#     sampling_rate = 500  # Sampling rate in Hz
#     fig, ax = plt.subplots(figsize=fig_size, dpi=100)

#     # Adjust vertical limits and lead spacing for better visualization
#     ax.set_ylim(-50, 90)  # Vertical limit for better signal visualization
#     lead_spacing = 20  # Vertical spacing between leads
#     base_y_left = 70  # Starting Y position for left leads
#     base_y_right = 70  # Starting Y position for right leads
#     mid_x = 5  # Mid-point for dividing left and right leads on the time axis

#     # Draw ECG Grid (you can customize this function for gridlines)
#     setup_ecg_subplot(ax)
#     ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
#     ax.set_xlabel("")
#     ax.set_ylabel("")

#     # Get the maximum length of the ECG signal from any of the leads in `all_leads_data`
#     max_length = max([len(all_leads_data[lead]) for lead in all_leads_data])

#     # Generate time scale to fit the full signal
#     left_x_scale = np.linspace(0, mid_x-0.1, max_length)  # Expanded to fit full width for left leads
#     right_x_scale = np.linspace(mid_x+0.1, 10, max_length)  # Expanded to fit full width for right leads

#     # Apply MinMaxScaler globally to normalize all leads' data for consistent scaling
#     scaler = MinMaxScaler(feature_range=(0, 16))  # Adjusting range to 0-8 for better visual representation
#     for lead in all_leads_data:
#         ecg_signal = np.array(all_leads_data[lead]) * voltage_gain
#         all_leads_data[lead] = scaler.fit_transform(ecg_signal.reshape(-1, 1)).squeeze()

#     # Plot Limb Leads (Left Side)
#     for idx, lead in enumerate(limb_leads):
#         if lead in all_leads_data:
#             ecg_signal = all_leads_data[lead]
#             y_offset = base_y_left - (idx * lead_spacing)
#             ax.plot(left_x_scale, ecg_signal + y_offset, label=lead, color='black', linewidth=1)
#             ax.text(left_x_scale[0] + 0.1, y_offset + 1, lead, fontsize=12, fontweight='bold', color='blue')

#     # Plot Chest Leads (Right Side)
#     for idx, lead in enumerate(chest_leads):
#         if lead in all_leads_data:
#             ecg_signal = all_leads_data[lead]
#             y_offset = base_y_right - (idx * lead_spacing)
#             ax.plot(right_x_scale, ecg_signal + y_offset, label=lead, color='black', linewidth=1)
#             ax.text(right_x_scale[0] + 0.1, y_offset + 1, lead, fontsize=12, fontweight='bold', color='blue')

#     # Handle heart rate for further analysis
#     hr_value = arrhythmia_result.get("hr")
#     if hr_value is not None:
#         if local_name not in label_counts:
#             label_counts[local_name] = {"hr_list": []}
#         label_counts[local_name]["hr_list"].append(hr_value)

#     # plt.suptitle(f"{local_name}_{arrhythmia_result['vifib_result']}")
# #    csv_name = 'vifib_testing_model_result.csv'
# #
# #    with open(save_path + csv_name, 'a') as csv:
# #         np.savetxt(csv, np.array([local_name, arrhythmia_result['vifib_result'],'II, III, aVF, v5']).reshape(1,-1), delimiter= ',', fmt = '%s', comments ='')
    
#     footer_text = f"{arrhythmia_result['vifib_result']}"
#     plt.figtext(0.5, 0.01, footer_text, wrap=True, ha='center', fontsize=12)
#     plt.tight_layout(rect=[0, 0.03, 1, 0.97])
#     plt.show()
#     fig.savefig(f"{save_path}{local_name}.pdf")
#     plt.close()


#     labels = arrhythmia_result.get("vifib_result", "").lower().split(";")
#     for label in labels:
#         label = label.strip()
#         if label:
#             if local_name not in label_counts:
#                 label_counts[local_name] = {}
#             label_counts[local_name][label] = label_counts[local_name].get(label, 0) + 1
# def model_check_for_ecg_data(file_path, is_lead_for, save_result):
#     let_inf_moedel = load_tflite_model(os.getenv("VIFIB")) # add your model path
#     frequency = 200
#     label_counts = {}
#     hr_counts={}
#     pdf = PDF()
#     for fn in glob.glob(file_path):
#         file_name = fn.split("\\")[-1].split(".csv")[0]
#         all_lead_data = pd.read_csv(fn, header=None).fillna(0)[:10000]
#         # print("is_lead_for: ",is_lead_for)
#         if any(str(_).isalpha() for _ in all_lead_data.iloc[0, :].values):
            
#             if is_lead_for == '2_Lead':
                
#                 all_lead_data = pd.read_csv(fn, usecols=['ECG']).fillna(0) #[:1000]
#                 all_lead_data = all_lead_data.rename(columns={'ECG': 'II'})
#             elif is_lead_for == '7_Lead':
#                 all_lead_data = pd.read_csv(fn, usecols=['I', 'II', 'III', 'aVR', 'aVL', 'aVF', 'v5']).fillna(0)
#             elif is_lead_for == '12_Lead':
#                 all_lead_data = pd.read_csv(fn, usecols=['I', 'II', 'III', 'aVR', 'aVL', 'aVF', 'v1', 'v2','v3', 'v4', 'v5', 'v6']).fillna(0)
#         else:
#             if is_lead_for == '2_Lead':
#                 all_lead_data = all_lead_data.rename(columns={'ECG':'II'})
#             elif is_lead_for == '7_Lead':
#                 all_lead_data = all_lead_data.rename(columns={0:'I', 1:'II', 2:'III', 3: 'aVR', 4: 'aVL', 5:'aVF',  6:'v5'})
#             elif is_lead_for == '12_Lead':  
#                 all_lead_data = all_lead_data.rename(columns={0:'I', 1:'II', 2:'III', 3: 'aVR', 4: 'aVL', 5:'aVF', 6: 'v1', 7:'v2', 8:'v3', 9:'v4', 10:'v5', 11:'v6'})  
           
#         i = 0
#         if all_lead_data.shape[0] <= 2500:
#             steps = all_lead_data.shape[0]
#         else:
#             steps = round(frequency * 10)
 
#         while i < all_lead_data.shape[0]:
#             ecg_data = all_lead_data[i : i+steps]
#             if ecg_data.shape[0] < frequency*2.5:
#                 print("<<<<<<<<<<<<<<<<<<<<<<<<< Less data for analysis >>>>>>>>>>>>>>>>>>>>>>>>>>>")
#                 break
#             local_name = f"{file_name}_{i}"
#             arrhythmia_detector = check_vfib_model(ecg_data, let_inf_moedel)
#             plotting(arrhythmia_detector ,ecg_data, save_result, local_name,is_lead_for, pdf, label_counts,hr_counts)
#             i += steps
#     final_counts = preprocess_label_counts(label_counts)
#     pdf.add_summary(final_counts,hr_counts)
# #    pdf.add_summary(label_counts)
#     final_report_path = save_result + "Final_Combined_Report.pdf"
#     pdf.output(final_report_path)
#     return arrhythmia_detector
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
        self.set_font("Arial", "", 12)
        self.cell(0, 10, "Summary", ln=True)
        self.ln(8)
  
      # Stretch table to full page width
        page_width = self.w - self.l_margin - self.r_margin
        col_widths = [
            page_width * 0.40,  # Filename
            page_width * 0.30,  # Label
            page_width * 0.15,  # Occurrence
            page_width * 0.15,  # HR
        ]
  
      # Table headers
        self.set_font("Arial", "B", 12)
        headers = ["Filename", "Label", "Occurrence", "HR"]
        for w, h in zip(col_widths, headers):
                self.cell(w, 10, h, 1, align="C")
        self.ln()
  
    
        # Table content
        self.set_font("Arial", "", 11)
        for filename, labels in label_counts.items():
            for label, count in labels.items():
                if label == "hr":
                    continue  # skip HR from label rows
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

def plotting(arrhythmia_result ,all_leads_data, local_name,is_lead_for, pdf, label_counts,hr_counts):
    if is_lead_for == '7_Lead':
        limb_leads = ['I', 'III', 'aVL', 'v5']
        chest_leads = ['II', 'aVR', 'aVF']
    else:
        limb_leads = ['I', 'II', 'III', 'aVR', 'aVL', 'aVF']  # Left side leads
        chest_leads = ['v1', 'v2', 'v3', 'v4', 'v5', 'v6']
    # Dynamic figure size based on lead configuration
    if is_lead_for == "2_Lead":
        fig_size=(8,4)  # Smaller size for 2 leads
    else:
        fig_size = (14, 10)  # Standard size for 7 leads

    voltage_gain = 10  # mm/mV for voltage gain
    sampling_rate = 500  # Sampling rate in Hz
    fig, ax = plt.subplots(figsize=fig_size, dpi=100)

    # Adjust vertical limits and lead spacing for better visualization
    ax.set_ylim(-50, 90)  # Vertical limit for better signal visualization
    lead_spacing = 20  # Vertical spacing between leads
    base_y_left = 70  # Starting Y position for left leads
    base_y_right = 70  # Starting Y position for right leads
    mid_x = 5  # Mid-point for dividing left and right leads on the time axis

    # Draw ECG Grid (you can customize this function for gridlines)
    setup_ecg_subplot(ax)
    ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
    ax.set_xlabel("")
    ax.set_ylabel("")

    # Get the maximum length of the ECG signal from any of the leads in `all_leads_data`
    max_length = max([len(all_leads_data[lead]) for lead in all_leads_data])

    # Generate time scale to fit the full signal
    left_x_scale = np.linspace(0, mid_x-0.1, max_length)  # Expanded to fit full width for left leads
    right_x_scale = np.linspace(mid_x+0.1, 10, max_length)  # Expanded to fit full width for right leads

    # Apply MinMaxScaler globally to normalize all leads' data for consistent scaling
    scaler = MinMaxScaler(feature_range=(0, 16))  # Adjusting range to 0-8 for better visual representation
    for lead in all_leads_data:
        ecg_signal = np.array(all_leads_data[lead]) * voltage_gain
        all_leads_data[lead] = scaler.fit_transform(ecg_signal.reshape(-1, 1)).squeeze()

    # Plot Limb Leads (Left Side)
    for idx, lead in enumerate(limb_leads):
        if lead in all_leads_data:
            ecg_signal = all_leads_data[lead]
            y_offset = base_y_left - (idx * lead_spacing)
            ax.plot(left_x_scale, ecg_signal + y_offset, label=lead, color='black', linewidth=1)
            ax.text(left_x_scale[0] + 0.1, y_offset + 1, lead, fontsize=12, fontweight='bold', color='blue')

    # Plot Chest Leads (Right Side)
    for idx, lead in enumerate(chest_leads):
        if lead in all_leads_data:
            ecg_signal = all_leads_data[lead]
            y_offset = base_y_right - (idx * lead_spacing)
            ax.plot(right_x_scale, ecg_signal + y_offset, label=lead, color='black', linewidth=1)
            ax.text(right_x_scale[0] + 0.1, y_offset + 1, lead, fontsize=12, fontweight='bold', color='blue')

    # Handle heart rate for further analysis
    hr_value = arrhythmia_result.get("hr")
    if hr_value is not None:
        if local_name not in label_counts:
            label_counts[local_name] = {"hr_list": []}
        label_counts[local_name]["hr_list"].append(hr_value)

    footer_text = f"{arrhythmia_result['vifib_result']}"
    plt.figtext(0.5, 0.01, footer_text, wrap=True, ha='center', fontsize=12)
    plt.tight_layout(rect=[0, 0.03, 1, 0.97])
    plt.show()
    pdf_buffer = BytesIO()
    fig.savefig(pdf_buffer, format="pdf")
    pdf_buffer.seek(0)
    plt.close()
   
    # Update 
    hr_value = arrhythmia_result.get("hr")
    label_counts.setdefault(local_name, {}).setdefault("hr_list", []).append(hr_value)

    labels = arrhythmia_result.get("vifib_result", "").lower().split(";")
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
def store_final_pdf_in_db(patient_id, arrhythmia, pdf_buffer):
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
    let_inf_moedel = load_tflite_model(os.getenv("VIFIB")) # add your model path
    frequency = 200
    label_counts = {}
    hr_counts={}
    ecg_pdf_buffers = []
    for fn in glob.glob(file_path):
        file_name = fn.split("\\")[-1].split(".csv")[0]
        patient_id = file_name.split("_")[0]
        all_lead_data = pd.read_csv(fn, header=None).fillna(0)[:10000]
        # print("is_lead_for: ",is_lead_for)
        if any(str(_).isalpha() for _ in all_lead_data.iloc[0, :].values):
            
            if is_lead_for == '2_Lead':
                
                all_lead_data = pd.read_csv(fn, usecols=['ECG']).fillna(0) #[:1000]
                all_lead_data = all_lead_data.rename(columns={'ECG': 'II'})
            elif is_lead_for == '7_Lead':
                all_lead_data = pd.read_csv(fn, usecols=['I', 'II', 'III', 'aVR', 'aVL', 'aVF', 'v5']).fillna(0)
            elif is_lead_for == '12_Lead':
                all_lead_data = pd.read_csv(fn, usecols=['I', 'II', 'III', 'aVR', 'aVL', 'aVF', 'v1', 'v2','v3', 'v4', 'v5', 'v6']).fillna(0)
        else:
            if is_lead_for == '2_Lead':
                all_lead_data = all_lead_data.rename(columns={'ECG':'II'})
            elif is_lead_for == '7_Lead':
                all_lead_data = all_lead_data.rename(columns={0:'I', 1:'II', 2:'III', 3: 'aVR', 4: 'aVL', 5:'aVF',  6:'v5'})
            elif is_lead_for == '12_Lead':  
                all_lead_data = all_lead_data.rename(columns={0:'I', 1:'II', 2:'III', 3: 'aVR', 4: 'aVL', 5:'aVF', 6: 'v1', 7:'v2', 8:'v3', 9:'v4', 10:'v5', 11:'v6'})  
           
        i = 0
        if all_lead_data.shape[0] <= 2500:
            steps = all_lead_data.shape[0]
        else:
            steps = round(frequency * 10)
 
        while i < all_lead_data.shape[0]:
            ecg_data = all_lead_data[i : i+steps]
            if ecg_data.shape[0] < frequency*2.5:
                break
            local_name = f"{file_name}_{i}"
            arrhythmia_detector = check_vfib_model(ecg_data, let_inf_moedel)
            buf = plotting(arrhythmia_detector ,ecg_data, local_name,is_lead_for, None, label_counts,hr_counts)
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
        arrhythmia="Vifib",
        pdf_buffer=final_pdf_buffer
    )

    print("Final report stored in DB:", file_id)
    return file_id
