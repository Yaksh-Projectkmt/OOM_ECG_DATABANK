import base64
import math
import time
import random
import uuid
from itertools import groupby
from operator import itemgetter
from pathlib import Path
import pandas as pd
import matplotlib.ticker as tickero
import numpy as np
import tensorflow as tf
import shutil
import easyocr
from scipy.ndimage import gaussian_filter1d
from matplotlib.gridspec import GridSpec
from skimage import io, color
from skimage.transform import rotate, hough_line, hough_line_peaks
from scipy import signal
from scipy import sparse
from numpy import trapz
from scipy.sparse.linalg import spsolve
from scipy.signal import (find_peaks, firwin, medfilt, butter, filtfilt)
import pywt
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from matplotlib import colormaps
from scipy.interpolate import interp1d
import glob
from analysis_tool import tools as st
from analysis_tool import utils
import neurokit2 as nk
import cv2
import warnings
import threading
from biosppy.signals import ecg as hami
import scipy
import os
from PIL import Image, ExifTags
from collections import Counter
from .detectron2.data import MetadataCatalog
from .detectron2.data.datasets import register_coco_instances
from .detectron2.engine import DefaultPredictor
from .detectron2.config import get_cfg
from .detectron2.utils.visualizer import Visualizer
from .detectron2.structures import Instances, Boxes
import torch
import re
from collections import defaultdict
from django.conf import settings

# Ignore specific FutureWarnings
warnings.filterwarnings("ignore")
base_dir_path = os.path.join(settings.BASE_DIR, 'analysis_tool', 'oea')
results_lock = threading.RLock()


def load_tflite_model(model_path):
    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    return interpreter, input_details, output_details

img_interpreter, img_input_details, img_output_details = load_tflite_model(base_dir_path +"\\Models\\restingecgModel_autoGrid_9.tflite")
interpreter = tf.lite.Interpreter(model_path= base_dir_path +"\\Models\\PVC_Trans_mob_40_test_tiny_iter1_OEA.tflite")
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
pac_model = load_tflite_model(base_dir_path + "\\Models\\PAC_TRANS_GRU_mob_23_OEA.tflite")
afib_model = load_tflite_model(base_dir_path + "\\Models\\afib_flutter_12_lead_3_9.tflite")
vfib_vfl_model = load_tflite_model(base_dir_path + "\\Models\\VFIB_Model_07JUN2024_1038.tflite")
block_model = load_tflite_model(base_dir_path + "\\Models\\Block_convex_2.tflite")
noise_model = load_tflite_model(base_dir_path + "\\Models\\NOISE_16_GPT.tflite")
let_inf_moedel = load_tflite_model(base_dir_path + "\\Models\\ST_21_10.tflite")
reader = easyocr.Reader(['en'])

# For grid in lead detection
Lead_list = ["I", "II", "III", "V1", "V2", "V3", "V4", "V5", "V6", "aVF", "aVL", "aVR"]
# register_coco_instances("my_dataset", {}, '20-09_PM_updated_annotations.coco.json', "New PM Cardio Train")
# MetadataCatalog.get("my_dataset").set(thing_classes=Lead_list)

MODEL_PATHS = {
    "6_2": base_dir_path + "\\Models\\model_final_29_01_R_101_FPN_3x.pth",
    "3_4": base_dir_path + "\\detectron_part_8OCT\\model_final_3X4_29_05_25.pth",
    "12_1":base_dir_path + "\\detectron_part_8OCT\\model_final_12X1_19_05_25.pth"
}

def load_object_detection_model(grid_type, img):
    cfg = get_cfg()
    if grid_type in ["3_4", "12_1", "6_2"]:
        cfg.merge_from_file(base_dir_path+"\\detectron2\\configs\\COCO-Detection\\faster_rcnn_R_101_FPN_3x.yaml")
    cfg.MODEL.WEIGHTS = MODEL_PATHS[grid_type]
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = len(Lead_list)
    cfg.DATASETS.TEST = ("my_dataset",)
    cfg.MODEL.DEVICE = "cpu"
    MetadataCatalog.get("my_dataset").set(thing_classes=Lead_list)
    return DefaultPredictor(cfg)


def lowpass(file, cutoff=0.4):
    b, a = signal.butter(3, cutoff, btype='lowpass', analog=False)
    low_passed = signal.filtfilt(b, a, file)
    return low_passed


def baseline_construction_200(ecg_signal, kernel_size=101):
    s_corrected = signal.detrend(ecg_signal)
    baseline_corrected = s_corrected - signal.medfilt(s_corrected, kernel_size)
    return baseline_corrected


def force_remove_folder(folder_path):
    """Forcefully remove a folder, even if files are read-only."""

    def onerror(func, path, exc_info):
        # Change the permission and retry
        os.chmod(path, 0o777)  # Grant full permissions
        func(path)

    shutil.rmtree(folder_path, onerror=onerror)


class NoiseDetection:
    def __init__(self, raw_data, class_name, frequency=200):
        self.frequency = frequency
        self.raw_data = raw_data
        self.class_name = class_name

    def prediction_model(self, input_arr):
        classes = ['Noise', 'Normal']
        input_arr = tf.cast(input_arr, dtype=tf.float32)
        input_arr = tf.image.resize(input_arr, size=(224, 224), method=tf.image.ResizeMethod.BILINEAR)
        input_arr = (tf.expand_dims(input_arr, axis=0),)
        model_pred = predict_tflite_model(noise_model, input_arr)[0]
        idx = np.argmax(model_pred)
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

    def noise_model_check(self):
        # Noise detection logic for individual lead
        if self.class_name == '12_1':
            steps_data = int(self.frequency * 5)
        else:
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

            # plt.plot(img_data);plt.show()

            # Get noise model result for the image
            model_result = self.prediction_model(data1)
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
            elif percentage['Noise'] / percentage['total_slice'] >= 0.6:
                noise_label = 'ARTIFACTS'

        return noise_label


# Peak detection
class pqrst_detection:

    def __init__(self, ecg_signal, class_name='6_2', fs=200, thres=0.5, lp_thres=0.2, rr_thres=0.12, width=(5, 50),
                 JR=False):
        self.ecg_signal = ecg_signal
        self.fs = fs
        self.thres = thres
        self.lp_thres = lp_thres
        self.rr_thres = rr_thres
        self.width = width
        self.JR = JR
        self.class_name = class_name

    def hamilton_segmenter(self):

        # check inputs
        if self.ecg_signal is None:
            print("Please specify an input signal.")
        ##            raise TypeError("Please specify an input signal.")

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

    def hr_count(self, class_name):
        cal_sec = 5
        if class_name == '6_2':
            cal_sec = 5
        elif class_name == '12_1':
            cal_sec = 10
        elif class_name == '3_4':
            cal_sec = 2.5
        if cal_sec != 0:
            hr = round(self.r_index.shape[0] * 60 / cal_sec)
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

    # def find_q_index(self, d):
    #     """The Q wave index in an ECG signal given the R wave index and a specified
    #     distance.
    #
    #     Args:
    #         ecg (array): ECG signal values
    #         R_index (array/list): R peak indices in the ECG signal
    #         d (int): The maximum distance (in samples) between the R peak and the Q wave onset that we want to find.
    #
    #     Returns:
    #         list: Q-wave indices for each R-wave index in the ECG signal.
    #     """
    #     d = int(d) + 1
    #     q = []
    #     for i in self.r_index:
    #         if i == 0:
    #             q.append(i)
    #             continue
    #         elif 0 <= i - d:
    #             q_array = self.ecg_signal[i - d:i]
    #         else:
    #             q_array = self.ecg_signal[:i]
    #         if self.ecg_signal[i] > 0:
    #             q_index = i - (len(q_array) - np.where(q_array == min(q_array))[0][0])
    #         else:
    #             q_index = i - (len(q_array) - np.where(q_array == max(q_array))[0][0])
    #         q.append(q_index)
    #     return np.sort(q)

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
        """Finds R-peaks in an ECG signal using the Hamilton segmenter algorithm.

        Args:
            ecg_signal (array): The ECG signal of numpy array
            fs (int, optional): sampling rate of the ECG signal, defaults to 200 (optional)

        Returns:
            list: the R-peak indices of the ECG signal using the Hamilton QRS complex detector algorithm.
        """
        r_ = []
        out = self.hamilton_segmenter()
        self.r_index = out["rpeaks"]
        heart_rate = self.hr_count(self.class_name)
        if self.JR:  # ---------------------
            diff_indexs = abs(round((self.fs * 0.4492537) + (heart_rate * -1.05518351) + 40.40601032654332))
        else:  # ---------------------
            diff_indexs = abs(round((self.fs * 0.4492537) + (heart_rate * -1.009) + 58.40601032654332))

        for r in self.r_index:
            if r - diff_indexs >= 0 and len(self.ecg_signal) >= r + diff_indexs:
                data = self.ecg_signal[r - diff_indexs:r + diff_indexs]
                abs_data = np.abs(data)
                r_.append(np.where(abs_data == max(abs_data))[0][0] + r - diff_indexs)
            else:
                r_.append(r)

        new_r = np.unique(r_) if r_ else self.r_index
        fs_diff = int((25 * self.fs) / 200)
        final_r = []
        if new_r.any(): final_r = [new_r[0]] + [new_r[j + 1] for j, i in enumerate(np.diff(new_r)) if i >= fs_diff]
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
        # if self.rr_thres == 0.15:
        #     self.ecg_signal = lowpass(self.ecg_signal,0.2)
        self.hr_ = self.hr_count(self.class_name)
        sd, qd = int(self.fs * 0.115), int(self.fs * 0.08)
        self.s_index = self.find_s_index(sd)
        # q_index = find_q_index(ecg_signal, r_index, qd)
        # s_index = find_new_s_index(ecg_signal,r_index,sd)
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

        if self.fs != 200:
            self.ecg_signal = MinMaxScaler(feature_range=(0, 4)).fit_transform(self.ecg_signal.reshape(-1, 1)).squeeze()

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


# PVC detection
class PVC_detection:
    def __init__(self, ecg_signal, fs=100):  # 200
        self.ecg_signal = ecg_signal
        self.fs = fs

    def lowpass(self, file):
        b, a = signal.butter(3, 0.3, btype='lowpass', analog=False)
        low_passed = signal.filtfilt(b, a, file)
        return low_passed

    def baseline_construction_200(self, kernel_size=101):
        s_corrected = signal.detrend(self.ecg_signal)
        baseline_corrected = s_corrected - signal.medfilt(s_corrected, kernel_size)
        return baseline_corrected

    def detect_beats(self,ecg, rate, ransac_window_size=3.35, lowfreq=5.0, highfreq=15.0):
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
            elif abs(max_amplitude + 0.11) < abs(min_amplitude):
                rpeak = np.argmin(local_signal) + search_window.start
            else:
                if max_amplitude >= 0:
                    rpeak = np.argmax(local_signal) + search_window.start
                else:
                    rpeak = np.argmin(local_signal) + search_window.start

            rpeaks.append(rpeak)
        return np.array(rpeaks)


    def calculate_surface_area(self, ecg_signal, qrs_start_index, qrs_end_index, sampling_rate):
        if qrs_start_index == 0 or qrs_end_index == 0:
            surface_area = 0
        else:
            qrs_complex = ecg_signal[qrs_start_index:qrs_end_index]
            absolute_qrs = np.abs(qrs_complex)
            time = np.arange(len(qrs_complex)) / sampling_rate
            surface_area = trapz(absolute_qrs, time)

        return surface_area

    def wide_qrs_find(self):
        wideQRS = []
        difference = []
        surface_area_list = []
        pvc = []
        above_r_peaks = []
        below_r_peaks = []
        for idx in self.r_index:
            if idx < len(self.low_pass_signal):
                if self.low_pass_signal[idx] >= 0:
                    above_r_peaks.append(idx)
                else:
                    below_r_peaks.append(idx)
        # if self.hr_count <= 88:
        #     thresold = round(self.fs * 0.08)  # 0.10
        # else:
        thresold = round(self.fs * 0.12)  # 0.12
        for k in range(len(self.r_index)):
            diff = self.s_index[k] - self.q_index[k]
            if self.r_index[k] in above_r_peaks:
                surface_thres = 0.02  # 0.14 #for OEA=0.02
                wideqs_thres = 0.01  # 0.13 #for OEA=0.01
            elif self.r_index[k] in below_r_peaks:
                surface_thres = 0.02  # for OEA=0.02
                wideqs_thres = 0.02  # for OEA=0.02
            if diff > thresold:
                difference.append(diff)
                wideQRS.append(self.r_index[k])
                surface_area = self.calculate_surface_area(self.low_pass_signal, self.q_index[k], self.s_index[k],
                                                           self.fs)
                if (diff / 100) >= wideqs_thres:
                    surface_area_list.append(round(surface_area, 3))
                    if surface_area >= surface_thres:
                        pvc.append(self.r_index[k])

        if len(difference) != 0:
            q_s_difference = [i / 100 for i in difference]  # 200
        else:
            q_s_difference = np.array([])
        return np.array(wideQRS), q_s_difference, pvc

    def PVC_CLASSIFICATION(self, PVC_R_Peaks):
        vt_counter = 0
        couplet_counter = 0
        triplet_counter = 0
        bigeminy_counter = 0
        trigeminy_counter = 0
        quadrigeminy_counter = 0
        vt = 0
        i = 0
        while i < len(PVC_R_Peaks):
            count = 0
            ones_count = 0
            while i < len(PVC_R_Peaks) and PVC_R_Peaks[i] == 1:
                count += 1
                ones_count += 1
                i += 1

            if count >= 4:
                vt_counter += 1
                vt += ones_count
                count = 0
                ones_count = 0
            if count == 3:
                triplet_counter += 1
            elif count == 2:
                couplet_counter += 1

            i += 1
        j = 0
        while j < len(PVC_R_Peaks) - 1:
            if PVC_R_Peaks[j] == 1:
                k = j + 1
                spaces = 0
                while k < len(PVC_R_Peaks) and PVC_R_Peaks[k] == 0:
                    spaces += 1
                    k += 1

                if k < len(PVC_R_Peaks) and PVC_R_Peaks[k] == 1:
                    if spaces == 1:
                        bigeminy_counter += 1
                    elif spaces == 2:
                        trigeminy_counter += 1
                    elif spaces == 3:
                        quadrigeminy_counter += 1
                j = k

            else:
                j += 1

        total_one = (1 * vt) + (couplet_counter * 2) + (triplet_counter * 3) + (bigeminy_counter * 2) + (
                trigeminy_counter * 2) + (quadrigeminy_counter * 2)
        total = vt_counter + couplet_counter + triplet_counter + bigeminy_counter + trigeminy_counter + quadrigeminy_counter
        ones = PVC_R_Peaks.count(1)
        if total == 0:
            Isolated = ones
        else:
            Common = total - 1
            Isolated = ones - (total_one - Common)
        if vt_counter > 1:
            vt_counter = 1
        return vt_counter, couplet_counter, triplet_counter, bigeminy_counter, trigeminy_counter, quadrigeminy_counter, Isolated, vt

    def VT_confirmation(self, ecg_signal, r_index):
        VTC = []
        pqrst_data = pqrst_detection(ecg_signal)
        for i in range(0, len(r_index) - 1):
            aoi = ecg_signal[r_index[i] - 5:r_index[i + 1]]
            low = pqrst_data.fir_lowpass_filter(aoi, 0.2, 30)
            if aoi.any():
                peaks, _ = find_peaks(low, prominence=0.2, width=(40))
                VTC.append(peaks)
        if round(len(VTC) / len(r_index)) >= .7:
            label = 'VT'
        else:
            label = 'Abnormal'

        return label, len(VTC)

    def prediction_model(self, image_path, target_shape=[224, 224], class_name=True):
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

    def model_r_detectron(self, e_signal, r_index, heart_rate, fs=100):
        pvc_0 = []
        lbbb, rbbb = [], []
        model_pred = []
        counter = 0
        # detect_rpeaks = self.detect_beats(e_signal, float(self.fs))
        detect_rpeaks = r_index


        for i in glob.glob(base_dir_path + "/temp_pvc_img/*.jpg"):
            os.remove(i)
        for r in detect_rpeaks:
            if r == detect_rpeaks[0]:
                windo_start = int(r) - 30
                if windo_start < 0:
                    windo_start = int(r) + 50
                windo_end = int(r) + 100
            elif r == detect_rpeaks[-1]:
                windo_start = int(r) - 50
                windo_end = int(r) + 130
            else:
                if int(r) - 50 > 0:
                    windo_start = int(r) - 50
                else:
                    windo_start = 0
                windo_end = int(r) + 80
            aa = pd.DataFrame(e_signal[windo_start:windo_end])
            plt.plot(aa, color='blue')
            plt.axis("off")
            plt.savefig(f"{base_dir_path}/temp_pvc_img/{r}.jpg")
            aq = cv2.imread(f"{base_dir_path}/temp_pvc_img/{r}.jpg")
            aq = cv2.resize(aq, (360, 720))
            cv2.imwrite(f"{base_dir_path}/temp_pvc_img/{r}.jpg", aq)
            plt.close()

        files = sorted(glob.glob(base_dir_path + "/temp_pvc_img/*.jpg"), key=len)
        for p_files in files:
            predictions, model_label = self.prediction_model(p_files)
            r_peack = int(p_files.split("\\")[-1].split(".")[0])
            if str(model_label) == 'PVC' and float(predictions[3]) > 0.78:  # 0.75
                pvc_0.append(r_peack)
            if str(model_label) == 'LBBB' and float(predictions[0]) > 0.78:
                lbbb.append(r_peack)
            if str(model_label) == 'RBBB' and float(predictions[4]) > 0.78:
                rbbb.append(r_peack)
            model_pred.append((r_peack, (float(predictions[3]), model_label)))
        return pvc_0, lbbb, rbbb, model_pred, detect_rpeaks

    def get_pvc_data(self):
        self.baseline_signal = self.baseline_construction_200(kernel_size=131)
        self.low_pass_signal = self.lowpass(self.baseline_signal)
        lbbb_rbbb_label = "Abnormal"
        pqrst_data = pqrst_detection(self.baseline_signal, fs=self.fs).get_data()
        self.r_index = pqrst_data['R_index']
        self.q_index = pqrst_data['Q_Index']
        self.s_index = pqrst_data['S_Index']
        self.hr_count = pqrst_data['HR_Count']
        self.p_t = pqrst_data['P_T List']
        self.ex_index = pqrst_data['Ex_Index']
        wide_qrs, q_s_difference, surface_index = self.wide_qrs_find()
        # wide_qrs = np.array([])
        model_pred = model_pvc = []
        lbbb_index, rbbb_index = [], []

        pvc_onehot = np.zeros(len(self.r_index)).tolist()  # r_index

        if len(wide_qrs) > 0:
            if self.fs == 200:
                model_pvc, lbbb_index, rbbb_index, model_pred, detect_rpeaks = self.model_r_detectron(
                    self.low_pass_signal, wide_qrs, self.hr_count, fs=self.fs)
            else:

                model_pvc, lbbb_index, rbbb_index, model_pred, detect_rpeaks = self.model_r_detectron(self.ecg_signal,
                                                                                                      wide_qrs,
                                                                                                      self.hr_count,
                                                                                                      fs=self.fs)
            label = "PVC" if len(model_pvc) > 0 else "Abnormal"
            pvc_onehot = [1 if r in model_pvc else 0 for r in detect_rpeaks]
            if len(detect_rpeaks)!= 0:
                if len(lbbb_index) / len(detect_rpeaks) > 0.3:
                    lbbb_rbbb_label = "LBBB"
                if len(rbbb_index) / len(detect_rpeaks) > 0.3:
                    lbbb_rbbb_label = "RBBB"
            else:
                lbbb_rbbb_label = "Abnormal"
        else:
            label = "Abnormal"
            lbbb_rbbb_label = "Abnormal"

        pvc_count = pvc_onehot.count(1)
        vt_counter, couplet_counter, triplet_counter, bigeminy_counter, trigeminy_counter, quadrigeminy_counter, remaining_ones, v_bit_vt = self.PVC_CLASSIFICATION(
            pvc_onehot)
        conf_vt_count = 0
        if vt_counter > 0:
            confirmation = self.VT_confirmation(self.low_pass_signal, detect_rpeaks)

            if self.hr_count > 100 and v_bit_vt > 12:
                conf_vt_count = 1
            if confirmation == "Abnormal":
                vt_counter = 0
            else:
                pass
        data = {'PVC-Label': label,
                'PVC-Count': pvc_count,
                'PVC-Index': model_pvc,
                'VT_counter': conf_vt_count,
                'PVC-Couplet_counter': couplet_counter,
                'PVC-Triplet_counter': triplet_counter,
                'PVC-Bigeminy_counter': bigeminy_counter,
                'PVC-Trigeminy_counter': trigeminy_counter,
                'PVC-Quadrigeminy_counter': quadrigeminy_counter,
                'PVC-Isolated_counter': remaining_ones,
                'PVC-wide_qrs': wide_qrs,
                'PVC-QRS_difference': q_s_difference,
                'PVC-model_pred': model_pred,
                "IVR_counter": 0,
                "NSVT_counter": 0,
                "lbbb_rbbb_label": lbbb_rbbb_label,
                "lbbb_index": lbbb_index,
                "rbbb_index": rbbb_index
                }
        if vt_counter > 0:
            if 60 <= self.hr_count < 100:
                data['VT_counter'] = 0
                data["NSVT_counter"] = vt_counter
            elif self.hr_count < 60 and v_bit_vt > 3:
                data['VT_counter'] = 0
                data["IVR_counter"] = vt_counter
        return data


# Bock detection
class BlockDetected:
    def __init__(self, ecg_signal, fs):
        self.ecg_signal = ecg_signal
        self.fs = fs
        self.block_processing()

    def block_processing(self):
        self.baseline_signal, self.lowpass_signal = filter_signal(self.ecg_signal, self.fs).get_data()
        pqrst_data = pqrst_detection(self.baseline_signal, fs=self.fs).get_data()
        self.r_index = pqrst_data["R_index"]
        self.q_index = pqrst_data["Q_Index"]
        self.s_index = pqrst_data["S_Index"]
        self.p_index = pqrst_data["P_Index"]
        self.hr_counts = pqrst_data["HR_Count"]
        self.p_t = pqrst_data["P_T List"]
        self.pr = pqrst_data["PR_Interval"]

    def third_degree_block_deetection(self):
        label = 'Abnormal'
        third_degree = []
        possible_mob_3rd = False
        if self.hr_counts <= 100 and len(self.p_t) != 0:  # 60 70
            constant_2 = all(map(lambda innerlist: len(innerlist) == 2, self.p_t))
            cons_2_1 = all(len(inner_list) in {1, 2} for inner_list in self.p_t)
            ampli_val = list(
                map(lambda inner_list: sum(self.baseline_signal[i] > 0.05 for i in inner_list) / len(inner_list),
                    self.p_t))
            count_above_threshold = sum(1 for value in ampli_val if value > 0.7)
            percentage_above_threshold = count_above_threshold / len(ampli_val)
            count = 0
            if percentage_above_threshold >= 0.7:
                inc_dec_count = 0
                for i in range(0, len(self.pr)):
                    if self.pr[i] > self.pr[i - 1]:
                        inc_dec_count += 1
                if len(self.pr) != 0:
                    if round(inc_dec_count / (len(self.pr)), 2) >= 0.50:  # if posibale to change more then 0.5
                        possible_mob_3rd = True
                # if cons_2_1 == False:
                #     for i in range(0, len(self.pr)):
                #         if self.pr[i] > self.pr[i-1]:
                #             count += 1
                #     if round(count/len(self.pr), 2) >= 0.5:
                #         possible_3rd = True
                for inner_list in self.p_t:
                    if len(inner_list) in [3, 4]:
                        ampli_val = [self.baseline_signal[i] for i in inner_list]
                        if ampli_val and (sum(value > 0.05 for value in ampli_val) / len(ampli_val)) > 0.7:
                            differences = np.diff(inner_list).tolist()
                            diff_list = [x for x in differences if x >= 70]
                            if len(diff_list) != 0:
                                third_degree.append(1)
                            else:
                                third_degree.append(0)
                    elif len(inner_list) in [3, 4] and possible_mob_3rd == True and constant_2 == False:
                        differences = np.diff(inner_list).tolist()
                        if all(diff > 70 for diff in differences):
                            third_degree.append(1)
                        else:
                            third_degree.append(0)
                    else:
                        third_degree.append(0)
        if len(third_degree) != 0:
            if third_degree.count(1) / len(third_degree) >= 0.4 or possible_mob_3rd:  # 0.5 0.4
                label = "3rd Degree block"
        return label

    def second_degree_block_detection(self):
        label = 'Abnormal'
        constant_3_peak = []
        possible_mob_1 = False
        possible_mob_2 = False
        mob_count = 0
        if self.hr_counts <= 100:  # 80
            if len(self.p_t) != 0:
                constant_2 = all(map(lambda innerlist: len(innerlist) == 2, self.p_t))
                rhythm_flag = all(len(inner_list) in {1, 2, 3} for inner_list in self.p_t)
                ampli_val = list(
                    map(lambda inner_list: sum(self.baseline_signal[i] > 0.05 for i in inner_list) / len(inner_list),
                        self.p_t))
                count_above_threshold = sum(1 for value in ampli_val if value > 0.7)
                percentage_above_threshold = count_above_threshold / len(ampli_val)
                if percentage_above_threshold >= 0.7:
                    if rhythm_flag and constant_2 == False:
                        pr_interval = []
                        for i, r_element in enumerate(self.r_index[1:], start=1):
                            if i <= len(self.p_t):
                                inner_list = self.p_t[i - 1]
                                last_element = inner_list[-1]
                                result = r_element - last_element
                                pr_interval.append(result)

                        counts = {}
                        count_2 = 0
                        for i in range(0, len(pr_interval)):
                            counts[i] = 1
                            if i in counts:
                                counts[i] += 1
                            if pr_interval[i] > pr_interval[i - 1]:
                                count_2 += 1
                        most_frequent = max(counts.values())
                        if round(count_2 / (len(pr_interval)), 2) >= 0.50:
                            possible_mob_1 = True
                        elif round(most_frequent / len(pr_interval), 2) >= 0.4:
                            possible_mob_2 = True

                        for inner_list in self.p_t:
                            if len(inner_list) == 3:
                                differences = np.diff(inner_list).tolist()
                                if differences[0] <= 0.5 * differences[1] or differences[1] <= 0.5 * differences[0]:
                                    if possible_mob_1 or possible_mob_2:
                                        mob_count += 1
                                    else:
                                        constant_3_peak.append(1)
                            else:
                                constant_3_peak.append(0)
                    else:
                        for inner_list in self.p_t:
                            if len(inner_list) == 3:
                                differences = np.diff(inner_list).tolist()
                                if differences[0] <= 0.5 * differences[1] or differences[1] <= 0.5 * differences[0]:
                                    constant_3_peak.append(1)
                                else:
                                    constant_3_peak.append(0)
                            else:
                                constant_3_peak.append(0)
        if len(constant_3_peak) != 0 and constant_3_peak.count(1) != 0:

            if constant_3_peak.count(1) / len(constant_3_peak) >= 0.4:  # 0.4 0.5
                label = "Mobitz_II"
        elif possible_mob_1 and mob_count > 1:  # 0 1 4
            label = "Mobitz_I"
        elif possible_mob_2 and mob_count > 1:  # 0  4
            label = "Mobitz_II"
        return label

    # Block new trans model for added
    def prediction_model_block(self, input_arr):
        classes = ['1st_deg', '2nd_deg', '3rd_deg', 'abnormal', 'normal']
        input_arr = tf.io.decode_jpeg(tf.io.read_file(input_arr), channels=3)
        input_arr = tf.image.resize(input_arr, size=(224, 224), method=tf.image.ResizeMethod.BILINEAR)
        input_arr = (tf.expand_dims(input_arr, axis=0),)
        model_pred = predict_tflite_model(block_model, input_arr)[0]
        idx = np.argmax(model_pred)
        return model_pred, classes[idx]

    def check_block_model(self, low_ecg_signal):
        label = 'Abnormal'
        for i in glob.glob(base_dir_path+'/temp_block_img' + "/*.jpg"):
            os.remove(i)

        randome_number = random.randint(200000, 1000000)
        temp_img = low_ecg_signal
        plt.figure()
        plt.plot(temp_img)
        plt.axis("off")
        plt.savefig(f"{base_dir_path}/temp_block_img/p_{randome_number}.jpg")
        aq = cv2.imread(f"{base_dir_path}/temp_block_img/p_{randome_number}.jpg")      
        aq = cv2.resize(aq, (2400, 360))
        cv2.imwrite(f"{base_dir_path}/temp_block_img/p_{randome_number}.jpg", aq)        
        plt.close()
        ei_ti_label = []
        files = sorted(glob.glob(base_dir_path+'/temp_block_img' + "/*.jpg"), key=extract_number)        
        for pvcfilename in files:
            predictions, ids = self.prediction_model_block(pvcfilename)
            label = "Abnormal"  # "Normal"
            if str(ids) == "3rd_deg" and float(predictions[2]) > 0.8:
                label = "3rd degree"
            if str(ids) == "2nd_deg" and float(predictions[1]) > 0.8:
                label = "2nd degree"
            if str(ids) == "1st_deg" and float(predictions[0]) > 0.8:
                label = "1st degree"

            if 0.40 < float(predictions[1]) < 0.70:
                ei_ti_label.append('2nd degree')
            if 0.40 < float(predictions[0]) < 0.70:
                ei_ti_label.append('1st degree')
            if 0.40 < float(predictions[2]) < 0.70:
                ei_ti_label.append('3rd degree')
        return label, ei_ti_label, predictions

def block_model_check(ecg_signal, frequency, abs_result):
    model_label = 'Abnormal'
    ei_ti_block = []
    lowpass_signal = lowpass(ecg_signal)
    baseline_signal = baseline_construction_200(lowpass_signal,131)
    get_block = BlockDetected(ecg_signal, frequency)
    block_result, ei_ti_label, model_pre = get_block.check_block_model(baseline_signal)
    if block_result == '1st degree' and abs_result != 'Abnormal':
        model_label = 'I_Degree'
    if block_result == '2nd degree' and (abs_result == 'Mobitz II' or abs_result == 'Mobitz I'):
        if abs_result == "Mobitz I":
            model_label = 'MOBITZ_I'
        if abs_result == "Mobitz II":
            model_label = 'MOBITZ_II'
    if block_result == '3rd degree' and abs_result != "Abnormal":
        model_label = 'III_Degree'
    if abs_result in ['1st deg. block', "3rd Degree block", 'Mobitz II', 'Mobitz I']:
        if block_result == '2nd degree':
            model_label = 'MOBITZ_I'
        elif block_result == '3rd degree':
            model_label = 'III_Degree'
    if ei_ti_label:
        if '1st degree' in ei_ti_label and abs_result != "Abnormal":
            model_label = 'I_Degree'
            ei_ti_block.append({"Arrhythmia": "I_Degree", "percentage": model_pre[0] * 100})
        if '2nd degree' in ei_ti_label and (abs_result == 'Mobitz I' or abs_result == 'Mobitz II'):
            if abs_result == "Mobitz I":
                model_label = 'MOBITZ_I'
                ei_ti_block.append({"Arrhythmia": "MOBITZ_I", "percentage": model_pre[1] * 100})
            if abs_result == "Mobitz II":
                model_label = 'MOBITZ_II'
                ei_ti_block.append({"Arrhythmia": "MOBITZ_II", "percentage": model_pre[1] * 100})
        if '3rd degree' in ei_ti_label and abs_result != "Abnormal":
            model_label = 'III_Degree'
            ei_ti_block.append({"Arrhythmia": "III_Degree", "percentage": model_pre[2] * 100})
    return model_label, ei_ti_block


# Vfib & VFL detection
def resampled_ecg_data(ecg_signal, original_freq, desire_freq):
    original_time = np.arange(len(ecg_signal)) / original_freq
    new_time = np.linspace(original_time[0], original_time[-1], int(len(ecg_signal) * (desire_freq / original_freq)))
    interp_func = interp1d(original_time, ecg_signal, kind='linear')
    scaled_ecg_data = interp_func(new_time)
    return scaled_ecg_data


def image_array_vfib(signal):
    scales = np.arange(1, 50, 1)
    coef, freqs = pywt.cwt(signal, scales, 'mexh')
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


def vfib_predict_tflite_model(model: tuple, input_data: tuple):
    with results_lock:
        if type(model) != tuple and type(input_data) != tuple:
            print("Error")
        #raise TypeError
        interpreter, input_details, output_details = model
        for i in range(len(input_data)):
            interpreter.set_tensor(input_details[i]['index'], input_data[i])
        interpreter.invoke()
        output = interpreter.get_tensor(output_details[0]['index'])
        return output


def vfib_model_pred_tfite(raw_signal, model, fs):
    if fs == 200 and (np.max(raw_signal) > 4.1 or np.min(raw_signal) < 0):
        raw_signal = MinMaxScaler(feature_range=(0, 4)).fit_transform(raw_signal.reshape(-1, 1)).squeeze()
    seconds = 2.5
    steps_data = int(fs * seconds)
    total_data = raw_signal.shape[0]
    start = 0
    normal, vfib_vflutter, asys, noise = [], [], [], []
    percentage = {'NORMAL': 0, 'VFIB-VFLUTTER': 0, 'ASYS': 0, 'NOISE': 0}
    model_prediction = []
    while start < total_data:
        end = start + steps_data
        if end - start == steps_data and end < total_data:
            _raw_s_ = raw_signal[start:end]
            if _raw_s_.any():
                raw = image_array_vfib(_raw_s_)
            else:
                raw = np.array([])
        else:
            _raw_s_ = raw_signal[start:end]
            if _raw_s_.any():
                _raw_s_ = raw_signal[-steps_data:total_data]
                raw = image_array_vfib(_raw_s_)
                end = total_data - 1
            else:
                raw = np.array([])
        if raw.any():
            raw = raw.astype(np.float32) / 255
            rs_raw = resampled_ecg_data(_raw_s_, fs, 500 / seconds)
            if rs_raw.shape[0] != 500:
                rs_raw = signal.resample(rs_raw, 500)
            image_data = (tf.expand_dims(raw, axis=0),)  # tf.constant(rs_raw.reshape(1, -1, 1).astype(np.float32)))
            # image_data = (tf.cast(image_data[0],dtype=tf.float32), )
            model_pred = vfib_predict_tflite_model(model, image_data)[0]
            label = np.argmax(model_pred)
            model_prediction.append(f'{(start, end)}={model_pred}')
            if label == 0:
                normal.append(((start, end), model_pred));
                percentage['NORMAL'] += (end - start) / total_data
            elif label == 1:
                vfib_vflutter.append(((start, end), model_pred));
                percentage['VFIB-VFLUTTER'] += (
                                                       end - start) / total_data
            elif label == 2:
                asys.append(((start, end), model_pred));
                percentage['ASYS'] += (end - start) / total_data
            else:
                noise.append(((start, end), model_pred));
                percentage['NOISE'] += (end - start) / total_data
        start = start + steps_data

    return normal, vfib_vflutter, asys, noise, model_prediction, percentage


def vfib_model_check(ecg_signal, baseline_signal, lowpass_signal, model, fs):
    normal, vfib_vflutter, asys, noise, model_prediction, percentage = vfib_model_pred_tfite(ecg_signal, model, fs)

    final_label_index = np.argmax([percentage['NORMAL'], percentage['VFIB-VFLUTTER'],
                                   percentage['ASYS'], percentage['NOISE']])
    final_label = "NORMAL"
    return final_label

def prediction_model_vfib_vfl(input_arr, vfib_vfl_model):
    classes = ['VFIB', 'asystole', 'noise', 'normal']
    input_arr = tf.io.decode_jpeg(tf.io.read_file(input_arr), channels=3)
    input_arr = tf.image.resize(input_arr, size=(224, 224), method=tf.image.ResizeMethod.BILINEAR)
    input_arr = (tf.expand_dims(input_arr, axis=0),)
    model_pred = predict_tflite_model(vfib_vfl_model, input_arr)[0]
    idx = np.argmax(model_pred)
    return model_pred, classes[idx]

def check_vfib_vfl_model(ecg_signal, vfib_vfl_model):
    baseline_signal = baseline_construction_200(ecg_signal)
    low_ecg_signal = lowpass(baseline_signal, cutoff=0.2)
    label = 'Abnormal'
    temp_uuid = str(uuid.uuid1())
    folder_path = os.path.join("vflutter_img/", temp_uuid)
    os.makedirs(folder_path)

    plt.figure()
    plt.plot(low_ecg_signal)
    plt.axis('off')
    plt.savefig(f'{folder_path}/temp_img.jpg')
    aq = cv2.imread(f'{folder_path}/temp_img.jpg')
    aq = cv2.resize(aq,(1080,460))
    cv2.imwrite(f'{folder_path}/temp_img.jpg',aq)
    plt.close()

    combine_result = []
    label = 'Abnormal'

    files = sorted(glob.glob(f"{folder_path}/*.jpg"), key=extract_number)
    for vfib_file in files:
        with tf.device("CPU"):
            predictions, ids = prediction_model_vfib_vfl(vfib_file, vfib_vfl_model)
        label = "Abnormal"  # "Normal"
        if str(ids) == "VFIB" and float(predictions[0]) > 0.75:
            label = "VFIB/Vflutter"
            combine_result.append(label)
        if str(ids) == "asystole" and float(predictions[1]) > 0.75:
            label = "ASYS"
            combine_result.append(label)

        if str(ids) == "noise" and float(predictions[2]) > 0.75:
            label = "Noise"
            combine_result.append(label)

        if str(ids) == "normal" and float(predictions[3]) > 0.75:
            label = "Normal"
            combine_result.append(label)
    for img_path in glob.glob(f'{folder_path}/*.jpg'):
        os.remove(img_path)

    force_remove_folder(folder_path)
    temp_label = list(set(combine_result))
    if temp_label:
        if len(temp_label) > 1:

            label = 'Abnormal'
            if 'ASYS' in temp_label:
                label = 'ASYS'
            elif 'Noise' in temp_label:
                label = 'Noise'
        else:
            label = temp_label[0]

    return label

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
        rpeak_diff = np.diff(self.r_index)
        more_then_3_rhythm_per = len(list(filter(lambda x: len(x) >= 3, self.p_t))) / len(self.r_index)
        inner_list_less_2 = len(list(filter(lambda x: len(x) < 2, self.p_t))) / len(self.r_index)

        zeros_count = self.p_index.count(0)
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
            q_new = self.q_index[:-4:4].tolist()
            s_new = self.s_index[4::4].tolist()
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


# Wide-qrs detection
def wide_qrs(q_index, r_index, s_index, hr, fs=100):
    label = 'Abnormal'
    wideQRS = []
    recheck_wide_qrs = []

    thresold = round(fs * 0.12)  # 0.10
    if len(r_index) != 0:

        for k in range(len(r_index)):
            diff = s_index[k] - q_index[k]
            if diff > thresold:
                wideQRS.append(r_index[k])
        if len(wideQRS) / len(r_index) >= 0.90:  # .50
            final_thresh = round(fs * 0.20)  # 0.18
            for k in range(len(r_index)):
                if diff > final_thresh:
                    recheck_wide_qrs.append(r_index[k])

        if len(recheck_wide_qrs) / len(r_index) >= 2.5:
            label = 'WIDE_QRS'
    return label, wide_qrs


def wide_qrs_find_pac(q_index, r_index, s_index, hr_count, fs=200):
    max_indexs = 0
    if hr_count <= 88:
        ms = 0.18  # 0.10
    else:
        ms = 0.16  # 0.12
    max_indexs = int(fs * ms)
    pvc = []
    difference = []
    pvc_index = []
    wide_qs_diff = []
    for k in range(len(r_index)):
        diff = s_index[k] - q_index[k]
        difference.append(diff)
        if max_indexs != 0:
            if diff >= max_indexs:
                pvc.append(r_index[k])
    if hr_count <= 88 and len(r_index) != 0:
        wide_r_index_per = len(pvc) / len(r_index)
        if wide_r_index_per < 0.8:
            pvc_index = np.array(pvc)
        else:
            ms = 0.12
            max_indexs = int(fs * ms)
            for k in range(len(r_index)):
                diff = s_index[k] - q_index[k]
                wide_qs_diff.append(diff)
                if max_indexs != 0:
                    if diff >= max_indexs:
                        pvc_index.append(r_index[k])
            difference = wide_qs_diff
    else:
        pvc_index = np.array(pvc)
    q_s_difference = [i / fs for i in difference]
    return np.array(pvc_index), q_s_difference


def predict_tflite_model(model: tuple, input_data: tuple):
    with results_lock:
        interpreter, input_details, output_details = model
        for i in range(len(input_data)):
            interpreter.set_tensor(input_details[i]['index'], input_data[i])
        interpreter.invoke()
        output = interpreter.get_tensor(output_details[0]['index'])

    return output


# PAC detection
class PAC_detedction:
    def __init__(self, ecg_signal, fs, hr_counts):
        self.ecg_signal = ecg_signal
        self.fs = fs
        self.hr_counts = hr_counts

    def detect_beats_for_pac(self, ecg, rate, ransac_window_size=3.35, lowfreq=5.0, highfreq=15.0):
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
            elif abs(max_amplitude + 0.11) < abs(min_amplitude):
                rpeak = np.argmin(local_signal) + search_window.start
            else:
                if max_amplitude >= 0:

                    rpeak = np.argmax(local_signal) + search_window.start
                else:
                    rpeak = np.argmin(local_signal) + search_window.start

            rpeaks.append(rpeak)

        return np.array(rpeaks)


    def PACcounter(self, PAC_R_Peaks, hr_counts):
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

    def predict_pac_model(self, input_arr, target_shape=[224, 224], class_name=True):
        try:
            classes = ['Abnormal', 'Junctional', 'Normal', 'PAC']
            input_arr = tf.keras.preprocessing.image.img_to_array(input_arr)
            input_arr = tf.convert_to_tensor(input_arr, dtype=tf.float32)
            # input_arr = tf.cast(input_arr, dtype=tf.float32)
            # input_arr = tf.convert_to_tensor(input_arr, dtype=tf.float32)
            input_arr = tf.image.resize(input_arr, size=(224, 224), method=tf.image.ResizeMethod.BILINEAR)
            input_arr = (tf.expand_dims(input_arr, axis=0),)
            model_pred = predict_tflite_model(pac_model, input_arr)[0]

            idx = np.argmax(model_pred)
            if class_name:
                idx = np.argmax(model_pred)
                return model_pred, classes[idx]
            else:
                return model_pred
        except Exception as e:

            print("PAC ERROR", e)
            return [0, 0, 0, 0], "Normal"

    def get_pac_data(self):
        baseline_signal = baseline_construction_200(self.ecg_signal, kernel_size=131)  # 101
        lowpass_signal = lowpass(baseline_signal, cutoff=0.2)
        r_peaks = self.detect_beats_for_pac(lowpass_signal, self.fs)
        pqrst_data = pqrst_detection(baseline_signal, fs=200, thres=0.37, lp_thres=0.1, rr_thres=0.15).get_data()
        junc_r_label = pqrst_data['R_Label']
        p_index = pqrst_data['P_Index']
        p_t = pqrst_data['P_T List']
        updated_union, junc_union, pac_list = [], [], []
        pac_detect, junc_index = [], []
        pac_label, jr_label = "Abnormal", "Abnormal"
        for i in range(len(r_peaks) - 1):
            try:
                # time.sleep(0.1)
                with results_lock:
                    fig, ax = plt.subplots(num=1, clear=True)
                    segment = lowpass_signal[r_peaks[i] - 25:r_peaks[i + 1] + 20]
                    ax.plot(segment, color='blue')
                    ax.axis(False)
                    fig.canvas.draw()
                    data = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
                    data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
                    image = Image.fromarray(data)
                    resized_image = image.resize((360, 720), Image.LANCZOS)

                    # timestamp = int(time.time() * 1000)
                    # resized_image.save(f'pac_temp/segment_{timestamp}.jpg', quality=95)

                    plt.close(fig)
                    # with tf.device('/GPU:0'):

                    predictions, ids = self.predict_pac_model(resized_image)

                    if str(ids) == "PAC" and float(predictions[3]) > 0.93:  # 0.91
                        updated_union.append(1)
                        junc_union.append(0)
                        # pac_detect.append(int(r_peaks[i]))
                        # pac_detect.append(int(r_peaks[i + 1]))
                        pac_detect.append((int(r_peaks[i]), int(r_peaks[i + 1])))
                    elif str(ids) == "Junctional" and float(predictions[1]) > 0.90:
                        junc_union.append(1)
                        updated_union.append(0)
                        # junc_index.append(int(r_peaks[i]))
                        # junc_index.append(int(r_peaks[i + 1]))
                        junc_index.append((int(r_peaks[i]), int(r_peaks[i + 1])))
                    else:
                        updated_union.append(0)
                        junc_union.append(0)
            except Exception as e:
                print(e)

        if junc_r_label == "Regular" and self.hr_counts <= 60:
            if len(r_peaks) != 0:
                junc_count = junc_union.count(1)
                if junc_count / len(r_peaks) >= 0.5:
                    jr_label = "JN_RHY" if self.hr_counts > 40 else "JN_BR"

        pac_data = self.PACcounter(updated_union, self.hr_counts)
        pac_data['PAC_Union'] = updated_union
        pac_data['PAC_Index'] = pac_detect
        pac_data['jr_label'] = jr_label
        return pac_data


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


# First-deg block detection
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


# For RR regular or Irregular
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
    if len(variation) != 0:
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

    if (mean_percentage_diff > 75) and (irrgular_per_r > 80):
        r_label = "Irregular"
    return r_label


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
        # "noOfPauseList":[a/1000 for a in SAf if a>=2000 and a<=2900 ]
    return pause_label


def combine(ecg_signal, is_lead, class_name, fs=200, skip_afib_flutter=False):
    print(f"------------------------------- {is_lead} ---------------------------------")
    baseline_signal, lowpass_signal = filter_signal(ecg_signal, fs).get_data()
    pace_label, pacemaker_index = pacemake_detect(baseline_signal, fs=fs)

    pac_data = {
        'PAC_Union': [],
        "PAC_Index": [],
        "PAC_Isolated": 0,
        "PAC_Bigeminy": 0,
        "PAC_Trigeminy": 0,
        "PAC_Quadrigeminy": 0,
        "PAC_Couplet": 0,
        "PAC_Triplet": 0,
        "PAC_SVT": 0}

    vfib_or_asystole_output = vfib_model_check(ecg_signal, baseline_signal, lowpass_signal, vfib_vfl_model, fs)

    if vfib_or_asystole_output == "Abnormal" or vfib_or_asystole_output == "NORMAL":

        pqrst_data = pqrst_detection(baseline_signal, class_name=class_name, fs=fs).get_data()
        r_label = pqrst_data['R_Label']
        r_index = pqrst_data['R_index']
        q_index = pqrst_data['Q_Index']
        s_index = pqrst_data['S_Index']
        j_index = pqrst_data['J_Index']
        p_t = pqrst_data['P_T List']

        if pace_label != 'False':
            temp_list = pacemaker_index
            for sublist in p_t:
                for val in temp_list:
                    if val in sublist:
                        sublist.remove(val)
                        temp_list.remove(val)

        pt = pqrst_data['PT PLot']
        hr_counts = pqrst_data['HR_Count']
        t_index = pqrst_data['T_Index']
        p_index = pqrst_data['P_Index']
        ex_index = pqrst_data['Ex_Index']
        pr_interval = pqrst_data['PR_Interval']
        p_label = pqrst_data['P_Label']
        pr_label = pqrst_data['PR_label']
        r_check_1 = new_rr_check(r_index)
        r_check_2 = check_r_irregular(r_index)
        r_label = "Regular"
        if r_check_1 == 'Irregular' and r_check_2 == 'Irregular':
            r_label = "Irregular"

        afib_label = jr_label = first_deg_block_label = second_deg_block = third_deg_block = aflutter_label = longqt_label = first_degree_block = PAC_label = abs_result = final_block_label = check_pause = 'Abnormal'
        temp_index = wide_qrs_list = []
        pvc_class = []
        pac_class = ['Abnormal']
        pvc_data = {'PVC-Index': [], "PVC-QRS_difference": [], "PVC-wide_qrs": np.array([]), 'PVC-model_pred': [],
                    'lbbb_rbbb_label': 'Abnormal'}

        if len(r_index) != 0 or len(s_index) != 0 or len(q_index) != 0:
            if (is_lead == 'II' or is_lead == 'III' or is_lead == "I" or is_lead == 'V1'
                    or is_lead == 'V2' or is_lead == 'V5' or is_lead == 'V6'):
                pvc_data = PVC_detection(ecg_signal, fs).get_pvc_data()

                pvc_count = pvc_data['PVC-Count']

                temp_pvc = []
                for key, val in pvc_data.items():
                    if 'counter' in key and val > 0:
                        temp_pvc.append(key.split('_')[0])
                if len(temp_pvc) != 0:
                    pvc_class = [label.replace('-', '_') for label in temp_pvc]
                else:
                    pvc_class = temp_pvc

            wide_qrs_label, _ = wide_qrs(q_index, r_index, s_index, hr_counts, fs=fs) if len(pvc_class) == 0 else (
                "Abnormal", [])
            temp_index, wide_qrs_list = wide_qrs_find_pac(q_index, r_index, s_index, hr_counts, fs=fs)
        else:
            wide_qrs_label = 'Abnormal'

        if all(p not in ['VT', 'IVR', 'NSVT', 'PVC-Triplet', 'PVC-Couplet'] for p in pvc_class) and len(
                r_index) > 0:  # 'PVC-Triplet', 'PVC-Couplet'
            if hr_counts <= 60:
                check_pause = check_long_short_pause(r_index)
            if r_label == 'Regular':
                if is_lead == 'II' or is_lead == 'III' or is_lead == "I" or is_lead == "V1" or is_lead == "V2" or is_lead == "V5" or is_lead == "V6":
                    pac_data = PAC_detedction(ecg_signal, fs, hr_counts).get_pac_data()
                    if r_label == 'Regular':
                        jr_label = pac_data['jr_label']
                    if is_lead == "II" or is_lead == "III" or is_lead == "I" or is_lead == "V1":
                        if all('PVC' not in p for p in pvc_class) and all(
                                'Abnormal' in l for l in [afib_label, aflutter_label]):
                            if hr_counts >= 55:
                                temp_pac = '; '.join(
                                    [key.split('_')[0] for key, val in pac_data.items() if
                                     'counter' in key and val > 0])
                                pac_class = temp_pac.replace('-', '_')
                            else:
                                pac_class = ""
                                pac_data = {'Model_Check': [],
                                            'PAC_Union': [],
                                            'PAC_Index': []}

                if is_lead == "II" or is_lead == "III" or is_lead == "I" or is_lead == "V1" or is_lead == "V2" or is_lead == "V4" or is_lead == 'V5':
                    if all('Abnormal' in l for l in
                           [afib_label, aflutter_label]):  # and len(pac_class) == 0 and len(pvc_class) == 0
                        lowpass_signal = lowpass(baseline_signal, 0.3)
                        first_deg_block_label, first_deg_block_index = first_degree_detect(lowpass_signal, fs)
                        abs_result = first_deg_block_label
                    if hr_counts <= 80:
                        if all('Abnormal' in l for l in [afib_label, aflutter_label, first_deg_block_label,
                                                         jr_label]):  # and len(pac_class) == 0 and len(pvc_class) == 0
                            second_deg_block = BlockDetected(ecg_signal, fs).second_degree_block_detection()
                            if second_deg_block != 'Abnormal':
                                abs_result = second_deg_block
                        if all('Abnormal' in l for l in
                               [afib_label, aflutter_label, first_deg_block_label, second_deg_block,
                                jr_label]):
                            third_deg_block = BlockDetected(ecg_signal, fs).third_degree_block_deetection()
                            if third_deg_block != 'Abnormal':
                                abs_result = third_deg_block
                    if abs_result != 'Abnormal':
                        final_block_label, block_ei_ti = block_model_check(ecg_signal, fs, abs_result)
                if all('Abnormal' in l for l in [afib_label, aflutter_label]) and len(pac_class) == 0 and len(
                        pvc_class) == 0:
                    lowpass_signal = lowpass(baseline_signal, 0.3)
                    longqt_label = detection_long_qt(lowpass_signal, r_index, fs)
            else:
                if not skip_afib_flutter:
                    if is_lead == 'II' or is_lead == 'III' or is_lead == 'I' or is_lead == 'V5' or is_lead == 'V6':
                        afib_flutter_check = afib_flutter_detection(lowpass_signal, r_index, q_index, s_index, p_index,
                                                                    p_t,
                                                                    pr_interval, afib_model)
                        is_afib_flutter = afib_flutter_check.abs_afib_flutter_check()
                        afib_model_per = flutter_model_per = 0
                        if is_afib_flutter:
                            afib_flutter_per, afib_indexs, flutter_indexs = afib_flutter_check.get_data()
                            afib_model_per = int(afib_flutter_per['AFIB'] * 100)
                            flutter_model_per = int(afib_flutter_per['FLUTTER'] * 100)
                        if afib_model_per >= 40:
                            afib_label = 'AFIB'

                        if afib_label != 'AFIB':
                            if flutter_model_per >= 60:
                                aflutter_label = 'AFL'
                if afib_label != 'AFIB':
                    if is_lead == "II" or is_lead == "III" or is_lead == "I" or is_lead == "V1" or is_lead == 'V2' or is_lead == 'V5' or is_lead == 'V6':
                        pac_data = PAC_detedction(ecg_signal, fs, hr_counts).get_pac_data()
                        if is_lead == 'II' or is_lead == 'III' or is_lead == "aVF":
                            jr_label = pac_data['jr_label']
                        if all('PVC' not in p for p in pvc_class) and all('Abnormal' in l for l in
                                                                          [afib_label, aflutter_label,
                                                                           check_pause]) and hr_counts <= 100:
                            temp_pac = '; '.join(
                                [key.split('_')[0] for key, val in pac_data.items() if 'counter' in key and val > 0])
                            pac_class = temp_pac.replace('-', '_')
                    if is_lead == "II" or is_lead == "III" or is_lead == "I" or is_lead == "V1" or is_lead == "V2" or is_lead == "V4" or is_lead == 'V5':
                        if all('Abnormal' in l for l in
                               [afib_label, aflutter_label]):  # and len(pac_class) == 0 and len(pvc_class) == 0
                            lowpass_signal = lowpass(baseline_signal, 0.3)
                            first_deg_block_label, first_deg_block_index = first_degree_detect(lowpass_signal, fs)
                            abs_result = first_deg_block_label

                        if hr_counts <= 80:
                            if all('Abnormal' in l for l in
                                   [afib_label, aflutter_label, first_deg_block_label, jr_label,
                                    check_pause]):  # and len(pac_class) == 0 and len(pvc_class) == 0
                                second_deg_block = BlockDetected(ecg_signal, fs).second_degree_block_detection()
                            if second_deg_block != 'Abnormal':
                                abs_result = second_deg_block
                            if all('Abnormal' in l for l in
                                   [afib_label, aflutter_label, first_deg_block_label, second_deg_block, jr_label,
                                    check_pause]):  # and len(pac_class) == 0 and len(pvc_class) == 0
                                third_deg_block = BlockDetected(ecg_signal, fs).third_degree_block_deetection()
                            if third_deg_block != 'Abnormal':
                                abs_result = third_deg_block
                        if abs_result != 'Abnormal':
                            final_block_label, block_ei_ti = block_model_check(ecg_signal, fs, abs_result)

            pac_class = "Abnormal" if pac_class == '' else pac_class
            label = {'Afib_label': afib_label,
                     'Aflutter_label': aflutter_label,
                     'JR_label': jr_label,
                     'wide_qrs_label': wide_qrs_label,
                     'longqt_label': longqt_label,
                     'final_block_label': final_block_label,
                     'check_pause': check_pause,
                     'pac_class': pac_class}
            if pvc_class:
                c_label = "; ".join(pvc_class) + "; " + "; ".join([l for l in label.values() if 'Abnormal' not in l])
            else:
                c_label = "; ".join([l for l in label.values() if 'Abnormal' not in l])
        else:
            c_label = "; ".join(pvc_class)

        c_label = c_label + f"; {pace_label}" if pace_label != "False" else c_label

        if c_label in ["", "; "]: c_label = 'NORMAL'

        data = {'Input_Signal': ecg_signal,
                'Baseline_Signal': baseline_signal,
                'Lowpass_signal': lowpass_signal,
                # 'Combine_Label':c_label.upper().replace("_","-"),
                'Combine_Label': c_label,
                'RR_Label': r_label,
                'R_Index': r_index,
                'Q_Index': q_index,
                'S_Index': s_index,
                'J_Index': j_index,
                'T_Index': t_index,
                'P_Index': p_index,
                'Ex_Index': ex_index,
                'P_T': pt,
                'HR_Count': hr_counts,
                'PVC_DATA': pvc_data,
                'PAC_DATA': pac_data,
                'PaceMaker': pace_label, }
    else:
        data = {'Input_Signal': ecg_signal,
                'Baseline_Signal': baseline_signal,
                'Lowpass_signal': lowpass(baseline_signal, 0.3),
                'Combine_Label': vfib_or_asystole_output.upper(),
                'RR_Label': 'Not Defined',
                'R_Index': np.array([]),
                'Q_Index': [],
                'S_Index': [],
                'J_Index': [],
                'T_Index': [],
                'P_Index': [],
                'Ex_Index': [],
                'P_T': [],
                'HR_Count': 0,
                'PVC_DATA': {'PVC-Index': [], "PVC-QRS_difference": [], "PVC-wide_qrs": np.array([]),
                             'PVC-model_pred': [], 'lbbb_rbbb_label': 'Abnormal'},
                'PAC_DATA': pac_data,
                'PaceMaker': pace_label, }

    return data


# R peak detection using biosppy
class RPeakDetection:
    def __init__(self, baseline_signal, fs=200):
        self.baseline_signal = baseline_signal
        self.fs = fs

    def butter_bandpass(self, lowcut, highcut, fs, order=5):
        nyq = 0.5 * fs
        low = lowcut / nyq
        high = highcut / nyq
        b, a = butter(order, [low, high], btype='band')
        return b, a

    def butter_bandpass_filter(self, data, lowcut, highcut, fs, order=5):
        b, a = self.butter_bandpass(lowcut, highcut, fs, order=order)
        y = filtfilt(b, a, data)
        return y

    def find_r_peak(self):
        lowcut = 0.5
        highcut = 50.0
        filtered_signal = self.butter_bandpass_filter(self.baseline_signal, lowcut, highcut, self.fs, order=6)
        # Step 3: R-Peak Detection
        out = hami.hamilton_segmenter(filtered_signal, sampling_rate=self.fs)
        rpeaks = hami.correct_rpeaks(filtered_signal, out[0], sampling_rate=self.fs, tol=0.1)
        r_peaks = rpeaks[0].tolist()
        return r_peaks


def find_new_q_index(ecg, R_index, d):
    q = []
    for i in R_index:
        q_ = []
        if i == 0:
            q.append(i)
            continue
        if ecg[i] > 0:
            c = i
            while c > 0 and ecg[c - 1] < ecg[c]:
                c -= 1
            if ecg[i] * 0.01 > ecg[c] or ecg[c] < 0 or c == 0:
                if abs(i - c) <= d:
                    q.append(c)
                    continue
                else:
                    q_.append(c)
            while c > 0:
                while c > 0 and ecg[c - 1] > ecg[c]:
                    c -= 1
                # q_.append(c)
                while c > 0 and ecg[c - 1] < ecg[c]:
                    c -= 1
                if q_ and q_[-1] == c:
                    break
                q_.append(c)
                if ecg[i] * 0.01 > ecg[c] or ecg[c] < 0 or c == 0:
                    break
        else:
            c = i
            while c > 0 and ecg[c - 1] > ecg[c]:
                c -= 1
            if ecg[i] * 0.01 < ecg[c] or ecg[c] > 0 or c == 0:
                if abs(i - c) <= d:
                    q.append(c)
                    continue
                else:
                    q_.append(c)
            while c > 0:
                while c > 0 and ecg[c - 1] < ecg[c]:
                    c -= 1
                # q_.append(c)
                while c > 0 and ecg[c - 1] > ecg[c]:
                    c -= 1
                if q_ and q_[-1] == c:
                    break
                q_.append(c)
                if ecg[i] * 0.01 < ecg[c] or ecg[c] > 0 or c == 0:
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


def extract_number(filename):
    match = re.search(r'(\d+)', os.path.basename(filename))
    return int(match.group(1)) if match else float('inf')


def prediction_model_mi(input_arr):
    classes = ['Noise', 'Normal', 'STDEP', 'STELE', 'TAB']
    input_arr = tf.io.decode_jpeg(tf.io.read_file(input_arr), channels=3)
    input_arr = tf.image.resize(input_arr, size=(224, 224), method=tf.image.ResizeMethod.BILINEAR)
    input_arr = (tf.expand_dims(input_arr, axis=0),)
    model_pred = predict_tflite_model(let_inf_moedel, input_arr)[0]
    idx = np.argmax(model_pred)
    return model_pred, classes[idx]


def inf_lat_model_check(low_ecg_signal, r_index, lead):
    with results_lock:
        label = 'Abnormal'
        time.sleep(0.1)
        try:
            if not os.path.exists('STsegimages/'):
                os.mkdir('STsegimages/')
        except Exception as r:
            print(r)

        try:
            randome_number = random.randint(200000, 1000000)
            plt.plot(low_ecg_signal)
            plt.axis("off")
            plt.savefig(f"STsegimages/p_{randome_number}.jpg")
            aq = cv2.imread(f"STsegimages/p_{randome_number}.jpg")
            aq = cv2.resize(aq, (1080, 460))
            cv2.imwrite(f"STsegimages/p_{randome_number}.jpg", aq)
            plt.close()
            pvc = []
            lbbb = []
            rbbb = []
            pvclist = []
            lbbblist = []
            rbbblist = []
            # mainstart = time.time()
            total_r_index = len(r_index) - 1
            label = 'Abnormal'

            files = sorted(glob.glob('STsegimages/' + "/*.jpg"), key=extract_number)
            for pvcfilename in files:
                # start = time.time()
                # with tf.device('/CPU:0'):
                predictions, ids = prediction_model_mi(pvcfilename)
                label = "Normal"
                if str(ids) == "STDEP" and float(predictions[2]) > 0.91:
                    pvc.append(1)
                    label = "STDEP"
                    pvclist.append(int(pvcfilename.split("_")[1].split(".jpg")[0]))
                else:
                    pvc.append(0)

                if str(ids) == "STELE" and float(predictions[3]) > 0.78:
                    lbbb.append(1)
                    label = "STELE"
                    lbbblist.append(int(pvcfilename.split("_")[1].split(".jpg")[0]))
                else:
                    lbbb.append(0)

                if str(ids) == "TAB" and float(predictions[4]) > 0.78:
                    label = "TAB"
                    rbbb.append(1)
                    rbbblist.append(int(pvcfilename.split("_")[1].split(".jpg")[0]))
                else:
                    rbbb.append(0)
            for i in glob.glob('STsegimages/' + "/*.jpg"):
                os.remove(i)
            force_remove_folder(os.path.join('STsegimages/'))
            return label
        except Exception as e:
            print('Inf_lat_error', e)


# Define a function to validate 'digits / digits ms' pattern
def is_single_format(value):
    return re.match(r'^\d+\s?[\|/]\s?\d+\s?ms$', value.strip(), re.IGNORECASE)


# Function to validate HR values
def validate_hr(hr_values):
    validated_hr = []
    for hr in hr_values:
        # Remove non-digit characters
        hr_cleaned = re.sub(r'\D', '', hr)
        # Check if the cleaned HR is a 2 or 3 digit number
        if hr_cleaned.isdigit() and 10 <= int(hr_cleaned) <= 250:
            validated_hr.append(int(hr_cleaned))
        else:
            validated_hr.append(None)
    return validated_hr


# Define a function to classify MI results
def classify_mi_result(input_list):
    keywords = ['lateral', 'inferior', 'abnormality']
    mi_result = [item for item in input_list if any(keyword in item.lower() for keyword in keywords)]
    result_classification = []
    for item in mi_result:
        if 'abnormality' in item.lower():
            result_classification.append('T_wave_Abnormality')
        if ' lateral' in item.lower() or item.lower().startswith('lateral'):
            result_classification.append('Lateral_MI')
        if 'inferior' in item.lower():
            result_classification.append('Inferior_MI')
    result_classification = list(set(result_classification))
    return result_classification


def classify_arrhythmia(input_list):
    afib_afl = []
    afib_keywords = ['atrial fibrillation', 'atrial fib', 'fibrillation']
    aflutter_keywords = ['atrial flutter', 'flutter']
    # Search for AFIB keywords
    if any(keyword in " ".join(input_list).lower() for keyword in afib_keywords):
        afib_afl.append('AFIB')
    # Search for AFLUTTER keywords
    if any(keyword in " ".join(input_list).lower() for keyword in aflutter_keywords):
        afib_afl.append('AFL')
    return afib_afl


def classify_hypertrophy(input_list):
    hypertrophy = []
    input_text = " ".join(input_list).lower()
    input_text = re.sub(r'[_/-]', ' ', input_text)
    input_text = re.sub(r'\s+', ' ', input_text)

    # Check specifically for 'right ventricular hypertrophy' or 'rvh' first
    if ('right ventricular hypertrophy' in input_text) or ('rvh' in input_text):
        hypertrophy.append('RVH')
     # Check for LVH
    if ('left ventricular hypertrophy' in input_text) or ('lvh' in input_text):
        hypertrophy.append('LVH')
    # Check for generic "ventricular hypertrophy"
    if 'ventricular hypertrophy' in input_text:
        # Use regex to extract a window of 3–4 words around the match
        match = re.search(r'(.{0,30}ventricular hypertrophy.{0,30})', input_text)
        if match:
            context = match.group(1)
            if 'left' not in context and 'right' not in context and 'lvh' not in context and 'rvh' not in context:
                hypertrophy.append('LVH')

    return list(set(hypertrophy))  # Remove duplicates
# Function to validate QT/QTcBaz and RR/PP values with duplicate and format check
def validate_intervals(interval_list):
    validated_intervals = []
    # Updated regex pattern to match 'xxx / xxx' or 'xxxx / xxxx' with or without 'ms'
    pattern = re.compile(r'(\d{3,4})\s*[/|]\s*(\d{3,4})', re.IGNORECASE)
    for interval in interval_list:
        match = pattern.search(interval)
        if match:
            # Format as 'xxx / xxx' or 'xxxx / xxxx' (removes 'ms' dependency)
            formatted_interval = f"{match.group(1)} / {match.group(2)}"
            validated_intervals.append(formatted_interval)
        else:
            validated_intervals.append(None)

    # Remove duplicates
    return list(dict.fromkeys(validated_intervals))


# Function to remove 'ms', split values, and remove square brackets
def process_and_split(values):
    processed_dict = {"Part1": [], "Part2": []}
    for value in values:
        if value:
            # Remove 'ms', clean value, and split by '/'
            cleaned_value = re.sub(r'\s?ms$', '', value).strip()
            parts = cleaned_value.split('\\')
            # Add split parts to respective keys, trimming extra spaces
            if len(parts) == 2:
                processed_dict["Part1"].append(parts[0].strip())
                processed_dict["Part2"].append(parts[1].strip())
    return processed_dict


def text_detection(img_path):
    img = cv2.imread(img_path)
    result = reader.readtext(img)
    HR = []
    QT_QTcBaz = []
    RR_PP = []
    QRS_values = []
    PR_values = []
    output_dict = {}
    extracted_text = [detection[1] for detection in result]

    for i, text in enumerate(extracted_text):

        match = re.search(r'(\d+)\s?(BPM|bpm|opm|bprn|bpr)', text)  # Match '137bpm' or '137 bpm'
        if match:
            HR.append(match.group(1))  # Extract only the numeric part
            print(f"Debug: Found HR -> {match.group(1)}")
        elif any(keyword in text for keyword in ['BPM', 'bpm', 'opm', 'bprn', 'bpr']) and i > 0:
            prev_value = extracted_text[i - 1].strip()
            if prev_value.isdigit():
                HR.append(prev_value)
                print(f"Debug: Found HR -> {prev_value}")

        # Detect 'QRS' (case-insensitive) and validate its next value
        if text.lower() in ['qrs', 'qrs duration'] and i + 1 < len(extracted_text):
            next_value = extracted_text[i + 1]
            # Check if the next value is in the format 'xx ms' or 'xxx ms'
            if re.match(r'^\d{2,3}\s?ms$', next_value.strip(), re.IGNORECASE):
                QRS_values.append(next_value.strip())
                print(f"Debug: Found QRS -> {next_value.strip()}")
            elif next_value.isdigit():
                QRS_values.append(next_value)
                print(f"Debug: Found QRS (without ms) -> {next_value}")

        # Detect 'PR' (case-insensitive) and validate its next value
        if text.lower() in ['pr', 'pr interval'] and i + 1 < len(extracted_text):
            next_value = extracted_text[i + 1]
            # Check if the next value is in the format 'xx ms' or 'xxx ms'
            if re.match(r'^\d{2,3}\s?ms$', next_value.strip(), re.IGNORECASE):
                PR_values.append(next_value.strip())
                print(f"Debug: Found PR -> {next_value.strip()}")
            elif next_value.isdigit():
                PR_values.append(next_value)
                print(f"Debug: Found PR (without ms) -> {next_value}")

        if (
                'QT / QTcBaz' in text or 'QT/ QTcB' in text or 'QT / QTcB' in text or 'QT/QTcB' in text or 'QTI/ QTcBaz' in text
                or 'QTIQTc-Baz' in text or 'QTQTc-Baz' in text or (
                text == 'QT' and i + 1 < len(extracted_text) and extracted_text[i + 1] == 'QTcBaz')):

            if text == 'QT' and i + 1 < len(extracted_text) and extracted_text[i + 1] == 'QTcBaz':
                qt_index = i + 1
            else:
                qt_index = i

            if qt_index + 1 < len(extracted_text) and is_single_format(extracted_text[qt_index + 1]):
                QT_QTcBaz.append(extracted_text[qt_index + 1])
            elif qt_index + 2 < len(extracted_text):
                value = f"{extracted_text[qt_index + 1]} / {extracted_text[qt_index + 2]}"
                QT_QTcBaz.append(value)

        if text == 'QT:' and i + 2 < len(extracted_text) and extracted_text[i + 1] == 'QTcBaz':
            next_value = extracted_text[i + 2]
            if is_single_format(next_value):
                QT_QTcBaz.append(next_value)

        if 'RR / PP' in text or 'RR/PP' in text or 'RR/ PP' in text or 'RRIPP' in text or 'PP' in text or 'RR / Pp' in text:
            if i + 1 < len(extracted_text):
                next_value = extracted_text[i + 1]
                if is_single_format(next_value):
                    RR_PP.append(next_value)
                elif re.match(r'^\d+:\s*/\s*\d+\.ms$', next_value):
                    match = re.search(r'^(\d+):\s*/\s*(\d+)\.ms$', next_value)
                    if match:
                        normalized_value = f"{match.group(1)} / {match.group(2)} ms"
                        RR_PP.append(normalized_value)
                elif i + 2 < len(extracted_text):
                    first_value = extracted_text[i + 1]
                    second_value = extracted_text[i + 2]
                    if second_value.lower().endswith('ms'):
                        value = f"{first_value} / {second_value}"
                        RR_PP.append(value)
                    elif first_value.isdigit() and second_value.isdigit():
                        value = f"{first_value} / {second_value} ms"
                        RR_PP.append(value)

        if text == 'RR' and i + 2 < len(extracted_text) and extracted_text[i + 1] == 'PP':
            next_value = extracted_text[i + 2]
            if is_single_format(next_value):
                RR_PP.append(next_value)
            elif i + 3 < len(extracted_text):
                first_value = extracted_text[i + 2]
                second_value = extracted_text[i + 3]
                if second_value.lower().endswith('ms'):
                    value = f"{first_value} / {second_value}"
                    RR_PP.append(value)
                elif first_value.isdigit() and second_value.isdigit():
                    value = f"{first_value} / {second_value} ms"
                    RR_PP.append(value)

        if text == 'RR:' and i + 2 < len(extracted_text) and extracted_text[i + 1] == 'PP':
            next_value = extracted_text[i + 2]
            if is_single_format(next_value):
                RR_PP.append(next_value)
            elif i + 3 < len(extracted_text):
                first_value = extracted_text[i + 2]
                second_value = extracted_text[i + 3]
                if second_value.lower().endswith('ms'):
                    value = f"{first_value} / {second_value}"
                    RR_PP.append(value)

        if text == 'RR /: PP' and i + 1 < len(extracted_text):
            next_value = extracted_text[i + 1]
            if is_single_format(next_value):
                RR_PP.append(next_value)

    validated_hr_list = validate_hr(HR)
    validated_qt_qtc_list = validate_intervals(QT_QTcBaz)
    validated_rr_pp_list = validate_intervals(RR_PP)
    classification = classify_mi_result(extracted_text)
    classify_arrhy = classify_arrhythmia(extracted_text)
    hypertrophy = classify_hypertrophy(extracted_text)

    if validated_hr_list:
        output_dict["HR"] = validated_hr_list[0]
    if validated_qt_qtc_list or validated_rr_pp_list:
        try:
            qt_split = process_and_split(validated_qt_qtc_list)
            rr_split = process_and_split(validated_rr_pp_list)
            output_dict["QT"] = int(qt_split["Part1"][0]) if qt_split['Part1'] else 0
            output_dict['QTcBaz'] = int(qt_split["Part2"][0]) if qt_split['Part2'] else 0
            output_dict["RR"] = int(rr_split["Part1"][0]) if rr_split['Part1'] else 0
            output_dict["PP"] = int(rr_split["Part2"][0]) if rr_split['Part2'] else 0
        except Exception as e:
            print(e)

    if classification:
        output_dict["MI"] = classification

    if classify_arrhy:
        output_dict["Arrhythmia"] = list(set(classify_arrhy))

    if hypertrophy:
        output_dict['Hypertrophy'] = hypertrophy

    if QRS_values:
        try:
            output_dict["QRS"] = int(QRS_values[0].split(' ')[0])
        except:
            output_dict["QRS"] = int(QRS_values[0].split('ms')[0])
    if PR_values:
        try:
            output_dict["PR"] = int(PR_values[0].split(' ')[0])
        except:
            output_dict["PR"] = int(PR_values[0].split('ms')[0])
    return output_dict


def find_ecg_info(ecg_signal, img_type, image_path):
    if img_type == '12_1':
        fa = 130
    elif img_type == '3_4':
        fa = 60
    else:
        fa = 110
    ocr_results = {}
    if img_type == '6_2':
        ocr_results = text_detection(image_path)
    rpeaks = detect_beats(ecg_signal, float(fa))
    rr_interval = []
    data_dic = {"rr_interval": 0,
                "PRInterval": 0,
                "QTInterval": 0,
                "QRSComplex": 0,
                "STseg": 0,
                "PRseg": 0,
                "QTc": 0}
    try:
        _, waves_peak = nk.ecg_delineate(ecg_signal, rpeaks, sampling_rate=fa, method="peaks")
        signal_dwt, waves_dwt = nk.ecg_delineate(ecg_signal, rpeaks, sampling_rate=fa, method="dwt")
        Tpeaks = np.where(np.isnan(waves_peak['ECG_T_Peaks']), 0, waves_peak['ECG_T_Peaks']).astype('int64').tolist()
        Qpeaks = np.where(np.isnan(waves_peak['ECG_Q_Peaks']), 0, waves_peak['ECG_Q_Peaks']).astype('int64').tolist()
    except Exception as e:
        print('Nurokit error:', e)
    for i in range(len(rpeaks) - 1):
        try:
            RRpeaks = abs(int(rpeaks[i]) * 3 - int(rpeaks[i + 1]) * 3)
            rr_interval.append(RRpeaks)
        except:
            rr_interval.append(0)
            RRpeaks = "0"
    try:
        data_dic['rr_interval'] = rr_interval[0]
    except:
        data_dic['rr_interval'] = "100"
    try:
        Ppeak = waves_peak['ECG_P_Peaks'][1]
        Rpeak = rpeaks[1]
        Ppeak = int(Ppeak) * 3
        Rpeak = int(Rpeak) * 3
        PRpeaks = abs(Rpeak - Ppeak)
    except:
        PRpeaks = "0"
    data_dic['PRInterval'] = PRpeaks
    try:
        Tpeak = waves_peak['ECG_T_Peaks'][1]
        Qpeak = waves_peak['ECG_Q_Peaks'][1]
        Tpeak = int(Tpeak) * 3
        Qpeak = int(Qpeak) * 3
        QTpeaks = abs(Tpeak - Qpeak)
    except:
        QTpeaks = "0"
    data_dic['QTInterval'] = QTpeaks
    try:
        Speak = waves_peak['ECG_S_Peaks'][1]
        Qpeak = waves_peak['ECG_Q_Peaks'][1]
        Speak = int(Speak) * 3
        Qpeak = int(Qpeak) * 3
        SQpeaks = abs(Speak - Qpeak)
    except:
        SQpeaks = "0"
    data_dic['QRSComplex'] = SQpeaks
    try:
        Spa = waves_peak['ECG_S_Peaks'][1]
        Ton = waves_dwt['ECG_T_Onsets'][1]
        Spa = int(Spa) * 3
        Ton = int(Ton) * 3
        STseg = abs(Ton - Spa)
    except:
        STseg = "0"
    data_dic['STseg'] = STseg
    try:
        PP = waves_dwt['ECG_P_Offsets']
        RRO = waves_dwt['ECG_R_Onsets']
        if math.isnan(PP[2]) or math.isnan(RRO[2]):
            PRseg = "0"
        else:
            PPIn = int(PP[1]) * 3
            RRon = int(RRO[1]) * 3
            PRseg = abs(PPIn - RRon)
    except:
        PRseg = "0"
    data_dic['PRseg'] = PRseg

    QTint = []
    finallist = []
    try:
        for i in range(len(Qpeaks) - 1):
            try:
                if Qpeaks[i] == 0 or Tpeaks[i] == 0:
                    QTint.append(0)
                else:
                    QT = abs(int(Qpeaks[i]) - int(Tpeaks[i])) / 200
                    QTint.append(QT * 1000)
                    # if QT>0.5: finallist.append(QT)  #0.2
            except:
                QTint.append(0)
    except:
        QTint.append(0)
    data_dic['QTc'] = QTint[0]
    if ocr_results:
        if 'QTcBaz' in ocr_results and ocr_results['QTcBaz'] != 0:
            data_dic['QTc'] = ocr_results['QTcBaz']

        if 'QT' in ocr_results and ocr_results['QT'] != 0:
            data_dic['QTInterval'] = ocr_results['QT']

        if 'RR' in ocr_results and ocr_results['RR'] != 0:
            data_dic['rr_interval'] = ocr_results['RR']

        if 'QRS' in ocr_results and ocr_results['QRS'] != 0:
            data_dic['QRSComplex'] = ocr_results['QRS']

        if 'HR' in ocr_results and ocr_results['HR'] != 0:
            data_dic['HR'] = ocr_results['HR']

        if 'PR' in ocr_results and ocr_results['PR'] != 0:
            data_dic['PRInterval'] = ocr_results['PR']
        if 'MI' in ocr_results:
            data_dic['MI'] = ocr_results['MI']
        if 'Hypertrophy' in ocr_results:
            data_dic['Hypertrophy'] = ocr_results['Hypertrophy']
        if 'Arrhythmia' in ocr_results:
            data_dic['Arrhythmia'] = list(set(ocr_results['Arrhythmia']))
    return data_dic


def detect_beats(ecg, rate, ransac_window_size=3.35, lowfreq=5.0, highfreq=9.0):
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
        elif abs(max_amplitude + 0.11) < abs(min_amplitude):
            rpeak = np.argmin(local_signal) + search_window.start
        else:
            if max_amplitude >= 0:
                rpeak = np.argmax(local_signal) + search_window.start
            else:
                rpeak = np.argmin(local_signal) + search_window.start

        rpeaks.append(rpeak)

    return np.array(rpeaks)


def lad_rad_detect_beats(ecg,rate,ransac_window_size=3.0,lowfreq=5.0,highfreq=10.0,lp_thresh=0.16):
    ransac_window_size = int(ransac_window_size * rate)

    lowpass = scipy.signal.butter(1, highfreq / (rate / 2.0), 'low')
    highpass = scipy.signal.butter(1, lowfreq / (rate / 2.0), 'high')
    ecg_low = scipy.signal.filtfilt(*lowpass, x=ecg)
    ecg_band = scipy.signal.filtfilt(*highpass, x=ecg_low)
    decg = np.diff(ecg_band)
    decg_power = decg ** 2
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

    shannon_energy = -square_decg_power * np.log(square_decg_power)
    shannon_energy[~np.isfinite(shannon_energy)] = 0.0

    mean_window_len = int(rate * 0.125 + 1)
    lp_energy = np.convolve(shannon_energy, [1.0 / mean_window_len] * mean_window_len, mode='same')

    lp_energy = scipy.ndimage.gaussian_filter1d(lp_energy, rate / lp_thresh)  # 14.0 for pos or neg
    lp_energy_diff = np.diff(lp_energy)

    zero_crossings = (lp_energy_diff[:-1] > 0) & (lp_energy_diff[1:] < 0)
    zero_crossings = np.flatnonzero(zero_crossings)
    zero_crossings -= 1
    return zero_crossings


def modify_arrhythmias(arr_final_result):
    allowed_if_afib_present = {'AFIB', 'PVC_Couplet', 'PVC-Triplet'}
    if 'AFIB' in arr_final_result:
        arr_final_result = [arr for arr in arr_final_result if arr in allowed_if_afib_present]

    return arr_final_result


def detect_rpeaks_eq(ecg, rate, ransac_window_size=3.35, lowfreq=5.0, highfreq=15.0):
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
        elif abs(max_amplitude + 0.11) < abs(min_amplitude):
            rpeak = np.argmin(local_signal) + search_window.start
        else:
            if max_amplitude >= 0:
                rpeak = np.argmax(local_signal) + search_window.start
            else:
                rpeak = np.argmin(local_signal) + search_window.start

        rpeaks.append(rpeak)
    return np.array(rpeaks)


def hr_count(r_index, class_name='6_2'):
    if class_name == '6_2':
        cal_sec = 5
    elif class_name == '12_1':
        cal_sec = 10
    elif class_name == '3_4':
        cal_sec = 2.5
    if cal_sec != 0:
        hr = round(r_index.shape[0] * 60 / cal_sec)
        return hr
    return 0


def is_rhythm_pos_neg(baseline_signal, fs):
    det_r_index = lad_rad_detect_beats(baseline_signal, fs, ransac_window_size=3.0, lowfreq=5.0, highfreq=10.0,
                                       lp_thresh=14.0)
    pos_neg_ind = []
    rhy_label = 'Positive'
    for r_idx in det_r_index:
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


def is_positive_r_wave(ecg_signal, fs):
    baseline_signal, lowpass_signal = filter_signal(ecg_signal, fs=fs).get_data()
    pqrst_data = pqrst_detection(baseline_signal, fs=fs).get_data()

    r_index = pqrst_data['R_index']
    q_index = pqrst_data['Q_Index']

    # Check if r_index and q_index are non-empty
    if len(r_index) > 0 and len(q_index) > 0:
        count_positive_r = 0
        total_r = min(len(r_index), len(q_index))  # Ensure comparison of equal length

        for i in range(total_r):
            r_amplitude = ecg_signal[r_index[i]]
            q_amplitude = ecg_signal[q_index[i]]

            if r_amplitude > q_amplitude:
                count_positive_r += 1

        # Check if 80% or more of the R amplitudes are greater than Q amplitudes
        if count_positive_r / total_r >= 0.6:
            return True
    return False


def is_negative_r_wave(ecg_signal, fs):
    baseline_signal, lowpass_signal = filter_signal(ecg_signal, fs=fs).get_data()
    pqrst_data = pqrst_detection(baseline_signal, fs).get_data()

    r_index = pqrst_data['R_index']
    q_index = pqrst_data['Q_Index']

    # Check if r_index and q_index are non-empty
    if len(r_index) > 0 and len(q_index) > 0:
        count_negative_r = 0
        total_r = min(len(r_index), len(q_index))  # Ensure comparison of equal length

        for i in range(total_r):
            r_amplitude = ecg_signal[r_index[i]]
            q_amplitude = ecg_signal[q_index[i]]

            if r_amplitude < q_amplitude:
                count_negative_r += 1

        # Check if 80% or more of the R amplitudes are less than Q amplitudes
        if count_negative_r / total_r >= 0.60:
            return True
    return False


class arrhythmia_detection:
    def __init__(self, pd_data: pd.DataFrame, fs: int, img_type: str, image_path: str):
        self.all_leads_data = pd_data
        self.fs = fs
        self.img_type = img_type
        self.image_path = image_path

    def find_repeated_elements(self, nested_list, test_for='Arrhythmia'):
        flat_list = []
        for element in nested_list:
            if isinstance(element, list):
                flat_list.extend(element)
            else:
                flat_list.append(element)
        counts = Counter(flat_list)
        threshold = 3
        if test_for == 'Arrhythmia':
            pac_related_found = any(item for item, count in counts.items() if 'PAC' in item and count >= 2)
            ivr_related_found = any(item for item, count in counts.items() if 'IVR' in item and count >= 2)
            if pac_related_found or ivr_related_found:
                threshold = 2

        repeated_elements = [item for item, count in counts.items() if count >= threshold]
        # else:
        #     repeated_elements = [item for item, count in counts.items() if count >= threshold]
        if "PVC_Couplet" in repeated_elements and counts["PVC_Couplet"] <= 2:
            repeated_elements.remove("PVC_Couplet")

        return repeated_elements

    def ecg_signal_processing(self):
        self.leads_pqrst_data = {}
        arr_final_result = mi_final_result = 'Abnormal'

        # Check if 'Arrhythmia' exists and contains only duplicates of 'AFIB'
        skip_afib_flutter = False

        if self.img_type == '6_2':
            ocr = text_detection(self.image_path)

            if "Arrhythmia" in ocr and ocr["Arrhythmia"]:
                arr_list = ocr["Arrhythmia"]
                unique_arr = set(arr_list)  # Convert to set to remove duplicates

                if unique_arr == {"AFIB"} or unique_arr == {"AFL"}:  # If the only unique element is 'AFIB'
                    skip_afib_flutter = True  # Set flag to skip afib_flutter_check
        # try:

        for lead in self.all_leads_data.columns:
            lead_data = {}
            let_inf_label = 'Abnormal'
            # st_t_abn_label = 'Abnormal'
            mi_data = {}
            ecg_signal = self.all_leads_data[lead].values
            if ecg_signal.any():
                arrhythmia_result = combine(ecg_signal, lead, self.img_type, self.fs,
                                            skip_afib_flutter=skip_afib_flutter)
                baseline_signal = arrhythmia_result['Baseline_Signal']
                lowpass_signal = arrhythmia_result['Lowpass_signal']
                r_index = arrhythmia_result['R_Index']
                lead_data['check_pos'] = is_positive_r_wave(ecg_signal, self.fs)
                lead_data['check_neg'] = is_negative_r_wave(ecg_signal, self.fs)
                is_rhythm = is_rhythm_pos_neg(baseline_signal, self.fs)
                lead_data['is_rhythm'] = is_rhythm
                print(f"{lead} : {arrhythmia_result['Combine_Label']}, Rhythm: {is_rhythm}")
                # st_t_abn_label = st_t_abn_model_check(baseline_signal, lowpass_signal, self.fs)
                if lead in ['II', 'III', 'aVF', 'I', 'aVL', 'V5',
                            'V6']:  # and (st_t_abn_label == 'NSTEMI' or st_t_abn_label == 'STEMI')
                    let_inf_label = inf_lat_model_check(lowpass_signal, r_index, lead)
                    print("MI :", let_inf_label)
                    lab = ''
                    if let_inf_label == "TAB":
                        lab = let_inf_label
                    if lead in ['II', 'III', 'aVF'] and let_inf_label == 'STELE':  # let_inf_label == 'TAB'
                        let_inf_label = 'Inferior_MI'
                    if lead in ['III', 'aVF', "II"] and let_inf_label == 'STDEP':
                        let_inf_label = 'Lateral_MI'
                    if lab == "TAB" and let_inf_label != "Lateral_MI" and let_inf_label != "Inferior_MI":
                        let_inf_label = "T_wave_Abnormality"

                lead_data['arrhythmia_data'] = arrhythmia_result

                lbbb_rbbb_data = PVC_detection(ecg_signal, self.fs).get_pvc_data()
                if lbbb_rbbb_data['lbbb_rbbb_label'] != 'Abnormal':
                    mi_data['lbbb_rbbb_label'] = lbbb_rbbb_data['lbbb_rbbb_label']
                    lead_data['mi_data'] = mi_data
                # if st_t_abn_label != 'Abnormal':
                #     mi_data['st_t_abn_label'] = st_t_abn_label
                #     lead_data['mi_data'] = mi_data
                if let_inf_label != 'Abnormal':
                    mi_data['let_inf_label'] = let_inf_label
                    lead_data['mi_data'] = mi_data
                self.leads_pqrst_data[lead] = lead_data
        if self.leads_pqrst_data:
            mi_labels, comm_arrhy_label, all_lead_hr = [], [], []
            try:
                for lead in self.leads_pqrst_data.keys():
                    comm_arrhy_label.append(self.leads_pqrst_data[lead]['arrhythmia_data']['Combine_Label'].split(';'))
                    if self.leads_pqrst_data[lead]['arrhythmia_data']['HR_Count'] > 50:
                        all_lead_hr.append(self.leads_pqrst_data[lead]['arrhythmia_data']['HR_Count'])
                    if 'mi_data' in self.leads_pqrst_data[lead] and 'let_inf_label' in self.leads_pqrst_data[lead][
                        'mi_data']:
                        if self.leads_pqrst_data[lead]['mi_data']['let_inf_label'] != 'Normal':
                            mi_labels.append(self.leads_pqrst_data[lead]['mi_data']['let_inf_label'])
                    if 'mi_data' in self.leads_pqrst_data[lead] and 'lbbb_rbbb_label' in self.leads_pqrst_data[lead][
                        'mi_data']:
                        mi_labels.append(self.leads_pqrst_data[lead]['mi_data']['lbbb_rbbb_label'])

                mi_labels = [condition for condition in mi_labels if condition not in ['STELE', 'STDEP']]
                if len(mi_labels) != 0:
                    check_inf_label = self.find_repeated_elements(mi_labels, test_for="mi")
                    if check_inf_label != 'Abnormal':
                        # temp_list = check_inf_label
                        mi_final_result = check_inf_label

                if self.leads_pqrst_data:
                    if len(all_lead_hr) != 0:
                        total_hr = int(sum(all_lead_hr) / len(all_lead_hr))
                    else:
                        total_hr = 0
                        if self.leads_pqrst_data:
                            temp_lead = next(iter(self.leads_pqrst_data))
                            total_hr = self.leads_pqrst_data[temp_lead]['arrhythmia_data']['HR_Count']
                else:
                    if "II" in self.leads_pqrst_data.keys():
                        get_r_temp_lead = 'II'
                    else:
                        if self.leads_pqrst_data:
                            get_r_temp_lead = next(iter(self.leads_pqrst_data))
                    es = self.all_leads_data[get_r_temp_lead].values
                    base_ecgs = baseline_construction_200(es, 105)
                    lowpass_ecgs = np.array(lowpass(base_ecgs, cutoff=0.3))
                    new_r_index = detect_rpeaks_eq(lowpass_ecgs, self.fs)
                    self.leads_pqrst_data[get_r_temp_lead]['arrhythmia_data']['R_Index'] = new_r_index
                    total_hr = hr_count(new_r_index, self.img_type)

            except Exception as e:
                print("Error: ", e, 'on line_no:', e.__traceback__.tb_lineno)
                total_hr = 0
                mi_final_result = 'Abnormal'

            mod_comm_arrhy = [[item.strip() for item in sublist if item.strip()] for sublist in comm_arrhy_label]
            all_arrhy_result = self.find_repeated_elements(mod_comm_arrhy, test_for="Arrhythmia")
            check_lead_dic = lambda keys, dic: all(key in dic for key in keys)
            lad_rad_keys = ['I', 'II', 'III', 'aVL', 'aVF']

            # For LAD and RAD
            axis_davi = []
            if check_lead_dic(lad_rad_keys, self.leads_pqrst_data):
                if (self.leads_pqrst_data['I']['is_rhythm'] == "Positive" and self.leads_pqrst_data['aVL'][
                    'is_rhythm'] == "Positive" and
                        self.leads_pqrst_data["II"]["is_rhythm"] == "Negative" and self.leads_pqrst_data["aVF"][
                            "is_rhythm"] == "Negative"):
                    axis_davi.append("Left_Axis_Deviation")
                elif (self.leads_pqrst_data["I"]["is_rhythm"] == "Negative" and self.leads_pqrst_data["aVL"][
                    "is_rhythm"] == "Negative" and self.leads_pqrst_data["II"]["is_rhythm"] == "Positive" and
                      self.leads_pqrst_data["aVF"]["is_rhythm"] == "Positive" and self.leads_pqrst_data["III"][
                          "is_rhythm"] == "Positive"):
                    axis_davi.append("Right_Axis_Deviation")
                elif (self.leads_pqrst_data["I"]["is_rhythm"] == "Negative" and
                      self.leads_pqrst_data["aVF"]["is_rhythm"] == "Negative"):
                    axis_davi.append("Extreme_Axis_Deviation")
                else:
                    axis_davi.append('Normal')
            else:
                axis_davi.append('Normal')

            # For LAFB and LPFB
            lafb_lpfb_result = []
            if check_lead_dic(lad_rad_keys, self.leads_pqrst_data):
                if (self.leads_pqrst_data['I']['check_pos'] == True and self.leads_pqrst_data['aVL'][
                    'check_pos'] == True and
                        self.leads_pqrst_data["II"]["check_neg"] == True and self.leads_pqrst_data["III"][
                            "check_neg"] == True and self.leads_pqrst_data["aVF"]["check_neg"] == True):
                    lafb_lpfb_result.append("LAFB")
                elif (self.leads_pqrst_data["I"]["check_neg"] == True and self.leads_pqrst_data["aVL"][
                    "check_neg"] == True and self.leads_pqrst_data["II"]["check_pos"] == True and
                      self.leads_pqrst_data["aVF"]["check_pos"] == True and self.leads_pqrst_data["III"][
                          "check_pos"] == True):
                    lafb_lpfb_result.append("LPFB")
                else:
                    lafb_lpfb_result.append('Normal')
            else:
                lafb_lpfb_result.append('Normal')

            # If all_arrhy_result has more than 1 element and contains "Normal" in any case, remove it
            if len(all_arrhy_result) > 1:
                all_arrhy_result = [arr for arr in all_arrhy_result if arr.lower() != "normal"]

            all_arrhy_result = [item for item in all_arrhy_result if item != '']
            all_arrhy_result = list(set(modify_arrhythmias(all_arrhy_result)))
            arr_final_result = ' '.join(all_arrhy_result)

            if "II" in self.leads_pqrst_data.keys():
                get_temp_lead = 'II'
                get_pro_lead = self.all_leads_data["II"]
            else:
                if self.leads_pqrst_data:
                    get_temp_lead = next(iter(self.leads_pqrst_data))
                    get_pro_lead = self.all_leads_data[get_temp_lead]
            lead_info_data = find_ecg_info(get_pro_lead, self.img_type, self.image_path)

            if 'HR' in lead_info_data:
                if lead_info_data['HR'] is not None:
                    total_hr = lead_info_data['HR']

            if len(arr_final_result) == 0:
                if "II" in self.leads_pqrst_data.keys():
                    if self.leads_pqrst_data['II']['arrhythmia_data']['RR_Label'] == 'Regular':
                        arr_final_result = 'NORMAL'
                    if total_hr < 60:
                        arr_final_result = "BR"
                    if total_hr > 100:
                        arr_final_result = "TC"

                else:
                    arr_final_result = 'NORMAL'
                    if total_hr < 60:
                        arr_final_result = "BR"
                    if total_hr > 100:
                        arr_final_result = "TC"

            if 'MI' in lead_info_data:
                if mi_final_result != 'Abnormal' and type(mi_final_result) == list:

                    ecr_mi = lead_info_data['MI']

                    if 'T_wave_Abnormality' not in mi_final_result and 'T_wave_Abnormality' in ecr_mi:
                        mi_final_result.append('T_wave_Abnormality')

                    if 'Lateral_MI' not in mi_final_result and 'Lateral_MI' in ecr_mi:
                        mi_final_result.append('Lateral_MI')

                    if 'Inferior_MI' not in mi_final_result and 'Inferior_MI' in ecr_mi:
                        mi_final_result.append('Inferior_MI')

                else:

                    mi_final_result = lead_info_data['MI']

            detections = []

            if "Arrhythmia" in lead_info_data and lead_info_data["Arrhythmia"]:
                lead_info_data["Arrhythmia"] = list(set(lead_info_data["Arrhythmia"]))  # Remove duplicates
            # **Prepare Unique Detections List**
            unique_detections = set()  # Use a set to avoid duplicates
            # **Include Unique Arrhythmias from lead_info_data**
            if "Arrhythmia" in lead_info_data and lead_info_data["Arrhythmia"]:
                for arr in lead_info_data["Arrhythmia"]:
                    unique_detections.add(arr)

            # **Include Unique Arrhythmias from all_arrhy_result**
            for lab in all_arrhy_result:
                unique_detections.add(lab)
            existing_detects = {d['detect'].lower() for d in detections}
            for detect in unique_detections:
                if detect.lower() not in existing_detects:
                    detections.append({"detect": detect, "detectType": "Arrhythmia", "confidence": 100})

            if isinstance(all_arrhy_result, list) and len(all_arrhy_result) > 1:
                for lab in all_arrhy_result:
                    if lab.lower() == "normal" and lab == '':
                        if total_hr < 60:
                            lab = "BR"
                        if total_hr > 100:
                            lab = "TC"
                    if lab:
                        detections.append({"detect": lab, "detectType": "Arrhythmia", "confidence": 100})
            else:
                if all_arrhy_result:
                    detect_value = all_arrhy_result if not isinstance(all_arrhy_result, list) else all_arrhy_result[0]
                else:
                    detect_value = "Normal"
                if detect_value.lower() == "normal" or detect_value == '':
                    if total_hr < 60:
                        detect_value = "BR"
                    elif total_hr > 100:
                        detect_value = "TC"
                    elif detect_value == "Normal":
                        detect_value = "NORMAL"
                if detect_value:
                    detections.append({"detect": detect_value, "detectType": "Arrhythmia", "confidence": 100})

            # Ensure Normal is not added if TC or BR exists
            arr_labels = {d["detect"].lower() for d in detections}

            if "normal" in arr_labels and ("tc" in arr_labels or "br" in arr_labels):
                detections = [d for d in detections if d["detect"].lower() != "normal"]
            if any(d["detect"].lower() in ["afib", "afl"] for d in detections):
                detections = [d for d in detections if d["detect"].lower() not in ["normal", "tc"]]
            if any(d["detect"].lower() in ["afib", "afl"] for d in detections):
                detections = [d for d in detections if not d["detect"].lower().startswith(("pvc", "pac"))]

            # **Remove ALL Duplicate Entries from `detections` while Preserving Order**
            seen = set()  # Use a set to track unique detection names (case-insensitive)
            final_detections = []  # List to store final unique detections

            for d in detections:
                key = d["detect"].lower()  # Convert to lowercase to make it case-insensitive
                if key not in seen:
                    seen.add(key)
                    final_detections.append(d)  # Keep original formatting in final output

            # **Update detections list**
            detections = final_detections

            # Handling MI detections
            if mi_final_result != 'Abnormal':
                if isinstance(mi_final_result, list) and len(mi_final_result) > 1:
                    for mi_lab in mi_final_result:
                        detections.append({"detect": mi_lab, "detectType": "MI", "confidence": 100})
                elif mi_final_result != []:
                    # If there's only one result, append it directly as a string
                    detect_value = mi_final_result if not isinstance(mi_final_result, list) else mi_final_result[0]
                    detections.append({"detect": detect_value, "detectType": "MI", "confidence": 100})

            if 'Hypertrophy' in lead_info_data and isinstance(lead_info_data['Hypertrophy'], list):
                for hypertrophy_label in lead_info_data['Hypertrophy']:
                    if hypertrophy_label and hypertrophy_label.lower() != 'normal':
                        detections.append({"detect": hypertrophy_label, "detectType": "Hypertrophy", "confidence": 100})

            if 'Normal' not in axis_davi:
                detections.append({"detect": axis_davi[0], "detectType": "axisDeviation", "confidence": 100})

            if 'Normal' not in lafb_lpfb_result:
                detections.append({"detect": lafb_lpfb_result[0], "detectType": "MI", "confidence": 100})


            if self.leads_pqrst_data:
                check_pvc_detect = lambda detections: bool(list(filter(lambda x: "PVC" in x["detect"], detections)))
                check_pac_detect = lambda detections: bool(list(filter(lambda x: "PAC" in x["detect"], detections)))
                detect_values = {d['detect'] for d in detections}
                matching_keys = [
                    key for key, value in self.leads_pqrst_data.items()
                    if any(
                        detect in value.get('arrhythmia_data', {}).get('Combine_Label', '').split(';')
                        for detect in detect_values
                    )
                ]
                if check_pac_detect(detections):
                    if matching_keys:
                        get_temp_lead = matching_keys[0]
                    total_pac = self.leads_pqrst_data[get_temp_lead]['arrhythmia_data']['PAC_DATA']['PAC_Union']
                    self.leads_pqrst_data['pacQrs'] = \
                    self.leads_pqrst_data[get_temp_lead]['arrhythmia_data']['PAC_DATA']['PAC_Index']
                else:
                    total_pac = []
                    self.leads_pqrst_data['pacQrs'] = []
                if check_pvc_detect(detections):
                    if matching_keys:
                        get_temp_lead = matching_keys[0]
                    self.leads_pqrst_data['pvcQrs'] = \
                    self.leads_pqrst_data[get_temp_lead]['arrhythmia_data']['PVC_DATA']['PVC-Index']
                    self.leads_pqrst_data['Vbeat'] = len(
                        self.leads_pqrst_data[get_temp_lead]['arrhythmia_data']['PVC_DATA']['PVC-Index'])
                else:
                    self.leads_pqrst_data['pvcQrs'] = []
                    self.leads_pqrst_data['Vbeat'] = 0
                self.leads_pqrst_data['beats'] = len(self.leads_pqrst_data[get_temp_lead]['arrhythmia_data']['R_Index'])

            else:
                total_pac = []
                self.leads_pqrst_data['beats'] = 0
                self.leads_pqrst_data['pvcQrs'] = []
                self.leads_pqrst_data['Vbeat'] = 0
                self.leads_pqrst_data['pacQrs'] = []

            if 'HR' in lead_info_data:
                if lead_info_data['HR'] is not None:
                    self.leads_pqrst_data['avg_hr'] = lead_info_data['HR']
                else:
                    self.leads_pqrst_data['avg_hr'] = total_hr
            else:
                self.leads_pqrst_data['avg_hr'] = total_hr
            if arr_final_result == "Normal":
                arr_final_result = "NORMAL"
            self.leads_pqrst_data['arr_final_result'] = arr_final_result
            self.leads_pqrst_data['mi_final_result'] = mi_final_result
            self.leads_pqrst_data['detections'] = detections
            self.leads_pqrst_data['RRInterval'] = lead_info_data['rr_interval']
            self.leads_pqrst_data['PRInterval'] = lead_info_data['PRInterval']
            self.leads_pqrst_data['QTInterval'] = lead_info_data['QTInterval']
            self.leads_pqrst_data['QRSComplex'] = lead_info_data['QRSComplex']
            self.leads_pqrst_data['STseg'] = lead_info_data['STseg']
            self.leads_pqrst_data['PRseg'] = lead_info_data['PRseg']
            self.leads_pqrst_data['QTc'] = lead_info_data['QTc']
            self.leads_pqrst_data['Abeat'] = total_pac.count(1) if len(total_pac) != 0 else 0
            self.leads_pqrst_data['color_dict'] = {}
        else:
            self.leads_pqrst_data = {"avg_hr": 0,
                                     "arr_final_result": 'Abnormal',
                                     "mi_final_result": 'Abnormal',
                                     "beats": 0,
                                     "detections": [],
                                     "RRInterval": 0,
                                     "PRInterval": 0,
                                     "QTInterval": 0,
                                     "QRSComplex": 0,
                                     "STseg": 0,
                                     "PRseg": 0,
                                     "QTc": 0,
                                     "pvcQrs": [],
                                     "pacQrs": [],
                                     "Vbeat": 0,
                                     "Abeat": 0,
                                     "color_dict": {},
                                     }
        return self.leads_pqrst_data


def check_noise(all_leads_data, class_name, fs):
    noise_result = []
    final_result = 'Normal'

    for lead in all_leads_data.keys():
        ecg_signal = all_leads_data[lead]
        ecg_signal = np.asarray(ecg_signal).ravel()
        get_noise = NoiseDetection(ecg_signal, class_name, frequency=fs).noise_model_check()
        noise_result.append(get_noise)

    noise_cou = noise_result.count('ARTIFACTS')

    if noise_cou >= len(all_leads_data.keys()) / 2:
        final_result = 'ARTIFACTS'

    return final_result

def process_and_plot_leads(ecg_df, file_name, result,top_label, class_name="6_2", mm_per_sec=25, mm_per_mV=10, signal_scale=0.01):
    leads = ['I', 'II', 'III', 'aVR', 'aVL', 'aVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']
    df = ecg_df

    # Define layouts
    if class_name == "6_2":
        lead_layout = [
            ['I', 'V1'], ['II', 'V2'], ['III', 'V3'],
            ['aVR', 'V4'], ['aVL', 'V5'], ['aVF', 'V6']
        ]
        rows, cols = 6, 2
        sampling_rate = 200
        fig_width_px, fig_height_px = 2800, 1770
    elif class_name == "3_4":
        lead_layout = [
            ['I', 'aVR', 'V1', 'V4'],
            ['II', 'aVL', 'V2', 'V5'],
            ['III', 'aVF', 'V3', 'V6']
        ]
        rows, cols = 3, 4
        sampling_rate = 300
        fig_width_px, fig_height_px = 1100, 1100
    elif class_name == "12_1":
        lead_layout = [[lead] for lead in leads]
        rows, cols = 12, 1
        sampling_rate = 325
        fig_width_px, fig_height_px = 2495, 3545
    else:
        raise ValueError("Invalid layout. Use '6_2', '3_4','12_1'")

    # Label remapping if top_label is 'avl'
    if str(top_label).lower() == 'avl':
        remap = {
            'I': 'aVL',
            'II': 'I',
            'III': 'aVR',
            'aVR': 'II',
            'aVL': 'aVF',
            'aVF': 'III'
        }     
    else:
        remap = {}

    # Time axis in mm
    time_sec = np.arange(df.shape[0]) / sampling_rate
    time_mm = time_sec * mm_per_sec
    box_height_mm = 25
    box_width_mm = time_mm[-1] + 10
    fig_width_mm = box_width_mm * cols
    grid_padding_mm = 20 if class_name == "3_4" else 0
    fig_height_mm = box_height_mm * rows + grid_padding_mm

    dpi = 100
    fig_width_in = fig_width_px / dpi
    fig_height_in = fig_height_px / dpi
    fig, ax = plt.subplots(figsize=(fig_width_in, fig_height_in), dpi=dpi)

    def draw_ecg_grid(ax, width_mm, height_mm):
        ax.set_xlim(0, width_mm)
        ax.set_ylim(0, height_mm)
        ax.set_aspect('equal')
        ax.axis('off')
        for x in np.arange(0, width_mm + 1, 1):
            ax.axvline(x=x, color='pink', linewidth=0.4, alpha=0.7)
        for y in np.arange(0, height_mm + 1, 1):
            ax.axhline(y=y, color='pink', linewidth=0.4, alpha=0.7)
        for x in np.arange(0, width_mm + 1, 5):
            ax.axvline(x=x, color='red', linewidth=0.6, alpha=0.7)
        for y in np.arange(0, height_mm + 1, 5):
            ax.axhline(y=y, color='red', linewidth=0.6, alpha=0.7)

    draw_ecg_grid(ax, fig_width_mm, fig_height_mm)

    # Extract label dictionary
    label_dict, color_dict = {}, {}
    for item in result.get('detections', []):
        if 'detectType' not in item or 'detect' not in item:
            continue
        key, value = item['detectType'], item['detect']
        label_dict[key] = f"{label_dict.get(key, '')}, {value}" if key in label_dict else value

    for r in range(rows):
        for c in range(cols):
            try:
                lead = lead_layout[r][c]
            except IndexError:
                continue

            if lead not in df.columns:
                continue

            raw = result[lead]['arrhythmia_data']['Baseline_Signal']

            if class_name == '3_4':
                amplitude_boost_boxes = 0.1
                y_offset = fig_height_mm - grid_padding_mm / 3 - (r + 1) * box_height_mm
            else:
                amplitude_boost_boxes = 4
                y_offset = fig_height_mm - grid_padding_mm / 2 - (r + 1) * box_height_mm

            amplitude_boost_mm = amplitude_boost_boxes * 1
            scale_factor = mm_per_mV + amplitude_boost_mm

            signal = (raw - np.mean(raw)) * signal_scale * scale_factor

            if class_name == '3_4':
                gap_mm = 5
                shift_left_mm = 3
                if c == 0:
                    x_offset = c * box_width_mm - shift_left_mm
                elif c > 0:
                    x_offset = c * box_width_mm + gap_mm
                else:
                    x_offset = c * box_width_mm
            else:
                x_offset = c * box_width_mm
            signal_shift_mm = 10 if c == 0 else 0

            x = time_mm + x_offset + signal_shift_mm
            y = signal + y_offset + box_height_mm / 2

            # Plot ECG waveform
            ax.plot(x, y, color='black', linewidth=1.5, alpha=1)

            arrhythmia_data = result.get(lead, {}).get('arrhythmia_data', {})
            pac_index, junc_index, pvc_index = [], [], []
            if lead in ['II', 'III','aVF', 'V1', 'V2', 'V5', 'V6']:
                if 'PAC' in label_dict['Arrhythmia']:
                    pac_index = arrhythmia_data.get('PAC_DATA', {}).get('PAC_Index', [])
                if 'Junctional' in label_dict['Arrhythmia']:
                    junc_index = arrhythmia_data.get('PAC_DATA', {}).get('junc_index', [])
                if 'PVC' in label_dict['Arrhythmia'] or 'NSVT' in label_dict['Arrhythmia']:
                    pvc_index = arrhythmia_data.get('PVC_DATA', {}).get('PVC-Index', [])

            # PAC
            if pac_index:
                color_dict['PAC'] = 'green'
                for st, ed in pac_index:
                    ax.plot(x[st:ed], y[st:ed], color='white', linewidth=5, alpha=0.3)
                    ax.plot(x[st:ed], y[st:ed], color='green', linewidth=1, alpha=0.6)

            # JUNCTIONAL
            if junc_index:
                color_dict['Junctional'] = 'brown'
                for st, ed in junc_index:
                    ax.plot(x[st:ed], y[st:ed], color='white', linewidth=5, alpha=0.3)
                    ax.plot(x[st:ed], y[st:ed], color='brown', linewidth=1, alpha=0.6)

            # PVC
            if pvc_index:
                color_dict['PVC'] = 'red'
                for idx in pvc_index:
                    st = max(idx - 20, 0)
                    ed = min(idx + 50, len(x))
                    ax.plot(x[st:ed], y[st:ed], color='white', linewidth=5, alpha=0.3)
                    ax.plot(x[st:ed], y[st:ed], color='red', linewidth=1, alpha=0.6)

            # Rhythm background coloring
            rhythm_color = None
            if lead in ['I', 'II', 'III', 'V1', 'V2', 'V5', 'V6']:
                if 'BR' in label_dict.get('Arrhythmia', ''):
                    rhythm_color = 'orangered'; color_dict['BR'] = rhythm_color
                elif 'TC' in label_dict.get('Arrhythmia', ''):
                    rhythm_color = 'magenta'; color_dict['TC'] = rhythm_color
                elif any(x in label_dict.get('Arrhythmia', '') for x in ['I_Degree', 'III_Degree', 'MOBITZ_I', 'MOBITZ_II']):
                    rhythm_color = 'blue'; color_dict['block'] = rhythm_color
                elif any(x in label_dict.get('Arrhythmia', '') for x in ['VFIB/Vflutter', 'ASYS']):
                    rhythm_color = 'aqua'; color_dict['VFIB_Asystole'] = rhythm_color
            if rhythm_color:
                ax.plot(x, y, color=rhythm_color, linewidth=0.8)

            lead_color = 'blue' if lead in ['II', 'III', 'aVF', 'I', 'aVL', 'V5','V6'] and 'MI' in label_dict else 'black'
            display_label = remap.get(lead, lead)

            label_shift_right_mm = 5
            if c == 0:
                label_x = x_offset + 3 + label_shift_right_mm
                label_align = 'left'
            else:
                label_x = x_offset + time_mm[0] - 3
                label_align = 'right'

            ax.text(label_x, np.median(y) + 5, display_label, fontsize=16,
                    verticalalignment='center', horizontalalignment=label_align,
                    fontweight='bold', color=lead_color)

    # Title section
    # Construct title
    title_lines = ["12-Lead ECG Plot"]
    if 'Arrhythmia' in label_dict:
        title_lines.append(f"Arrhythmia: {label_dict['Arrhythmia']}")
    if 'MI' in label_dict:
        title_lines.append(f"MI: {label_dict['MI']}")
    if 'axisDeviation' in label_dict:
        title_lines.append(f"Axis Deviation: {label_dict['axisDeviation']}")
    if 'Hypertrophy' in label_dict:
        title_lines.append(f"Hypertrophy: {label_dict['Hypertrophy']}")
    fig.subplots_adjust(top=0.90)
    fig.suptitle('\n'.join(title_lines), fontsize=16, fontweight='bold')


    # Add text labels at the bottom
    bottom_label = (
        # f"HR: {result.get('avg_hr', '--')} BPM   |   "
        # f"Image Type: {class_name}   |   "
        # f"RR: {result.get('RRInterval', '--')} ms   |   "
        # f"PR: {result.get('PRInterval', '--')} ms   |   "
        # f"QT: {result.get('QTInterval', '--')} ms   |   "
        # f"QRS: {result.get('QRSComplex', '--')} ms   |   "
        # f"ST: {result.get('STseg', '--')} ms   |   "
        # f"PRseg: {result.get('PRseg', '--')} ms   |   "
        # f"QTc: {result.get('QTc', '--')} ms"
        f'''HR: {result['avg_hr']}, image_type: {class_name},
        RRInterval: {result['RRInterval']}, PRInterval: {result['PRInterval']},
        QTInterval: {result['QTInterval']}, QRSComplex: {result['QRSComplex']},
        STseg: {result['STseg']}, PRseg: {result['PRseg']}, QTc: {result['QTc']}'''
    )
    plt.figtext(0.5, 0.015, bottom_label, ha='center', fontsize=14, wrap=True,fontweight='bold')
    plt.tight_layout(rect=[0.02, 0.03, 0.98, 0.90])
    # plt.tight_layout(rect=[0.02, 0.03, 0.98, 0.93]) # Leave space at the bottom for text_label

    # Save the plot
    result['color_dict'] = color_dict
    save_img = os.path.join(settings.MEDIA_ROOT, 'analysis_tool','uploads')
    temp_path = base_dir_path + f"\\cropped_lead_images\\temp_path\\{file_name}_temp.jpg"
    
    fig.savefig(temp_path, dpi=dpi, bbox_inches='tight', pad_inches=0)
    plt.close()

    final_path = f"{save_img}\\{file_name}.jpg"
    img = Image.open(temp_path)
    img = img.resize((fig_width_px, fig_height_px), Image.LANCZOS)
    img.save(final_path)
    os.remove(temp_path)

    # ================= Return & Attach Summary ==================
    metrics = {
        "HR": result.get("avg_hr"),
        "RRInterval": result.get("RRInterval"),
        "PRInterval": result.get("PRInterval"),
        "QTInterval": result.get("QTInterval"),
        "QRSComplex": result.get("QRSComplex"),
        "STseg": result.get("STseg"),
        "PRseg": result.get("PRseg"),
        "QTc": result.get("QTc"),
    }

    summary = {
        "title_lines": title_lines,
        "bottom_label": bottom_label,
        "metrics": metrics,
        "image_path": final_path
    }

    # Attach to result so check_oea_analysis can access it
    result["summary_text"] = summary

    return summary

def setup_ecg_subplot(fig):

    total_time = 10  # seconds for full width
    ax = fig.add_axes([0, 0, 1, 1], zorder=0)

    ax.set_xticks(np.arange(0, total_time, 0.2))  # Major grid (5mm = 0.2s)
    ax.set_yticks(np.arange(-50, 91, 5))  # Major voltage grid (5mm = 0.5mV)
    ax.grid(True, which='major', color='red', linestyle='-', linewidth=0.8, alpha=0.9)  # Dark grid
    ax.grid(True, which='minor', color='red', linestyle='-', linewidth=0.3, alpha=0.6)  # Light grid
    ax.minorticks_on()




def correct_orientation(image_path):
    img = Image.open(image_path)
    for orientation in ExifTags.TAGS.keys():
        if ExifTags.TAGS[orientation] == 'Orientation':
            break
    if img._getexif() is None:
        return np.array(img.convert('RGB'))

    exif = dict(img._getexif().items())

    orientation_value = exif.get(orientation, None)
    if orientation_value == 3:
        img = img.rotate(180, expand=True)
    elif orientation_value == 6:
        img = img.rotate(270, expand=True)
    elif orientation_value == 8:
        img = img.rotate(90, expand=True)

    return np.array(img.convert('RGB'))


def orientation_image(image_path):
    image = correct_orientation(image_path)
    gray = color.rgb2gray(image)
    edges = cv2.Canny((gray * 255).astype(np.uint8), 50, 150)
    h, theta, d = hough_line(edges)
    angles = []
    for _, angle, dist in zip(*hough_line_peaks(h, theta, d)):
        angle_deg = np.degrees(angle)
        if -15 < angle_deg < 15:
            angles.append(angle_deg)

    if len(angles) > 0:
        median_angle = np.median(angles)
    else:
        median_angle = 0

    estimated_skew_angle = median_angle
    corrected_image = rotate(image, median_angle, resize=True)
    corrected_image = (corrected_image * 255).astype(np.uint8)
    return corrected_image

def expectation_maximization(image, max_iter=1, tol=1e-6, div_thresh = 2):
    # Flatten the image into a 1D array

    # add to input gray image

    pixel_values = image.flatten()

    # Initialize parameters for two Gaussian distributions
    # Initial means (just as an example, could be improved)
    mean_0 = np.mean(pixel_values) - np.std(pixel_values)
    mean_1 = np.mean(pixel_values) + np.std(pixel_values)

    # Initial variances (arbitrary small values)
    var_0 = np.var(pixel_values) / 2
    var_1 = np.var(pixel_values) / 2

    # Initial mixing coefficients
    weight_0 = 0.5
    weight_1 = 0.5

    # Initialize a list to store probabilities
    probabilities = np.zeros((len(pixel_values), 2))

    # E-step and M-step iterations
    for iteration in range(max_iter):
        # E-step: Calculate responsibilities (probabilities)
        gaussian_0 = (1 / np.sqrt(2 * np.pi * var_0)) * np.exp(- (pixel_values - mean_0) ** 2 / (2 * var_0))
        gaussian_1 = (1 / np.sqrt(2 * np.pi * var_1)) * np.exp(- (pixel_values - mean_1) ** 2 / (2 * var_1))

        # Compute responsibilities
        weighted_gaussian_0 = weight_0 * gaussian_0
        weighted_gaussian_1 = weight_1 * gaussian_1
        total = weighted_gaussian_0 + weighted_gaussian_1

        # Normalize
        probabilities[:, 0] = weighted_gaussian_0 / total
        probabilities[:, 1] = weighted_gaussian_1 / total

        # M-step: Update parameters
        # Update weights (mixing coefficients)
        weight_0 = np.mean(probabilities[:, 0])
        weight_1 = np.mean(probabilities[:, 1])

        # Update means
        mean_0 = np.sum(probabilities[:, 0] * pixel_values) / np.sum(probabilities[:, 0])
        mean_1 = np.sum(probabilities[:, 1] * pixel_values) / np.sum(probabilities[:, 1])

        # Update variances
        var_0 = np.sum(probabilities[:, 0] * (pixel_values - mean_0) ** 2) / np.sum(probabilities[:, 0])
        var_1 = np.sum(probabilities[:, 1] * (pixel_values - mean_1) ** 2) / np.sum(probabilities[:, 1])

        # Check for convergence
        if iteration > 0:
            mean_diff = abs(mean_1 - mean_0)
            if mean_diff < tol:
                print(f"Converged after {iteration} iterations.")
                break

    # After convergence, the threshold is chosen as the mean between the two Gaussian components
    threshold = (mean_0 + mean_1) / 2
    threshold = threshold / div_thresh
    return threshold


def extract_black_on_white(image_path, ecg_type,orig_height=None, orig_width=None):
    # Load image in color
    original_img = cv2.imread(image_path)

    if ecg_type == "12_1":
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        threshold = expectation_maximization(image, div_thresh = 4)  # 70
    elif ecg_type == "6_2":
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if orig_height is not None and orig_width is not None and orig_height < 2000 and orig_width < 2000:
            threshold = 240
        else:
            threshold = expectation_maximization(image)
    elif ecg_type == "3_4":
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        threshold = expectation_maximization(image, div_thresh=1)
        # Use original image size for threshold logic
        if orig_height is not None and orig_width is not None and orig_height > 2000 and orig_width > 2000:
            threshold = 60
        else:
            threshold = int(threshold)
            threshold = np.clip(threshold, 185, 190)
        lower_black = np.array([0, 0, 0], dtype=np.uint8)
        upper_black = np.array([int(threshold), int(threshold), int(threshold)], dtype=np.uint8)
        black_mask = cv2.inRange(original_img, lower_black, upper_black)
        white_canvas = np.ones_like(original_img) * 255
        result_on_white = np.where(black_mask[:, :, None] == 255, original_img, white_canvas)
        return result_on_white

    if ecg_type == "6_2" or ecg_type == '12_1':
        height, width = image.shape
        pixels = image.reshape(-1, 1)
        labels = np.zeros_like(pixels)
        labels[pixels > threshold] = 1
        lda = LinearDiscriminantAnalysis()
        lda.fit(pixels, labels.ravel())
        predicted_labels = lda.predict(pixels)
        binary_image = predicted_labels.reshape(height, width)
        binary_image = binary_image * 255
        result_on_white = cv2.bitwise_not(binary_image)

        return result_on_white


class ImageFilter:
    """Wraps an image with easy pixel access."""

    def __init__(self, file_path):
        self.image = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
        _, self.image = cv2.threshold(self.image, 128, 255, cv2.THRESH_BINARY_INV)
        self.height, self.width = self.image.shape

    def __getitem__(self, key):
        """Allows access using image[y0:y1, x0:x1]."""
        return self.image[key]


class Point:
    """Represents a point with x and y coordinates."""

    def __init__(self, x, y):
        self.x = x
        self.y = y


class SignalExtractor:
    """
    Signal extractor of an ECG image.
    """

    def __init__(self, n: int) -> None:
        """Initialization of the signal extractor.
        Args:
            n (int): Number of signals to extract.
        """
        self.__n = n

    def extract_signals(self, ecg: ImageFilter):
        """
        Extract the signals from the ECG image.
        Args:
            ecg (Image): ECG image.
        Returns:
            List of extracted signals as lists of Point objects.
        """
        N = ecg.width
        LEN, SCORE = (2, 3)  # Cache values
        rois = self.__get_roi(ecg)
        mean = lambda cluster: (cluster[0] + cluster[-1]) / 2
        cache = {}

        for col in range(1, N):
            prev_clusters = self.__get_clusters(ecg, col - 1)
            if not prev_clusters:
                continue
            clusters = self.__get_clusters(ecg, col)
            for c in clusters:
                cache[col, c] = [None] * self.__n
                for roi_i in range(self.__n):
                    costs = {}
                    for pc in prev_clusters:
                        node = (col - 1, pc)
                        ctr = math.ceil(mean(pc))
                        if node not in cache:
                            cache[node] = [[ctr, None, 1, 0]] * self.__n
                        ps = cache[node][roi_i][SCORE]  # Previous score
                        d = abs(ctr - rois[roi_i])  # Vertical distance to ROI
                        g = self.__gap(pc, c)  # Disconnection level
                        costs[pc] = ps + d + N / 10 * g

                    best = min(costs, key=costs.get)
                    y = math.ceil(mean(best))
                    p = (col - 1, best)
                    l = cache[p][roi_i][LEN] + 1
                    s = costs[best]
                    cache[col, c][roi_i] = (y, p, l, s)

        raw_signals = self.__backtracking(cache, rois)
        return raw_signals

    def __get_roi(self, ecg: ImageFilter):
        """Get the coordinates of the ROI from the ECG image."""
        WINDOW = 10
        SHIFT = (WINDOW - 1) // 2
        stds = np.zeros(ecg.height)

        for i in range(ecg.height - WINDOW + 1):
            std = ecg[i: i + WINDOW, :].reshape(-1).std()
            stds[i + SHIFT] = std

        min_distance = int(ecg.height * 0.1)
        peaks, _ = find_peaks(stds, distance=min_distance)
        rois = sorted(peaks, key=lambda x: stds[x], reverse=True)

        if len(rois) < self.__n:
            self.__n = len(rois)
            # raise ("The indicated number of ROIs could not be detected.")
            pass

        return sorted(rois[: self.__n])

    def __get_clusters(self, ecg: ImageFilter, col: int):
        """Get clusters of black pixels in a given column."""
        BLACK = 0
        clusters = []
        black_p = np.where(ecg[:, col] == BLACK)[0]

        for _, g in groupby(enumerate(black_p), lambda idx_val: idx_val[0] - idx_val[1]):
            clusters.append(tuple(map(itemgetter(1), g)))

        return clusters

    def __gap(self, pc, c):
        """Compute the gap between two clusters (vertical white space)."""
        pc_min, pc_max = pc[0], pc[-1]
        c_min, c_max = c[0], c[-1]
        if pc_min <= c_min and pc_max <= c_max:
            return len(range(pc_max + 1, c_min))
        elif pc_min >= c_min and pc_max >= c_max:
            return len(range(c_max + 1, pc_min))
        return 0

    def __backtracking(self, cache, rois):
        """Extracts signals using backtracking."""
        X_COORD, CLUSTER = (0, 1)
        Y_COORD, PREV, LEN = (0, 1, 2)
        raw_signals = [None] * self.__n

        for roi_i in range(self.__n):
            roi = rois[roi_i]
            max_len = max([v[roi_i][LEN] for v in cache.values()])
            cand_nodes = [node for node, stats in cache.items() if stats[roi_i][LEN] == max_len]
            best = min(cand_nodes, key=lambda node: abs(math.ceil(np.mean(node[CLUSTER])) - roi))
            raw_s = []

            while best is not None:
                y = cache[best][roi_i][Y_COORD]
                raw_s.append(Point(best[X_COORD], y))
                best = cache[best][roi_i][PREV]

            raw_s.reverse()
            raw_signals[roi_i] = raw_s

        return raw_signals


# Predict the ECG grid type using TFLite model
def predict_grid_type(image_path):
    with results_lock:
        with tf.device('cpu'):
            classes = ['12_1', '3_4', '6_2', 'No ECG']
            image = Image.open(image_path).convert('RGB')
            input_arr = np.array(image, dtype=np.float32)
            input_arr = tf.image.resize(input_arr, size=(224, 224), method=tf.image.ResizeMethod.BILINEAR)
            input_arr = tf.expand_dims(input_arr, axis=0)
            img_interpreter.set_tensor(img_input_details[0]['index'], input_arr)
            img_interpreter.invoke()
            output_data = img_interpreter.get_tensor(img_output_details[0]['index'])
            idx = np.argmax(output_data[0])
            return output_data[0], classes[idx]


# Function to get bounding box for left-side leads and extend it properly
def get_left_bounding_box(leads, labels_and_boxes, right_leads_x_min, image_height):
    selected_boxes = [box for label, box in labels_and_boxes if label in leads]

    if not selected_boxes:
        return None  # No leads detected for this side

    x_min = min(box[0] for box in selected_boxes)
    y_min = max(min(box[1] for box in selected_boxes) - 50, 0)  # Expand top, ensure within bounds
    x_max = right_leads_x_min - 10  # Extend up to 10 pixels before right-side leads
    y_max = min(max(box[3] for box in selected_boxes) + 250, image_height)  # Expand bottom, ensure within bounds

    return (x_min, y_min, x_max, y_max)

# Function to get bounding box for right-side leads and extend it to 90% width
def get_right_bounding_box(leads, labels_and_boxes, image_width, image_height):
    selected_boxes = [box for label, box in labels_and_boxes if label in leads]

    if not selected_boxes:
        return None  # No leads detected for this side

    x_min = min(box[0] for box in selected_boxes)
    y_min = max(min(box[1] for box in selected_boxes) - 60, 0)  # Expand top, ensure within bounds
    x_max = int(image_width * 0.9)  # Extend right box to 90% of the image width
    y_max = min(max(box[3] for box in selected_boxes) + 250, image_height)  # Expand bottom, ensure within bounds

    return (x_min, y_min, x_max, y_max)


def clamp_bbox(x1, y1, x2, y2, image_width, image_height):
    return max(0, x1), max(0, y1), min(image_width, x2), min(image_height, y2)


# Function to group bounding boxes into 4 columns
def group_boxes_into_columns(boxes, num_columns=4, threshold=50):
    """Ensure exactly 4 columns, merging or splitting as needed."""
    boxes = sorted(boxes, key=lambda b: b[0])  # Sort by X-coordinate
    columns = [[] for _ in range(num_columns)]

    # Find width ranges for 4 columns
    x_positions = sorted(set(box[0] for box in boxes))
    column_width = (max(x_positions) - min(x_positions)) // num_columns

    for box in boxes:
        x1, _, x2, _ = box
        column_idx = min(num_columns - 1, int((x1 - min(x_positions)) / column_width))
        columns[column_idx].append(box)

    return columns


# Function to fix incomplete bounding boxes in each column
def fix_column_boxes(columns, image_width):
    """Ensure each column forms a complete rectangle."""
    fixed_columns = []

    for col in columns:
        if col:
            x1 = min(box[0] for box in col)
            x2 = max(box[2] for box in col)
        else:
            x1, x2 = 0, 0  # Empty column

        fixed_columns.append([x1, x2])

    return fixed_columns


# Function to get bounding box for the entire 12×1 column format
def get_12x1_bounding_box(labels_and_boxes, image_width, image_height):
    if not labels_and_boxes:
        return None  # No leads detected

    x_min = min(box[0] for _, box in labels_and_boxes)
    y_min = max(min(box[1] for _, box in labels_and_boxes) - 50, 0)  # Expand top within bounds
    x_max = int(image_width * 0.92)  # Extend width to 85% of the image
    y_max = min(max(box[3] for _, box in labels_and_boxes) + 100, image_height)  # Expand bottom within bounds

    return (x_min, y_min, x_max, y_max)

def img_signle_extraction(crop_imgs_path, class_name, orig_height=None, orig_width=None):
    ecg_signle_dic = {}

    # Mapping for lead names
    lead_mapping = {
        "top.jpg": ['I', 'II', 'III', 'aVR', 'aVL', 'aVF'],
        "bottom.jpg": ['V1', 'V2', 'V3', 'V4', 'V5', 'V6'],
        "left.jpg": ['I', 'II', 'III', 'aVR', 'aVL', 'aVF'],
        "right.jpg": ['V1', 'V2', 'V3', 'V4', 'V5', 'V6'],
        "c_1.jpg": ['I', 'II', 'III'],
        "c_2.jpg": ['aVR', 'aVL', 'aVF'],
        "c_3.jpg": ['V1', 'V2', 'V3'],
        "c_4.jpg": ['V4', 'V5', 'V6'],
    }

    for img_path in crop_imgs_path:
        file_name = Path(img_path).name
        if file_name not in lead_mapping:
            continue  # Skip unknown files

        # Preprocess image
        filter_img = extract_black_on_white(img_path, class_name,orig_height, orig_width)
        if class_name == "3_4":
            gray = cv2.cvtColor(filter_img, cv2.COLOR_BGR2GRAY)
            _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        else:
            binary = filter_img
        cv2.imwrite(img_path, binary)

        # Extract signals
        ecg_image = ImageFilter(img_path)
        if class_name == '3_4':
            extractor = SignalExtractor(n=3)
        elif class_name == '12_1':
            extractor = SignalExtractor(n=6)
        else:
            extractor = SignalExtractor(n=7)
        signals = extractor.extract_signals(ecg_image)


        for idx, ecg_signal in enumerate(signals):

            if class_name == '3_4':
                trimmed_signal = np.array([-p.y for p in ecg_signal])[10:-10] # 30,-50
                if file_name in ['c_1.jpg', 'c_2.jpg', 'c_3.jpg', 'c_4.jpg']:
                    ecg_signle_dic[lead_mapping[file_name][idx]] = trimmed_signal
            elif class_name == '6_2':
                trimmed_signal = np.array([-p.y for p in ecg_signal])[50:-50] # 22
                if idx<6:
                    if file_name in ['left.jpg', 'right.jpg']:
                        ecg_signle_dic[lead_mapping[file_name][idx]] = trimmed_signal
            else:
                trimmed_signal = np.array([-p.y for p in ecg_signal])[170:-120] #250:-180
                if idx < 6:
                    if file_name in ['top.jpg', 'bottom.jpg']:
                        ecg_signle_dic[lead_mapping[file_name][idx]] = trimmed_signal


    return ecg_signle_dic


def convert_png_to_jpeg(input_path):
    """
    Converts a PNG image to JPEG format with .jpeg extension and overwrites the original file.
    Does nothing if the file is not a PNG.

    Parameters:
    - input_path: str - Path to the input image file.
    """
    try:
        # Check file extension
        if not input_path.lower().endswith('.png'):
            print(f"Skipped: '{input_path}' is not a PNG file.")
            return input_path

        # Open the PNG image
        with Image.open(input_path) as img:
            # Convert to RGB (JPEG does not support transparency)
            img = img.convert("RGB")

            # Save as .jpeg
            base = os.path.splitext(input_path)[0]
            jpeg_path = f"{base}.jpeg"
            img.save(jpeg_path, "JPEG")
            print(f"Converted to JPEG: {jpeg_path}")

        # Delete the original PNG file
        os.remove(input_path)
        print(f"Removed original PNG: {input_path}")

        return jpeg_path

    except Exception as e:
        print(f"Error processing '{input_path}': {e}")
    return input_path
def ensure_min_image_size(image_path: str, min_size: int = 1000) -> str:
    """
    Ensure the image has at least min_size x min_size pixels.
    If not, it will be upscaled by 2x or 3x and overwrite the original image.

    Args:
        image_path (str): Path to the image file.
        min_size (int): Minimum pixel size for width and height (default 1000).

    Returns:
        str: Path to the updated image (same as input, overwritten if resized).
    """
    with Image.open(image_path) as img:
        width, height = img.size

        if width >= min_size and height >= min_size:
            print(f"[✓] Image already meets size requirements: {width}x{height}")
            return image_path

        # Calculate the scaling factor (2x, 3x, ...)
        scale_factor = 2
        while width * scale_factor < min_size or height * scale_factor < min_size:
            scale_factor += 1

        # Resize the image
        new_size = (width * scale_factor, height * scale_factor)
        resized_img = img.resize(new_size, Image.BICUBIC)
        resized_img.save(image_path)  # Overwrite original

        print(f"[↑] Image resized to {new_size} with scale factor {scale_factor}")
        return image_path
def auto_upright_ecg(img, min_improvement=1):
    # reader = easyocr.Reader(['en'], gpu=False)
    # img = cv2.imread(image_path)
    h, w = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    inv = 255 - gray  # Invert for easier OCR

    def ocr_score(image):
        # EasyOCR expects a file or numpy array (BGR or grayscale)
        result = reader.readtext(image)
        keywords = ['bpm', 'QRS', 'ECG', 'PR', 'QT', 'ms', 'GE']
        found = 0
        for _, text, _ in result:
            if any(word.lower() in text.lower() for word in keywords):
                found += 1
        return found

    # Check if current orientation already contains ECG keywords
    top_region = inv[:h // 3, :]
    base_score = ocr_score(top_region)
    if base_score >= 2:
        return img  # Already upright

    # Try all 4 rotations (excluding 0, already checked)
    best_score = base_score
    best_img = img
    for k in range(1, 4):  # 90, 180, 270 degrees
        rotated_inv = np.rot90(inv, k)
        region = rotated_inv[:h // 3, :]
        score = ocr_score(region)
        if score > best_score + min_improvement:
            best_score = score
            best_img = np.rot90(img, k)  # Rotate original image, not inverted one

    return best_img

def image_crop_and_save(image_path, class_name, output_folder):
    img = cv2.imread(image_path)
    orig_height, orig_width = img.shape[:2]
    top_label = ''
    if img is None:
        print(f"Error reading {image_path}. Skipping...")
        pass
    if class_name in MODEL_PATHS:
        predictor = load_object_detection_model(class_name, img)
        lead_result = predictor(img)
        instances = lead_result["instances"].to("cpu")
        scores = instances.scores.tolist()
        pred_classes = instances.pred_classes.tolist()
        boxes = instances.pred_boxes.tensor
        # Dictionary to track the highest confidence detection per class
        best_detections = {}
        for i, cls in enumerate(pred_classes):
            if cls not in best_detections or scores[i] > best_detections[cls]['score']:
                best_detections[cls] = {'score': scores[i], 'box': boxes[i]}

        # Extract filtered detections
        if best_detections:
            filtered_scores = [det['score'] for det in best_detections.values()]
            filtered_boxes = torch.stack([det['box'] for det in best_detections.values()])
            filtered_classes = list(best_detections.keys())
        else:
            filtered_scores = []
            filtered_boxes = torch.empty((0, 4))
            filtered_classes = []

        # Store detected labels and boxes
        labels_and_boxes = []
        for i, (box, label) in enumerate(zip(filtered_boxes, filtered_classes)):
            x1, y1, x2, y2 = box.tolist()
            label_name = Lead_list[label] if Lead_list is not None else str(label)
            labels_and_boxes.append((label_name, (x1, y1, x2, y2)))


        # Sort labels_and_boxes based on vertical position (y1)
        labels_and_boxes.sort(key=lambda x: x[1][1])

        if labels_and_boxes:
            top_label, top_box = labels_and_boxes[0]
        else:
            print("No leads detected.")


    if class_name == '6_2':
        left_side_leads = ["I", "II", "III", "aVR", "aVL", "aVF"]
        right_side_leads = ["V1", "V2", "V3", "V4", "V5", "V6"]
        image_height, image_width = img.shape[:2]
        # Find the leftmost x-coordinate of right-side leads
        right_leads_x_min = min((box[0] for label, box in labels_and_boxes if label in right_side_leads),
                                default=image_width)

        # Get bounding boxes for left and right leads
        left_bbox = get_left_bounding_box(left_side_leads, labels_and_boxes, right_leads_x_min, image_height)
        right_bbox = get_right_bounding_box(right_side_leads, labels_and_boxes, image_width, image_height)
        try:
            # Draw bounding boxes on the image and save cropped results
            if left_bbox is not None:
                x1, y1, x2, y2 = map(int, left_bbox)  # Ensure integer values
                left_cropped = img[y1:y2, x1:x2]
                if left_cropped.size > 0:
                    cv2.imwrite(os.path.join(output_folder, f"left.jpg"), left_cropped)
                    cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), 5)
                    cv2.putText(img, "Left Leads", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 0, 0), 5)
                else:
                    print(f"No left-side leads detected for.")

            if right_bbox is not None:
                x1, y1, x2, y2 = map(int, right_bbox)  # Ensure integer values
                right_cropped = img[y1:y2, x1:x2]
                if right_cropped.size > 0:
                    cv2.imwrite(os.path.join(output_folder, f"right.jpg"), right_cropped)
                    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 5)
                    cv2.putText(img, "Right Leads", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 5)
                else:
                    print(f"No right-side leads detected for .")
        except Exception as e:
            print("Error : ", e, "Image not processs")
            pass

    elif class_name == '3_4':
        image_height, image_width = img.shape[:2]

        # Convert tensor to list of lists
        box_list = filtered_boxes.tolist()

        # Group into columns
        grouped_columns = group_boxes_into_columns(box_list, num_columns=4)
        fixed_columns = fix_column_boxes(grouped_columns, image_width)

        skipped_columns = []
        column_coords = []

        # Compute global top and bottom for alignment
        all_boxes = [box for col in grouped_columns for box in col]
        if all_boxes:
            global_top = max(0, min(box[1] for box in all_boxes) - 120) #80
            global_bottom = min(image_height, max(box[3] for box in all_boxes) + 150)
        else:
            print("No boxes detected in any column.")


        for i, col_boxes in enumerate(grouped_columns):
            x1, _ = fixed_columns[i]

            if not col_boxes:
                print(f"Skipping column {i + 1} due to no boxes.")
                skipped_columns.append(i + 1)
                continue

            col_top = global_top
            col_bottom = global_bottom

            # Extend to next column or end of image
            if i == len(fixed_columns) - 1:
                x2 = int(image_width * 0.98)
            else:
                x2 = fixed_columns[i + 1][0] - 1

            # Clamp
            x1, col_top, x2, col_bottom = clamp_bbox(int(x1), int(col_top), int(x2), int(col_bottom), image_width,
                                                     image_height)

            if x1 >= x2 or col_top >= col_bottom:
                print(f"Skipping invalid crop for column {i + 1}")
                skipped_columns.append(i + 1)
                continue

            column_coords.append((x1, x2, col_top, col_bottom))

            # Crop and save
            cropped_region = img[col_top:col_bottom, x1:x2]
            if cropped_region.size == 0:
                print(f"Skipping empty crop for column {i + 1}")
                skipped_columns.append(i + 1)
                continue

            cv2.imwrite(os.path.join(output_folder, f"c_{i + 1}.jpg"), cropped_region)
            # Draw rectangle and label
            cv2.rectangle(img, (x1, col_top), (x2, col_bottom), (255, 0, 0), 4)
            cv2.putText(img, f"Col {i + 1}", (x1 + 10, col_top + 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

        # Attempt to fix Column 1 by splitting Column 2
        if 1 in skipped_columns and 2 not in skipped_columns:
            if len(column_coords) > 0:
                x1, x2, col_top, col_bottom = column_coords[0]
                mid_x = (x1 + x2) // 2

                if x1 < mid_x < x2 and col_top < col_bottom:
                    col1_region = img[col_top:col_bottom, x1:mid_x]
                    col2_region = img[col_top:col_bottom, mid_x:x2]

                    # Save or draw
                    cv2.imwrite(os.path.join(output_folder, f"c_1.jpg"), col1_region)
                    cv2.rectangle(img, (x1, col_top), (mid_x, col_bottom), (0, 255, 0), 4)
                    cv2.putText(img, "Col 1", (x1 + 10, col_top + 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

                    cv2.imwrite(os.path.join(output_folder, f"c_2.jpg"), col2_region)
                    cv2.rectangle(img, (mid_x, col_top), (x2, col_bottom), (0, 255, 255), 4)
                    cv2.putText(img, "Col 2", (mid_x + 10, col_top + 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

                    print("Column 1 was missing, so Column 2 was split into two parts.")
                else:
                    print("ERROR: Invalid dimensions while splitting column 2.")
            else:
                print("ERROR: No valid column coordinates to split column 2.")


    elif class_name == '12_1':
        image_height, image_width = img.shape[:2]
        full_bbox = get_12x1_bounding_box(labels_and_boxes, image_width, image_height)
        if full_bbox is not None:
            x1, y1, x2, y2 = map(int, full_bbox)
            # Create label → box mapping
            label_box_dict = {label: box for label, box in labels_and_boxes}
            if top_label == 'aVL' and all(k in label_box_dict for k in ['aVL', 'III', 'V1', 'V6']):
                x1_exp = max(0, x1)
                x2_exp = min(image_width, x2 + 320)
                # Top: aVL to III
                avl_box = label_box_dict['aVL']
                iii_box = label_box_dict['III']
                y1_top = int(min(avl_box[1], iii_box[1])) - 30
                y2_top = int(max(avl_box[3], iii_box[3])) + 40
                y1_top = max(0, y1_top)
                y2_top = min(image_height, y2_top)
                top_cropped = img[y1_top:y2_top, x1_exp:x2_exp]
                cv2.imwrite(os.path.join(output_folder, f"top.jpg"), top_cropped)
                # Bottom: V1 to V6
                v1_box = label_box_dict['V1']
                v6_box = label_box_dict['V6']
                y1_bot = int(min(v1_box[1], v6_box[1])) - 30
                y2_bot = int(max(v1_box[3], v6_box[3])) + 40
                y1_bot = max(0, y1_bot)
                y2_bot = min(image_height, y2_bot)
                bottom_cropped = img[y1_bot:y2_bot, x1_exp:x2_exp]
                cv2.imwrite(os.path.join(output_folder, f"bottom.jpg"), bottom_cropped)


            else:
                # Define top and bottom leads (standard 12-lead split)
                top_leads = ['I', 'II', 'III', 'aVR', 'aVL', 'aVF']
                bottom_leads = ['V1', 'V2', 'V3', 'V4', 'V5', 'V6']

                # Check if we have at least 4 leads from both sets (to be safe)
                if sum(1 for l in top_leads if l in label_box_dict) >= 4 and \
                        sum(1 for l in bottom_leads if l in label_box_dict) >= 4:
                    image_height, image_width = img.shape[:2]
                    # Get boxes for each group
                    top_boxes = [label_box_dict[l] for l in top_leads if l in label_box_dict]
                    bottom_boxes = [label_box_dict[l] for l in bottom_leads if l in label_box_dict]
                    # Compute left X (fixed) and right X (expandable)
                    x1_exp = max(0, int(min([box[0] for box in top_boxes + bottom_boxes])))
                    x2_initial = int(max([box[2] for box in top_boxes + bottom_boxes]) + 100)
                    # Compute vertical bounds
                    top_y1 = max(0, int(min(box[1] for box in top_boxes) - 30))
                    top_y2 = int(max(box[3] for box in top_boxes) + 30)
                    bottom_y1 = int(min(box[1] for box in bottom_boxes) - 30)
                    bottom_y2 = min(image_height, int(max(box[3] for box in bottom_boxes) + 30))
                    # Adjust vertical overlap or gap
                    if bottom_y1 < top_y2:
                        # They overlap — find midpoint and split
                        mid_y = (top_y2 + bottom_y1) // 2
                        top_y2 = mid_y
                        bottom_y1 = mid_y
                    else:
                        # Soften the gap if too wide
                        gap = bottom_y1 - top_y2
                        if gap > 10:
                            shrink = gap // 2
                            top_y2 = min(top_y2 + shrink, image_height)
                            bottom_y1 = max(bottom_y1 - shrink, 0)
                    # Enforce 75% width rule (expand x2 if needed)
                    min_required_width = int(image_width * 0.75)
                    current_width = x2_initial - x1_exp
                    if current_width < min_required_width:
                        x2_exp = min(image_width, x1_exp + min_required_width)
                    else:
                        x2_exp = min(image_width, x2_initial)
                    # Crop using fixed left (x1_exp) and adjusted right (x2_exp)
                    top_crop = img[top_y1:top_y2 + 10, x1_exp:x2_exp]
                    bottom_crop = img[bottom_y1:bottom_y2 + 10, x1_exp:x2_exp]
                    # Optional: save crops
                    cv2.imwrite(os.path.join(output_folder, f"top.jpg"), top_crop)
                    cv2.imwrite(os.path.join(output_folder, f"bottom.jpg"), bottom_crop)


    croped_img_get = glob.glob(os.path.join(output_folder, "*"))
    ecg_raw_signals = {}
    if croped_img_get:
        ecg_raw_signals = img_signle_extraction(croped_img_get, class_name, orig_height, orig_width)
    return ecg_raw_signals,top_label



def signal_extraction_and_arrhy_detection(image_path):

    crop_img = base_dir_path+ '/cropped_lead_images'
    img_id = "temp_11"
    mk_img_path = os.path.join(crop_img, img_id)

    for i in glob.glob(f'{crop_img}/temp_path/*.jpg'):
        os.remove(i)
    try:
        if os.path.exists(mk_img_path):
            force_remove_folder(mk_img_path)  # Use the force remove function
        else:
            print(f"Folder does not exist: {mk_img_path}")
    except Exception as e:
        print(f"Error removing folder: {e}")

    try:
        if not os.path.exists(mk_img_path):
            os.mkdir(mk_img_path)
            output_folder = mk_img_path
        else:
            output_folder = mk_img_path
    except Exception as r:
        print("Mack crop img folder error :", r)
        output_folder = crop_img
    file_name = os.path.splitext(os.path.basename(image_path))[0]
    save_img = os.path.join(settings.MEDIA_ROOT,'analysis_tool','uploads')
    csv_output = os.path.join(settings.BASE_DIR, "analysis_tool", "analysis_result",f"{file_name}.csv")
    output_data, class_name = predict_grid_type(image_path)
    ecg_raw_signals = image_crop_and_save(image_path, class_name, output_folder)
    image_path =convert_png_to_jpeg(image_path)
    image_path = ensure_min_image_size(image_path)
    
    # class_name = '6_2'
    if class_name == '6_2':
        corrected_image = orientation_image(image_path)
        temp_path = os.path.join(f'{crop_img}\\{file_name}.jpg')
        cv2.imwrite(temp_path,corrected_image)
        ecg_raw_signals,top_label = image_crop_and_save(temp_path, class_name, output_folder)
    else:
        ecg_raw_signals,top_label = image_crop_and_save(image_path, class_name, output_folder)
    not_use_lead = []
    results = {"avg_hr": 0,
               "arr_final_result": 'Normal',
               "mi_final_result": 'Abnormal',
               "beats": 0,
               "detections": [],
               "RRInterval": 0,
               "PRInterval": 0,
               "QTInterval": 0,
               "QRSComplex": 0,
               "STseg": 0,
               "PRseg": 0,
               "QTc": 0,
               "pvcQrs": [],
               "pacQrs": [],
               "Vbeat": 0,
               "Abeat": 0,
               "arr_analysis_leads": [],
               "arr_not_analysis_leads": [],
               "color_dict": {}
               }

    if ecg_raw_signals:
        noise_result = check_noise(ecg_raw_signals, class_name, 200)
        if noise_result == 'ARTIFACTS':
            results['detections'] = [{"detect": 'ARTIFACTS', "detectType": "Arrhythmia", "confidence": 100}]

        if ecg_raw_signals and len(ecg_raw_signals) > 5 and noise_result == 'Normal':
            lead_sequence = ["I", "II", "III","aVF", "aVL", "aVR", "V1", "V2", "V3", "V4", "V5", "V6"]
            analysis_leads = ecg_raw_signals.keys()
            results['arr_analysis_leads'] = analysis_leads
            results['arr_not_analysis_leads'] = list(filter(lambda x: x not in ecg_raw_signals, lead_sequence))
            ordered_ecg_data = {lead: pd.Series(ecg_raw_signals.get(lead, [])) for lead in analysis_leads}
            ecg_df = pd.DataFrame(ordered_ecg_data)#.fillna(0)
            ecg_df = ecg_df.apply(lambda col: col.fillna(col.median()), axis=0)
            ecg_df.to_csv(f"{save_img}\\{file_name}.csv", index=False)
            arrhythmia_detector = arrhythmia_detection(ecg_df, fs=200, img_type=class_name, image_path=image_path)
            results = arrhythmia_detector.ecg_signal_processing()

            # --- MI/Arrhythmia base64 image logic ---
            save_dir = base_dir_path+ '/cropped_lead_images'
            os.makedirs(save_dir, exist_ok=True)

            # Remove old images
            for old_img_path in glob.glob(os.path.join(save_dir, "*.jpg")):
                try:
                    os.remove(old_img_path)
                except Exception as e:
                    print(f"Error removing file {old_img_path}: {e}")
            # Mapping from detection type/label to leads
            detection_leads_map = {
                "Arrhythmia": ['I', 'II', 'III'],
                "Lateral_MI": ['I', 'aVL', 'V5', 'V6'],
                "Inferior_MI": ['II', 'III', 'aVF', 'aVL'],
                "LBBB": ['I', 'aVL', 'V1', 'V5', 'V6'],
                "RBBB": ['I', 'aVL', 'V1', 'V2', 'V3', 'V5', 'V6'],
                "LAD": ['I', 'II', 'aVF'],
                "RAD": ['I', 'II', 'aVF'],
                "EAD": ['I', 'II', 'aVF'],
                "LVH": ['I', 'III', 'aVL', 'aVF', 'aVR', 'V4', 'V5', 'V6'],
                "RVH": ['II', 'III', 'aVF', 'V1', 'V2', 'V3', 'V4'],
                "LAFB": ['I', 'II', 'III', 'aVL', 'aVF'],
                "LPFB": ['I', 'II', 'III', 'aVL', 'aVF'],
                "T_wave_Abnormality": ['I', 'II', 'III', 'aVF', 'V5'],
            }

            def get_detection_key(detectType, detect):
                detect = detect.lower()
                if detectType == "Arrhythmia":
                    return "Arrhythmia"
                if detectType == "MI":
                    if "lateral" in detect:
                        return "Lateral_MI"
                    if "inferior" in detect:
                        return "Inferior_MI"
                    if "lbbb" in detect:
                        return "LBBB"
                    if "rbbb" in detect:
                        return "RBBB"
                    if "lafb" in detect:
                        return "LAFB"
                    if "lpfb" in detect:
                        return "LPFB"
                    if "t_wave_abnormality" in detect or "t wave abnormality" in detect:
                        return "T_wave_Abnormality"
                if detectType == "axisDeviation":
                    if "left_axis_deviation" in detect:
                        return "LAD"
                    if "right_axis_deviation" in detect:
                        return "RAD"
                    if "extreme_axis_deviation" in detect:
                        return "EAD"
                if detectType == "Hypertrophy":
                    if "lvh" in detect:
                        return "LVH"
                    if "rvh" in detect:
                        return "RVH"
                return None

            for detection in results.get("detections", []):
                key = get_detection_key(detection.get("detectType", ""), detection.get("detect", ""))
                if not key:
                    continue
                leads = detection_leads_map.get(key, [])
                detection["leads"] = []
                for lead in leads:
                    if lead in ecg_df.columns:
                        plt.figure(figsize=(6, 2))
                        plt.plot(ecg_df[lead].values, color='black')
                        plt.axis('off')
                        img_path = os.path.join(save_dir, f"{file_name}_{lead}_signal.jpg")
                        plt.savefig(img_path, bbox_inches='tight', pad_inches=0)
                        plt.close()
                        with open(img_path, "rb") as image_file:
                            encoded_string = base64.b64encode(image_file.read()).decode('utf-8')
                        detection["leads"].append({
                            "lead": lead,
                            "Image": encoded_string
                        })
            process_and_plot_leads(ecg_df, file_name, results,top_label, class_name=class_name, mm_per_sec=25, mm_per_mV=10, signal_scale=0.01)


    else:
        not_use_lead = ["I", "II", "III", "V1", "V2", "V3", "V4", "V5", "V6", "aVF", "aVL", "aVR"]
        results['arr_analysis_leads'] = []
        results['arr_not_analysis_leads'] = not_use_lead
    return results