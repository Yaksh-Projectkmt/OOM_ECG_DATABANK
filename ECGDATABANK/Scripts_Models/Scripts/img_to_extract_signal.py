import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from math import ceil
from itertools import groupby
from operator import itemgetter
import csv
from django.conf import settings
from PIL import Image

# -- EM Thresholding --
def expectation_maximization(image, max_iter=1, tol=1e-3):
    pixel_values = image.flatten().astype(np.float64)
    mean_0 = np.mean(pixel_values) - np.std(pixel_values)
    mean_1 = np.mean(pixel_values) + np.std(pixel_values)
    var_0 = var_1 = np.var(pixel_values) / 2
    weight_0 = weight_1 = 0.5
    probabilities = np.zeros((len(pixel_values), 2))

    for _ in range(max_iter):
        gaussian_0 = (1 / np.sqrt(2 * np.pi * var_0)) * np.exp(- (pixel_values - mean_0)**2 / (2 * var_0))
        gaussian_1 = (1 / np.sqrt(2 * np.pi * var_1)) * np.exp(- (pixel_values - mean_1)**2 / (2 * var_1))
        weighted_gaussian_0 = weight_0 * gaussian_0
        weighted_gaussian_1 = weight_1 * gaussian_1
        total = weighted_gaussian_0 + weighted_gaussian_1
        probabilities[:, 0] = weighted_gaussian_0 / total
        probabilities[:, 1] = weighted_gaussian_1 / total
        weight_0 = np.mean(probabilities[:, 0])
        weight_1 = np.mean(probabilities[:, 1])
        mean_0_new = np.sum(probabilities[:, 0] * pixel_values) / np.sum(probabilities[:, 0])
        mean_1_new = np.sum(probabilities[:, 1] * pixel_values) / np.sum(probabilities[:, 1])
        var_0 = np.sum(probabilities[:, 0] * (pixel_values - mean_0_new) ** 2) / np.sum(probabilities[:, 0])
        var_1 = np.sum(probabilities[:, 1] * (pixel_values - mean_1_new) ** 2) / np.sum(probabilities[:, 1])
        if np.abs(mean_0_new - mean_0) < tol and np.abs(mean_1_new - mean_1) < tol:
            break
        mean_0, mean_1 = mean_0_new, mean_1_new

    return (mean_0 + mean_1) / 2

def binarize_image(image_path, output_folder):
    original_img = cv2.imread(image_path)
    if original_img is None:
        return None

    gray_img = cv2.cvtColor(original_img, cv2.COLOR_BGR2GRAY)
    threshold = expectation_maximization(gray_img)
    threshold = int(np.clip(threshold, 100, 190))

    lower_black = np.array([0, 0, 0], dtype=np.uint8)
    upper_black = np.array([threshold, threshold, threshold], dtype=np.uint8)
    black_mask = cv2.inRange(original_img, lower_black, upper_black)
    white_canvas = np.ones_like(original_img, dtype=np.uint8) * 255
    result_on_white = np.where(black_mask[:, :, None] == 255, original_img, white_canvas)
    result_gray = cv2.cvtColor(result_on_white, cv2.COLOR_BGR2GRAY)
    _, binary_output = cv2.threshold(result_gray, 1, 255, cv2.THRESH_OTSU)
    binary_output = cv2.bitwise_not(binary_output)

    os.makedirs(output_folder, exist_ok=True)
    image_name = os.path.basename(image_path)
    output_path = os.path.join(output_folder, image_name)
    cv2.imwrite(output_path, binary_output)
    return output_path

# -- Digitization Classes --
class DigitizationError(Exception):
    pass

class Point:
    def __init__(self, x, y):
        self.x = x
        self.y = y

class ImageWrapper:
    def __init__(self, file_path):
        self.image = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
        _, self.image = cv2.threshold(self.image, 128, 255, cv2.THRESH_BINARY_INV)
        self.height, self.width = self.image.shape

    def __getitem__(self, key):
        return self.image[key]

class SignalExtractor:
    def __init__(self, n: int) -> None:
        self.__n = n

    def extract_signals(self, ecg: ImageWrapper):
        N = ecg.width
        LEN, SCORE = (2, 3)
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
                        ctr = ceil(mean(pc))
                        if node not in cache:
                            cache[node] = [[ctr, None, 1, 0]] * self.__n
                        ps = cache[node][roi_i][SCORE]
                        d = abs(ctr - rois[roi_i])
                        g = self.__gap(pc, c)
                        costs[pc] = ps + d + N / 10 * g
                    best = min(costs, key=costs.get)
                    y = ceil(mean(best))
                    p = (col - 1, best)
                    l = cache[p][roi_i][LEN] + 1
                    s = costs[best]
                    cache[col, c][roi_i] = (y, p, l, s)

        return self.__backtracking(cache, rois)

    def __get_roi(self, ecg: ImageWrapper):
        WINDOW = 10
        SHIFT = (WINDOW - 1) // 2
        stds = np.zeros(ecg.height)
        for i in range(ecg.height - WINDOW + 1):
            std = ecg[i:i + WINDOW, :].reshape(-1).std()
            stds[i + SHIFT] = std
        min_distance = int(ecg.height * 0.1)
        peaks, _ = find_peaks(stds, distance=min_distance)
        rois = sorted(peaks, key=lambda x: stds[x], reverse=True)
        if len(rois) < self.__n:
            raise DigitizationError("The indicated number of ROIs could not be detected.")
        return sorted(rois[:self.__n])

    def __get_clusters(self, ecg: ImageWrapper, col: int):
        BLACK = 0
        clusters = []
        black_p = np.where(ecg[:, col] == BLACK)[0]
        for _, g in groupby(enumerate(black_p), lambda idx_val: idx_val[0] - idx_val[1]):
            clusters.append(tuple(map(itemgetter(1), g)))
        return clusters

    def __gap(self, pc, c):
        pc_min, pc_max = pc[0], pc[-1]
        c_min, c_max = c[0], c[-1]
        if pc_min <= c_min and pc_max <= c_max:
            return len(range(pc_max + 1, c_min))
        elif pc_min >= c_min and pc_max >= c_max:
            return len(range(c_max + 1, pc_min))
        return 0

    def __backtracking(self, cache, rois):
        X_COORD, CLUSTER = (0, 1)
        Y_COORD, PREV, LEN = (0, 1, 2)
        raw_signals = [None] * self.__n
        for roi_i in range(self.__n):
            roi = rois[roi_i]
            max_len = max([v[roi_i][LEN] for v in cache.values()])
            cand_nodes = [node for node, stats in cache.items() if stats[roi_i][LEN] == max_len]
            best = min(cand_nodes, key=lambda node: abs(ceil(np.mean(node[CLUSTER])) - roi))
            raw_s = []
            while best is not None:
                y = cache[best][roi_i][Y_COORD]
                raw_s.append(Point(best[X_COORD], y))
                best = cache[best][roi_i][PREV]
            raw_s.reverse()
            raw_signals[roi_i] = raw_s
        return raw_signals

def save_signals_to_csv(signals, filename, height):
    with open(filename, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["index", "II"])
        for point in signals[0]:
            writer.writerow([point.x, height - point.y])
 
# -- Django-Compatible Image Processor --
def process_images(file_path):
    try:
        binarized_folder = os.path.join(settings.MEDIA_ROOT, "morphology_drow", 'binarized')
        csv_folder = os.path.join(settings.MEDIA_ROOT, "morphology_drow", 'csv_files')

        image_name = os.path.basename(file_path)
        bin_path = binarize_image(file_path, binarized_folder)
        if not bin_path:
            return f"Could not process: {file_path}"

        extractor = SignalExtractor(n=1)
        ecg_image = ImageWrapper(bin_path)
        signals = extractor.extract_signals(ecg_image)

        os.makedirs(csv_folder, exist_ok=True)
        csv_filename = os.path.splitext(image_name)[0] + ".csv"
        csv_path = os.path.join(csv_folder, csv_filename)
        save_signals_to_csv(signals, csv_path, ecg_image.height)

        os.remove(bin_path)
        return f"Processing completed: CSV saved at: {csv_path}"

    except DigitizationError as e:
        return f"Digitization error: {str(e)}"
    except Exception as e:
        return f"Error processing image: {str(e)}"

