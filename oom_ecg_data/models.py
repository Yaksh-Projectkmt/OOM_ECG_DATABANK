# import tensorflow as tf
# import numpy as np
# import json
# import glob
# import os

# LABEL_MAP = {'p': 1, 'q': 2, 'r': 3, 's': 4, 't': 5}

# def retrain_model_from_corrections():
#     model = tf.keras.models.load_model("models/pqrst_model.h5")

#     X_train = []
#     y_train = []

#     for file in glob.glob("corrected/corrected_*.json"):
#         with open(file, "r") as f:
#             data = json.load(f)
#             peaks_by_lead = data['corrected_peaks']
#             for lead_name, peaks in peaks_by_lead.items():
#                 signal_len = 1000  # ✅ Replace with real length if known
#                 ecg_signal = np.zeros(signal_len)  # TODO: Load real ECG
#                 label = np.zeros(signal_len, dtype=int)

#                 for peak_type, indices in peaks.items():
#                     for i in indices:
#                         if 0 <= i < signal_len:
#                             label[i] = LABEL_MAP[peak_type]

#                 X_train.append(ecg_signal)
#                 y_train.append(label)

#     X_train = np.array(X_train).reshape(-1, signal_len, 1)
#     y_train = np.array(y_train)
#     y_train = tf.keras.utils.to_categorical(y_train, num_classes=6)

#     model.fit(X_train, y_train, epochs=3, batch_size=4)
#     model.save("models/pqrst_model.h5")
