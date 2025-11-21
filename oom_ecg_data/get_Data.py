import pandas as pd
import numpy as np
import datetime
import pymongo
from bson import ObjectId
from collections import Counter
import warnings
import time
from pymongo import MongoClient
warnings.filterwarnings("ignore")


class Mongodb_Get_Data_Fast:
    def __init__(self, patientID, start_date, end_date,mongo_uri="mongodb://192.168.1.65:27017/"):
        self.patientID = patientID
        self.start_date = start_date
        self.end_date = end_date
        self.db_local = None
        self.lead_2 = self.lead_7 = self.lead_12 = False
        self.client = MongoClient(mongo_uri)  # <-- add this
        self.db_local = self.client["Patients"]
    # ------------------ MongoDB Connection ------------------
    def _connect(self, source=2):
        if source == 1:
            url, dbname = (
                "mongodb://admin:KmtOom2023@191.169.1.6:27017/ecgs1?authSource=admin",
                "oom-ecg"
            )
        elif source == 2:
            url, dbname = (
                "mongodb://readonly_user:9ikJ4Qn1YmG1l1EVF1OQ@192.168.2.131:27017/?authSource=admin",
                "ecgs"
            )
        else:
            url, dbname = (
                "mongodb://admin:HopsSlk2023@191.169.1.10:27017/ecgs1?authSource=admin",
                "oom-ecg-erp"
            )

        client = pymongo.MongoClient(url)
        # remote target DB (change if needed)
        self.db_local = pymongo.MongoClient("mongodb://192.168.1.65:27017/")["ecgarrhythmias"]
        return client[dbname]

    # ------------------ Vectorized Hex Decode ------------------
    @staticmethod
    def _decode_hex_series(series):
        raw = bytearray().join(bytes.fromhex("".join(x)) for x in series)
        val = np.frombuffer(raw, dtype="<i2")  # 16-bit signed little endian
        return val * (4.6 / 4095) / 4

    # ------------------ ECG Lead Processing ------------------
    def _process_leads(self, df):
        all_rows = []
        for _, row in df.iterrows():
            vII = self._decode_hex_series([row["data"]])
            vI = v5 = v1 = v2 = v3 = v4 = v6 = None

            if self.lead_7 or self.lead_12:
                if "data1" in row: vI = self._decode_hex_series([row["data1"]])
                if "data5" in row: v5 = self._decode_hex_series([row["data5"]])

            if self.lead_12:
                if "vOne" in row: v1 = self._decode_hex_series([row["vOne"]])
                if "vTwo" in row: v2 = self._decode_hex_series([row["vTwo"]])
                if "vThree" in row: v3 = self._decode_hex_series([row["vThree"]])
                if "vFour" in row: v4 = self._decode_hex_series([row["vFour"]])
                if "vSix" in row: v6 = self._decode_hex_series([row["vSix"]])

            n = len(vII)
            timestamps = pd.date_range(start=row["dateTime"], periods=n, freq="4ms")  # 250Hz

            if self.lead_12 and all(x is not None for x in [vI, vII, v1, v2, v3, v4, v5, v6]):
                vIII = vII - vI
                aVR = -(vI + vII) / 2
                aVL = (vI - vIII) / 2
                aVF = (vII - vIII) / 2
                df_row = pd.DataFrame({
                    "DateTime": timestamps,
                    "I": vI, "II": vII, "III": vIII,
                    "aVR": aVR, "aVL": aVL, "aVF": aVF,
                    "v1": v1, "v2": v2, "v3": v3,
                    "v4": v4, "v5": v5, "v6": v6
                })
            elif self.lead_7 and all(x is not None for x in [vI, vII, v5]):
                vIII = vII - vI
                aVR = -(vI + vII) / 2
                aVL = (vI - vIII) / 2
                aVF = (vII - vIII) / 2
                df_row = pd.DataFrame({
                    "DateTime": timestamps,
                    "I": vI, "II": vII, "III": vIII,
                    "aVR": aVR, "aVL": aVL, "aVF": aVF,
                    "v5": v5
                })
            else:
                df_row = pd.DataFrame({"DateTime": timestamps, "II": vII})

            all_rows.append(df_row)

        return pd.concat(all_rows).reset_index(drop=True)

    # ------------------ Insert into Remote DB ------------------
    def insert_processed_data_to_db(self, arrhythmia_name, df,source=2):
        # -------------------- Mapping --------------------
        mapping = {
            'Myocardial Infarction': ['T-wave abnormalities', 'Inferior MI', 'Lateral MI'],
            'Atrial Fibrillation & Atrial Flutter': ['AFIB', 'Aflutter', 'AFL'],
            'HeartBlock': ['I DEGREE', 'MOBITZ-I', 'MOBITZ-II', 'III Degree', 'III_Degree'],
            'Junctional Rhythm': ['Junctional Bradycardia', 'Junctional Rhythm', 'BR', 'JN-BR', 'JN-RHY'],
            'Premature Atrial Contraction': ['PAC-Isolated', 'PAC-Bigeminy', 'PAC-Couplet', 'PAC-Triplet',
                                            'SVT', 'PAC-Trigeminy', 'PAC-Quadrigeminy'],
            'Premature Ventricular Contraction': ['AIVR', 'PVC-Bigeminy', 'PVC-Couplet', 'PVC-Isolated',
                                                'PVC-Quadrigeminy', 'NSVT', 'PVC-Trigeminy',
                                                'PVC-Triplet', 'IVR', 'VT'],
            'Ventricular Fibrillation and Asystole': ['VFIB', 'VFL', 'ASYSTOLE'],
            'Noise': ['Noise'], 'Others': ['Others'],
            'LBBB': ['LBBB', 'RBBB'],
            'Abnormal': ['ABNORMAL'], 'Artifacts': ['Artifacts'],
            'Normal': ['Normal'], 'SINUS-ARR': ['SINUS-ARR'],
            'ShortPause': ['Short Pause', 'Long Pause'],
            'TC': ['TC'], 'WIDE-QRS': ['WIDE-QRS'],
        }

        # -------------------- Resolve parent arrhythmia --------------------
        parent = next(
            (k for k, v in mapping.items() if arrhythmia_name.strip() in v or arrhythmia_name.strip() == k),
            arrhythmia_name
        )
        collection = self.db_local[parent]

        # -------------------- Drop DateTime --------------------
        df = df.drop(columns=["DateTime"], errors="ignore")
        if df.empty:
            print(f"No data to insert for {arrhythmia_name}")
            return

        # -------------------- Determine lead count --------------------
        lead_count = df.shape[1]
        lead_value = {1: 2, 5: 7, 8: 12}.get(lead_count, lead_count)
        freq = 250 if lead_value == 12 else 200

        # -------------------- Chunking --------------------
        chunk_size = 2000
        total_rows = len(df)
        num_chunks = (total_rows // chunk_size) + (1 if total_rows % chunk_size else 0)

        # -------------------- Insert in chunks --------------------
        for i in range(num_chunks):
            start = i * chunk_size
            end = min((i + 1) * chunk_size, total_rows)
            df_chunk = df.iloc[start:end]

            leads = {col: df_chunk[col].tolist() for col in df_chunk.columns}
            datalength = len(next(iter(leads.values()))) if leads else 0
            if source == 2:
                server = "Live"
            elif source == 1:
                server = "Test"  
            else:
                server = "Dev"
            
            record = {
                "PatientID": self.patientID,
                "Arrhythmia": arrhythmia_name,
                "Lead": lead_value,
                "Frequency": freq,
                "datalength": datalength,
                "server": server,
                "created_at": datetime.datetime.now(datetime.timezone.utc),
                "Data": leads
            }
            collection.insert_one(record)

        print(f"Inserted {arrhythmia_name} ({lead_value} leads) in {num_chunks} chunks")

        # -------------------- Update patient_db --------------------
        patient_db = self.client["Patients"]
        patient_collection = patient_db[parent]

        total_time_minutes = (total_rows / freq) / 60

        patient_collection.update_one(
            {"PatientID": self.patientID},
            {
                "$set": {"PatientID": self.patientID},
                "$inc": {"total_records": num_chunks, "total_time": total_time_minutes}
            },
            upsert=True
        )
        print(f"  Updated patient_db.{parent} for {self.patientID}")

    # ------------------ Main get_data ------------------
    def get_data(self, source=2, arrythmia=None, mi=None, lead_2=False, lead_7=True, lead_12=False):
        db = self._connect(source)
        self.lead_2, self.lead_7, self.lead_12 = lead_2, lead_7, lead_12
        patients = db["patients"]
        ecgarr = db["ecgarrhythmias"]

        patient = patients.find_one({"patientId": self.patientID})
        if not patient:
            print("Patient not found.")
            return None
        pid = str(patient["_id"])
        ecgcoll = db[f"{pid}_ecgs"]

        arr_filter = {
            "patient": ObjectId(pid),
            "starttime": {"$gte": self.start_date, "$lte": self.end_date},
            "endtime": {"$gte": self.start_date, "$lte": self.end_date},
        }
        arr_list = list(ecgarr.find(arr_filter))

        if not arr_list and not mi:
            all_data = list(ecgcoll.find({"dateTime": {"$gte": self.start_date, "$lte": self.end_date}}, projection={"_id": 0}))
            if not all_data:
                print(" No ECG found.")
                return None
            df = pd.DataFrame(all_data).sort_values("dateTime").reset_index(drop=True)
            return self._process_leads(df)

        for doc in arr_list:
            st, et = doc["starttime"], doc["endtime"]
            seg = list(ecgcoll.find({"dateTime": {"$gte": st, "$lte": et}}, projection={"_id": 0}))
            if not seg: continue
            seg_df = pd.DataFrame(seg).sort_values("dateTime").reset_index(drop=True)
            ecg_df = self._process_leads(seg_df)
            self.insert_processed_data_to_db(doc["Arrhythmia"], ecg_df)

        return None


# ------------------ Public Entry ------------------
def get_the_data(patientID, starttime=None, endtime=None,
                 arrythmia=None, mi=None, source=2, datetime_format=24):
    if not starttime: starttime = "01-01-2000 00:00:00"
    if not endtime: endtime = "01-01-2100 23:59:59"
    fmt = "%b %d, %Y %I:%M:%S %p.%f" if datetime_format == 12 else "%d-%m-%Y %H:%M:%S.%f"
    if "." not in starttime: starttime += ".000"
    if "." not in endtime: endtime += ".999"
    sdt = datetime.datetime.strptime(starttime.strip(), fmt) - datetime.timedelta(hours=5, minutes=30)
    edt = datetime.datetime.strptime(endtime.strip(), fmt) - datetime.timedelta(hours=5, minutes=30)

    print(f"\n[START] PatientID: {patientID}")
    start_timer = time.time()

    mongo = Mongodb_Get_Data_Fast(patientID, sdt, edt)
    for config in [{"lead_12": True}, {"lead_7": True}, {"lead_2": True}]:
        try:
            df = mongo.get_data(source=source, arrythmia=arrythmia, mi=mi, **config)
            if df is not None:
                elapsed = time.time() - start_timer
                print(f"[DONE] PatientID: {patientID} | Time: {elapsed:.2f} sec")
                return df
        except Exception as e:
            continue

    elapsed = time.time() - start_timer
    print(f"[NO DATA] PatientID: {patientID} | Time: {elapsed:.2f} sec")
    return None
