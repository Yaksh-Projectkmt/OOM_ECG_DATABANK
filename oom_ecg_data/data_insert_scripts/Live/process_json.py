import pandas as pd
import numpy as np
import datetime
import pymongo
from bson import ObjectId
from collections import Counter
import warnings
import time
from datetime import datetime, timezone
from concurrent.futures import ThreadPoolExecutor
warnings.filterwarnings("ignore")
class MongodbJobProcessor:
    def __init__(self):
      # remote ECG DB
      self.db_remote = pymongo.MongoClient(
          "mongodb://readonly_user:9ikJ4Qn1YmG1l1EVF1OQ@192.168.2.131:27017/?authSource=admin"
      )["ecgs"]

      # local processed DB
      self.db_local = pymongo.MongoClient("mongodb://localhost:27017/")["ecgarrhythmias"]

      # job queue DB
      self.job_db = pymongo.MongoClient("mongodb://localhost:27017/")["test"]["Test3"]

      self.patient_db = pymongo.MongoClient("mongodb://localhost:27017/")["Patients"]
      self.lead_2 = self.lead_7 = self.lead_12 = False

    # ------------------ Hex Decode ------------------
    @staticmethod
    def _decode_hex_series(series):
      raw = bytearray().join(bytes.fromhex("".join(x)) for x in series)
      val = np.frombuffer(raw, dtype="<i2")  # 16-bit signed little endian
      return val * (4.6 / 4095) / 4

    # ------------------ Process Leads ------------------
    def _process_leads(self, df):
      all_rows = []
      for _, row in df.iterrows():
          vII = self._decode_hex_series([row["data"]])
          vI = v5 = v1 = v2 = v3 = v4 = v6 = None

          # Common extra leads (7/12)
          if self.lead_7 or self.lead_12:
              if "data1" in row:
                  vI = self._decode_hex_series([row["data1"]])
              if "data5" in row:
                  v5 = self._decode_hex_series([row["data5"]])

          # Extra chest leads (12 only)
          if self.lead_12:
              if "vOne" in row:
                  v1 = self._decode_hex_series([row["vOne"]])
              if "vTwo" in row:
                  v2 = self._decode_hex_series([row["vTwo"]])
              if "vThree" in row:
                  v3 = self._decode_hex_series([row["vThree"]])
              if "vFour" in row:
                  v4 = self._decode_hex_series([row["vFour"]])
              if "vSix" in row:
                  v6 = self._decode_hex_series([row["vSix"]])

          # Generate timestamps
          n = len(vII)
          timestamps = pd.date_range(start=row["dateTime"], periods=n, freq="4ms")  # 250Hz

          # ---- Lead 12 ----
          if self.lead_12:
              if all(x is not None for x in [vI, vII, v1, v2, v3, v4, v5, v6]):
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
              else:
                  df_row = pd.DataFrame({"DateTime": timestamps, "II": vII})

          # ---- Lead 7 ----
          elif self.lead_7:
              if vI is not None and v5 is not None:
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

          # ---- Lead 2 ----
          else:
              df_row = pd.DataFrame({"DateTime": timestamps, "II": vII})

          all_rows.append(df_row)

      return pd.concat(all_rows).reset_index(drop=True)

    # ------------------ Insert Processed ------------------
    def insert_processed_data_to_db(self, patient_id, arrhythmia_name, df, version, chunk_size=2000):
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

        # -------------------- Prepare leads --------------------
        leads = {col: df[col].tolist() for col in df.columns if col != "DateTime"}
        if not leads:
            print(f"No lead data found for {arrhythmia_name}")
            return

        # -------------------- Determine lead count --------------------
        if version == 5:
            lead_value = 7
        elif version == 8:
            lead_value = 12
        elif version == 2:
            lead_value = 2
        else:
            lead_value = version  # fallback

        frequency = 250 if lead_value == 12 else 200

        # -------------------- Insert --------------------
        total_length = len(next(iter(leads.values())))  # length of first lead
        num_chunks = (total_length // chunk_size) + (1 if total_length % chunk_size else 0)

        if total_length <= chunk_size:
            record = {
                "PatientID": patient_id,
                "Arrhythmia": arrhythmia_name,
                "Lead": lead_value,
                "Frequency": frequency,
                "datalength": total_length,
                "server":"Live",
                "created_at": datetime.now(timezone.utc),
                "Data": leads
            }
            collection.insert_one(record)
        else:
            for i in range(0, total_length, chunk_size):
                chunked_leads = {lead: values[i:i+chunk_size] for lead, values in leads.items()}
                datalength = len(next(iter(chunked_leads.values())))
                record = {
                    "PatientID": patient_id,
                    "Arrhythmia": arrhythmia_name,
                    "Lead": lead_value,
                    "Frequency": frequency,
                   "datalength": datalength,
                    "server":"Live",
                    "created_at": datetime.now(timezone.utc),
                    "Data": chunked_leads
                }
                collection.insert_one(record)

        # -------------------- Update Patients DB --------------------
        patient_collection = self.patient_db[parent]

        total_time_minutes = (total_length / frequency) / 60

        patient_collection.update_one(
            {"PatientID": patient_id},
            {
                "$set": {"PatientID": patient_id},
                "$inc": {
                    "total_records": num_chunks,
                    "total_time": total_time_minutes
                }
            },
            upsert=True
        )

        # -------------------- Remove pending job --------------------
        job = self.job_db.find_one({"_status": "pending"})
        if job:
            self.job_db.delete_one({"_id": job["_id"]})
            
    def process_job(self, job):
        patient_oid_str = job["patient"]
        arrhythmia = job["Arrhythmia"]
        version = job["version"]

        def handle_error(msg):
            """Helper to log and remove failed jobs from queue."""
            self.job_db.delete_one({"_id": job["_id"]})
            return

        # === Normalize timestamps ===
        sdt, edt = job["starttime"], job["endtime"]
        if isinstance(sdt, str):
            sdt = datetime.fromisoformat(sdt)
        if isinstance(edt, str):
            edt = datetime.fromisoformat(edt)
        if sdt.tzinfo is None:
            sdt = sdt.replace(tzinfo=timezone.utc)
        if edt.tzinfo is None:
            edt = edt.replace(tzinfo=timezone.utc)

        # === Lookup patient in patients collection ===
        patients = self.db_remote["patients"]
        try:
            patient_doc = patients.find_one({"_id": ObjectId(patient_oid_str)})
        except Exception as e:
            return handle_error(f"Invalid patient ObjectId: {e}")

        if not patient_doc:
            return handle_error("Patient not found")

        real_patient_id = patient_doc.get("patientId", patient_oid_str)
        ecgcoll = self.db_remote.get_collection(f"{patient_oid_str}_ecgs")

        # === Query ECG data ===
        seg = list(ecgcoll.find({"dateTime": {"$gte": sdt, "$lte": edt}}))
        if not seg:
            return handle_error("No data found in given time range")

        seg_df = pd.DataFrame(seg).sort_values("dateTime").reset_index(drop=True)

        # === Force lead config based on version ===
        self.lead_12 = (version == 8)
        self.lead_7 = (version == 5)
        self.lead_2 = (version == 2)

        try:
            ecg_df = self._process_leads(seg_df)
            self.insert_processed_data_to_db(real_patient_id, arrhythmia, ecg_df, version)
        except Exception as e:
            return handle_error(f"Processing error: {e}")

        # === Mark job as done ===
        self.job_db.update_one(
            {"_id": job["_id"]},
            {"$set": {"_status": "done"}}
        )
    def run(self, workers=4, batch_size=10, work_duration=300, rest_duration=600, check_interval=10):
        """
        Run job processor in cycles:
        - Work for `work_duration` seconds (default 5 min)
        - Then rest for `rest_duration` seconds (default 10 min)
        - If no pending jobs, sleep until at least one appears
        """
        while True:
            # Wait until at least one pending job appears
            while self.job_db.count_documents({"_status": "pending"}) == 0:
                time.sleep(check_interval)

            # Process for limited duration
            start_time = time.time()
            with ThreadPoolExecutor(max_workers=workers) as executor:
                while (time.time() - start_time) < work_duration:
                    pending_jobs = list(self.job_db.find({"_status": "pending"}).limit(batch_size))
                    if pending_jobs:
                        for job in pending_jobs:
                            executor.submit(self.process_job, job)
                    else:
                        # No jobs during active cycle — small wait before checking again
                        time.sleep(check_interval)

            # Rest before next cycle
            time.sleep(rest_duration)  # wait before checking again

if __name__ == "__main__":
    processor = MongodbJobProcessor()
    processor.run()
