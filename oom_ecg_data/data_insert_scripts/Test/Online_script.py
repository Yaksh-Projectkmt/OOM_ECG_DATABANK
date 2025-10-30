#------------------------
#Online Testing script 
#------------------------
import ssl
import json
import random
import threading
import pymongo
import pandas as pd
import numpy as np
import warnings
import time
import traceback
from datetime import datetime
from bson import ObjectId
from collections import Counter
from paho.mqtt import client as mqtt_client
print("online Test script run")
warnings.filterwarnings("ignore")
# --- CONFIG ---
MONGO_URI = "mongodb://localhost:27017/"
# MONGO_URI ="mongodb://192.168.1.65:27017/"
MONGO_DB = "test"
# MONGO_DB="Queue"
MONGO_COLLECTION = "Test2"
# MONGO_COLLECTION="Test_Queue"

BROKER = "oomcardiotest.projectkmt.com"
PORT = 8883
USERNAME = "kmt"
PASSWORD = "dVBbS3NxMtmzD438"
TOPIC_Y = "oom/ecg/processedData"
CLIENT_ID = f'{random.randint(1e16, 2e16):.0f}'

# --- MongoDB Setup ---
myclient = pymongo.MongoClient(MONGO_URI)
mydb = myclient[MONGO_DB]
mycol = mydb[MONGO_COLLECTION]
is_connected = False
is_disconnected = False
# --- Queue Counter ---
last_doc = mycol.find_one(sort=[("_queue", -1)])
queue_counter = (last_doc["_queue"] + 1) if last_doc and "_queue" in last_doc else 1

# ------------------ Insert JSON with status ------------------
from datetime import datetime
from bson import ObjectId

def insert_json(docs):
    global queue_counter
    for doc in docs:
        # Build a new document with only required fields
        new_doc = {}

        # patient
        if "patient" in doc:
            try:
                new_doc["patient"] = ObjectId(doc["patient"])
            except Exception as e:
                print("Error converting patient to ObjectId:", e)

        # arrhythmia
        if "Arrhythmia" in doc:
            new_doc["Arrhythmia"] = doc["Arrhythmia"]

        # starttime
        if "starttime" in doc:
            try:
                new_doc["starttime"] = datetime.utcfromtimestamp(int(doc["starttime"]) / 1000)
            except Exception as e:
                print("Error converting starttime:", e)

        # endtime
        if "endtime" in doc:
            try:
                new_doc["endtime"] = datetime.utcfromtimestamp(int(doc["endtime"]) / 1000)
            except Exception as e:
                print("Error converting endtime:", e)

        # version
        if "version" in doc:
            new_doc["version"] = doc["version"]

        # internal fields
        new_doc["_status"] = "pending"
        new_doc["_queue"] = queue_counter

        # insert
        mycol.insert_one(new_doc)
        queue_counter += 1



        
# ------------------ MQTT Callbacks ------------------
def on_connect(client, userdata, flags, rc, properties=None):
    global is_connected, is_disconnected
    if rc == 0:
        is_connected = True
        is_disconnected = False
        result, mid = client.subscribe(TOPIC_Y, qos=1)
    else:
        print(f"Failed to connect, return code {rc}")

def on_disconnect(client, userdata, rc, properties=None):
    global is_connected, is_disconnected
    is_connected = False
    is_disconnected = True
    print(f"Disconnected (code {rc}). Will attempt to reconnect...")



def connect_mqtt() -> mqtt_client.Client:
    client = mqtt_client.Client(client_id=CLIENT_ID, protocol=mqtt_client.MQTTv5)
    client.username_pw_set(USERNAME, PASSWORD)
    context = ssl.SSLContext(ssl.PROTOCOL_TLSv1_2)
    client.tls_set_context(context)
    client.on_connect = on_connect
    client.on_disconnect = on_disconnect
    client.reconnect_delay_set(min_delay=1, max_delay=60)
    return client

def subscribe(client):

    def on_message(client, userdata, msg):
        decoded_message = msg.payload.decode("utf-8", errors="replace").strip()
        if not decoded_message:
            return
        try:
            dd = json.loads(decoded_message)
            print("Testing online json",dd)
            # Use queue insert instead of direct insert
            if isinstance(dd, dict):
                insert_json([dd])
            elif isinstance(dd, list) and all(isinstance(doc, dict) for doc in dd):
                insert_json(dd)
            else:
                print("Unsupported JSON type")

        except json.JSONDecodeError:
            print("Invalid JSON received")

    client.on_message = on_message

def run():
    client = connect_mqtt()
    subscribe(client)
    while True:
        print("Connecting to broker...")
        client.connect(BROKER, PORT)
        client.loop_forever()
def start_mqtt_listener():
    t = threading.Thread(target=run, daemon=True)
    t.start()
# ------------------ MAIN ------------------
if __name__ == "__main__":
    
    start_mqtt_listener()
    threading.Event().wait()

