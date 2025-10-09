import ssl
import json
import random
import threading
import queue
import pymongo
import warnings
import time
from datetime import datetime
from bson import ObjectId
from paho.mqtt import client as mqtt_client

warnings.filterwarnings("ignore")

# --- CONFIG ---
MONGO_URI ="mongodb://192.168.1.65:27017/"
MONGO_DB="Queue"
MONGO_COLLECTION="Live_Queue"

BROKER = "mqtt.oomcardio.com"
PORT = 8883
USERNAME = "ranchodrai"
PASSWORD = "eSyk1b07B0x942R1cA0oc4cu"
TOPICS_Y = [
    "oom/ecg/processedData",
    "oom/ecg/processedDataOffline"
]
CLIENT_ID = f'{random.randint(1e16, 2e16):.0f}'

# --- MongoDB Setup ---
myclient = pymongo.MongoClient(MONGO_URI)
mydb = myclient[MONGO_DB]
mycol = mydb[MONGO_COLLECTION]

# --- Shared State ---
queue_counter = (mycol.find_one(sort=[("_queue", -1)]) or {}).get("_queue", 0) + 1
data_queue = queue.Queue(maxsize=5000)  # limit memory use
stop_event = threading.Event()


# ------------------ Insert JSON with status ------------------
def insert_json(docs):
    global queue_counter
    for doc in docs:
        new_doc = {}

        if "patient" in doc:
            try:
                new_doc["patient"] = ObjectId(doc["patient"])
            except Exception as e:
                print("Error converting patient:", e)

        for field in ["Arrhythmia", "version"]:
            if field in doc:
                new_doc[field] = doc[field]

        for field in ["starttime", "endtime"]:
            if field in doc:
                try:
                    new_doc[field] = datetime.utcfromtimestamp(int(doc[field]) / 1000)
                except Exception as e:
                    print(f"Error converting {field}:", e)

        new_doc["_status"] = "pending"
        new_doc["_queue"] = queue_counter
        queue_counter += 1
        mycol.insert_one(new_doc)


# ------------------ Background Mongo Worker ------------------
def mongo_worker():
    """Continuously inserts documents from queue to MongoDB."""
    while not stop_event.is_set():
        try:
            batch = []
            while not data_queue.empty():
                batch.append(data_queue.get_nowait())
                data_queue.task_done()

            if batch:
                insert_json(batch)
            else:
                time.sleep(0.1)  # sleep briefly when idle

        except Exception as e:
            print("Mongo Worker Error:", e)
            time.sleep(1)


# ------------------ MQTT Callbacks ------------------
def on_connect(client, userdata, flags, rc, properties=None):
    if rc == 0:
        print("MQTT connected.")
        for topic in TOPICS_Y:
            client.subscribe(topic, qos=1)
    else:
        print(f"MQTT connection failed: {rc}")


def on_disconnect(client, userdata, rc, properties=None):
    print(f"MQTT disconnected (code {rc}). Reconnecting...")


def connect_mqtt() -> mqtt_client.Client:
    client = mqtt_client.Client(client_id=CLIENT_ID, protocol=mqtt_client.MQTTv5)
    client.username_pw_set(USERNAME, PASSWORD)
    context = ssl.SSLContext(ssl.PROTOCOL_TLSv1_2)
    client.tls_set_context(context)
    client.on_connect = on_connect
    client.on_disconnect = on_disconnect
    return client


def subscribe(client):
    def on_message(client, userdata, msg):
        decoded = msg.payload.decode("utf-8", errors="replace").strip()
        if not decoded:
            return
        try:
            dd = json.loads(decoded)
            if isinstance(dd, dict):
                data_queue.put(dd)
            elif isinstance(dd, list):
                for d in dd:
                    if isinstance(d, dict):
                        data_queue.put(d)
            else:
                print("Unsupported JSON type")
        except json.JSONDecodeError:
            print("Invalid JSON received")

    client.on_message = on_message


def run():
    client = connect_mqtt()
    subscribe(client)
    client.connect(BROKER, PORT)
    client.loop_start()  # Non-blocking loop
    return client


# ------------------ MAIN ------------------
if __name__ == "__main__":
    client = run()

    # Start Mongo worker thread
    threading.Thread(target=mongo_worker, daemon=True).start()
    try:
        while True:
            time.sleep(5)
    except KeyboardInterrupt:
        stop_event.set()
        client.loop_stop()
