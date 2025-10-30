from django.apps import AppConfig
from threading import Thread
from oom_ecg_data.data_insert_scripts.DEV import Online_script, Ofline_script, resting_script
from oom_ecg_data.data_insert_scripts.Test import Online_script as TestOnline, Ofline_script as TestOfline, resting_script as TestResting
from oom_ecg_data.data_insert_scripts.DEV.process_json import MongodbJobProcessor as DevJobProcessor
from oom_ecg_data.data_insert_scripts.Test.process_json import MongodbJobProcessor as TestJobProcessor
from oom_ecg_data.data_insert_scripts.Live.process_json import MongodbJobProcessor as LiveJobProcessor
from oom_ecg_data.data_insert_scripts.Live import Online_script as LiveOnline, Ofline_script as LiveOfline

class OomEcgDataConfig(AppConfig):
    default_auto_field = 'django.db.models.BigAutoField'
    name = 'oom_ecg_data'

    def ready(self):
        import os
        # Prevent multiple executions on Django autoreload
        if os.environ.get('RUN_MAIN') != 'true':
            return

#         # Define all your script functions
        scripts = [
#             # DEV scripts
            # Online_script.start_mqtt_listener,
            # Ofline_script.start_mqtt_listener,
            # resting_script.start_mqtt_listener,
            # # Tmt_script.start_mqtt_listener,

#             # TEST scripts
            # TestOnline.start_mqtt_listener,
            # TestOfline.start_mqtt_listener,
            # TestResting.start_mqtt_listener,
            # TestTmt.start_mqtt_listener,

#             # LIVE script
            LiveOnline.start_mqtt_listener,
            LiveOfline.start_mqtt_listener,

#             # Mongo job processors
            DevJobProcessor().run,
            TestJobProcessor().run,
            LiveJobProcessor().run,
        ]

        # Start each in its own thread
        for script in scripts:
            t = Thread(target=script, daemon=True)
            t.start()
