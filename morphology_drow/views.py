from django.shortcuts import render
from django.http import JsonResponse
from django.core.files.storage import default_storage
import json,os
import pandas as pd
from django.conf import settings
import os
import pandas as pd
from django.shortcuts import render
import pymongo
from django.http import JsonResponse
from morphology_drow.img_to_extract_signal import process_images
import subprocess
from django.views.decorators.csrf import csrf_exempt
from pymongo import MongoClient
from django.utils import timezone
from scipy import signal
mongo_client = MongoClient("mongodb://192.168.1.65:27017/")
db = mongo_client['Morphology_data']
morphology_db = mongo_client['Morphology_Patients']

# Dashboard view
def index(request):
    return render(request, 'morphology_drow/morph_index.html')

# API view for ECG data (Example)
def api_ecg_data(request):
    data = {"message": "ECG data API response"}
    return JsonResponse(data)

def morphology_data_insert_db(patient_id, arrhythmia, datalength,frequency=200):
    patient_col = morphology_db[arrhythmia]

    # time in minutes
    time_minutes = (datalength / int(frequency)) / 60

    # increment if exists, else insert
    patient_col.update_one(
        {"PatientID": patient_id},
        {
            "$inc": {
                "total_records": 1,
                "total_time": time_minutes
            },
            "$setOnInsert": {
                "PatientID": patient_id
            }
        },
        upsert=True
    )

# Ensure correct import
def uploaded_file(request):
    
    if request.method == 'POST' and request.FILES.get('file'):
        uploaded_file = request.FILES['file']
        
        # Define the save path for temporary storage
        temp_folder = os.path.join(settings.MEDIA_ROOT,"morphology_drow", 'temp')
        os.makedirs(temp_folder, exist_ok=True)  # Create directory if it doesn't exist

        file_path = os.path.join(temp_folder, uploaded_file.name)

        # Save the uploaded file
        with default_storage.open(file_path, 'wb+') as destination:
            for chunk in uploaded_file.chunks():
                destination.write(chunk)

        # Process the image to extract ECG signal
        process_images(file_path)  

        # Define the CSV file path
        csv_name = os.path.splitext(uploaded_file.name)[0]  # Remove file extension
        csv_file_path = os.path.join(settings.MEDIA_ROOT, "morphology_drow", 'csv_files', f"{csv_name}.csv")
        
        os.remove(file_path)
        # Construct image URL (assuming the processed image is saved here)
        image_url = f"{uploaded_file.name}"
        
        return JsonResponse({'message': 'File uploaded successfully', 'image_url': image_url, 'csv_file': csv_name})

    return JsonResponse({'error': 'Invalid request'}, status=400)

def csv_data(request):
    global csv_name
    if request.method == "POST":
        try:
            data = request.body.decode('utf-8')  # Read incoming request data
            request_data = json.loads(data)
            csv_name = request_data.get("csv_file")

            # Define the CSV file path
            csv_file_path = os.path.join(settings.MEDIA_ROOT, "morphology_drow", 'csv_files', f"{csv_name}.csv")  
            
            # Load ECG data from CSV
            df = pd.read_csv(csv_file_path)

            # Ensure the CSV file has correct column names
            if df.shape[1] < 2:  # Check if at least two columns exist
                return JsonResponse({"error": "CSV file does not have enough columns"}, status=400)

            # Assume the first column is Index and the second column is Voltage
            x_values = df.iloc[:, 0].tolist()  # First column (Index)
            y_values = df.iloc[:, 1].tolist()  # Second column (Voltage)

            min_y = min(y_values)
            if min_y < 0:
                y_values = [y - min_y for y in y_values] 
            
            return JsonResponse({"x": x_values, "y": y_values})

        except Exception as e:
            return JsonResponse({"error": str(e)}, status=500)

    return JsonResponse({"error": "Invalid request method"}, status=405)

def upload_ecg(request):
    if request.method == "POST":
        # Define fixed file path (replace with your actual file path)
        csv_file_path = os.path.join(settings.MEDIA_ROOT, "morphology_drow", 'csv_files', f"{csv_name}.csv")  
        
        # Get Arrhythmia, Sub-Arrhythmia, and Lead values from the form
        arrhythmia = request.POST.get("arrhythmia")
        sub_arrhythmia = request.POST.get("sub_arrhythmia")
        lead_type = request.POST.get("lead")  # Corrected from "leads" to "lead"
        if not arrhythmia or not sub_arrhythmia or not lead_type:
            return JsonResponse({"error": "Arrhythmia, Sub-Arrhythmia, and Lead are required"}, status=400)

        try:
            # Read the CSV File
            all_lead_data = pd.read_csv(csv_file_path).fillna(0)
        
            # Ensure the correct data format
            if any(str(_).isalpha() for _ in all_lead_data.iloc[0, :].values):
                all_lead_data = pd.read_csv(csv_file_path, usecols=['Index']).fillna(0)
                all_lead_data = all_lead_data.rename(columns={'II': 'II'})
            else:
                all_lead_data = all_lead_data.rename(columns={0: 'II'})

            datalength=len(all_lead_data)
            # Create Data Dictionary
            ecg_data_dic = {
                'PatientID': csv_name,  
                'Arrhythmia': sub_arrhythmia,  
                'Lead': 2,  # Fixed Lead = 2
                'Frequency': 200,# Fixed Frequency = 200
                'leadtype': lead_type,   # Lead Type
                "server": "local",
                'datalength':datalength,
                "created_At": timezone.now(),
                'Data': {'II': all_lead_data['II'].to_list()},
            }
            
            # Dynamically Select the Correct MongoDB Collection
            collection = db[arrhythmia]# Collection name = Arrhythmia Type
            
            # Check for Duplicate Entry
            exists = collection.count_documents(
                {'PatientID': csv_name, 'Arrhythmia': sub_arrhythmia, 'leadtype': lead_type, 'Lead': 2, 'Frequency': 200}
            ) > 0
            if exists:
                return JsonResponse({"error": "Duplicate data found. Skipping insertion."}, status=400)  # Changed to error
            else:
                collection.insert_one(ecg_data_dic)
                morphology_data_insert_db(csv_name,arrhythmia,datalength)
                msg = f"ECG data inserted successfully into `{arrhythmia}` collection."
                return JsonResponse({"message": msg})

        except Exception as e:
            return JsonResponse({"error": "File processing failed"}, status=400)

    return JsonResponse({"error": "Invalid request"}, status=400)

def remove_all_csvs(request):
    if request.method == "POST":
        folder_path = os.path.join(settings.MEDIA_ROOT, "morphology_drow", 'csv_files')
        # , f"{csv_name}_output.csv"

        try:
            if not os.path.exists(folder_path):
                return JsonResponse({"error": "Folder not found."}, status=404)

            deleted_files = []
            for filename in os.listdir(folder_path):
                file_path = os.path.join(folder_path, filename)
                if os.path.isfile(file_path):
                    os.remove(file_path)
                    deleted_files.append(filename)

            return JsonResponse({"message": f"Data inserted successfully..", "files": deleted_files})

        except Exception as e:
            return JsonResponse({"error": str(e)}, status=500)

    return JsonResponse({"error": "Invalid request."}, status=400)   
     
def lowpass(file, cutoff=0.4):
    b, a = signal.butter(3, cutoff, btype='lowpass', analog=False)
    low_passed = signal.filtfilt(b, a, file)
    return low_passed
    

@csrf_exempt
def open_morphology_script(request):
    if request.method == 'POST':
        try:
            data = json.loads(request.body)
            image_data = data['points']
            pid = data["PatientID"]
            arrhythmia = data["arrhythmia"]
            # Convert canvas points to voltage
            voltages = [(200 - point['y']) / 200 * 5 for point in image_data]
            df = pd.DataFrame({'FilteredVoltage': lowpass(voltages)})  # only second column

            # Convert to list of floats
            filtered_voltage_list = df['FilteredVoltage'].tolist()
            datalength=len(filtered_voltage_list)
            # Prepare final document
            document = {
                "PatientID": pid,
                "Arrhythmia": arrhythmia,
                "Lead": 2,
                "Frequency": 200,
                "datalength":datalength,
                "server": "local",
                "created_At": timezone.now(),
                "Data": {
                    "II": filtered_voltage_list
                }
                
            }
            # Save to MongoDB
            collection = db[arrhythmia]
            
            # Check for duplicate data
            existing_document = collection.find_one({
                "PatientID": pid,
                "Lead": 2,
                "Frequency": 200,
                "Arrhythmia": arrhythmia
            })
            
            if existing_document:
                return JsonResponse({
                    "status": "error",
                    "message": "Duplicate data found for this PatientID"
                })
            
            # Insert if no duplicate found
            collection.insert_one(document)
            morphology_data_insert_db(pid, arrhythmia,datalength)
            return JsonResponse({"status": "success", "message": "Data inserted successfully!"})
        except Exception as e:
            return JsonResponse({'success': False, 'error': str(e)})
