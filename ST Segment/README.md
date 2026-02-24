# ECG ST-Segment Detection and Multi-Lead Analysis System

## 1. Overview

This project implements a complete automated ECG signal processing and
ST-segment analysis pipeline. The system processes ECG CSV files and
performs:

-   Deep learning--based R-peak detection (TensorFlow Lite)
-   Deep learning--based P and T segmentation
-   Q and S point detection
-   Beat-by-beat ST segment measurement
-   Isoelectric baseline estimation
-   ST elevation and depression calculation (in mm)
-   Multi-lead ECG PDF generation (2-lead, 7-lead, 12-lead)
-   Per-beat CSV export
-   Automatic PDF merging
-   Batch processing of entire folders

The pipeline is designed for clinical research, large-scale screening,
and automated ECG report generation.

------------------------------------------------------------------------

## 2. Key Features

-   End-to-end automated ECG processing pipeline
-   Deep learning--based peak and segmentation detection
-   Beat-wise ST-segment analysis in millimeters
-   Clinical-style multi-lead ECG visualization
-   Support for 2-lead, 7-lead, and 12-lead ECG formats
-   Scalable batch processing of large ECG datasets
-   Thread-safe TensorFlow Lite inference
-   Structured CSV and PDF reporting outputs

------------------------------------------------------------------------

## 3. System Architecture

### 3.1 Signal Preprocessing

Each lead undergoes the following preprocessing steps: - Median
filtering for baseline wander removal - Low-pass filtering for noise
suppression - Z-score normalization - Scaling to ECG grid units (small
boxes and mV)

### 3.2 Deep Learning Models (TensorFlow Lite)

Two TensorFlow Lite models are required: - R-peak detection model - P/T
peak segmentation model

Thread-local interpreters are used to ensure safe and efficient batch
execution.

### 3.3 Beat-Level Detection

For each detected beat, the system identifies: - P peak - Q point - R
peak - S point - T onset - J point

All detections are stored per beat and per lead.

------------------------------------------------------------------------

## 4. Lead Configurations

The system automatically determines the lead configuration based on
available columns in the input CSV file.

### 4.1 2-Lead Configuration

Supported: - Lead II (primary rhythm lead)

Use Case: - Limited ECG acquisition systems

Outputs: - Single-lead PDF layout - Beat-by-beat ST analysis - Per-beat
CSV export

### 4.2 7-Lead Configuration

Supported Leads: - I - II - III - aVR - aVL - aVF - V5

Use Case: - Compact ECG acquisition systems

Outputs: - Multi-panel PDF layout (7 leads per chunk) - Beat-wise ST
measurement for each lead - Lead-wise abnormal ST summary

### 4.3 12-Lead Configuration

Supported Leads: - I - II - III - aVR - aVL - aVF - V1 - V2 - V3 - V4 -
V5 - V6

Use Case: - Standard clinical 12-lead ECG format

Outputs: - Clinical-style multi-page PDF - Proper grid scaling (10
mm/mV) - Full PQRST annotation - ST shading and deviation labeling -
Per-lead and per-beat CSV analysis

------------------------------------------------------------------------

## 5. ST Segment Detection Methodology

For every beat: 1. J-point is detected after S wave recovery. 2. ST
center is calculated at J + 60 ms. 3. Median ST value between J and T
onset is computed. 4. ST deviation from the isoelectric baseline is
calculated. 5. Elevation or depression is determined. 6. Abnormal ST is
flagged if threshold ≥ 1.5 mm (default).

Each beat produces exactly one CSV row per lead.

------------------------------------------------------------------------

## 6. CSV Output Structure

Each row corresponds to one beat in one lead.

### Main Columns

-   file_name
-   lead
-   chunk_number
-   beat_index
-   p_index
-   q_index
-   r_index
-   s_index
-   t_onset_index
-   j_point_index
-   st_center_index
-   st_mm
-   st_elevation_mm
-   st_depression_mm
-   st_detected
-   isoelectric_baseline
-   shading_start_index
-   shading_end_index

All measurements are stored in millimeters and sample indices.

------------------------------------------------------------------------

## 7. PDF Output

For each ECG file, the system generates: - Chunk-based PDF pages -
Standard ECG grid background - Small box and large box scaling - P, Q,
R, S, T markers - ST shaded region (elevation or depression) - Lead
labels - Time scaling (10 seconds per chunk)

After processing: - All chunk PDFs are merged per ECG file - A final
combined PDF is generated across all processed ECGs

------------------------------------------------------------------------

## 8. Folder Processing Workflow

The system performs the following steps: 1. Recursively scans the input
folder for CSV files 2. Processes each ECG independently 3. Saves
outputs inside a dedicated folder per ECG 4. Merges chunk PDFs
automatically 5. Generates final merged reports

------------------------------------------------------------------------

## 9. Output Directory Structure

    <save_path>/
        <ecg_name>/
            <ecg_name>_chunk_001.pdf
            <ecg_name>_chunk_002.pdf
            ...
            <ecg_name>_MERGED.pdf
            <ecg_name>_ST_analysis.csv

Final combined report:

    FINAL_ALL_CSV_ALL_LEADS.pdf

------------------------------------------------------------------------

## 10. Requirements

### 10.1 Python Version

-   Python 3.10.10 (recommended)

### 10.2 Dependencies

``` bash
pip install numpy pandas matplotlib scipy tensorflow PyPDF2
```

### 10.3 Required Models

-   R-peak TensorFlow Lite model (.tflite)
-   P/T segmentation TensorFlow Lite model (.tflite)

------------------------------------------------------------------------

## 11. How to Run

### 11.1 Configure Paths

``` python
path      = r"INPUT_FOLDER"
save_path = r"OUTPUT_FOLDER"
r_index_model_path  = r"R_MODEL.tflite"
pt_index_model_path = r"PT_MODEL.tflite"
```

### 11.2 Execute the Pipeline

``` bash
python script_name.py
```

------------------------------------------------------------------------

## 12. Measurement Standards

-   1 small box (horizontal) = 0.04 seconds
-   1 small box (vertical) = 0.1 mV
-   10 mm = 1 mV scaling
-   Default sampling frequency = 200 Hz

------------------------------------------------------------------------

## 13. Thread Safety

-   Thread-local TensorFlow Lite interpreters
-   Lock-protected inference execution
-   Safe parallel file processing for batch workloads

------------------------------------------------------------------------

## 14. Design Characteristics

-   Each beat always generates exactly one CSV row
-   Leads without valid data are automatically skipped
-   Automatic detection of 2, 7, or 12-lead configuration
-   Clinical-style ECG visualization and annotation
-   Research-grade ST-segment screening pipeline

------------------------------------------------------------------------

## 15. Use Cases

-   Clinical research and ECG studies
-   Automated ST-elevation screening
-   Large-scale ECG dataset analysis
-   Medical AI validation pipelines
-   ECG report automation systems
