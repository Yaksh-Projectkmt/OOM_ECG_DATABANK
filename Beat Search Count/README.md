# ECG Pattern Search -- Hybrid (DL + Morphology Matching)

## 1. Overview

This project implements a hybrid ECG beat pattern search system that
combines deep learning and morphology-based similarity matching. The
pipeline is designed for automated detection of similar ECG beats across
large datasets using a reference beat selected via a GUI.

Core technologies used: - Deep Learning--based R-peak detection
(TensorFlow Lite) - Signal preprocessing (baseline removal, filtering,
normalization) - QRS detection (Q, R, S points) - Morphology-based
similarity scoring - Batch processing across multiple CSV files -
Automated matched beat image generation

Important: The GUI is used only for reference beat selection, batch
control, and basic visualization. All matched outputs are saved directly
to disk and can be accessed without the GUI.

------------------------------------------------------------------------

## 2. Key Features

-   Deep learning R-peak detection using TensorFlow Lite
-   Robust ECG preprocessing pipeline
-   R-anchored beat alignment
-   Multi-criteria morphology similarity scoring
-   Edge-beat handling with signal padding
-   Duplicate beat prevention logic
-   Multi-lead ECG support
-   Full folder batch processing
-   Automatic matched image generation per file and per lead
-   GUI-assisted reference beat selection (optional for viewing)

------------------------------------------------------------------------

## 3. System Architecture

### 3.1 Signal Preprocessing

The preprocessing pipeline includes: - Low-pass filtering for noise
suppression - Baseline wander removal - Signal normalization

Main functions: - `lowpass()` - `remove_baseline_wander()` -
`normalize()`

### 3.2 Deep Learning Module

R-peaks are detected using a TensorFlow Lite model.

Key functions: - `predict_r_peaks_dl()` - `refine_r_peaks()`

The model is loaded using:

``` python
tf.lite.Interpreter(model_path="path_to_your_model.tflite")
```

### 3.3 QRS Detection

After R-peak detection: - Q and S points are computed per beat - QRS
complexes are extracted per lead

Main function: - `detect_q_s()`

### 3.4 Similarity Engine

Beat similarity is evaluated using morphology-based scoring.

Core logic functions: - `is_similar_beat()` - `overlaps_existing_r()`

Scoring components: - Core QRS correlation - Extended QRS correlation -
Amplitude consistency - R-peak polarity check - Final weighted
similarity score

------------------------------------------------------------------------

## 4. GUI Role and Clarification

### 4.1 Purpose of GUI

The GUI is included only for: - Selecting the reference beat - Starting
batch processing - Stopping batch processing - Basic visualization

GUI Components: - `ECGPatternGUI` - `ECGPatternGUIApp`

### 4.2 What GUI Is NOT Required For

The GUI is NOT required for: - Loading results - Viewing matched
images - Accessing outputs - Running batch matching logic

All results are saved automatically to disk and can be used by external
scripts (e.g., data bank processing workflows).

------------------------------------------------------------------------

## 5. How the System Works

### Step 1: Select Reference Beat (GUI Only Purpose)

1.  Run the script.
2.  Click "Select Folder".
3.  Select a CSV file.
4.  Choose a lead.
5.  Drag on the plot to select one full beat containing an R peak
    (mandatory).
6.  Start batch search.

The selected beat becomes the reference morphology for matching.

### Step 2: Batch Search Process

During batch processing: - Each CSV file in the selected folder is
processed - Each numeric lead is analyzed - R peaks are detected using
the DL model - Q and S points are computed - Beats are aligned using
R-anchored alignment - Morphology similarity scores are calculated -
Similar beats are detected and shaded - Matched plots are generated and
saved as images

------------------------------------------------------------------------

## 6. Project Structure (Logical Components)

Main modules inside the script:

### Preprocessing

-   `lowpass()`
-   `remove_baseline_wander()`
-   `normalize()`

### Deep Learning

-   `predict_r_peaks_dl()`
-   `refine_r_peaks()`

### QRS Detection

-   `detect_q_s()`

### Similarity & Matching

-   `is_similar_beat()`
-   `overlaps_existing_r()`

### Batch Processing

-   `process_file()`
-   `start_batch()`

### GUI

-   `ECGPatternGUI`
-   `ECGPatternGUIApp`

------------------------------------------------------------------------

## 7. Output Structure

All matched images are automatically saved to:

    <Selected_Folder>/matched_images/

File naming format:

    <original_filename>_<lead>.png

Important: - Output images are always written to disk - GUI is not
required to access results - Images can be directly opened from the
folder - Outputs can be consumed by separate data bank scripts

------------------------------------------------------------------------

## 8. Viewing Results

### 8.1 Directly From Folder (Recommended)

Open the folder:

    matched_images/

All matched plots are available as PNG files.

### 8.2 From GUI

Use the following controls: - "Load Results" - "Next Plot" - "Previous
Plot"

Note: The GUI only visualizes grouped results. The primary outputs are
always saved to disk.

------------------------------------------------------------------------

## 9. Running the Script

### 9.1 GUI Mode

``` bash
python Final_beat_search_count.py
```

### 9.2 Batch Mode (Headless)

``` bash
python Final_beat_search_count.py --batch
```

In batch mode: - Matplotlib uses Agg backend - Images are generated and
saved automatically - GUI interaction is not required

------------------------------------------------------------------------

## 10. Requirements

### 10.1 Python Version

-   Python 3.10.10 (recommended)

### 10.2 Required Libraries

-   numpy
-   pandas
-   matplotlib
-   scipy
-   tensorflow
-   fastdtw
-   tkinter

Install dependencies:

``` bash
pip install numpy pandas matplotlib scipy tensorflow fastdtw
```

Note: `tkinter` is included with most standard Python installations.

------------------------------------------------------------------------

## 11. Model Requirement

The system requires a TensorFlow Lite R-peak detection model.

Example configuration:

``` python
tf.lite.Interpreter(model_path="path_to_your_model.tflite")
```

Ensure the model path is correctly updated inside the script before
execution.

------------------------------------------------------------------------

## 12. Similarity Logic Summary

Beat similarity is determined using: - R-peak polarity validation - Core
QRS correlation - Extended QRS correlation - Amplitude ratio
consistency - Final weighted similarity threshold

Only beats exceeding the defined similarity threshold are accepted as
matches.

Additional safeguards: - Edge-beat handling via signal padding - Minimum
RR distance checks for duplicate prevention

------------------------------------------------------------------------

## 13. Design Notes

-   Signals shorter than the minimum required length are skipped
-   Leads with insufficient data are automatically ignored
-   Batch processing runs in a background thread
-   Stop Batch safely terminates processing after the current file
-   Garbage collection is triggered after each file to manage memory
    usage
-   Outputs are designed for integration into downstream data bank
    processing workflows
