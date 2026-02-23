let currentFileType = null;
let uploadedImageData = null;

const fileInput = document.getElementById('fileInput');
const uploadBox = document.getElementById('uploadBox');
const uploadTitle = document.getElementById('uploadTitle');
const loadingOverlay = document.getElementById('loadingOverlay');
const fileInfo = document.getElementById('fileInfo');
const fileIcon = document.getElementById('fileIcon');
const fileName = document.getElementById('fileName');

const removeBtn = document.getElementById('removeBtn');

const defaultState = document.getElementById('defaultState');
const imageAnalysis = document.getElementById('imageAnalysis');
const pdfAnalysis = document.getElementById('pdfAnalysis');
const csvAnalysis = document.getElementById('csvAnalysis');

const viewPlotBtn = document.getElementById('viewPlotBtn');
const closePlotBtn = document.getElementById('closePlotBtn');
const plotModal = document.getElementById('plotModal');

const iconSVG = {
    image: `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
        <rect x="3" y="3" width="18" height="18" rx="2" ry="2"></rect>
        <circle cx="8.5" cy="8.5" r="1.5"></circle>
        <polyline points="21 15 16 10 5 21"></polyline>
    </svg>`,
    pdf: `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
        <path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"></path>
        <polyline points="14 2 14 8 20 8"></polyline>
        <line x1="16" y1="13" x2="8" y2="13"></line>
        <line x1="16" y1="17" x2="8" y2="17"></line>
    </svg>`,
    csv: `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
        <path d="M12 2h10v20H2V2h10z"></path>
        <line x1="12" y1="2" x2="12" y2="22"></line>
    </svg>`
};
function startTask(title, description) {
    const taskId = taskManager.createTask(
        "analysis",
        title,
        description
    );
    taskManager.startTaskTimer(taskId);
    return taskId;
}

fileInput.addEventListener('change', handleFileSelect);
removeBtn.addEventListener('click', resetUpload);

viewPlotBtn.addEventListener('click', () => {
    plotModal.style.display = 'block';
});
closePlotBtn.addEventListener('click', () => {
    plotModal.style.display = 'none';
});

uploadBox.addEventListener('dragover', (e) => {
    e.preventDefault();
    uploadBox.classList.add('dragging');
});

uploadBox.addEventListener('dragleave', () => {
    uploadBox.classList.remove('dragging');
});

uploadBox.addEventListener('drop', (e) => {
    e.preventDefault();
    uploadBox.classList.remove('dragging');
    const file = e.dataTransfer.files[0];
    if (file) {
        processFile(file);
    }
});
const getCSRFToken = () => {
    const cookie = document.cookie.split(';').find(c => c.trim().startsWith('csrftoken='));
    return cookie ? decodeURIComponent(cookie.split('=')[1]) : document.getElementById('csrfToken')?.value || '';
  };
function handleFileSelect(e) {
    const file = e.target.files[0];
    if (file) {
        processFile(file);
    }
}

let savedServerFilename = null;
let uploadedTmtFile = null;  // stores Django-saved filename
let lastTaskId = null;

const taskContext = {
    image: { filename: null, taskId: null },
    csv:   { filename: null, taskId: null },
    tmt:   { filename: null, taskId: null }
};
/* ===============================
   MAIN IMAGE FLOW
================================ */

function processFile(file) {

    fileName.textContent = file.name;
    uploadTitle.textContent = "Uploading...";
    loadingOverlay.style.display = "flex";

    // ---------- IMAGE PREVIEW ----------
    if (file.type.startsWith("image/")) {
        const reader = new FileReader();
        reader.onload = e => {
            uploadedImage.src = e.target.result;
        };
        reader.readAsDataURL(file);
    }

    // ---------- CSV PREVIEW (lead detect) ----------
    if (file.name.endsWith(".csv")) {
        const reader = new FileReader();
        reader.onload = evt => {
            const lines = evt.target.result.split(/\r?\n/);
            const header = lines[0]?.split(",").map(h => h.trim()) || [];
            const colCount = header.length;

            document.getElementById("TOtalRecords").textContent =
                Math.max(lines.length - 1, 0);

            let leadType = "Unknown";
            if ([2, 3].includes(colCount)) leadType = "2_Lead";
            else if ([7, 8].includes(colCount)) leadType = "7_Lead";
            else if ([12, 13].includes(colCount)) leadType = "12_Lead";

            document.getElementById("leadvalue").textContent = leadType;
        };
        reader.readAsText(file);
    }

    // ---------- SAVE FILE TO BACKEND ----------
    const formData = new FormData();
    formData.append("file", file);

    fetch(uploadFileUrl, {
        method: "POST",
        body: formData,
        headers: { "X-CSRFToken": getCSRFToken() }
    })
    .then(res => {
        if (!res.ok) throw new Error("Upload failed");
        return res.json();
    })
    .then(data => {
        if (data.error) throw new Error(data.error);

        /* ===============================
           TASK-WISE STORE
        ================================ */

        // IMAGE
        if (file.type.startsWith("image/")) {
            taskContext.image.filename = data.filename;
            taskContext.image.taskId   = data.task_id;

            showAnalysis("image");
            alertSystem.success("Success", "Image uploaded");

        // CSV
        } else if (file.name.endsWith(".csv")) {
            savedServerFilename = data.filename.split("/").pop();
            console.log("Saved server filename:", savedServerFilename);
            taskContext.csv.filename = data.filename;
            taskContext.csv.taskId   = data.task_id ?? null;

            showAnalysis("csv");
            alertSystem.success("Success", "CSV uploaded");

        // TMT (PDF)
        } else if (file.type === "application/pdf") {
            uploadedTmtFile = file;               // ? REQUIRED
            taskContext.tmt.filename = data.filename;
            taskContext.tmt.taskId   = data.task_id;

            showAnalysis("pdf");
            alertSystem.success("Success", "TMT PDF uploaded");
        }

        // UI common
        fileIcon.innerHTML = iconSVG[currentFileType];
        fileInfo.style.display = "flex";
    })
    .catch(err => {
        console.error(err);
        alertSystem.error("Error", err.message);
    })
    .finally(() => {
        loadingOverlay.style.display = "none";
        uploadTitle.textContent = "Drop your file here";
    });
}


/* ===============================
   START IMAGE ANALYSIS (BUTTON)
================================ */

function startImageAnalysis() {

    const { filename, taskId } = taskContext.image;

    if (!filename || !taskId) {
        alertSystem.info("Info", "Upload an ECG image first.");
        return;
    }

    const uiTaskId = startTask(
        "Image Analysis",
        "Running ECG image analysis"
    );
    
    alertSystem.info(
        "Info",
        "Data analyzing started. Please check the Task Notification."
    );
    resetUpload();

    fetch(`/analysis/process_image/${filename}/${taskId}`, {
        method: "POST",
        headers: { "X-CSRFToken": getCSRFToken() }
    })
    .then(res => res.json())
    .then(data => {
        if (data.error) throw new Error(data.error);

        taskManager.completeTask(uiTaskId, {
            result: "Image Analysis Completed",
            task_id: taskId
        });

        alertSystem.success("Success", "ECG analysis completed");
    })
    .catch(err => {
        taskManager.errorTask(uiTaskId, err.message);
        alertSystem.error("Error", err.message);
    });
}

function showAnalysis(type) {
    currentFileType = type;

    defaultState.style.display = 'none';
    imageAnalysis.style.display = 'none';
    pdfAnalysis.style.display = 'none';
    csvAnalysis.style.display = 'none';

    if (type === 'image') {
        imageAnalysis.style.display = 'block';
    } else if (type === 'pdf') {
        pdfAnalysis.style.display = 'block';
    } else if (type === 'csv') {
        csvAnalysis.style.display = 'block';
    }
}

function resetUpload() {
    currentFileType = null;
    uploadedImageData = null;

    fileInput.value = '';
    fileInfo.style.display = 'none';

    // Hide viewers
    imageViewer.style.display = 'none';
    const plotModal = document.getElementById("plotModal");
    if (plotModal) plotModal.style.display = 'none';

    // Reset main UI
    defaultState.style.display = 'block';
    imageAnalysis.style.display = 'none';
    pdfAnalysis.style.display = 'none';
    csvAnalysis.style.display = 'none';

    uploadTitle.textContent = "Drop your file here";
}
function uploadAndPlot() {
    const fileInput = document.getElementById('fileInput');
    if (!fileInput) {
        console.error('Error: fileInput element not found.');
        alertSystem.error('Error', 'File input not found.');
        return;
    }

    const file = fileInput.files[0];
    if (!file) {
        alertSystem.info('Info', 'Please select a file.');
        return;
    }

    const formData = new FormData();
    formData.append('ecg_file', file);

    fetch('/analysis/plot_csv_view/', {
        method: 'POST',
        body: formData,
        headers: { 'X-CSRFToken': document.querySelector("input[name='csrfmiddlewaretoken']")?.value }
    })
    .then(response => response.json())
    .then(data => {
        if (data.error) {
            alertSystem.error('Error', data.error);
            return;
        }

        const plotDiv = document.getElementById('plotContainer');
        if (!plotDiv) {
            console.error('Error: chart-container element not found.');
            alertSystem.error('Error', 'Plot container not found.');
            return;
        }
        plotDiv.style.display = 'block';

        let plotElement = document.getElementById('plot');
        if (!plotElement) {
            plotElement = document.createElement('div');
            plotElement.id = 'plot';
            plotElement.style.cssText = 'width: 100%; height: 500px; border-radius: 8px;';
            plotDiv.appendChild(plotElement);
        }

        // Clear existing content
        const existingImages = plotDiv.querySelectorAll('img');
        existingImages.forEach(el => el.remove());
        Plotly.purge(plotElement); // Clear previous Plotly plot

        // Plot all leads with Plotly
        plotECG(data);

        const viewBtn = document.getElementById('viewBtn');
        if (viewBtn) viewBtn.style.display = 'none';
    })
    .catch(error => {
        console.error('Error:', error);
        alertSystem.error('Error', error.message);
    });
}
const minMaxScale = (data, minVal = -1, maxVal = 1) => {
    const min = Math.min(...data);
    const max = Math.max(...data);
    if (max === min) return data.map(() => (maxVal + minVal) / 2);
    return data.map(v => ((v - min) / (max - min)) * (maxVal - minVal) + minVal);
};

function plotECG(data) {
    const plotDiv = document.getElementById('plot');
    if (!plotDiv) {
        console.error('Error: Missing plot container');
        return;
    }

    const leadNames = Object.keys(data.leads);
    if (!leadNames.length) {
        console.error('No leads found');
        return;
    }

    const traces = [];
    const rows = leadNames.length;

    leadNames.forEach((lead, idx) => {
        const rawX = data.leads[lead].x;
        const rawY = data.leads[lead].y;

        if (!rawX || !rawY || rawY.length === 0) return;

        // Normalize Y for consistent display
        const scaledY = minMaxScale(rawY, -1, 1);

        traces.push({
            x: rawX,
            y: scaledY,
            name: lead.toUpperCase(),
            type: "scatter",
            mode: "lines",
            line: { width: 1.3, color: "#000" },
            xaxis: "x" + (idx + 1),
            yaxis: "y" + (idx + 1)
        });
    });

    const layout = {
        grid: { rows: rows, columns: 1, pattern: "independent" },
        height: rows * 320,
        margin: { t: 40, b: 40, l: 65, r: 40 },
        plot_bgcolor: document.body.dataset.theme === "dark" ? "#1e1e2f" : "#ffffff",
        paper_bgcolor: document.body.dataset.theme === "dark" ? "#1e1e2f" : "#ffffff",
        font: {
            color: document.body.dataset.theme === "dark" ? "#fff" : "#000"
        },
        showlegend: false
    };

    leadNames.forEach((lead, idx) => {
        const xAxisId = "xaxis" + (idx + 1);
        const yAxisId = "yaxis" + (idx + 1);

        layout[xAxisId] = {
            title: { text: "Time (samples)", font: { size: 12 } },
            showgrid: true,
            gridcolor: "rgba(255,0,0,0.4)",
            zeroline: false,
            dtick: 50,
            minor: {
                showgrid: true,
                gridcolor: "rgba(255,150,150,0.3)"
            }
        };

        layout[yAxisId] = {
            title: { text: lead.toUpperCase(), standoff: 8, font: { size: 14 } },
            range: [-1.5, 1.5],
            showgrid: true,
            gridcolor: "rgba(255,0,0,0.4)",
            zeroline: false,
            dtick: 0.5,
            minor: {
                showgrid: true,
                gridcolor: "rgba(255,150,150,0.25)"
            }
        };
    });

    const config = {
        responsive: true,
        displaylogo: false,
        modeBarButtonsToAdd: [
            "pan2d", "zoom2d", "autoScale2d", "resetScale2d",
            "zoomIn2d", "zoomOut2d", "toImage"
        ]
    };

    Plotly.newPlot(plotDiv, traces, layout, config);

    plotDiv.on("plotly_selected", (eventData) => {
        if (eventData) {
            window.selectedData = eventData.points.map(pt => ({
                lead: pt.data.name,
                x: pt.x,
                y: pt.y
            }));
        }
    });
}

// Close chart container
function closeChart() {
    const chartContainer = document.getElementById('chart-container');
    const viewBtn = document.getElementById('viewBtn');
    if (chartContainer) chartContainer.style.display = 'none';  
    if (viewBtn) viewBtn.style.display = 'block';
}


async function uploadCSVFile() {
    const arrhythmia = document.getElementById("arrhythmiaSelect")?.value;
    if (!arrhythmia) {
        alertSystem.info("Info", "Select Arrhythmia");
        return;
    }

    const is_lead = document.getElementById("leadvalue")?.innerText?.trim();
    if (!is_lead) {
        alertSystem.error("Error", "Lead value missing");
        return;
    }

    if (!savedServerFilename) {
        alertSystem.error("Error", "CSV file not uploaded");
        return;
    }

    // Disable upload button
    const uploadBtn = document.getElementById("uploadCSVBtn");
    if (uploadBtn) uploadBtn.disabled = true;

    // Start task
    const taskId = startTask(
        "CSV Analysis",
        `Running ${arrhythmia} model`
    );

    alertSystem.info(
        "Info",
        "Data analyzing started. Please check the Task Notification."
    );

    const formData = new FormData();
    formData.append("csv_name", savedServerFilename);
    formData.append("is_lead", is_lead);

    try {
        const response = await fetch(
            `/analysis/run_model_arrhythmia/${arrhythmia}/${savedServerFilename}/`,
            {
                method: "POST",
                headers: { "X-CSRFToken": getCSRFToken() },
                body: formData
            }
        );

        const data = await response.json();

        if (!response.ok || data.error) {
            throw new Error(data.error || "Analysis failed");
        }

        // STORE RESULT UUID (GLOBAL FOR DOWNLOAD)
        window.lastResultUUID = data.file_id;

        // Task success
        taskManager.completeTask(taskId, {
            result: "CSV Analysis Completed",
            task_id: data.task_id
        });

        alertSystem.success("Success", "Analysis completed successfully");

    } catch (error) {

        taskManager.errorTask(taskId, error.message);
        alertSystem.error("Error", error.message);

    } finally {
        if (uploadBtn) uploadBtn.disabled = false;
    }
}

function uploadTMTFile() {

    if (!uploadedTmtFile) {
        alertSystem.info("Info", "Please upload a PDF first.");
        return;
    }

    // START UI TASK
    const uiTaskId = startTask(
        "TMT Analysis",
        "Processing TMT PDF"
    );

    alertSystem.info(
        "Info",
        "Data analyzing started. Please check the Task Notification."
    );

    const formData = new FormData();
    formData.append("file", uploadedTmtFile);

    // Optional: reset UI AFTER request is sent
    resetUpload();

    fetch("/analysis/upload_tmt_pdf/", {
        method: "POST",
        body: formData,
        headers: { "X-CSRFToken": getCSRFToken() }
    })
    .then(res => {
        if (!res.ok) {
            throw new Error("TMT upload failed");
        }
        return res.json();
    })
    .then(data => {

        if (data.error) {
            throw new Error(data.error);
        }

        console.log("TMT Response:", data);

        // STORE BACKEND RESULT
        window.tmtTaskId  = data.task_id;
        window.tmtZipFile = data.zip_file;

        // MARK TASK SUCCESS
        taskManager.completeTask(uiTaskId, {
            result: "TMT PDF Processed",
            task_id: data.task_id
        });

        // Clean preview UI (safe)
        const container = document.getElementById("tmtPreviewContainer");
        if (container) container.style.display = "none";

        const imagesDiv = document.getElementById("tmtPreviewImages");
        if (imagesDiv) imagesDiv.innerHTML = "";

        // Enable download button
        const downloadBtn = document.getElementById("DownloadTmtBtn");
        if (downloadBtn) {
            downloadBtn.style.display = "inline-flex";
        }

        alertSystem.success(
            "Success",
            `TMT Report processed (Task ID: ${data.task_id})`
        );

    })
    .catch(err => {

        console.error("TMT ERROR:", err);

        // MARK TASK FAILED
        taskManager.errorTask(uiTaskId, err.message);

        alertSystem.error(
            "Error",
            err.message || "Failed to process TMT PDF."
        );
    });
}


document.querySelector('.Download-History').addEventListener('click', function () {
    window.location.href = this.dataset.url;
});
