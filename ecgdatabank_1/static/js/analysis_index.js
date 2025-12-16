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

const viewImageBtn = document.getElementById('viewImageBtn');
const closeImageBtn = document.getElementById('closeImageBtn');
const imageViewer = document.getElementById('imageViewer');
const uploadedImage = document.getElementById('uploadedImage');

const viewPlotBtn = document.getElementById('viewPlotBtn');
const closePlotBtn = document.getElementById('closePlotBtn');
const plotModal = document.getElementById('plotModal');
const pageLoader = document.getElementById('page-loader');

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

fileInput.addEventListener('change', handleFileSelect);
removeBtn.addEventListener('click', resetUpload);
viewImageBtn.addEventListener('click', () => {
    imageViewer.style.display = 'block';
});
closeImageBtn.addEventListener('click', () => {
    imageViewer.style.display = 'none';
});
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
function processFile(file) {

fileName.textContent = file.name;
uploadTitle.textContent = 'Processing...';
loadingOverlay.style.display = 'flex';

// Preview image (client side)
if (file.type.startsWith('image/')) {
    const reader = new FileReader();
    reader.onload = (e) => {
        uploadedImageData = e.target.result;
        uploadedImage.src = uploadedImageData;
    };
    reader.readAsDataURL(file);
}

// STEP 1: Upload file to Django ? store server filename
const formData = new FormData();
formData.append("file", file);

fetch(uploadFileUrl, {
    method: "POST",
    body: formData,
    headers: {
        "X-CSRFToken": document.querySelector("input[name='csrfmiddlewaretoken']")?.value
    }
})
.then(res => res.json())
.then(data => {
    if (data.error) {
        alertSystem.error("Error", data.error);
        return;
    }
    console.log("UPLOAD API RESPONSE:", data);
    // Save backend-generated filename
    savedServerFilename = data.filename.split("/").pop();
    console.log("Saved server filename:", savedServerFilename);

})
.catch(err => {
    console.error("Upload error:", err);
    alertSystem.error("Error", "Upload failed.");
});

// STEP 2: After UI loading animation
setTimeout(() => {
    loadingOverlay.style.display = 'none';
    uploadTitle.textContent = 'Drop your file here';
    

    const type = file.type;

    if (type.startsWith('image/')) {
        showAnalysis('image');

    } else if (type === 'application/pdf') {
        showAnalysis('pdf');
        uploadedTmtFile = file;

    } else if (type === 'text/csv' || file.name.endsWith('.csv')) {
        showAnalysis('csv');

    // Parse CSV & detect lead type
    const reader = new FileReader();
    reader.onload = function(evt) {
        const lines = evt.target.result.split(/\r?\n/);
        const header = lines[0].split(",").map(h => h.trim());
        const colCount = header.length;

        document.getElementById("TOtalRecords").textContent = lines.length - 1;

        let leadType = "Unknown";
        if (colCount === 2 || colCount === 3) leadType = "2_Lead";
        else if (colCount === 7 || colCount === 8) leadType = "7_Lead";
        else if (colCount === 12) leadType = "12_Lead";
        else if (colCount === 13) leadType = "12_Lead";

        document.getElementById("leadvalue").textContent = leadType;
    };
        reader.readAsText(file);
    }

        fileIcon.innerHTML = iconSVG[currentFileType];
        fileInfo.style.display = 'flex';

        }, 1500);
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
function runImageUpload() {
    if (!savedServerFilename) {
        alertSystem.info("info", "Please upload an ECG image first.");
        return;
    }

    uploadimage(savedServerFilename);
}

function updateWalletBalance() {
    fetch("/auth/get-balance/")
        .then(res => res.json())
        .then(data => {
            document.getElementById("walletBalance").textContent =
                data.balance || "0.00";
        });
}

function uploadimage(savedServerFilename) {
    return new Promise((resolve, reject) => {

        // --- SHOW PAGE LOADER ---
        if (pageLoader) pageLoader.style.display = 'flex';

        document.getElementById("outputPlaceholder").style.display = "flex";
        document.getElementById("outputImage").style.display = "none";
        document.getElementById("DownloadImageBtn").style.display = "none";

        const url = processImageUrl.replace("FILENAME_PLACEHOLDER", savedServerFilename);
        console.log("Uploading image to:", url);

        fetch(url)
            .then(response => {
                const contentType = response.headers.get("content-type");
                if (!contentType) throw new Error("No content type in response.");

                // JSON response (success / error)
                if (contentType.includes("application/json")) {
                    return response.json().then(jsonData => {
                        if (jsonData.error) {
                            if (jsonData.error.toLowerCase().includes("no ecg")) {
                                throw new Error("No ECG detected in the uploaded image.");
                            }
                            if (jsonData.error.toLowerCase().includes("artifact")) {
                                throw new Error("Artifacts detected in the ECG.");
                            }
                            throw new Error(jsonData.error);
                        }
                        return jsonData;
                    });
                }

                // ZIP (ignored)
                if (
                    contentType.includes("application/zip") ||
                    contentType.includes("application/octet-stream")
                ) {
                    return response.blob();
                }

                // HTML (unexpected)
                if (contentType.includes("text/html")) {
                    throw new Error("Unexpected HTML response.");
                }

                throw new Error(`Unsupported content type: ${contentType}`);
            })
            .then(data => {

                // --- HIDE PAGE LOADER ---
                if (pageLoader) pageLoader.style.display = "none";
                document.getElementById("outputPlaceholder").style.display = "none";
                const outputImage = document.getElementById("outputImage");
                const downloadBtn = document.getElementById("DownloadImageBtn");

                const fullName = savedServerFilename.split("/").pop();
                const cleanName = fullName.substring(0, fullName.lastIndexOf(".")) || fullName;

                const processedImageUrl = `/media/analysis_tool/uploads/${cleanName}.jpg`;
                console.log("Processed Image URL:", processedImageUrl);

                // Update output image
                outputImage.src = processedImageUrl;
                outputImage.style.display = "block";

                // Enable download button
                downloadBtn.style.display = "inline-flex";
                downloadBtn.setAttribute("data-download", processedImageUrl);

                updateWalletBalance();
                resolve();
            })

            .catch(error => {
                console.error("Error image processing:", error);

                // --- HIDE PAGE LOADER (ERROR CASE) ---
                if (pageLoader) pageLoader.style.display = "none";

                setTimeout(() => {
                    if (error.message.includes("No ECG detected")) {
                        alertSystem.info("info", "No ECG signal found. Processing stopped.");
                        resetUpload();
                    } else if (error.message.includes("Artifacts detected")) {
                        alertSystem.info("info", "Artifacts detected. Please upload a cleaner ECG image.");
                    } else {
                        alertSystem.info("info", `Error: ${error.message}`);
                    }
                }, 100);

                reject(error);
            });
    });
}

document.getElementById("DownloadImageBtn").addEventListener("click", async function () {

    const url = this.getAttribute("data-download");
    if (!url) {
        alertSystem.info("info", "No processed output to download.");
        return;
    }

    // Deduct wallet BEFORE download
    const okimage = await deductWallet("pdf");
    if (!okimage) return;

    // Wallet success ? continue download
    const a = document.createElement("a");
    a.href = url;
    a.download = url.split("/").pop();
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);

    updateWalletBalance();
});

function uploadCSVFile() {

    const arrhythmia = document.getElementById("arrhythmiaSelect").value;

    if (!arrhythmia) {
        alertSystem.info("Info", "Select Arrhythmia first.");
        return;
    }

    const is_lead = document.getElementById("leadvalue").innerText;
    console.log("Detected lead type:", is_lead);

    if (pageLoader) pageLoader.style.display = "flex";

    const formData = new FormData();
    formData.append("csv_name", savedServerFilename);
    formData.append("is_lead", is_lead);

    //ONLY ONE MODEL RUNNER
    const endpoint = `/analysis/run_model_arrhythmia/${arrhythmia}/${savedServerFilename}/`;

    fetch(endpoint, {
        method: "POST",
        body: formData,
        headers: {
            "X-CSRFToken": getCSRFToken()
        }
    })
    .then(res => res.json())
    .then(data => {

        if (pageLoader) pageLoader.style.display = "none";

        console.log("Analysis result:", data);

        if (data.error) {
            alertSystem.error("Error", data.error);
            return;
        }

        alertSystem.success("Success", "Analysis completed!");

        const downloadBtn = document.getElementById("DownloadCSVBtn");
        downloadBtn.style.display = "inline-flex";

        const baseName = savedServerFilename.replace(".csv", "");
        const processedCSV = `/media/analysis_tool/analysis_result/${baseName}_result.csv`;

        downloadBtn.setAttribute("data-download", processedCSV);
    })
    .catch(err => {

        console.error("CSV Error:", err);
        if (pageLoader) pageLoader.style.display = "none";

        alertSystem.error("Error", "Something went wrong while analyzing the CSV.");
    });
}

document.getElementById("DownloadCSVBtn").addEventListener("click", async function () {

    const patientId  = savedServerFilename;
    const arrhythmia = document.getElementById("arrhythmiaSelect").value;
    const leadType   = document.getElementById("leadvalue").innerText;
    
    // Wallet deduction (PDF)
    const ok = await deductWallet("pdf", patientId, arrhythmia, leadType);
    if (!ok) return;

    // Download PDF directly (NO ZIP, NO LOG API)
    const downloadUrl =
        `/analysis/download_patient_pdf/?patient_id=${encodeURIComponent(patientId)}` +
        `&arrhythmia=${encodeURIComponent(arrhythmia)}` +
        `&lead_type=${encodeURIComponent(leadType)}`;

    // Open PDF (browser will download or preview)
    window.open(downloadUrl, "_blank");

    //  Update wallet UI
    updateWalletBalance();
});
function uploadTMTFile() {

    if (!uploadedTmtFile) {
        alertSystem.info("info", "Please upload a PDF first.");
        return;
    }

    // Show loader
    if (pageLoader) pageLoader.style.display = "flex";

    const formData = new FormData();
    formData.append("file", uploadedTmtFile);

    fetch("/analysis/upload_tmt_pdf/", {
        method: "POST",
        body: formData,
        headers: { "X-CSRFToken": getCSRFToken() }
    })
    .then(res => res.json())
    .then(data => {

        // Hide loader
        if (pageLoader) pageLoader.style.display = "none";

        if (data.error) {
            alertSystem.error("Error", data.error);
            return;
        }

        console.log("TMT Response:", data);

        // Save ZIP for download
        window.tmtZipFile = data.zip_file;

        // Remove preview UI (ensuring no preview appears)
        const container = document.getElementById("tmtPreviewContainer");
        if (container) container.style.display = "none";

        const imagesDiv = document.getElementById("tmtPreviewImages");
        if (imagesDiv) imagesDiv.innerHTML = "";

        // Enable download button
        document.getElementById("DownloadTmtBtn").style.display = "inline-flex";

        alertSystem.success("Success", "TMT Report processed!");
    })
    .catch(err => {

        console.error("TMT ERROR:", err);

        // Hide loader on error
        if (pageLoader) pageLoader.style.display = "none";

        alertSystem.error("Error", "Failed to process TMT PDF.");
    });
}

document.getElementById("DownloadTmtBtn").addEventListener("click", async () => {

    if (!window.tmtZipFile) {
        alertSystem.info("info", "No report ZIP available.");
        return;
    }

    const ok = await deductWallet("pdf");
    if (!ok) return;

    const url = `/media/analysis_tool/uploads/${window.tmtZipFile}`;
    const a = document.createElement("a");
    a.href = url;
    a.download = window.tmtZipFile;

    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);

    updateWalletBalance();
});

async function deductWallet(fileType, patientId, arrhythmia, leadType) {
    try {
        // Build payload WITHOUT null/undefined values
        const payload = { file_type: fileType };

        if (patientId !== undefined && patientId !== null && patientId !== "")
            payload.patient_id = patientId;

        if (arrhythmia !== undefined && arrhythmia !== null && arrhythmia !== "")
            payload.arrhythmia = arrhythmia;

        if (leadType !== undefined && leadType !== null && leadType !== "")
            payload.lead_type = leadType;

        console.log("Deduct Wallet Payload:", payload);

        const response = await fetch('/ommecgdata/deduct_wallet_before_download/', {
            method: 'POST',
            credentials: 'same-origin',
            headers: {
                'Content-Type': 'application/json',
                'X-CSRFToken': getCSRFToken()
            },
            body: JSON.stringify(payload)
        });

        const data = await response.json();
        console.log("Wallet deduction response:", data);

        if (response.ok && (data.status === "success" || data.message === "OK" || data.success === true)) {
            return true;
        }

        alertSystem.error("Wallet Error", data.error || "Wallet Error");
        return false;

    } catch (err) {
        console.error("Wallet Error:", err);
        alertSystem.error("Error", "Failed to communicate with server.");
        return false;
    }
}
