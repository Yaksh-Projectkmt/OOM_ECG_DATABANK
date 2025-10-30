// ui js 
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
const plotViewer = document.getElementById('plotViewer');

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
    plotViewer.style.display = 'block';
});
closePlotBtn.addEventListener('click', () => {
    plotViewer.style.display = 'none';
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

function handleFileSelect(e) {
    const file = e.target.files[0];
    if (file) {
        processFile(file);
    }
}

function processFile(file) {
    if (!file) return;

    // Reset UI before processing
    uploadTitle.textContent = 'Processing...';
    fileInfo.style.display = 'none';
    loadingOverlay.style.display = 'flex';

    // Clear previous data
    uploadedImageData = null;
    uploadedImage.src = '';
    document.getElementById('TOtalRecords').textContent = '-';
    document.getElementById('leadvalue').textContent = '-';

    // Show filename immediately
    fileName.textContent = file.name;

    // Detect type early
    const type = file.type;
    currentFileType = null;

    if (type.startsWith('image/')) {
        const reader = new FileReader();
        reader.onload = (e) => {
            uploadedImageData = e.target.result;
            uploadedImage.src = uploadedImageData;
        };
        reader.readAsDataURL(file);
    }

    setTimeout(() => {
        loadingOverlay.style.display = 'none';
        uploadTitle.textContent = 'Drop your file here';

        // Trigger correct analysis section
        if (type.startsWith('image/')) {
            showAnalysis('image', file);
        } else if (type === 'application/pdf') {
            showAnalysis('pdf', file);
        } else if (type === 'text/csv' || file.name.toLowerCase().endsWith('.csv')) {
            showAnalysis('csv', file);
        } else {
            defaultState.style.display = 'block';
        }

        // Update file icon safely
        if (typeof iconSVG !== 'undefined' && iconSVG[currentFileType]) {
            fileIcon.innerHTML = iconSVG[currentFileType];
        }

        fileInfo.style.display = 'flex';
    }, 500); // shorter delay is enough
}


function showAnalysis(type, file = null) {
    currentFileType = type;

    // Hide all sections first
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

        // If file is provided, read it immediately
        if (file) {
            const reader = new FileReader();
            reader.onload = function (event) {
                const csvText = event.target.result.trim();
                if (!csvText) return;

                const lines = csvText.split('\n').filter(line => line.trim());
                const headers = lines[0].split(',').map(h => h.trim());
                const data = lines.slice(1).map(line => line.split(',').map(v => v.trim()));

                // Update metrics
                document.getElementById('TOtalRecords').textContent = data.length;
                document.getElementById('leadvalue').textContent = headers.length; // count columns
            };
            reader.readAsText(file);
        }
    }
}



function resetUpload() {
    currentFileType = null;
    uploadedImageData = null;
    fileInput.value = '';
    fileInfo.style.display = 'none';
    imageViewer.style.display = 'none';
    plotViewer.style.display = 'none';

    defaultState.style.display = 'block';
    imageAnalysis.style.display = 'none';
    pdfAnalysis.style.display = 'none';
    csvAnalysis.style.display = 'none';
}


// functionality js 
function uploadimage() {
    const fileInput = document.getElementById('fileInput');
    const section = document.getElementById('uploadTitle')?.textContent;

    if (!fileInput || !section) {
        console.error('Error: fileInput or uploadTitle element not found.');
        alertSystem.error('Error', 'Required elements not found.');
        return;
    }

    const file = fileInput.files[0];
    if (!file) {
        alertSystem.info('Warning', 'Please select an image file.');
        return;
    }

    const fileType = file.name.split('.').pop().toLowerCase();
    const validImageTypes = ['jpg', 'jpeg', 'png' ,'pdf'];

    // Validate image type
    if (!validImageTypes.includes(fileType)) {
        alertSystem.warning('Warning', `Invalid file type. Please upload only: ${validImageTypes.join(', ')}`);
        return;
    }

    // Prepare form data
    const formData = new FormData();
    formData.append('file', file);

    // Upload image
    fetch(uploadFileUrl, {
        method: 'POST',
        body: formData,
        headers: {
            'X-CSRFToken': document.querySelector("input[name='csrfmiddlewaretoken']")?.value
        },
        credentials: 'same-origin'
    })
    .then(response => response.json())
    .then(data => {
        if (data.error) {
            alertSystem.error('Error', `Image upload failed: ${data.error}`);
            return;
        }

        // Image uploaded successfully
        alertSystem.success('Success', 'Image uploaded successfully!');

        // Optionally: run post-processing after upload
        if (typeof processImage === 'function') {
            processImage(file.name)
                .catch(err => console.error('Process Image Error:', err));
        }
    })
    .catch(error => {
        console.error('Upload Error:', error);
        alertSystem.error('Error', `Upload failed: ${error.message}`);
    });
}

// TMT Upload Function — Handles only PDF uploads and ZIP response
function uploadTMTFile() {
    const fileInput = document.getElementById('fileInput');
    const section = document.getElementById('uploadTitle')?.textContent;

    if (!fileInput || !section) {
        console.error('Error: fileInput or uploadTitle element not found.');
        alertSystem.error('Error', 'Required elements not found.');
        return;
    }

    const file = fileInput.files[0];
    if (!file) {
        alertSystem.info('Warning', 'Please select a TMT PDF file.');
        return;
    }

    const fileType = file.name.split('.').pop().toLowerCase();
    if (fileType !== 'pdf') {
        alertSystem.warning('Warning', 'Invalid file type. Please upload only PDF files for TMT Data.');
        return;
    }

    const formData = new FormData();
    formData.append('file', file);

    fetch('/analysis/upload_tmt_pdf/', {
        method: 'POST',
        body: formData,
        headers: {
            'X-CSRFToken': document.querySelector("input[name='csrfmiddlewaretoken']")?.value
        },
        credentials: 'same-origin'
    })
    .then(response => {
        const contentType = response.headers.get('content-type') || '';
        if (contentType.includes('application/zip') || contentType.includes('application/octet-stream')) {
            return response.blob();
        }
        if (contentType.includes('application/json')) {
            return response.json();
        }
        throw new Error('Unexpected response type for TMT upload');
    })
    .then(data => {

        // Handle JSON error response
        if (data && data.error) {
            alertSystem.error('Error', `TMT Upload Error: ${data.error}`);
            return;
        }

        // Handle ZIP blob response
        if (data instanceof Blob) {
            const zipUrl = window.URL.createObjectURL(data);
            const a = document.createElement('a');
            a.href = zipUrl;
            a.download = `${file.name.replace('.pdf', '')}_processed.zip`;
            document.body.appendChild(a);
            a.click();
            document.body.removeChild(a);
            window.URL.revokeObjectURL(zipUrl);

            setTimeout(() => {
                alertSystem.success('Success', 'TMT PDF processed and ZIP downloaded successfully.');
                hide(); // Reset file input and UI
            }, 100);
        }
    })
    .catch(error => {
        console.error('TMT Upload Error:', error);
        alertSystem.error('Error', `TMT upload failed: ${error.message}`);
    });
}
async function uploadCSVFile() {
    const fileInput = document.getElementById('fileInput');
    const arrhythmiaSelect = document.getElementById('arrhythmiaSelect');
    const versionSelect = document.getElementById('versionSelect');
    const leadValueElement = document.getElementById('leadvalue');

    if (!fileInput || !fileInput.files[0]) {
        alertSystem.info('Info', 'Please select a CSV file.');
        return;
    }

    const file = fileInput.files[0];

    // Prepare form data
    const formData = new FormData();
    formData.append('file', file);

    try {
        const response = await fetch(uploadFileUrl, {
            method: 'POST',
            body: formData,
            headers: { 'X-CSRFToken': document.querySelector("input[name='csrfmiddlewaretoken']")?.value },
            credentials: 'same-origin'
        });

        if (!response.ok) throw new Error(`HTTP error! Status: ${response.status}`);
        const data = await response.json();

        uploadedFileName = file.name;
        alertSystem.success('Success', 'CSV uploaded successfully!');

        // Get values for arrhythmia processing
        const category = arrhythmiaSelect.value;
        const modelFileName = versionSelect.value;
        const leadType = leadValueElement.textContent.trim();

        if (!category || !modelFileName || !leadType) {
            alertSystem.info('Info', 'Please select arrhythmia, version, and lead type.');
            return;
        }

        // Call processArrhythmia with correct URL
        processArrhythmia(category, modelFileName, leadType);

    } catch (err) {
        console.error(err);
        alertSystem.error('Error', `Upload failed: ${err.message}`);
    }
}

// Process arrhythmia function
function processArrhythmia(category, modelFileName, leadType) {
    if (!uploadedFileName) {
        alertSystem.info('Info', 'No uploaded file found!');
        return;
    }

    // Use placeholder-based URL like old JS
    const url = processArrhythmiaUrl
        .replace('CATEGORY_PLACEHOLDER', encodeURIComponent(category))
        .replace('FILENAME_PLACEHOLDER', encodeURIComponent(modelFileName))
        .replace('IMAGE_NAME_PLACEHOLDER', encodeURIComponent(uploadedFileName))
        .replace('LEAD_TYPE_PLACEHOLDER', encodeURIComponent(leadType));

    fetch(url, {
        method: 'POST',
        headers: { 'X-CSRFToken': document.querySelector("input[name='csrfmiddlewaretoken']")?.value },
        credentials: 'same-origin'
    })
    .then(res => res.json())
    .then(data => {
        if (data.error) throw new Error(data.error);
        alertSystem.success('Success', `Arrhythmia Check Result: ${data.result}`);
        window.location.href = '/analysis/download_result/';
    })
    .catch(err => {
        console.error(err);
        alertSystem.error('Error', `Failed: ${err.message}`);
    });
}

// function uploadCSVFile() {
//     const fileInput = document.getElementById('fileInput');
//     const section = document.getElementById('uploadTitle')?.textContent;

//     if (!fileInput) {
//         console.error('Error: fileInput element not found.');
//         alertSystem.error('Error','File input not found.');
//         return;
//     }

//     const file = fileInput.files[0];
//     if (!file) {
//         alertSystem.info('Warning','Please select a CSV file.');
//         return;
//     }

//     const fileType = file.name.split('.').pop().toLowerCase();
//     if (fileType !== 'csv') {
//         alertSystem.warning('Warning', 'Only CSV files are allowed.');
//         return;
//     }

//     // Prepare form data
//     const formData = new FormData();
//     formData.append('file', file);

//     fetch(uploadFileUrl, {  // uploadFileUrl should point to your CSV upload endpoint
//         method: 'POST',
//         body: formData,
//         headers: { 'X-CSRFToken': document.querySelector("input[name='csrfmiddlewaretoken']")?.value },
//         credentials: 'same-origin'
//     })
//     .then(response => response.json())
//     .then(data => {

//         uploadedFileName = file.name;
//         uploadedLeadCount = data.lead_count; // optional if your backend returns it
//         uploadedPlotImage = data.plot_image; // optional if your backend returns it

//         alertSystem.success('Success','CSV file uploaded successfully!');
//     })
//     .catch(error => {
//         console.error('Upload Error:', error);
//         alertSystem.error('Error',`Upload failed: ${error.message}`);
//     });
// }
// // Process arrhythmia for a file
// function processArrhythmia(category, filename) {
//     let leadType = getSelectedLead();
//     if (!leadType) {
//         alertSystem.info('info',"Please select a lead type.");
//         return;
//     }
//     if (!uploadedFileName) {
//         alertSystem.info('info',"No uploaded file found!");
//         return;
//     }

//     //  Hide modal before processing
//     const modalElement = document.getElementById('fileModal');
//     if (modalElement) {
//         const modal = bootstrap.Modal.getInstance(modalElement) || new bootstrap.Modal(modalElement);
//         modal.hide();
//     }

//     //  Show loader after modal is hidden
//     setTimeout(() => {
//         showLoader('Processing Arrhythmia... Please Wait');

//         const url = processArrhythmiaUrl
//             .replace('CATEGORY_PLACEHOLDER', category)
//             .replace('FILENAME_PLACEHOLDER', filename)
//             .replace('IMAGE_NAME_PLACEHOLDER', uploadedFileName)
//             .replace('LEAD_TYPE_PLACEHOLDER', leadType);

//         fetch(url, {
//             method: 'POST',
//             headers: { 'X-CSRFToken': document.querySelector("input[name='csrfmiddlewaretoken']")?.value },
//             credentials: 'same-origin'
//         })
//             .then(response => {
//                 if (!response.ok) {
//                     throw new Error(`HTTP error! Status: ${response.status}`);
//                 }
//                 const contentType = response.headers.get('content-type');
//                 if (!contentType || !contentType.includes('application/json')) {
//                     throw new Error(`Unexpected response type: ${contentType || 'unknown'}. Expected JSON.`);
//                 }
//                 return response.json();
//             })
//             .then(data => {
//                 if (data.error) {
//                     throw new Error(`Server error: ${data.error}`);
//                 }
//                 const downloadUrl = '/analysis/download_result/';
//                 window.location.href = downloadUrl;

//            
//                 setTimeout(() => {
//                     alertSystem.success('Success',`Arrhythmia Check Result: ${data.result}`);
//                     hide();
//                 }, 100);
//             })
//             .catch(error => {
//                 console.error('Processing error:', error);
//                 hide();
//            
//                 setTimeout(() => {
//                     alertSystem.error('Error',`Failed to process arrhythmia: ${error.message}`);
//                 }, 100);
//             });

//     }, 300); // small delay to allow modal to close cleanly
// }

// Process uploaded ECG image
function processImage(filename) {
    return new Promise((resolve, reject) => {
 
        const url = processImageUrl.replace('FILENAME_PLACEHOLDER', filename);

        fetch(url)
            .then(response => {
                const contentType = response.headers.get('content-type');
                if (!contentType) throw new Error('No content type in response.');

                if (contentType.includes('application/json')) {
                    return response.json().then(jsonData => {
                        if (jsonData.error) {
                            const errorMessage = jsonData.error.toLowerCase();
                            if (errorMessage.includes('no ecg')) {
                                throw new Error('No ECG detected in the uploaded image.');
                            } else if (errorMessage.includes('artifact')) {
                                throw new Error('Artifacts detected in the ECG.');
                            } else {
                                throw new Error(jsonData.error);
                            }
                        }
                        return jsonData;
                    });
                }
                if (contentType.includes('application/zip') || 
                    contentType.includes('application/octet-stream') || 
                    contentType.includes('application/x-zip-compressed')) {
                    return response.blob();
                }
                if (contentType.includes('text/html')) {
                    return response.text().then(html => {
                        throw new Error(`Unexpected HTML response: ${html}`);
                    });
                }
                throw new Error(`Unsupported content type: ${contentType}`);
            })
            .then(data => {
                
                const outputButton = document.getElementById('showOutputButton');
                if (outputButton) outputButton.style.display = 'block';

                if (data instanceof Blob) {
                    const blobUrl = URL.createObjectURL(data);
                    const a = document.createElement('a');
                    a.href = blobUrl;
                    a.download = `${filename}_results.zip`;
                    a.style.display = 'none';
                    document.body.appendChild(a);
                    a.click();
                    URL.revokeObjectURL(blobUrl);
                    document.body.removeChild(a);
                } else if (data.image_url) {
                    const outputButton = document.getElementById('showOutputButton');
                    if (outputButton) outputButton.setAttribute('data-image', data.image_url);
                } else {
                    throw new Error('Unexpected data format in response.');
                }
                resolve();
            })
            .catch(error => {
                console.error('Error while processing image:', error);
                const outputButton = document.getElementById('showOutputButton');
                if (outputButton) outputButton.style.display = 'none';


                setTimeout(() => {
                    if (error.message.includes('No ECG detected')) {
                        alertSystem.info('info','No ECG signal found in the uploaded image. Processing stopped.');
                        hide();
                    } else if (error.message.includes('Artifacts detected')) {
                        alertSystem.info('info','Artifacts detected in the ECG. Processing stopped. Please upload a cleaner image.');
                        hide();
                    } else {
                        alertSystem.info('info',`Processing stopped due to error: ${error.message}`);
                        hide();
                    }
                }, 100);

                reject(error);
            });
    });
    
}

// Upload and plot ECG data
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

        const plotDiv = document.getElementById('plotModal');
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
function plotECG(data) {
    const plotDiv = document.getElementById('plot');
    if (!plotDiv) {
        console.error('Error: No DOM element with id "plot" exists on the page.');
        alertSystem.error('Error', 'Plot container not found. Please try again.');
        return;
    }

    const leadNames = Object.keys(data.leads);
    if (!leadNames.length) {
        console.error('Error: No lead data found in response.');
        alertSystem.error('Error', 'No ECG lead data available.');
        return;
    }

    const traces = [];
    const gridRows = leadNames.length;

    leadNames.forEach((lead, idx) => {
        const x = data.leads[lead].x;
        const y = data.leads[lead].y;
        const xAxisName = `x${idx + 1}`;
        const yAxisName = `y${idx + 1}`;

        traces.push({
            x: x,
            y: y,
            name: lead.toUpperCase(),
            type: 'scatter',
            mode: 'lines',
            line: { width: 1.2, color: 'black' },
            xaxis: xAxisName,
            yaxis: yAxisName,

        });
    });

    const layout = {
        grid: { rows: gridRows, columns: 1, pattern: 'independent' },
        height: 450 * gridRows, // Reduced height per subplot for better fit
        margin: { t: 40, b: 40, l: 50, r: 40 },
        plot_bgcolor: document.body.dataset.theme === 'dark' ? '#1e1e2f' : 'white',
        paper_bgcolor: document.body.dataset.theme === 'dark' ? '#1e1e2f' : 'white',
        font: { color: document.body.dataset.theme === 'dark' ? '#ffffff' : '#000000' },
        showlegend: false // Disable legends
    };

    leadNames.forEach((lead, idx) => {
        const axisX = `xaxis${idx + 1}`;
        const axisY = `yaxis${idx + 1}`;
        layout[axisX] = {
            range: [0, data.leads[lead].x.length],
            title: { text: 'Time Index', standoff: 20, font: { size: 12 } },
            showgrid: true,
            gridcolor: 'rgba(255, 0, 0, 0.7)',
            gridwidth: 0.6,
            zeroline: false,
            dtick: 100,
            tickfont: { size: 12 },
            minor: { showgrid: true, gridcolor: 'rgba(255, 192, 203, 0.6)', gridwidth: 0.5 }
        };
        layout[axisY] = {
            range: [0,4],
            title: { text: lead.toUpperCase(), standoff: 15, font: { size: 14 } },
            showgrid: true,
            gridcolor: 'rgba(255, 0, 0, 0.7)',
            gridwidth: 0.6,
            zeroline: false,
            dtick: 0.5,
            tickfont: { size: 12 },
            minor: { showgrid: true, gridcolor: 'rgba(255, 192, 203, 0.6)', gridwidth: 0.3 }
        };
    });

    const config = {
        responsive: true,
        displaylogo: false,
        modeBarButtonsToAdd: ['pan2d', 'zoom2d', 'autoScale2d', 'resetScale2d', 'zoomIn2d', 'zoomOut2d', 'toImage']
    };

    Plotly.newPlot('plot', traces, layout, config);

    plotDiv.on('plotly_selected', function (eventData) {
        if (eventData) {
            window.selectedData = eventData.points.map(pt => ({ x: pt.x, y: pt.y }));
        }
    });
}
const modal = document.getElementById('plotModal');
const closeBtn = document.getElementById('closePlotBtn');
closeBtn.addEventListener('click', () => {
    modal.style.display = 'none';
});

// Close when clicking outside the modal content
window.addEventListener('click', (e) => {
    if (e.target == modal) {
        modal.style.display = 'none';
    }
});