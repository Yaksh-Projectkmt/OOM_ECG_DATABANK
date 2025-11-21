const leads = ['I', 'II', 'III', 'aVR', 'aVL', 'aVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6'];
const arrhythmiaDict = {
'Myocardial Infarction': ['T-wave abnormalities', 'Inferior MI', 'Lateral MI'],
'Atrial Fibrillation & Atrial Flutter': ['AFIB', 'Aflutter', 'AFL'],
'HeartBlock': ['I DEGREE', 'MOBITZ-I', 'MOBITZ-II', 'III Degree'],
'Junctional Rhythm': ['Junctional Bradycardia', 'Junctional Rhythm'],
'Premature Atrial Contraction': ['PAC-Isolated', 'PAC-Bigeminy', 'PAC-Couplet', 'PAC-Triplet',
                                'SVT', 'PAC-Trigeminy', 'PAC-Quadrigeminy'],
'Premature Ventricular Contraction': ['AIVR', 'PVC-Bigeminy', 'PVC-Couplet', 'PVC-Isolated',
                                                'PVC-Quadrigeminy', 'NSVT', 'PVC-Trigeminy',
                                                'PVC-Triplet', 'IVR', 'VT'],
'Ventricular Fibrillation and Asystole': ['VFIB', 'VFL', 'ASYSTOLE'],
'Noise':['Noise'],
'LBBB':['LBBB','RBBB'],
'Artifacts': ['Artifacts'],
'SINUS-ARR': ['SINUS-ARR'],
'ShortPause': ['Short Pause', 'Long Pause'],
'TC': ['TC'],
'WIDE-QRS': ['WIDE-QRS'],
'Abnormal': ['ABNORMAL'],
'Normal':['Normal'],
'Others':['Others']
};

function showUploadSection() {
    document.getElementById('uploadSection').style.display = 'block';
    document.getElementById('morphologySection').style.display = 'none';
    document.getElementById('buttonSection').style.display = 'none';
    document.getElementById('ecgForm').style.display = 'none';
    document.getElementById('plot').style.display = 'none';
    document.getElementById('mainTitle').style.display = 'flex';
}

function showMorphologySection() {
    document.getElementById('morphologySection').style.display = 'block';
    document.getElementById('uploadSection').style.display = 'none';
    document.getElementById('buttonSection').style.display = 'none';
    document.getElementById('ecgForm').style.display = 'none';
    document.getElementById('plot').style.display = 'none';
    initCanvas();
    document.getElementById('mainTitle').style.display = 'none';
}

function showFileName(input) {
    const fileLabel = document.getElementById('fileLabel');
    fileLabel.textContent = input.files.length > 0 ? input.files[0].name : 'Click to Upload ECG Image';
}

function handleFileSelection(input) {
    // Update the file name display
    showFileName(input);
    // Trigger upload if a file is selected
    if (input.files.length > 0) {
        uploadFile();
    }
}

function handleContainerClick(event) {
    event.stopPropagation(); // Prevent event from reaching the label
    document.getElementById('fileInput').click(); // Trigger file input dialog
}

async function uploadFile() {
    const fileInput = document.getElementById('fileInput');
    const file = fileInput.files[0];
    const loader = document.getElementById('page-loader');

    const allowedExtensions = ['jpg', 'jpeg', 'png'];
    const fileExtension = file.name.split('.').pop().toLowerCase();
    if (!allowedExtensions.includes(fileExtension)) {
        alertSystem.error('Error', `File "${file.name}" not allowed.`);
        return;
    }

    loader.style.display = 'block';

    try {
        const img = new Image();
        const objectURL = URL.createObjectURL(file);
        await new Promise((resolve) => {
            img.onload = () => {
                URL.revokeObjectURL(objectURL);
                resolve();
            };
            img.src = objectURL;
        });

        const formData = new FormData();
        formData.append('file', file);

        const response = await fetch(uploadFileUrl, {
            method: 'POST',
            body: formData,
            headers: {
                'X-CSRFToken': document.querySelector("input[name='csrfmiddlewaretoken']").value
            },
            credentials: 'same-origin'
        });

        const data = await response.json();
        loader.style.display = 'none';

        if (data.error) {
            alertSystem.error('Error', data.error);
            return;
        }

        if (data.image_url && data.csv_file) {
            alertSystem.success('Success', `File "${file.name}" uploaded successfully!`);
            await fetchCSVData(data.csv_file);
        }
    } catch (error) {
        loader.style.display = 'none';
        console.error('Upload Error:', error);
        alertSystem.error('Error', 'File upload failed. Please try again.');
    }
}

async function fetchCSVData(csvFileName) {
    const loader = document.getElementById('page-loader');
    loader.style.display = 'block';

    try {
        const response = await fetch('/morphology/csv_data/', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
                'X-CSRFToken': document.querySelector("input[name='csrfmiddlewaretoken']").value
            },
            body: JSON.stringify({ csv_file: csvFileName }),
            credentials: 'same-origin'
        });

        const data = await response.json();
        loader.style.display = 'none';

        if (data.error) {
            alertSystem.error('Error', data.error);
            return;
        }

        plotECG(data);
        document.getElementById('ecgForm').style.display = 'block';
        document.getElementById('plot').style.display = 'block';
    } catch (error) {
        loader.style.display = 'none';
        console.error('Fetch CSV Error:', error);
        alertSystem.error('Error', 'Failed to fetch ECG data.');
    }
}

function plotECG(data) {
    const trace = {
        x: data.x,
        y: data.y,
        type: 'scatter',
        line: { color: '#22c55e', width: 2 },
        mode: 'lines'
    };

    const layout = {
        xaxis: {
            title: 'Time Index',
            titlefont: { size: 14, color: 'var(--text)' },
            tickfont: { size: 12 }
        },
        yaxis: {
            title: 'ECG Value (mV)',
            titlefont: { size: 14, color: 'var(--text)' },
            tickfont: { size: 12 }
        },
        plot_bgcolor: 'var(--background)',
        paper_bgcolor: 'var(--background)',
        margin: { t: 40, b: 60, l: 60, r: 20 },
        showlegend: false
    };

    const config = {
        responsive: true,
        displaylogo: false,
        modeBarButtons: [['autoScale2d', 'resetScale2d'], ['zoomIn2d', 'zoomOut2d', 'toImage']],
        toImageButtonOptions: {
            format: 'png',
            filename: 'ecg_signal'
        }
    };

    Plotly.newPlot('ecgChart', [trace], layout, config);
    window.dispatchEvent(new Event('resize'));

    document.getElementById('ecgChart').on('plotly_selected', (eventData) => {
        if (eventData) {
            window.selectedData = eventData.points.map(pt => ({
                x: pt.x,
                y: pt.y
            }));
        }
    });
}

async function uploadECG() {
    const form = document.getElementById('ecgForm');
    const formData = new FormData(form);
    const loader = document.getElementById('page-loader');

    loader.style.display = 'block';

    try {
        const response = await fetch('/morphology/upload_ecg/', {
            method: 'POST',
            body: formData,
            headers: {
                'X-CSRFToken': document.querySelector("input[name='csrfmiddlewaretoken']").value
            },
            credentials: 'same-origin'
        });

        const data = await response.json();
        loader.style.display = 'none';

        if (data.message) {
            alertSystem.success('Success', data.message);
            document.getElementById('leads').value = '';
            document.getElementById('arrhythmia').value = '';
            document.getElementById('subArrhythmia').value = '';
        } else if (data.error) {
            alertSystem.error('Error', data.error);
        }
    } catch (error) {
        loader.style.display = 'none';
        console.error('Upload ECG Error:', error);
        alertSystem.error('Error', 'Failed to upload ECG data.');
    }
}

async function handleECGSubmit() {
    const lead = document.getElementById('leads').value.trim();
    const arrhythmia = document.getElementById('arrhythmia').value.trim();
    const subArrhythmia = document.getElementById('subArrhythmia').value.trim();
    const loader = document.getElementById('page-loader');

    if (!lead || !arrhythmia || !subArrhythmia) {
        alertSystem.warning('Warning', 'Please fill in all fields: Lead, Arrhythmia, and Sub-arrhythmia.');
        return;
    }

    loader.style.display = 'block';

    try {
        const form = document.getElementById('ecgForm');
        const formData = new FormData(form);
        const response = await fetch('/morphology/upload_ecg/', {
            method: 'POST',
            body: formData,
            headers: {
                'X-CSRFToken': document.querySelector("input[name='csrfmiddlewaretoken']").value
            },
            credentials: 'same-origin'
        });

        const data = await response.json();
        loader.style.display = 'none';

        if (data.message) {
            alertSystem.success('Success', data.message);
            hide();
        } else if (data.error) {
            alertSystem.error('Error', data.error);
        }
    } catch (error) {
        loader.style.display = 'none';
        console.error('Submit ECG Error:', error);
        alertSystem.error('Error', 'Failed to submit ECG data.');
    }
}

async function hide() {
    const fileInput = document.getElementById('fileInput');
    const fileLabel = document.getElementById('fileLabel');
    const morphologySection = document.getElementById('morphologySection');
    const ecgForm = document.getElementById('ecgForm');
    const plot = document.getElementById('plot');
    const buttonSection = document.getElementById('buttonSection');
    const leadSelect = document.getElementById('leads');
    const arrhythmiaSelect = document.getElementById('arrhythmia');
    const subArrhythmiaSelect = document.getElementById('subArrhythmia');

    // Reset inputs and labels
    fileInput.value = '';
    fileLabel.textContent = 'Click to Upload ECG Image';

    // Hide sections
    morphologySection.style.display = 'none';
    ecgForm.style.display = 'none';
    if (plot.style.display !== 'none') {
        plot.style.display = 'none';
        Plotly.purge('ecgChart');
    }

    // Reset selects
    leadSelect.selectedIndex = 0;
    arrhythmiaSelect.selectedIndex = 0;
    subArrhythmiaSelect.selectedIndex = 0;

    // Show button section and main title
    buttonSection.style.display = 'block';
    document.getElementById('mainTitle').style.display = 'flex';

    // Cleanup CSVs via POST
    try {
        const response = await fetch('/morphology/remove_all_csvs/', {
            method: 'POST',
            headers: {
                'X-CSRFToken': document.querySelector("input[name='csrfmiddlewaretoken']").value,
                'Content-Type': 'application/json'
            },
            credentials: 'same-origin'
        });
        const data = await response.json();
    } catch (error) {
        console.error('Cleanup fetch error:', error);
    }
}
function originalimage() {
    const fileInput = document.getElementById('fileInput');
    const loader = document.getElementById('page-loader');

    if (fileInput.files.length > 0) {
        loader.style.display = 'flex';  // Show loader

        const file = fileInput.files[0];
        const reader = new FileReader();

        reader.onload = function(event) {
            // Create overlay
            const overlay = document.createElement('div');
            overlay.id = 'image-overlay';
            overlay.style.position = 'fixed';
            overlay.style.top = 0;
            overlay.style.left = 0;
            overlay.style.width = '100vw';
            overlay.style.height = '100vh';
            overlay.style.backgroundColor = 'rgba(0, 0, 0, 0.6)';
            overlay.style.zIndex = 999;
            overlay.style.display = 'flex';
            overlay.style.justifyContent = 'center';
            overlay.style.alignItems = 'center';
            overlay.style.transition = 'opacity 0.3s ease';
            overlay.style.opacity = 0;

            // Create modal container
            const modal = document.createElement('div');
            modal.style.position = 'relative';
            modal.style.backgroundColor = '#fff';
            modal.style.padding = '20px';
            modal.style.borderRadius = '10px';
            modal.style.boxShadow = '0 8px 16px rgba(0, 0, 0, 0.3)';
            modal.style.maxWidth = '90vw';
            modal.style.maxHeight = '90vh';
            modal.style.overflow = 'auto';
            modal.style.animation = 'fadeIn 0.3s ease';

            // Close button
            const closeButton = document.createElement('button');
            closeButton.innerHTML = '&times;';
            closeButton.style.position = 'absolute';
            closeButton.style.top = '10px';
            closeButton.style.right = '15px';
            closeButton.style.background = 'transparent';
            closeButton.style.border = 'none';
            closeButton.style.fontSize = '24px';
            closeButton.style.cursor = 'pointer';

            // Append close handler
            closeButton.onclick = () => document.body.removeChild(overlay);

            // Append ESC key close support
            const escListener = (e) => {
                if (e.key === 'Escape') {
                    document.body.removeChild(overlay);
                    document.removeEventListener('keydown', escListener);
                }
            };
            document.addEventListener('keydown', escListener);

            // Create image
            const img = new Image();
            img.src = event.target.result;
            img.style.maxWidth = '100%';
            img.style.maxHeight = '80vh';
            img.style.borderRadius = '5px';

            // Assemble
            modal.appendChild(closeButton);
            modal.appendChild(img);
            overlay.appendChild(modal);
            document.body.appendChild(overlay);

            // Trigger fade-in
            setTimeout(() => overlay.style.opacity = 1, 50);

            // Hide loader
            loader.style.display = 'none';
        };

        reader.readAsDataURL(file);
    }
}
const canvas = document.getElementById('drawingCanvas');
if (canvas) {
    canvas.style.touchAction = 'none';
    canvas.width = canvas.offsetWidth;
    canvas.height = canvas.offsetHeight;
}

let painting = false;
let rawPoints = [];
// Initialize the canvas and grid
function initCanvas() {
    if (!canvas) return;
    canvas.width = canvas.offsetWidth;
    canvas.height = canvas.offsetHeight;
    const ctx = canvas.getContext('2d');
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    drawECGGrid(ctx, canvas.width, canvas.height);
}

// Draw ECG-style grid
function drawECGGrid(ctx, width, height) {
    const majorGridSpacing = 50;  // large grid every 50px
    const minorGridSpacing = 10;  // small grid every 10px

    ctx.save();
    ctx.lineWidth = 0.7;
    ctx.strokeStyle = 'rgba(244, 210, 216, 0.92)'; // light pink minor grid
    for (let x = 0; x <= width; x += minorGridSpacing) {
        ctx.beginPath();
        ctx.moveTo(x, 0);
        ctx.lineTo(x, height);
        ctx.stroke();
    }
    for (let y = 0; y <= height; y += minorGridSpacing) {
        ctx.beginPath();
        ctx.moveTo(0, y);
        ctx.lineTo(width, y);
        ctx.stroke();
    }

    // Draw thicker major grid lines
    ctx.lineWidth = 1.3;
    ctx.strokeStyle = 'rgba(245, 121, 121, 0.93)'; // darker red major grid
    for (let x = 0; x <= width; x += majorGridSpacing) {
        ctx.beginPath();
        ctx.moveTo(x, 0);
        ctx.lineTo(x, height);
        ctx.stroke();
    }
    for (let y = 0; y <= height; y += majorGridSpacing) {
        ctx.beginPath();
        ctx.moveTo(0, y);
        ctx.lineTo(width, y);
        ctx.stroke();
    }

    ctx.restore();
}

function getPointerPos(e) {
    const rect = canvas.getBoundingClientRect();
    return { x: e.clientX - rect.left, y: e.clientY - rect.top };
}

function startDrawing(e) {
    painting = true;
    const pos = getPointerPos(e);
    rawPoints.push(pos);
    drawLoop();
}

function addPoint(e) {
    if (!painting) return;
    const pos = getPointerPos(e);
    rawPoints.push(pos);
}

function stopDrawing() {
    painting = false;
}

function drawLoop() {
    if (!painting) return;
    const ctx = canvas.getContext('2d');
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    drawECGGrid(ctx, canvas.width, canvas.height); // redraw grid each frame
    ctx.lineWidth = 3;
    ctx.lineCap = 'round';
    ctx.strokeStyle = '#FF0000';

    if (rawPoints.length > 0) {
        ctx.beginPath();
        ctx.moveTo(rawPoints[0].x, rawPoints[0].y);
        for (let i = 1; i < rawPoints.length; i++) {
            ctx.lineTo(rawPoints[i].x, rawPoints[i].y);
        }
        ctx.stroke();
    }
    requestAnimationFrame(drawLoop);
}
// Show grid immediately when page loads
window.addEventListener('load', initCanvas);

// Optional: add event listeners for drawing
canvas.addEventListener('mousedown', startDrawing);
canvas.addEventListener('mousemove', addPoint);
canvas.addEventListener('mouseup', stopDrawing);
canvas.addEventListener('mouseleave', stopDrawing);

function resetDrawing() {
    const ctx = canvas.getContext('2d');
    ctx.clearRect(0, 0, canvas.width, canvas.height);

    // Reset data arrays
    rawPoints = [];
    augmentedData = [];

    // Reset form/UI fields
    document.getElementById('replicationCount').value = '';
    document.getElementById('signalPlot').style.display = 'none';
    document.getElementById('subarrhythmiaSelect').style.display = 'none';
    document.getElementById('patientIdInput').value = '';
    document.getElementById('arrhythmiaSelect').selectedIndex = 0;
    document.getElementById('subarrhythmiaSelect').selectedIndex = 0;

    // Redraw ECG grid
    drawECGGrid(ctx, canvas.width, canvas.height);
}

function augmentSignal() {
    const count = parseInt(document.getElementById('replicationCount').value);
    if (isNaN(count) || count < 1 || count >= 10) {
        alertSystem.warning('Warning', 'Please enter a valid number between 1 and 10.');
        return;
    }
    if (rawPoints.length < 2) {
        alertSystem.warning('Warning', 'Draw the ECG signal first.');
        return;
    }

    // Calculate signal width (span of drawn points)
    const xMin = Math.min(...rawPoints.map(p => p.x));
    const xMax = Math.max(...rawPoints.map(p => p.x));
    const signalWidth = xMax - xMin;

    // Build repeated (augmented) signal segments
    augmentedData = [];
    for (let i = 0; i < count; i++) {
        rawPoints.forEach(pt => {
            augmentedData.push({
                x: pt.x + i * signalWidth, // shift horizontally
                y: pt.y                    // keep same height
            });
        });
    }

    alertSystem.success('Success', `Signal augmented ${count} times (same height preserved).`);
}


function saveData() {
    const patientId = document.getElementById('patientIdInput').value.trim();
    const arrhythmia = document.getElementById('arrhythmiaSelect').value;
    const subarrhythmia = document.getElementById('subarrhythmiaSelect').value;
    const pointsToSave = augmentedData.length > 0 ? augmentedData : rawPoints;

    if (!arrhythmia) {
        alertSystem.warning('Warning', 'Please select an Arrhythmia.');
        return;
    }

    if (!subarrhythmia) {
        alertSystem.warning('Warning', 'Please select a Subarrhythmia.');
        return;
    }
    if (!patientId) {
        alertSystem.warning('Warning', 'Please enter Patient ID.');
        return;
    }
    if (pointsToSave.length === 0) {
        alertSystem.warning('Warning', 'Please draw the signal before saving.');
        return;
    }

    fetch('/morphology/open_morphology_script/', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
            'X-CSRFToken': document.querySelector("input[name='csrfmiddlewaretoken']").value
        },
        body: JSON.stringify({
            points: pointsToSave,
            arrhythmia: arrhythmia,
            subarrhythmia: subarrhythmia,
            PatientID: patientId,
        })
    })
    .then(response => response.json())
    .then(data => {
        if (data.status === 'success') {
            alertSystem.success('Success', data.message);
            resetDrawing();
        } else {
            alertSystem.error('Error', data.message || 'Failed to save data.');
        }
    })
    .catch(error => {
        console.error('Error:', error);
        alertSystem.error('Error', 'Failed to save data.');
    });
}

function showGraph() {
    const signalPlotDiv = document.getElementById('signalPlot');
    const pointsToPlot = augmentedData.length > 0 ? augmentedData : rawPoints;

    if (pointsToPlot.length < 2) {
        alertSystem.warning('Warning', 'Draw or augment the signal first.');
        return;
    }

    // Show container
    signalPlotDiv.style.display = 'block';
    signalPlotDiv.style.overflowX = 'auto';  // allow horizontal scroll
    signalPlotDiv.style.whiteSpace = 'nowrap'; // prevent wrapping
    const containerWidth = signalPlotDiv.offsetWidth;
    const containerHeight = 400; // fixed height (you can adjust)

    const x = pointsToPlot.map(p => p.x);
    const y = pointsToPlot.map(p => canvas.height - p.y);
    const ySmoothed = lowpassFilter(y);

     // Determine signal boundaries
    const xMin = Math.min(...x);
    const xMax = Math.max(...x);
    const signalWidth = xMax - xMin;

    // Use signal width or container width (whichever is greater)
    const plotWidth = Math.max(signalWidth + 100, containerWidth);
    const plotHeight = canvas.height;
    const majorSpacing = 50;
    const minorSpacing = 10;

    const gridShapes = [];

    // Minor grid lines
    for (let i = 0; i <= plotWidth; i += minorSpacing) {
        gridShapes.push({
            type: 'line',
            x0: i, x1: i, y0: 0, y1: plotHeight,
            line: { color: 'rgba(244, 210, 216, 0.92)', width: 0.5 }
        });
    }
    for (let j = 0; j <= plotHeight; j += minorSpacing) {
        gridShapes.push({
            type: 'line',
            x0: 0, x1: plotWidth, y0: j, y1: j,
            line: { color: 'rgba(244, 210, 216, 0.92)', width: 0.5 }
        });
    }

    // Major grid lines
    for (let i = 0; i <= plotWidth; i += majorSpacing) {
        gridShapes.push({
            type: 'line',
            x0: i, x1: i, y0: 0, y1: plotHeight,
            line: { color: 'rgba(245,121,121,0.9)', width: 1 }
        });
    }
    for (let j = 0; j <= plotHeight; j += majorSpacing) {
        gridShapes.push({
            type: 'line',
            x0: 0, x1: plotWidth, y0: j, y1: j,
            line: { color: 'rgba(245,121,121,0.9)', width: 1 }
        });
    }

    // Plot with augmentation + ECG grid
    Plotly.newPlot('signalPlot', [
        {
            x, y,
            type: 'scatter',
            mode: 'lines',
            name: 'Original',
            line: { color: 'gray', width: 2 }
        },
        {
            x, y: ySmoothed,
            type: 'scatter',
            mode: 'lines',
            name: 'Filtered',
            line: { color: 'blue', width: 2 }
        }
    ], {
         width: plotWidth,
        height: plotHeight,
        plot_bgcolor: 'rgba(255,255,255,1)',
        paper_bgcolor: 'rgba(255,255,255,1)',
        margin: { t: 10, b: 10, l: 10, r: 10 },
        shapes: gridShapes,
        xaxis: {
            range: [0, plotWidth],
            showgrid: false,
            zeroline: false,
            fixedrange: true,
             visible: false, 
        },
        yaxis: {
            range: [0, plotHeight],
            showgrid: false,
            zeroline: false,
            visible: false,
            fixedrange: true,
            scaleanchor: "x",
            scaleratio: 1
        },
        autosize: false
    }, {
        displayModeBar: false,
        responsive: true
    });

 // Adjust layout on resize
    window.addEventListener('resize', () => {
        const newContainerWidth = signalPlotDiv.offsetWidth;
        const newPlotWidth = Math.max(signalWidth + 100, newContainerWidth);
        Plotly.relayout('signalPlot', {
            width: newPlotWidth,
            height: plotHeight
        });
    });
    }

function lowpassFilter(data) {
    const alpha = 0.3;
    const filtered = [];
    let prev = data[0];
    for (let i = 0; i < data.length; i++) {
        const val = alpha * data[i] + (1 - alpha) * prev;
        filtered.push(val);
        prev = val;
    }
    return filtered;
}

function updateSubarrhythmia() {
    const arrhythmia = document.getElementById('arrhythmiaSelect').value;
    const subSelect = document.getElementById('subarrhythmiaSelect');
    
    subSelect.innerHTML = '<option value="">Select Subarrhythmia</option>';
    if (arrhythmia && arrhythmiaDict[arrhythmia]) {
        arrhythmiaDict[arrhythmia].forEach(sub => {
            const opt = document.createElement('option');
            opt.value = sub;
            opt.textContent = sub;
            subSelect.appendChild(opt);
        });
        subSelect.style.display = 'inline-block';
    } else {
        subSelect.style.display = 'none';
    }
}

document.addEventListener('DOMContentLoaded', () => {
    document.getElementById('ecgForm').style.display = 'none';
    document.getElementById('plot').style.display = 'none';
    document.getElementById('morphologySection').style.display = 'none';

    const leadSelect = document.getElementById('leads');
    leads.forEach(lead => {
        const option = document.createElement('option');
        option.value = lead;
        option.textContent = lead;
        leadSelect.appendChild(option);
    });

    const arrhythmiaSelect = document.getElementById('arrhythmia');
    const subArrhythmiaSelect = document.getElementById('subArrhythmia');
    arrhythmiaSelect.addEventListener('change', () => {
        const selectedMain = arrhythmiaSelect.value;
        subArrhythmiaSelect.innerHTML = '<option value="">Select Sub-Arrhythmia</option>';
        if (arrhythmiaDict[selectedMain]) {
            arrhythmiaDict[selectedMain].forEach(sub => {
                const option = document.createElement('option');
                option.value = sub;
                option.textContent = sub;
                subArrhythmiaSelect.appendChild(option);
            });
        }
    });

    const arrhythmiaMorphSelect = document.getElementById('arrhythmiaSelect');
    for (const key in arrhythmiaDict) {
        const option = document.createElement('option');
        option.value = key;
        option.textContent = key;
        arrhythmiaMorphSelect.appendChild(option);
    }

    // File input click handling
    const fileInputContainer = document.querySelector('.file-input-container');
    if (fileInputContainer) {
        fileInputContainer.replaceWith(fileInputContainer.cloneNode(true));
        const newFileInputContainer = document.querySelector('.file-input-container');
        newFileInputContainer.addEventListener('click', (event) => {
            event.stopPropagation();
            const fileInput = document.getElementById('fileInput');
            fileInput.click();
        });
    }

    //  Make Morphology Section the default active view
    showMorphologySection();
});