//Basic js 
  // CSRF Token Utility
  const getCSRFToken = () => {
    const cookie = document.cookie.split(';').find(c => c.trim().startsWith('csrftoken='));
    return cookie ? decodeURIComponent(cookie.split('=')[1]) : document.getElementById('csrfToken')?.value || '';
  };
    const safeFetch = async (url, options) => {
    try {
      const response = await fetch(url, options);
      if (!response.ok) throw new Error(`HTTP error! Status: ${response.status}`);
      return await response.json();
    } catch (error) {
      console.error(`Fetch error at ${url}:`, error);
      throw error;
    } 
  };

  const debounce = (func, wait) => {
    let timeout;
    return (...args) => {
        clearTimeout(timeout);
        timeout = setTimeout(() => func(...args), wait);
    };
};

const selectors = {
    ecgTableBody: '#ecgTableBody',
    editEcgData: '.edit-btn',
    saveData: '.save-btn',
    close: '.plot-close'
};

const record_id = "{{ record_id }}";
const patient_id = "{{ patient_id }}";
const collection = "{{ collection }}";
const leadNumeric = "{{ lead }}" === "II" ? 2 : "{{ lead }}"; 

  // ---- CSV helper ----
function downloadCSV(csvContent, filename) {
  const blob = new Blob([csvContent], { type: 'text/csv;charset=utf-8;' });
  const url = URL.createObjectURL(blob);
  const link = document.createElement('a');
  link.setAttribute('href', url);
  link.setAttribute('download', filename);
  document.body.appendChild(link);
  link.click();
  document.body.removeChild(link);
}

//Data plotting logic
let currentPlot = null;

// Min-Max scaling helper
const minMaxScale = (arr, minVal = 0, maxVal = 4) => {
    const min = Math.min(...arr);
    const max = Math.max(...arr);
    if (max === min) return arr.map(() => (maxVal + minVal) / 2);
    return arr.map(v => ((v - min) / (max - min)) * (maxVal - minVal) + minVal);
};

// Scaled Y values
const scaledY = minMaxScale(ecgSignal);

// Create ECG plot
function createECGPlot() {
    const timeAxis = ecgSignal.map((_, i) => i);
    const dataLength = timeAxis.length;
    const windowSize = 2000;

    const rawDtick = (dataLength < windowSize) ? Math.round(dataLength / 25) : 100;
    const xDtick = Math.ceil(rawDtick / 5) * 5;
    const xMinorDtick = xDtick / 5;

    // Start with empty data
    const trace = {
        x: [],
        y: [],
        mode: 'lines',
        name: 'ECG Signal',
        line: {
            color: 'black',
            width: 1
        },
        hovertemplate: '<b>Time:</b> %{x:.3f}s<br><b>Amplitude:</b> %{y:.3f}mV<extra></extra>'
    };

    const layout = {
        xaxis: {
            range: [0, dataLength <= windowSize ? dataLength : windowSize],
            title: { text: 'Data', font: { size: 14, color: '#6b7280' } },
            showgrid: true,
            gridcolor: 'rgba(245, 121, 121, 0.93)',
            zeroline: false,
            dtick: xDtick,
            minor: { showgrid: true, gridcolor: 'rgba(244, 210, 216, 0.92)', dtick: xMinorDtick, tick0: 0 },
        },
        yaxis: {
            range: [0, 4.1],
            title: { text: 'ECG (mV)', font: { size: 14, color: '#6b7280' } },
            showgrid: true,
            gridcolor: 'rgba(245, 121, 121, 0.93)',
            zeroline: false,
            dtick: 0.5,
            minor: { showgrid: true, gridcolor: 'rgba(244, 210, 216, 0.92)' }
        },
        plot_bgcolor: '#ffffffff',
        paper_bgcolor: 'white',
        margin: { l: 60, r: 30, t: 30, b: 60 },
        hovermode: 'x unified',
        font: { family: '-apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif' }
    };

    const config = {
        responsive: true,
        displaylogo: false,
        modeBarButtonsToAdd: [ 'pan2d', 'zoom2d', 'select2d', 'autoScale2d', 'resetScale2d', 'zoomIn2d', 'zoomOut2d', 'toImage']
      };

    const plotElement = document.getElementById("ecgPlot");

    currentPlot = Plotly.newPlot(plotElement, [trace], layout, config);

    // Animation logic
    let i = 0;
    const frameStep = 30;

    function animateWave() {
        if (i < dataLength) {
            const nextX = timeAxis.slice(i, i + frameStep);
            const nextY = scaledY.slice(i, i + frameStep);
            Plotly.extendTraces(plotElement, { x: [nextX], y: [nextY] }, [0]);
            i += frameStep;
            requestAnimationFrame(animateWave);
        }
    }
    requestAnimationFrame(animateWave);
}


    // Initialize the plot when page loads
    document.addEventListener('DOMContentLoaded', function() {
        if (ecgSignal && ecgSignal.length > 0) {
            createECGPlot();
        } else {
            document.getElementById('ecgPlot').innerHTML = `
                <div class="loading">
                    <div class="spinner"></div>
                    <p>Loading ECG data...</p>
                </div>
            `;
        }
    });

// Handle window resize
window.addEventListener('resize', function() {
    if (currentPlot) {
        Plotly.Plots.resize('ecgPlot');
    }
});

//functionality logics 
function resetEditState(container, editBtn, saveBtn, confirmBtn, select, downloadSelect,closeBtn=null) {
    if (container) container.style.display = 'none';
    if (editBtn) editBtn.style.display = 'inline-block';
    if (saveBtn) saveBtn.style.display = 'inline-block';
    if (confirmBtn) confirmBtn.style.display = 'none';
    if (select) select.value = '';
    if (downloadSelect) downloadSelect.style.display = 'inline-block';
    if (closeBtn) closeBtn.style.display = 'none';
}

// Initialize button references
let editButton = document.getElementById(`editEcgData`);
let saveButton = document.getElementById(`saveData`);
let confirmEditBtn = document.getElementById(`confirmEditBtn`);
let closeBtn = document.getElementById(`closeBtn`); // Query close button
let arrhythmiaContainer = document.getElementById(`arrhythmiaContainer`);
let arrhythmiaSelect = document.getElementById(`Arrhythmia`);
let downloadTypeSelect = document.getElementById(`downloadType`);

if (!closeBtn) {
    console.warn(`Close button closeBtn not found in DOM.`);
}
// Reset button states when opening the plot
resetEditState(arrhythmiaContainer, editButton, saveButton, confirmEditBtn, arrhythmiaSelect, downloadTypeSelect,closeBtn);

//Edit data logic
if (editButton) {
  // Remove existing listeners to prevent duplication
  const newEditButton = editButton.cloneNode(true);
  editButton.parentNode.replaceChild(newEditButton, editButton);
  editButton = newEditButton;

  editButton.addEventListener('click', debounce(async () => {
    if (!arrhythmiaContainer || !editButton || !saveButton || !confirmEditBtn || !arrhythmiaSelect || !downloadTypeSelect) {
      alertSystem.info('Info','Required elements not found.');
      return;
    }

    // Re-query closeBtn to ensure it’s available
    closeBtn = document.getElementById(`closeBtn`);
    if (!closeBtn) {
      console.error('Error',`Close button closeBtn not found.`);
      alertSystem.error('Error','Close button not found. Edit functionality may be limited.');
      return;
    }

    // Hide normal buttons, show edit controls
    downloadTypeSelect.style.display = 'none';
    editButton.style.display = 'none';
    saveButton.style.display = 'none';
    arrhythmiaContainer.style.display = 'flex';
    confirmEditBtn.style.display = 'inline-block';
    closeBtn.style.display = 'inline-block';

    // Set default to current arrhythmia
    const currentArrhythmia = sessionStorage.getItem('selectedArrhythmia') || '{{ arrhythmia }}';
    arrhythmiaSelect.value = currentArrhythmia;
    arrhythmiaSelect.setAttribute('data-original', currentArrhythmia);

    // Reset old listeners
    const newConfirmEditBtn = confirmEditBtn.cloneNode(true);
    confirmEditBtn.parentNode.replaceChild(newConfirmEditBtn, confirmEditBtn);
    confirmEditBtn = newConfirmEditBtn;

    const newCloseBtn = closeBtn.cloneNode(true);
    closeBtn.parentNode.replaceChild(newCloseBtn, closeBtn);
    closeBtn = newCloseBtn;

    // Confirm button
    confirmEditBtn.addEventListener('click', debounce(async () => {
    const newArrhythmia = arrhythmiaSelect.value.trim();
    const originalArrhythmia = (arrhythmiaSelect.getAttribute('data-original') || '').trim();

  if (!newArrhythmia) {
    alertSystem.info('Info','Invalid arrhythmia selection.');
    resetEditState(arrhythmiaContainer, editButton, saveButton, confirmEditBtn, arrhythmiaSelect, downloadTypeSelect, closeBtn);
    return;
  }

  if (newArrhythmia === originalArrhythmia) {
    alertSystem.info('Info','No changes made.');
    resetEditState(arrhythmiaContainer, editButton, saveButton, confirmEditBtn, arrhythmiaSelect, downloadTypeSelect, closeBtn);
    return;
  }

  try {
    const requestData = {
        object_id: record_id,
        old_collection: originalArrhythmia,
        new_collection: newArrhythmia,
        lead: leadNumeric,
        PatientID: patient_id
    };

    const data = await safeFetch('/report/edit_data/', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'X-CSRFToken': getCSRFToken()
      },
      body: JSON.stringify(requestData)
    });

    if (data.status === 'success') {
      alertSystem.success('Success','ECG data updated successfully!');

      const arrhythmiaBadge = document.querySelector('.meta-item .meta-value .alert-badge');
      if (arrhythmiaBadge) {
        arrhythmiaBadge.innerHTML = `<i class="fas fa-circle"></i> ${newArrhythmia}`;
      }

      sessionStorage.setItem('selectedArrhythmia', newArrhythmia);
      window.location.href = "/report/";
    } else {
      alertSystem.error('Error', data.message || 'Failed to update data.');
    }
  } catch (error) {
    console.error('Request failed:', error);
    alertSystem.error('Error','Duplicate ECG data found.');
  }
}, 100));

    // Close button
    closeBtn.addEventListener('click', debounce(() => {
      resetEditState(arrhythmiaContainer, editButton, saveButton, confirmEditBtn, arrhythmiaSelect, downloadTypeSelect, closeBtn);
    }, 100));

  }, 100));
}

// Download logic

// ---- Download logic ----
if (saveButton && downloadTypeSelect) {

    // Remove existing listeners by cloning once
    const newSaveButton = saveButton.cloneNode(true);
    saveButton.parentNode.replaceChild(newSaveButton, saveButton);
    saveButton = newSaveButton;

    saveButton.addEventListener('click', async () => {
        const downloadType = downloadTypeSelect.value;
        if (!downloadType) {
            alert('Please select a download type.');
            return;
        }

        if (!ecgSignal || ecgSignal.length === 0) {
            alert('ECG data not loaded.');
            return;
        }

        if (!record_id) {
            alert('No record selected.');
            return;
        }

        try {
            switch (downloadType) {
                case 'raw_data':
                    // For now only exporting current lead
                    let csv = `Index,Lead_${leadNumeric}\n`;
                    for (let i = 0; i < ecgSignal.length; i++) {
                        csv += `${i},${ecgSignal[i]}\n`;
                    }
                    downloadCSV(csv, `raw_ecg_data_${patient_id}_lead${leadNumeric}.csv`);
                    break;

                case 'plot_png':
                    const plotDiv = document.getElementById('ecgPlot'); // correct ID
                    if (!plotDiv || plotDiv.children.length === 0) {
                        alert("ECG plot not loaded yet.");
                        return;
                    }

                    let width = 1600, height = 1500;
                    if (leadNumeric == 2) { width = 1000; height = 400; }
                    else if (leadNumeric == 7) { width = 1200; height = 1000; }

                    Plotly.downloadImage(plotDiv, {
                        format: 'png',
                        filename: `ecg_plot_${patient_id}_lead${leadNumeric}`,
                        width: width,
                        height: height,
                        scale: 2
                    });
                    break;
                case 'selected_data':
                    const plotDivSel = document.getElementById('ecgPlot');
                    if (!plotDivSel || !plotDivSel.data || !plotDivSel.data[0]) {
                        alert("ECG plot not loaded yet.");
                        return;
                    }

                    // Get current x-axis range
                    const xRange = plotDivSel.layout?.xaxis?.range || [0, ecgSignal.length];
                    const x0 = xRange[0];
                    const x1 = xRange[1];

                    // Build the visible data from scaled plot
                    const visibleData = ecgSignal
                        .map((val, idx) => ({ x: idx, y: scaledY[idx] }))
                        .filter(p => p.x >= x0 && p.x <= x1);

                    if (visibleData.length === 0) {
                        alert("No data in the current view range.");
                        return;
                    }

                    // Create CSV
                    let csvContent = "Index,Scaled_Y\n" +
                        visibleData.map(row => `${row.x},${row.y}`).join("\n");

                    downloadCSV(csvContent, `selected_data_${patient_id}_lead${leadNumeric}.csv`);
                    break;
                default:
                    alert('Invalid download type.');
            }
        } catch (err) {
            console.error('Download error:', err);
            alert('Error downloading: ' + err.message);
        }
    });
}