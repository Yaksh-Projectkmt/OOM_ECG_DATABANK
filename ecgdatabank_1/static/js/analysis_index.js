// Global variables for file upload state
let uploadedFileName = '';
let uploadedLeadCount = null; // Store lead count from upload
let uploadedPlotImage = null;
let uploadedImageDataURL = null; // Store uploaded image URL

// Show the selected file name or reset label
function showFileName(input) {
    const fileLabel = document.getElementById('fileLabel');
    if (!fileLabel) {
        console.error('Error: fileLabel element not found.');
        return;
    }
    fileLabel.textContent = input.files.length > 0 
        ? input.files[0].name 
        : 'Click to Upload File';
}

// Navigate to a specific section and configure upload settings
function navigateToSection(section) {
    const elements = {
        mainHeader: document.getElementById('mainHeader'),
        uploadSection: document.getElementById('uploadSection'),
        isLeadForHolter: document.getElementById('isLeadForHolter'),
        isLeadForCSV: document.getElementById('isLeadForCSV'),
        cardsContainer: document.getElementById('cardsContainer'),
        oeaCard: document.getElementById('oeaCard'),
        tmtCard: document.getElementById('tmtCard'),
        holterCard: document.getElementById('holterCard'),
        csvCard: document.getElementById('csvCard'),
        uploadTitle: document.getElementById('uploadTitle'),
        uploadSubtitle: document.getElementById('uploadSubtitle'),
        fileInput: document.getElementById('fileInput'),
        viewBtn: document.getElementById('viewBtn')
    };

    // Check for missing critical elements
    for (const [key, el] of Object.entries(elements)) {
        if (!el && key !== 'viewBtn') {
            console.error(`Error: ${key} element not found.`);
            return;
        }
    }

    // Hide all sections and cards initially
    elements.mainHeader.style.display = 'none';
    elements.uploadSection.style.display = 'none';
    if (elements.isLeadForHolter) elements.isLeadForHolter.style.display = 'none';
    if (elements.isLeadForCSV) elements.isLeadForCSV.style.display = 'none';
    elements.cardsContainer.style.display = 'none';
    elements.oeaCard.style.display = 'none';
    elements.tmtCard.style.display = 'none';
    elements.holterCard.style.display = 'none';
    elements.csvCard.style.display = 'none';

    // Configure upload section based on selection
    if (['oea', 'tmt', 'holter', 'csv'].includes(section)) {
        elements.uploadSection.style.display = 'block';

        // Set allowed file types
        const fileAccept = {
            oea: 'image/png, image/jpeg, image/jpg',
            tmt: 'application/pdf',
            holter: '.csv',
            csv: '.csv'
        };
        elements.fileInput.accept = fileAccept[section];

        // Hide additional elements for specific sections
        if (section === 'oea' || section === 'tmt') {
            if (elements.isLeadForHolter) elements.isLeadForHolter.style.display = 'none';
            if (elements.isLeadForCSV) elements.isLeadForCSV.style.display = 'none';
            if (elements.viewBtn) elements.viewBtn.style.display = 'none';
        }

        // Update title and subtitle
        const titles = {
            oea: { title: 'OOM ECG Analyzer (OEA)', subtitle: 'Upload ECG Image' },
            tmt: { title: 'TMT Data Analysis', subtitle: 'Upload TMT Pdf' },
            holter: { title: 'Holter Data Analysis', subtitle: 'Upload Holter CSV File' },
            csv: { title: 'CSV File Upload', subtitle: 'Upload CSV File' }
        };
        elements.uploadTitle.textContent = titles[section].title;
        elements.uploadSubtitle.textContent = titles[section].subtitle;
    }
}

// Show all cards and hide other sections
function uploadFile() {
    const fileInput = document.getElementById('fileInput');
    const section = document.getElementById('uploadTitle')?.textContent;
    if (!fileInput || !section) {
        console.error('Error: fileInput or uploadTitle element not found.');
        alertSystem.error('Error','Required elements not found.');
        return;
    }

    const file = fileInput.files[0];
    if (!file) {
        alertSystem.info('Warning','Please select a file.');
        return;
    }

    const fileType = file.name.split('.').pop().toLowerCase();
    const validTypes = {
        'TMT Data': ['pdf'],
        'Holter Data': ['csv'],
        'CSV File': ['csv'],
        'OEA': ['jpg', 'png', 'jpeg']
    };

    // Validate file format
    for (const [key, types] of Object.entries(validTypes)) {
        if (section.includes(key) && !types.includes(fileType)) {
            alertSystem.warning('warning','Select the correct file format: Only ${types.join(', ')} files are allowed for ${key}.');
            return;
        }
    }

    // Show loader for TMT, Holter, or CSV sections
    if (section.includes('TMT Data') || section.includes('Holter Data') || section.includes('CSV File')) {
        showLoader();
    }

    // Show popup for OEA section, but don't show loader yet
    if (section.includes('OEA')) {
        showUploadedPopup();
    }

    // Prepare and send file
    const formData = new FormData();
    formData.append('file', file);
    const uploadUrl = section.includes('TMT Data') ? '/analysis/upload_tmt_pdf/' : uploadFileUrl;

    fetch(uploadUrl, {
        method: 'POST',
        body: formData,
        headers: { 'X-CSRFToken': document.querySelector("input[name='csrfmiddlewaretoken']")?.value },
        credentials: 'same-origin'
    })
    .then(response => {
        const contentType = response.headers.get('content-type');
        if (section.includes('TMT Data')) {
            if (contentType.includes('application/zip') || contentType.includes('application/octet-stream')) {
                return response.blob();
            }
            if (contentType.includes('application/json')) {
                return response.json();
            }
            throw new Error('Unexpected response type for TMT upload');
        }
        return response.json();
    })
    .then(data => {
        if (section.includes('TMT Data')) {
            if (data.error) {
                alertSystem.error('Error',`TMT Upload Error: ${data.error}`);
                hideLoader();
                return;
            }
            const zipUrl = `/media/analysis_tool/uploads/${data.zip_file}`;
            const a = document.createElement('a');
            a.href = zipUrl;
            a.download = data.zip_file;
            document.body.appendChild(a);
            a.click();
            document.body.removeChild(a);
            hideLoader();
            setTimeout(() => {
                     alertSystem.success('Success','TMT PDF processed and ZIP downloaded.');
                    // Reset file input and UI
                    hide();
                }, 100);
            return;
        }

        if (fileType === 'csv') {
            uploadedFileName = file.name;
            uploadedLeadCount = data.lead_count;
            uploadedPlotImage = data.plot_image;

            // Check for 12-lead requirement in Holter Data section
            if (section.includes('Holter Data') && uploadedLeadCount !== 12) {
                hideLoader();
                setTimeout(() => {
                    alertSystem.warning('Warning','Holter Data requires exactly 12-lead data. Please upload a valid 12-lead CSV file.');
                    // Reset file input and UI
                    hide();
                }, 100);
                return;
            }

            hideLoader();

            // setTimeout(() => {
            //         alert('CSV file uploaded successfully!');
            //         // Reset file input and UI
            //     }, 100);
            
            const cardsContainer = document.getElementById('cardsContainer');
            const viewBtn = document.getElementById('viewBtn');
            const openPopupButton = document.getElementById('openPopupButton');
            const isLeadForHolter = document.getElementById('isLeadForHolter');
            const isLeadForCSV = document.getElementById('isLeadForCSV');
                
            if (section.includes('Holter Data')) {
                if (isLeadForHolter) isLeadForHolter.style.display = 'none';
                if (isLeadForCSV) isLeadForCSV.style.display = 'none';
            } else if (section.includes('CSV File')) {
                if (isLeadForHolter) isLeadForHolter.style.display = 'none';
                if (isLeadForCSV) isLeadForCSV.style.display = 'block';
            }
            if (cardsContainer) cardsContainer.style.display = 'grid';
            if (viewBtn) viewBtn.style.display = 'inline-block';
            if (openPopupButton) openPopupButton.style.display = 'none';
        }

        else {
            if (isLeadForHolter) isLeadForHolter.style.display = 'none';
            if (isLeadForCSV) isLeadForCSV.style.display = 'none';
            processImage(file.name)
                .then(() => hideLoader())
                .catch(error => {
                    console.error('Process Image Error:', error);
                    hideLoader();
                });
        }
    })
    .catch(error => {
        console.error('Upload Error:', error);
        alertSystem.error('Error',`Upload failed: ${error.message}`);
        hideLoader();
    });
}
// Process uploaded ECG image
function processImage(filename) {
    return new Promise((resolve, reject) => {
        // // Show loader after the alert is confirmed
        showLoader();
 
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
                hideLoader();
                // setTimeout(() => {
                //     alert('ECG image processing completed successfully.');
                //     // Reset file input and UI
                // }, 100);
                
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
                hideLoader();
                resolve();
            })
            .catch(error => {
                console.error('Error while processing image:', error);
                const outputButton = document.getElementById('showOutputButton');
                if (outputButton) outputButton.style.display = 'none';

                hideLoader();

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

// Fetch files for a given category
function fetchFiles(category) {
    const url = fetchFilesUrl.replace('CATEGORY_PLACEHOLDER', category);

    fetch(url)
        .then(response => {
            if (!response.ok) throw new Error(`HTTP error! Status: ${response.status}`);
            return response.json();
        })
        .then(data => {
            if (data.error) {
                alertSystem.error('Error',data.error);
            } else {
                showModal(category, data.files);
            }
        })
        .catch(error => console.error('Fetch error:', error));
}

// Display modal with files
function showModal(category, files) {
    const modalTitle = document.getElementById('fileModalLabel');
    const modalBody = document.getElementById('fileModalBody');
    if (!modalTitle || !modalBody) {
        console.error('Error: Modal elements not found.');
        return;
    }

    modalTitle.textContent = `${category.toUpperCase()} Models`;
    modalTitle.style.color = 'black';
    modalBody.innerHTML = files.map(file => `
        <button class="btn btn-outline-primary m-2" onclick="processArrhythmia('${category}', '${file}')">${file}</button>
    `).join('');

    const modalElement = document.getElementById('fileModal');
    if (modalElement) {
        const modal = new bootstrap.Modal(modalElement);
        modal.show();
    } else {
        console.error('Error: fileModal element not found.');
    }
}

// Get selected lead type
function getSelectedLead() {
    const selectedLead = document.querySelector('input[name="lead"]:checked');
    return selectedLead ? selectedLead.value : null;
}

// Process arrhythmia for a file
function processArrhythmia(category, filename) {
    let leadType = getSelectedLead();
    if (!leadType) {
        alertSystem.info('info',"Please select a lead type.");
        return;
    }
    if (!uploadedFileName) {
        alertSystem.info('info',"No uploaded file found!");
        return;
    }

    //  Hide modal before processing
    const modalElement = document.getElementById('fileModal');
    if (modalElement) {
        const modal = bootstrap.Modal.getInstance(modalElement) || new bootstrap.Modal(modalElement);
        modal.hide();
    }

    //  Show loader after modal is hidden
    setTimeout(() => {
        showLoader('Processing Arrhythmia... Please Wait');

        const url = processArrhythmiaUrl
            .replace('CATEGORY_PLACEHOLDER', category)
            .replace('FILENAME_PLACEHOLDER', filename)
            .replace('IMAGE_NAME_PLACEHOLDER', uploadedFileName)
            .replace('LEAD_TYPE_PLACEHOLDER', leadType);

        fetch(url, {
            method: 'POST',
            headers: { 'X-CSRFToken': document.querySelector("input[name='csrfmiddlewaretoken']")?.value },
            credentials: 'same-origin'
        })
            .then(response => {
                if (!response.ok) {
                    throw new Error(`HTTP error! Status: ${response.status}`);
                }
                const contentType = response.headers.get('content-type');
                if (!contentType || !contentType.includes('application/json')) {
                    throw new Error(`Unexpected response type: ${contentType || 'unknown'}. Expected JSON.`);
                }
                return response.json();
            })
            .then(data => {
                if (data.error) {
                    throw new Error(`Server error: ${data.error}`);
                }
                const downloadUrl = '/analysis/download_result/';
                window.location.href = downloadUrl;

                hideLoader();
                setTimeout(() => {
                    alertSystem.success('Success',`Arrhythmia Check Result: ${data.result}`);
                    hide();
                }, 100);
            })
            .catch(error => {
                console.error('Processing error:', error);
                hide();
                hideLoader();
                setTimeout(() => {
                    alertSystem.error('Error',`Failed to process arrhythmia: ${error.message}`);
                }, 100);
            });

    }, 300); // small delay to allow modal to close cleanly
}


// Handle file input change and store image data URL
const fileInput = document.getElementById('fileInput');
if (fileInput) {
    fileInput.addEventListener('change', function () {
        const file = this.files[0];
        if (file) {
            const reader = new FileReader();
            reader.onload = function (e) {
                uploadedImageDataURL = e.target.result;
            };
            reader.readAsDataURL(file);
        }
    });
}

// Show uploaded image in popup
function showUploadedPopup() {
    const uploadPopup = document.getElementById('uploadPopup');
    const uploadPopupImage = document.getElementById('uploadPopupImage');
    if (!uploadPopup || !uploadPopupImage) {
        console.error('Error: uploadPopup or uploadPopupImage element not found.');
        return;
    }
    if (uploadedImageDataURL) {
        uploadPopupImage.src = uploadedImageDataURL;
        uploadPopup.style.display = 'block';
    }
}

// Show output image in popup
function showOutputPopup() {
    const uploadedFileInput = document.getElementById('fileInput');
    const popupImage = document.getElementById('outputPopupImage');
    if (!uploadedFileInput || !popupImage) {
        console.error('Error: fileInput or outputPopupImage element not found.');
        return;
    }

    const fullPath = uploadedFileInput.files[0]?.name;
    if (!fullPath) {
        console.error('Error: No file selected.');
        return;
    }

    const fileName = fullPath.substring(0, fullPath.lastIndexOf('.')) || fullPath;

    const imageUrl = `/media/analysis_tool/uploads/${fileName}.jpg`;

    popupImage.src = imageUrl;
    popupImage.style.display = 'block';
}

// Hide UI elements and reset file input
function hide() {
    const elements = {
        showOutputButton: document.getElementById('showOutputButton'),
        uploadPopup: document.getElementById('uploadPopup'),
        popcard: document.getElementById('popcard'),
        viewBtn: document.getElementById('viewBtn'),
        chartContainer: document.getElementById('chart-container'),
        isLeadForHolter: document.getElementById('isLeadForHolter'),
        isLeadForCSV: document.getElementById('isLeadForCSV'),
        cardsContainer: document.getElementById('cardsContainer'),
        fileInput: document.getElementById('fileInput'),
        fileLabel: document.getElementById('fileLabel'),
    };

    // Close the fileModal if it exists
    const modalElement = document.getElementById('fileModal');
    if (modalElement) {
        const modal = bootstrap.Modal.getInstance(modalElement) || new bootstrap.Modal(modalElement);
        modal.hide();
    } else {
        console.warn('Warning: fileModal element not found.');
    }

    for (const [key, el] of Object.entries(elements)) {
        if (!el) {
            console.error(`Error: ${key} element not found.`);
            continue;
        }
        if (key === 'fileInput') {
            el.value = '';
        } else if (key === 'fileLabel') {
            el.textContent = 'Click to Upload File';
        } else {
            el.style.display = 'none';
        }
    }

    const openPopupButton = document.getElementById('openPopupButton');
    if (openPopupButton) {
        openPopupButton.style.display = 'block';
    }
    deleteFiles();
}

// Show loader overlay
function showLoader(message = 'Please Wait......') {
    let overlay = document.getElementById('overlay');
    if (!overlay) {
        overlay = document.createElement('div');
        overlay.id = 'overlay';
        overlay.style.cssText = `
            position: fixed;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            background-color: rgba(0, 0, 0, 0.6);
            color: white;
            font-size: 2em;
            display: flex;
            justify-content: center;
            align-items: center;
            z-index: 9999;
        `;
        overlay.innerText = message;
        document.body.appendChild(overlay);
    } else {
        overlay.style.display = 'flex';
        overlay.innerText = message;
    }
}

// Hide loader overlay
function hideLoader() {
    const overlay = document.getElementById('overlay');
    if (overlay) overlay.style.display = 'none';
}

// Toggle info popup for both Holter and CSV
function toggleInfoPopup(event, popupId) {
    const popup = document.getElementById(popupId);
    if (!popup) {
        console.error(`Error: ${popupId} element not found.`);
        return;
    }
    popup.style.display = popup.style.display === 'none' ? 'block' : 'none';
    const rect = event.target.getBoundingClientRect();
    popup.style.top = `${window.scrollY + rect.bottom + 5}px`;
    popup.style.left = `${window.scrollX + rect.left}px`;
}

// Add event listeners for both info buttons
document.addEventListener('DOMContentLoaded', function () {
    const infoButtons = [
        { id: 'informations', popup: 'infoPopupHolter' },
        { id: 'informations_csv', popup: 'infoPopupCSV' }
    ];

    infoButtons.forEach(button => {
        const element = document.getElementById(button.id);
        if (element) {
            element.addEventListener('click', function (e) {
                toggleInfoPopup(e, button.popup);
            });
        } else {
            console.error(`Error: ${button.id} element not found.`);
        }
    });

    // Hide info popups on outside click
    document.addEventListener('click', function (event) {
        infoButtons.forEach(button => {
            const popup = document.getElementById(button.popup);
            const infoButton = document.getElementById(button.id);
            if (popup && popup.style.display === 'block' && 
                !popup.contains(event.target) && 
                infoButton && !infoButton.contains(event.target)) {
                popup.style.display = 'none';
            }
        });
    });

    const showOutputButton = document.getElementById('showOutputButton');
    if (showOutputButton) {
        showOutputButton.addEventListener('click', showOutputPopup);
    } else {
        console.error('Error: showOutputButton element not found.');
    }
});

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
            hideLoader();
            return;
        }

        const plotDiv = document.getElementById('chart-container');
        if (!plotDiv) {
            console.error('Error: chart-container element not found.');
            alertSystem.error('Error', 'Plot container not found.');
            hideLoader();
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
        hideLoader();
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

// Close chart container
function closeChart() {
    const chartContainer = document.getElementById('chart-container');
    const viewBtn = document.getElementById('viewBtn');
    if (chartContainer) chartContainer.style.display = 'none';  
    if (viewBtn) viewBtn.style.display = 'block';
}

async function deleteFiles() {  
    try {
        const response = await fetch('/analysis/delete-files/', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
                'X-CSRFToken': document.querySelector("input[name='csrfmiddlewaretoken']")?.value || ''
            },
            body: JSON.stringify({ delete_all: true }),
            credentials: 'same-origin'
        });

        // Check if response is JSON
        const contentType = response.headers.get('content-type');
        if (!contentType || !contentType.includes('application/json')) {
            const text = await response.text();
            console.error('Non-JSON response received:', text.slice(0, 100) + '...');
            console.error('Error: Expected JSON, got', contentType || 'unknown');
            return; // Silently exit, no frontend feedback
        }

        const data = await response.json();
        if (response.ok) {
            // Reset global variable
            uploadedFileName = '';
        } else {
            console.error('Deletion error:', data.error || 'Unknown server error');
        }
    } catch (error) {
        console.error('Fetch error:', error.message);
    }
}