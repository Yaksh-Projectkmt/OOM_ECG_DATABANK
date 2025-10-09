  const delay = ms => new Promise(resolve => setTimeout(resolve, ms));
  window.__ecgRowClickAttached = false;
  
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
  
  document.addEventListener('DOMContentLoaded', () => {
    // Selectors for DOM elements
    const selectors = {
      cards: '.card[data-arrhythmia]',
      cardTiles: '.card-tile[data-arrhythmia]',
      searchButton: '#searchButton',
      ecgTableBody: '#ecgTableBody',
      paginationControls: '#paginationControls',
      prevBtn: '#prevBtn',
      nextBtn: '#nextBtn',
      plotContainer: '#plot-container',
      editEcgData: '.edit-btn',
      saveData: '.save-btn',
      close: '.plot-close'
    };
  
    // Pagination state
    let currentPage = 1;
    let totalPages = parseInt(document.querySelector(selectors.paginationControls)?.dataset.totalPages) || 1;
    let searchContext = null;
  
    // Store ecgData per objectId to avoid overwriting
    window.ecgData = {};
  
    // CSRF Token Utility
    const getCSRFToken = () => {
      const cookie = document.cookie.split(';').find(c => c.trim().startsWith('csrftoken='));
      return cookie ? decodeURIComponent(cookie.split('=')[1]) : document.getElementById('csrfToken')?.value || '';
    };
  
    const safeFetch = async (url, options) => {
      const pageLoader = document.getElementById('page-loader');
      if (pageLoader) pageLoader.style.display = 'flex';
      try {
        const response = await fetch(url, options);
        if (!response.ok) throw new Error(`HTTP error! Status: ${response.status}`);
        return await response.json();
      } catch (error) {
        console.error(`Fetch error at ${url}:`, error);
        throw error;
      } 
      finally {
        if (pageLoader) pageLoader.style.display = 'none';
      }
    };
  
    // Enhanced safeFetch for background tasks
    const backgroundFetch = async (url, options, taskId) => {
      try {
        // Update task progress
        if (taskId) {
          taskManager.updateTask(taskId, { progress: 10 });
        }
  
        const response = await fetch(url, options);
        
        if (taskId) {
          taskManager.updateTask(taskId, { progress: 50 });
        }
  
        if (!response.ok) throw new Error(`HTTP error! Status: ${response.status}`);
        
        const result = await response.json();
        
        if (taskId) {
          taskManager.updateTask(taskId, { progress: 90 });
        }
  
        return result;
      } catch (error) {
        console.error(`Background fetch error at ${url}:`, error);
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
  
    const setupArrhythmiaDropdown = (arrhythmiaId, subArrhythmiaId) => {
      const arrhythmiaSelect = document.getElementById(arrhythmiaId);
      const subArrhythmiaSelect = document.getElementById(subArrhythmiaId);
      if (!arrhythmiaSelect || !subArrhythmiaSelect) return;
  
      arrhythmiaSelect.addEventListener('change', () => {
          const selectedArrhythmia = arrhythmiaSelect.value.trim();
          subArrhythmiaSelect.querySelectorAll('option').forEach(option => {
              const dataArr = (option.dataset.arrhythmia || '').trim();
              option.style.display = selectedArrhythmia ? (dataArr === selectedArrhythmia ? 'block' : 'none') : 'block';
          });
          subArrhythmiaSelect.value = '';
      });
    };
  
    setupArrhythmiaDropdown('arrhythmiaMI', 'subArrhythmia');
    setupArrhythmiaDropdown('newarrhythmiaMI', 'newsubArrhythmia');
    setupArrhythmiaDropdown('newarrhythmiaMI_multiple', 'newsubArrhythmia_multiple'); 
  
  const attachRowEventListeners = () => {
    // Only attach once
    if (window.__ecgRowClickAttached) return;
    window.__ecgRowClickAttached = true;
  
    if (!window.loadingPlots) window.loadingPlots = new Set();
  
    const tableBody = document.getElementById('ecgTableBody');
    if (!tableBody) return;
  
    tableBody.addEventListener('click', debounce(async (e) => {
      const row = e.target.closest('tr:not(.plot-row)');
      if (!row) return;
  
      const isDeleteBtn = e.target.closest('.delete-btn');
      if (isDeleteBtn) {
        await deleteData(e.target.dataset.id, e.target.dataset.collection);
        return;
      }
  
      const objectId = row.dataset.objectId;
      if (window.loadingPlots.has(objectId)) {
        return;
      }
  
      window.loadingPlots.add(objectId); // Lock
  
      const patientId = row.cells[1].textContent.trim();
      const leadDisplay = row.cells[2].textContent.trim();
      const leadNumeric = row.getAttribute('data-lead-value');
      const tableArrhythmia = row.cells[3].textContent.trim();
      const frequency = row.cells[4].textContent.trim();
      const samplesTaken = parseInt(row.getAttribute("data-samples-taken")) || 5000;
      sessionStorage.setItem('objectId', objectId);
  
      const plotRow = document.querySelector(`tr.plot-row[data-plot-id="${objectId}"]`);
      const plotContainer = document.getElementById(`plot-container-${objectId}`);
      const plotElement = document.getElementById(`plot-${objectId}`);
  
      if (!plotRow || !plotContainer || !plotElement) {
        alertSystem.error('Error','Plot container not found.');
        window.loadingPlots.delete(objectId);
        return;
      }
  
      const isVisible = plotRow.style.display === 'table-row';
      plotRow.style.display = isVisible ? 'none' : 'table-row';
  
      // Initialize button references
      let editButton = document.getElementById(`editEcgData-${objectId}`);
      let saveButton = document.getElementById(`saveData-${objectId}`);
      let closeButton = document.querySelector(`#plot-container-${objectId} .plot-close`);
      let confirmEditBtn = document.getElementById(`confirmEditBtn-${objectId}`);
      let closeBtn = document.getElementById(`closeBtn-${objectId}`); // Query close button
      let arrhythmiaContainer = document.getElementById(`arrhythmiaContainer-${objectId}`);
      let arrhythmiaSelect = document.getElementById(`Arrhythmia-${objectId}`);
      let downloadTypeSelect = document.getElementById(`downloadType-${objectId}`);
  
      if (!closeBtn) {
        console.warn(`Close button closeBtn-${objectId} not found in DOM.`);
      }
  
      // Reset button states when opening the plot
      resetEditState(arrhythmiaContainer, editButton, saveButton, confirmEditBtn, arrhythmiaSelect, downloadTypeSelect, closeBtn);
  
      if (!isVisible) {
        plotContainer.scrollIntoView({ behavior: 'smooth', block: 'start' });
        plotElement.innerHTML = '<p style="text-align:center;font-weight:bold;">Loading ECG data...</p>';
        Plotly.purge(plotElement);
  
        try {
          const data = await safeFetch('/ommecgdata/get_object_id/', {
            method: 'POST',
            headers: {
              'Content-Type': 'application/json',
              'X-CSRFToken': getCSRFToken()
            },
            body: JSON.stringify({
              objectId,
              patientId,
              lead: leadNumeric,
              tableArrhythmia,
              selectedArrhythmia: sessionStorage.getItem('selectedArrhythmia') || tableArrhythmia,
              frequency,
              samplesTaken: samplesTaken
            })
          });
  
          if (!data || typeof data !== 'object' || !data.ecgData) {
            throw new Error('No ECG data received from server');
          }
  
          let ecgData = null;
          let leadConfig = '';
  
          if (leadNumeric === '2') {
            leadConfig = '2_lead';
            ecgData = {
              x: data.x || Array.from({ length: data.ecgData.length }, (_, i) => i),
              y: data.ecgData,
              pqrst: data.pqrst || {} // Store PQRST indices if provided
            };
            if (!ecgData.y || ecgData.y.length === 0) throw new Error('Empty ECG signal');
          } else if (leadNumeric === '7' || leadNumeric === '12') {
            leadConfig = leadNumeric === '7' ? '7_lead' : '12_lead';
            ecgData = { 
              leadDict: data.ecgData,
              pqrst: data.pqrst || {} // Store PQRST indices if provided 
            };
            if (Object.keys(ecgData.leadDict).length === 0) throw new Error('No leads in ECG data');
          } else {
            throw new Error('Invalid lead configuration');
          }
  
          window.ecgData[objectId] = ecgData;
          await fetchAndPlotECG(ecgData, leadNumeric, patientId, objectId, leadConfig);
        } catch (error) {
          plotElement.innerHTML = `<p style="text-align:center; color: red;">Error loading ECG data: ${error.message}</p>`;
          console.error('Error fetching ECG data:', error);
        } finally {
          window.loadingPlots.delete(objectId);
        }
      } else {
        window.loadingPlots.delete(objectId);
      }
  
      // Re-query DOM elements after plot is loaded to ensure valid references
      editButton = document.getElementById(`editEcgData-${objectId}`);
      saveButton = document.getElementById(`saveData-${objectId}`);
      closeButton = document.querySelector(`#plot-container-${objectId} .plot-close`);
      confirmEditBtn = document.getElementById(`confirmEditBtn-${objectId}`);
      closeBtn = document.getElementById(`closeBtn-${objectId}`); // Re-query close button
      arrhythmiaContainer = document.getElementById(`arrhythmiaContainer-${objectId}`);
      arrhythmiaSelect = document.getElementById(`Arrhythmia-${objectId}`);
      downloadTypeSelect = document.getElementById(`downloadType-${objectId}`);
  
      if (!closeBtn) {
        console.warn(`Close button closeBtn-${objectId} not found after plot load.`);
      }
  
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
          // Re-query closeBtn to ensure its available
          closeBtn = document.getElementById(`closeBtn-${objectId}`);
          if (!closeBtn) {
            console.error('Error',`Close button closeBtn-${objectId} not found.`);
            alertSystem.error('Error','Close button not found. Edit functionality may be limited.');
            return;
          }
  
          downloadTypeSelect.style.display = 'none';
          editButton.style.display = 'none';
          saveButton.style.display = 'none';
          arrhythmiaContainer.style.display = 'flex';
          confirmEditBtn.style.display = 'inline-block';
          closeBtn.style.display = 'inline-block'; // Show the close button
          arrhythmiaSelect.value = tableArrhythmia || '';
  
          // Remove existing listeners from confirmEditBtn
          const newConfirmEditBtn = confirmEditBtn.cloneNode(true);
          confirmEditBtn.parentNode.replaceChild(newConfirmEditBtn, confirmEditBtn);
          confirmEditBtn = newConfirmEditBtn;
  
          // Remove existing listeners from closeBtn
          const newCloseBtn = closeBtn.cloneNode(true);
          closeBtn.parentNode.replaceChild(newCloseBtn, closeBtn);
          closeBtn = newCloseBtn;
  
          confirmEditBtn.addEventListener('click', debounce(async (e) => {
            if (e.target.classList.contains('close-action')) {
              resetEditState(arrhythmiaContainer, editButton, saveButton, confirmEditBtn, arrhythmiaSelect, downloadTypeSelect, closeBtn);
              return;
            }
  
            const newArrhythmia = arrhythmiaSelect.value;
            if (!newArrhythmia || newArrhythmia === tableArrhythmia) {
              alertSystem.info('Info','No changes made or invalid selection.');
              resetEditState(arrhythmiaContainer, editButton, saveButton, confirmEditBtn, arrhythmiaSelect, downloadTypeSelect, closeBtn);
              return;
            }
  
            const pageLoader = document.getElementById('page-loader');
            if (pageLoader) pageLoader.style.display = 'flex';
  
            try {
              const requestData = {
                object_id: objectId,
                old_collection: sessionStorage.getItem('selectedArrhythmia') || tableArrhythmia,
                new_collection: newArrhythmia,
                lead: leadNumeric,
                PatientID: patientId
              };
  
              const data = await safeFetch('/ommecgdata/edit_data/', {
                method: 'POST',
                headers: {
                  'Content-Type': 'application/json',
                  'X-CSRFToken': getCSRFToken()
                },
                body: JSON.stringify(requestData)
              });
  
              if (data.status === 'success') {
                alertSystem.success('Success','ECG data updated successfully!');
                row.cells[3].textContent = newArrhythmia;
                sessionStorage.setItem('selectedArrhythmia', newArrhythmia);
                window.location.href = "/ommecgdata/";
              } else {
                alertSystem.error('Error',` ${data.message || 'Failed to update data.'}`);
              }
            } catch (error) {
              alertSystem.error('Error','Duplicate ECG data found.');
            } finally {
              if (pageLoader) pageLoader.style.display = 'none';
              resetEditState(arrhythmiaContainer, editButton, saveButton, confirmEditBtn, arrhythmiaSelect, downloadTypeSelect, closeBtn);
            }
          }, 100));
  
          // Add event listener for the close button
          closeBtn.addEventListener('click', debounce(() => {
            resetEditState(arrhythmiaContainer, editButton, saveButton, confirmEditBtn, arrhythmiaSelect, downloadTypeSelect, closeBtn);
          }, 100));
        }, 100));
      }
  
      if (saveButton && downloadTypeSelect) {
        // Remove existing listeners
        const newSaveButton = saveButton.cloneNode(true);
        saveButton.parentNode.replaceChild(newSaveButton, saveButton);
        saveButton = newSaveButton;
  
        saveButton.addEventListener('click', debounce(async () => {
          const downloadType = downloadTypeSelect.value;
          if (!downloadType) {
            alertSystem.warning('Warning','Please select a download type.');
            return;
          }
          const pageLoader = document.getElementById('page-loader');
          if (pageLoader) pageLoader.style.display = 'flex';
  
          try {
            let dataToDownload, endpoint, filename, contentType = 'application/octet-stream';
  
            switch (downloadType) {
              case 'raw_data':
                const rawData = window.ecgData[objectId] || {};
                if (leadNumeric === '2') {
                  if (!rawData.x || !rawData.y) {
                    alertSystem.error('Error','No raw ECG data available.');
                    return;
                  }
                  let csvContent = 'DateTime,II\n';
                  for (let i = 0; i < rawData.x.length; i++) {
                    csvContent += `${rawData.x[i]},${rawData.y[i]}\n`;
                  }
                  downloadCSV(csvContent, `raw_ecg_data_${patientId}.csv`);
                  return;
                }
                if (leadNumeric === '7' || leadNumeric === '12') {
                  if (!rawData.leadDict) {
                    alertSystem.error('Error','No raw ECG data available.');
                    return;
                  }
                  const leadNames = Object.keys(rawData.leadDict);
                  const signalLength = rawData.leadDict[leadNames[0]].length;
                  let csvContent = leadNames.join(',') + '\n';
                  for (let i = 0; i < signalLength; i++) {
                    const row = leadNames.map(lead => rawData.leadDict[lead][i]).join(',');
                    csvContent += row + '\n';
                  }
                  downloadCSV(csvContent, `raw_ecg_data_${patientId}.csv`);
                  return;
                }
                alertSystem.warning('Warning','Unsupported lead type.');
                return;
  
            case 'plot_png':
                const plotDivId = `plot-${objectId}`;
                const plotDiv = document.getElementById(plotDivId);
                if (!plotDiv || plotDiv.children.length === 0) {
                  alertSystem.error('Error',"ECG plot not loaded yet. Please wait or try again.");
                  return;
                }
                let leadCount = leadNumeric || 12; // Default to 12 leads if not specified
                // Set dimensions based on lead count
                let width, height;
                switch (parseInt(leadCount)) {
                  case 2: // Single-lead ECG
                    width = 1000;
                    height = 400;
                    break;
                  case 7: // 7-lead ECG
                    width = 1200;
                    height = 1000;
                    break;
                  case 12: // 12-lead ECG (standard)
                    width = 1600;
                    height = 1500;
                    break;
                  default: // Fallback for other configurations
                    width = 1600;
                    height = 1000;
                }
        
                Plotly.downloadImage(plotDiv, {
                  format: 'png',
                  filename: `ecg_plot_${patientId}_leads_${leadCount}`,
                  width: width,
                  height: height,
                  scale: 2
                });
                return;
            case 'pqrst_csv':
                // Use updated PQRST indices if available, otherwise fall back to server-provided data
                let updatedPqrstData = {};
                if (leadNumeric === '2') {
                    updatedPqrstData = window.ecgData[objectId]?.updatedPQRST || window.ecgData[objectId]?.pqrst || {};
                } else if (leadNumeric === '7' || leadNumeric === '12') {
                    updatedPqrstData = window.ecgData[objectId]?.allPeaks || window.ecgData[objectId]?.pqrst || {};
                }
  
                // Validate PQRST data
                const requiredKeys = leadNumeric === '2' 
                    ? ['p_points', 'q_points', 'r_peaks', 's_points', 't_points']
                    : ['p', 'q', 'r', 's', 't'];
  
                const isValid = leadNumeric === '2'
                    ? requiredKeys.every(key => Array.isArray(updatedPqrstData[key]) && updatedPqrstData[key].length > 0)
                    : Object.keys(updatedPqrstData).length > 0 && requiredKeys.every(key => 
                        Array.isArray(updatedPqrstData[Object.keys(updatedPqrstData)[0]][key]) && 
                        updatedPqrstData[Object.keys(updatedPqrstData)[0]][key].length > 0);
  
                if (!isValid || Object.keys(updatedPqrstData).length === 0) {
                    console.error('Invalid PQRST data:', updatedPqrstData);
                    alertSystem.error('Error', 'No valid PQRST data available for download. Please ensure ECG data is loaded correctly.');
                    if (pageLoader) pageLoader.style.display = 'none';
                    return;
                }
  
                // Generate CSV content client-side
                let csvContent = 'P_index,Q_index,R_index,S_index,T_index\n';
                if (leadNumeric === '2') {
                    // For 2-lead ECG, generate column-based CSV
                    const { p_points, q_points, r_peaks, s_points, t_points } = updatedPqrstData;
                    
                    // Find the minimum length of PQRST arrays to align rows
                    const minLength = Math.min(
                        p_points.length,
                        q_points.length,
                        r_peaks.length,
                        s_points.length,
                        t_points.length
                    );
  
                    // Generate rows, padding with empty strings for missing values
                    for (let i = 0; i < minLength; i++) {
                        const p = (typeof p_points[i] === 'number' && !isNaN(p_points[i])) ? p_points[i] : '';
                        const q = (typeof q_points[i] === 'number' && !isNaN(q_points[i])) ? q_points[i] : '';
                        const r = (typeof r_peaks[i] === 'number' && !isNaN(r_peaks[i])) ? r_peaks[i] : '';
                        const s = (typeof s_points[i] === 'number' && !isNaN(s_points[i])) ? s_points[i] : '';
                        const t = (typeof t_points[i] === 'number' && !isNaN(t_points[i])) ? t_points[i] : '';
                        csvContent += `${p},${q},${r},${s},${t}\n`;
                    }
                } else if (leadNumeric === '7' || leadNumeric === '12') {
                    // For 7/12-lead ECG, generate column-based CSV like 2-lead
                    const firstLead = Object.keys(updatedPqrstData)[0]; // Use first lead's data
                    const { p, q, r, s, t } = updatedPqrstData[firstLead] || {};
  
                    // Find the minimum length of PQRST arrays to align rows
                    const minLength = Math.min(
                        (p || []).length,
                        (q || []).length,
                        (r || []).length,
                        (s || []).length,
                        (t || []).length
                    );
  
                    // Generate rows, padding with empty strings for missing values
                    for (let i = 0; i < minLength; i++) {
                        const pVal = (typeof p[i] === 'number' && !isNaN(p[i])) ? p[i] : '';
                        const qVal = (typeof q[i] === 'number' && !isNaN(q[i])) ? q[i] : '';
                        const rVal = (typeof r[i] === 'number' && !isNaN(r[i])) ? r[i] : '';
                        const sVal = (typeof s[i] === 'number' && !isNaN(s[i])) ? s[i] : '';
                        const tVal = (typeof t[i] === 'number' && !isNaN(t[i])) ? t[i] : '';
                        csvContent += `${pVal},${qVal},${rVal},${sVal},${tVal}\n`;
                    }
                }
  
                // Download CSV
                downloadCSV(csvContent, `pqrst_${patientId}.csv`);
                if (pageLoader) pageLoader.style.display = 'none';
                return;
                
            case 'selected_data':
                  if (!window.selectedData || !window.selectedData.x?.length || !window.selectedData.y?.length) {
                    alertSystem.warning('Warning', 'No ECG data selected for download.');
                    return;
                  }
  
                  try {
                    // For 2-lead, send x and y directly
                    if (leadNumeric === '2') {
                      dataToDownload = window.selectedData;
                    } else {
                      // For 7/12-lead, send leadDict structure
                      dataToDownload = { leadDict: window.selectedData };
                    }
  
                    const response = await fetch('/ommecgdata/selecteddownload/', {
                      method: 'POST',
                      headers: {
                        'Content-Type': 'application/json',
                        'X-CSRFToken': getCSRFToken()
                      },
                      body: JSON.stringify(dataToDownload)
                    });
  
                    if (!response.ok) throw new Error(`HTTP error! Status: ${response.status}`);
  
                    const blob = await response.blob();
                    const url = window.URL.createObjectURL(blob);
                    const a = document.createElement('a');
                    a.href = url;
                    a.download = `selected_ecg_data_${patientId}.csv`;
                    document.body.appendChild(a);
                    a.click();
                    window.URL.revokeObjectURL(url);
                    document.body.removeChild(a);
                  } catch (error) {
                    console.error('Error downloading file:', error);
                    alertSystem.error('Error', 'downloading file.');
                  }
                  return;
                  
              default:
                throw new Error('Invalid download type selected.');
            }
  } catch (error) {
      console.error('Error downloading:', error);
      alertSystem.error('Error', `downloading ${downloadType.replace('_', ' ')}: ${error.message}`);
    } finally {
      if (pageLoader) pageLoader.style.display = 'none';
    }
  }, 100));
      }
  let shareClickLock = false;
  
  document.addEventListener("click", async function (e) {
      // Toggle share menu
      if (e.target.classList.contains("share-icon")) {
          const objectId = e.target.getAttribute("data-id");
          const menu = document.getElementById(`shareMenu-${objectId}`);
          if (menu) {
              menu.style.display = menu.style.display === "block" ? "none" : "block";
          }
          e.stopPropagation();
      }
  
      // Handle any share option click
      else if (e.target.closest(".share-option")) {
          e.preventDefault();
          e.stopPropagation();  // prevent bubbling duplicates
  
          if (shareClickLock) return; //prevent multiple triggers
          shareClickLock = true;
          setTimeout(() => { shareClickLock = false; }, 500); // unlock after 0.5s
  
          const link = e.target.closest(".share-option");
          const platform = [...link.classList].find(cls =>
              ["whatsapp", "facebook", "teams", "email"].includes(cls)
          );
  
          if (!platform) return;
  
          const patientId = link.getAttribute("data-patient") || "";
          const objectId = link.closest(".share-dropdown").querySelector(".share-icon").dataset.id;
          const downloadType = downloadTypeSelect.value;
  
          if (!downloadType) {
              alertSystem.warning("Warning", "Please select a download type first.");
              return;
          }
  
          let fileContent = "";
  
          if (downloadType === "raw_data") {
              const rawData = window.ecgData[objectId];
              if (!rawData) {
                  alertSystem.error("Error", "No raw ECG data available to share.");
                  return;
              }
              // Generate CSV
              if (leadNumeric === "2") {
                  fileContent = "DateTime,II\n" + rawData.x.map((x, i) => `${x},${rawData.y[i]}`).join("\n");
              } else if (leadNumeric === "7" || leadNumeric === "12") {
                  const leadNames = Object.keys(rawData.leadDict);
                  const signalLength = rawData.leadDict[leadNames[0]].length;
                  fileContent = leadNames.join(",") + "\n";
                  for (let i = 0; i < signalLength; i++) {
                      fileContent += leadNames.map(lead => rawData.leadDict[lead][i]).join(",") + "\n";
                  }
              }
          } else if (downloadType === "pqrst_csv") {
              const pqrst = window.ecgData[objectId]?.updatedPQRST || window.ecgData[objectId]?.pqrst;
              if (!pqrst) {
                  alertSystem.error("Error", "No valid PQRST data to share.");
                  return;
              }
              const { p_points, q_points, r_peaks, s_points, t_points } = pqrst;
              const minLength = Math.min(p_points.length, q_points.length, r_peaks.length, s_points.length, t_points.length);
              fileContent = "P_index,Q_index,R_index,S_index,T_index\n";
              for (let i = 0; i < minLength; i++) {
                  fileContent += `${p_points[i]},${q_points[i]},${r_peaks[i]},${s_points[i]},${t_points[i]}\n`;
              }
          } else if (downloadType === "plot_png") {
              const plotDiv = document.getElementById(`plot-${objectId}`);
              if (!plotDiv) {
                  alertSystem.error("Error", "ECG Plot not found!");
                  return;
              }
              let width = 1000, height = 400;
              if (leadNumeric == 7) { width = 1200; height = 1000; }
              else if (leadNumeric == 12) { width = 1600; height = 1500; }
              fileContent = await Plotly.toImage(plotDiv, { format: "png", width, height });
          } else if (downloadType === "selected_data") {
              if (!window.selectedData) {
                  alertSystem.warning("Warning", "No ECG data selected for download.");
                  return;
              }
  
              if (leadNumeric === "2") {
                  // 2-lead selected data
                  if (!window.selectedData.x?.length || !window.selectedData.y?.length) {
                      alertSystem.warning("Warning", "Invalid 2-lead selected data.");
                      return;
                  }
                  fileContent = "DateTime,II\n" +
                      window.selectedData.x.map((x, i) => `${x},${window.selectedData.y[i]}`).join("\n");
  
              } else if (leadNumeric === "7" || leadNumeric === "12") {
                  // 7-lead or 12-lead selected data
                  if (!window.selectedData.leadDict) {
                      alertSystem.warning("Warning", "Invalid multi-lead selected data.");
                      return;
                  }
  
                  const leadNames = Object.keys(window.selectedData.leadDict);
                  if (leadNames.length === 0) {
                      alertSystem.warning("Warning", "No leads found in selected data.");
                      return;
                  }
  
                  const signalLength = window.selectedData.leadDict[leadNames[0]].length;
                  fileContent = leadNames.join(",") + "\n";
                  for (let i = 0; i < signalLength; i++) {
                      fileContent += leadNames.map(lead => window.selectedData.leadDict[lead][i]).join(",") + "\n";
                  }
  
              } else {
                  alertSystem.warning("Warning", "Unsupported lead type for selected data.");
                  return;
              }
          }
  
          // Upload to unified endpoint
          try {
              const response = await fetch("/ommecgdata/upload_plot/", {
                  method: "POST",
                  headers: {
                      "Content-Type": "application/json",
                      "X-CSRFToken": getCSRFToken()
                  },
                  body: JSON.stringify({
                      type: downloadType,
                      content: fileContent,
                      patientId
                  })
              });
  
              const result = await response.json();
              if (result.status === "success") {
                  const fileUrl = result.url;
  
      // Clean URL from backend (replace \ with /)
      const cleanUrl = fileUrl.replace(/\\/g, "/").replace(/%5C/g, "/");
      
      let shareUrl = "";
      switch (platform) {
          case "whatsapp":
              const whatsappBody = `Patient ID: ${patientId}\n\nClick the link below:\n${cleanUrl}`;
              shareUrl = `https://web.whatsapp.com/send?text=${encodeURIComponent(whatsappBody)}`;
              break;
      
          case "facebook":
              const facebookBody = `Patient ID: ${patientId}\n\nClick the link below:\n${cleanUrl}`;
              shareUrl = `https://www.facebook.com/sharer/sharer.php?u=${encodeURIComponent(cleanUrl)}&quote=${encodeURIComponent(facebookBody)}`;
              break;
      
          case "teams":
              const teamsBody = `Patient ID: ${patientId}\n\nClick the link below:\n${cleanUrl}`;
              shareUrl = `https://teams.microsoft.com/share?href=${encodeURIComponent(cleanUrl)}&text=${encodeURIComponent(teamsBody)}`;
              break;
      
          case "email":
              const emailBody = `Patient ID: ${patientId}\n\nClick the link below:\n${cleanUrl}`;
              shareUrl = `mailto:?subject=ECG Report - ${patientId}&body=${encodeURIComponent(emailBody)}`;
              break;
      }
      
      if (shareUrl) window.open(shareUrl, "_blank");           
                  } else {
                      alertSystem.error("Error", "Failed to upload file: " + (result.message || ""));
                  }
              } catch (err) {
                  console.error(err);
                  alertSystem.error("Error", "Uploading file failed.");
              }
          }
      });
  
     if (closeButton) {
        // Remove existing listeners
        const newCloseButton = closeButton.cloneNode(true);
        closeButton.parentNode.replaceChild(newCloseButton, closeButton);
        closeButton = newCloseButton;
  
        closeButton.addEventListener('click', debounce(() => {
          plotRow.style.display = 'none';
          Plotly.purge(plotElement);
          resetEditState(arrhythmiaContainer, editButton, saveButton, confirmEditBtn, arrhythmiaSelect, downloadTypeSelect, closeBtn);
          document.getElementById('mainContent').scrollIntoView({ behavior: 'smooth', block: 'start' });
        }, 100));
      }
  
      function resetEditState(container, editBtn, saveBtn, confirmBtn, select, downloadSelect,closeBtn=null) {
        if (container) container.style.display = 'none';
        if (editBtn) editBtn.style.display = 'inline-block';
        if (saveBtn) saveBtn.style.display = 'inline-block';
        if (confirmBtn) confirmBtn.style.display = 'none';
        if (select) select.value = '';
        if (downloadSelect) downloadSelect.style.display = 'inline-block';
        if (closeBtn) closeBtn.style.display = 'none';
      }
    }, 100));
  };
  
    const handleCardClick = (card, selectorType) => {
      card.style.cursor = 'pointer';
      card.addEventListener('click', debounce(async () => {
        const arr = card.dataset.arrhythmia;
        if (!arr) return console.error(`No data-arrhythmia attribute found on ${selectorType}!`);
  
        sessionStorage.removeItem('randomData');
        sessionStorage.removeItem('searchResults');
        sessionStorage.removeItem('totalPages');
        sessionStorage.removeItem('selectedArrhythmia');
        sessionStorage.setItem('selectedArrhythmia', arr);
        sessionStorage.setItem('dataSource', 'random');
  
        try {
          const json = await safeFetch(`/ommecgdata/fetch_random_ecg_data/${arr}/`, {
            method: 'GET',
            headers: { 'X-Requested-With': 'XMLHttpRequest' }
          });
  
          if (json.data?.length > 0) {
            sessionStorage.setItem('randomData', JSON.stringify(json.data));
            currentPage = 1;
            totalPages = json.total_pages || 1;
            sessionStorage.setItem('totalPages', totalPages);
            updateTableWithData(json.data);
            updatePaginationDisplay();
            window.location.href = card.dataset.redirect || `/ommecgdata/ecg_details/${arr}/`;
          } else {
            alertSystem.info('Info',json.message || `No random ECG data found for ${arr}`);
          }
        } catch (error) {
          alertSystem.error('Error','fetching random ECG data. Please try again.');
        }
      }, 100));
    };
  
    document.querySelectorAll(selectors.cards).forEach(card => handleCardClick(card, 'card'));
    document.querySelectorAll(selectors.cardTiles).forEach(card => handleCardClick(card, 'card-tile'));
  
  const updateTableWithData = async (data) => {
    const tableBody = document.getElementById('ecgTableBody');
    if (!tableBody) {
    console.error('Table body not found! Cannot update table with data.');
    return;  // Prevent further execution
    }
  
    tableBody.innerHTML = '';  
    data.forEach((row, index) => {
      const rowHtml = `
        <tr data-object-id="${row.object_id || ''}" data-lead-value="${row.LeadNumeric || row.Lead || ''}" data-samples-taken="${row.samples_taken || 0}">
          <td>${(currentPage - 1) * 10 + (index + 1)}</td>
          <td>${row.PatientID || ''}</td>
          <td>${row.Lead || ''}</td>
          <td>${row.Arrhythmia || ''}</td>
          <td>${row.Frequency || ''}</td>
          <td>${row.duration}</td>
          <td>  
                 <button class="icon-btn delete"><i class="fas fa-trash delete-btn" data-id="${row.object_id || ''}" data-collection="${row.collection_name || ''}" title="Delete"></i></button>
          </td>
        </tr>
        <tr class="plot-row" data-plot-id="${row.object_id || ''}" style="display: none;">
          <td colspan="7">
            <div id="plot-container-${row.object_id || ''}" class="plot-container">
              <div class="modal-header">
                   <h5 class="modal-title" style="display:flex; align-items:center; gap:8px;">
                    <i class="fas fa-heartbeat" style="color: var(--danger);"></i>
                              ECG Signal - ${row.PatientID || ''}
  <!-- Share dropdown -->
                  <div class="share-dropdown">
                    <i class="fas fa-share share-icon" 
                      data-id="${row.object_id || ''}" 
                      data-patient="${row.PatientID || ''}"></i>
  
                    <div class="share-menu" id="shareMenu-${row.object_id || ''}">
                      <a href="#" class="share-option whatsapp" data-patient="${row.PatientID || ''}">
                        <i class="fab fa-whatsapp"></i>
                      </a>
                      <a href="#" class="share-option facebook" data-patient="${row.PatientID || ''}">
                        <i class="fab fa-facebook"></i>
                      </a>
                      <a href="#" class="share-option teams" data-patient="${row.PatientID || ''}">
                        <i class="fab fa-microsoft"></i>
                      </a>
                      <a href="#" class="share-option email" data-patient="${row.PatientID || ''}">
                        <i class="fas fa-envelope"></i>
                      </a>
                    </div>
                  </div>
                </h5>
  
                <button type="button" class="btn-close plot-close" id="close-${row.object_id || ''}" data-id="${row.object_id || ''}" aria-label="Close"></button>
              </div>
              <div class="plot" id="plot-${row.object_id || ''}"></div>
              <div class="modal-footer">
                <div class="arrhythmia-label" id="arrhythmiaContainer-${row.object_id || ''}" style="display: none;">
                  <select class="form-control" id="Arrhythmia-${row.object_id || ''}" name="ArrhythmiaType">
                    <option value=" " selected>Select Arrhythmia Type</option>
                        <option value="Myocardial Infarction">Myocardial Infarction</option>
                        <option value="Atrial Fibrillation & Atrial Flutter">Atrial Fibrillation & Atrial Flutter</option>
                        <option value="HeartBlock">HeartBlock</option>
                        <option value="Junctional Rhythm">Junctional Rhythm</option>
                        <option value="Premature Atrial Contraction">Premature Atrial Contraction</option>
                        <option value="Premature Ventricular Contraction">Premature Ventricular Contraction</option>
                        <option value="Ventricular Fibrillation and Asystole">Ventricular Fibrillation and Asystole</option>
                        <option value="Noise">Noise</option>
                        <option value="Others">Others</option>
                        <option value="LBBB & RBBB">LBBB & RBBB</option>
                        <option value="Artifacts">Artifacts</option>
                        <option value="SINUS-ARR">SINUS-ARR</option>
                        <option value="ShortPause">ShortPause</option>
                        <option value="TC">TC</option>
                        <option value="WIDE-QRS">WIDE-QRS</option>
                        <option value="Abnormal">Abnormal</option>
                        <option value="Normal">Normal</option>
                  </select>
                </div>
                <div class="download-label" id="downloadContainer-${row.object_id || ''}">
                  <select class="form-control" id="downloadType-${row.object_id || ''}" name="downloadType">
                    <option value="" disabled selected>Download Type</option>
                    <option value="raw_data">Raw Data</option>
                    <option value="plot_png">Plot PNG</option>
                    <option value="pqrst_csv">PQRST CSV</option>
                    <option value="selected_data">Selected Data</option>
                  </select>
                </div>
                <div class="button-group">
                  <button class="btn btn-warning edit-btn" id="editEcgData-${row.object_id || ''}" data-id="${row.object_id || ''}">Edit</button>
                  <button class="btn btn-success save-btn" id="saveData-${row.object_id || ''}" data-id="${row.object_id || ''}">Save</button>
                  <button class="btn btn-primary confirm-edit-btn" id="confirmEditBtn-${row.object_id || ''}" data-id="${row.object_id || ''}" style="display: none;">Confirm</button>
                  <button class="btn btn-secondary close-btn" id="closeBtn-${row.object_id || ''}" style="display: none;">Close</button>              
                  </div>
              </div>
            </div>
          </td>
        </tr>`;
      tableBody.insertAdjacentHTML('beforeend', rowHtml);
    });
    updatePaginationDisplay();
    attachRowEventListeners();
  };
  
    const updatePaginationDisplay = () => {
      const paginationDiv = document.getElementById('paginationControls');
      if (!paginationDiv) return;
  
      paginationDiv.style.display = totalPages > 1 ? 'flex' : 'none';
      const span = paginationDiv.querySelector('.page-info') || document.createElement('span');
      span.className = 'page-info';
      span.textContent = `Page ${currentPage} of ${totalPages}`;
      if (!span.parentNode) {
        const prevBtn = document.getElementById('prevBtn');
        prevBtn?.insertAdjacentElement('afterend', span);
      }
  
      const prevBtn = document.getElementById('prevBtn');
      const nextBtn = document.getElementById('nextBtn');
      if (prevBtn) prevBtn.disabled = currentPage === 1;
      if (nextBtn) nextBtn.disabled = currentPage === totalPages;
    };
  
    const loadSessionData = () => {
      const randomDataString = sessionStorage.getItem('randomData');
      if (randomDataString) {
        try {
          const randomData = JSON.parse(randomDataString);
          if (Array.isArray(randomData) && randomData.length > 0) {
            currentPage = 1;
            totalPages = parseInt(sessionStorage.getItem('totalPages')) || 1;
            updateTableWithData(randomData);
          }
        } catch (error) {
          console.error('Error parsing randomData from sessionStorage:', error);
        }
      }
  
      const searchResultsString = sessionStorage.getItem('searchResults');
      if (searchResultsString) {
        try {
          const searchResults = JSON.parse(searchResultsString);
          totalPages = parseInt(sessionStorage.getItem('totalPages')) || 1;
          if (Array.isArray(searchResults) && searchResults.length > 0) {
            currentPage = 1;
            updateTableWithData(searchResults);
            searchContext = {
              patientId: document.getElementById('patientId')?.value.trim() || '',
              leadType: document.getElementById('leadType')?.value || '',
              arrhythmia: sessionStorage.getItem('selectedArrhythmia') || '',
              frequency: document.getElementById('frequency')?.value.trim() || ''
            };
          }
        } catch (error) {
          console.error('Error parsing searchResults from sessionStorage:', error);
        }
      }
    };
  
    // Enhanced single search with task management
    if (document.getElementById('searchButton')) {
      document.getElementById('searchButton').addEventListener('click', debounce(async () => {
        const fields = {
          patientId: document.getElementById('patientId')?.value.trim() || '',
          leadType: document.getElementById('leadType')?.value || '',
          arrhythmia: document.getElementById('arrhythmiaMI')?.value.trim() || '',
          frequency: document.getElementById('frequency')?.value.trim() || ''
        };
  
        if (Object.values(fields).some(value => !value)) {
  // alertSystem.warning('Warning','Please fill all search fields.');
          return;
        }
        cleanupModalBackdrops();
  
        // Try to close unifiedSearchModal
        const unifiedModal = document.getElementById('unifiedSearchModal');
        if (unifiedModal) {
          const unifiedInstance = bootstrap.Modal.getInstance(unifiedModal) || new bootstrap.Modal(unifiedModal);
          unifiedInstance.hide(); 
        }
  
          // Extra safety: also cleanup getPageModal if it exists
        const pageModal = document.getElementById("getPageModal");
        if (pageModal) {
          const pageInstance = bootstrap.Modal.getInstance(pageModal);
          if (pageInstance) {
            pageInstance.hide();
            pageInstance.dispose();
          }
        }
        // Create a task for this search
        const taskId = taskManager.createTask(
          'single_search',
          'Single ECG Search',
          `Searching for Patient: ${fields.patientId}, Lead: ${fields.leadType}, Arrhythmia: ${fields.arrhythmia}`,
          fields
        );
  
        // alertSystem.info('Task Created', 'Your search has been added to the task bar. You can continue working while it processes.');
  
        // Clear session storage
        sessionStorage.removeItem('randomData');
        sessionStorage.removeItem('searchResults');
        sessionStorage.removeItem('totalPages');
        sessionStorage.removeItem('selectedArrhythmia');
        sessionStorage.setItem('selectedArrhythmia', fields.arrhythmia);
        sessionStorage.setItem('dataSource', 'search');
  
        // Prepare form data
        const formData = new FormData();
        Object.entries(fields).forEach(([key, value]) => formData.append(key, value));
        formData.append('csrfmiddlewaretoken', document.querySelector('[name=csrfmiddlewaretoken]')?.value || '');
  
        // Execute search in background
        try {
          taskManager.updateTask(taskId, { progress: 20 });
          
          const result = await backgroundFetch('/ommecgdata/fetch_ecg_data/', {
            method: 'POST',
            body: formData,
            headers: { 'X-Requested-With': 'XMLHttpRequest' }
          }, taskId);
  
          if (result.status === 'success' && Array.isArray(result.data) && result.data.length > 0) {
            // Task completed successfully
            taskManager.completeTask(taskId, {
              data: result.data,
              total_pages: result.total_pages,
              arrhythmia: fields.arrhythmia
            });
  
  // alertSystem.success('Search Complete', `Found ${result.data.length} ECG records. Click the task to view results.`);
          } else {
            // No data found
            taskManager.errorTask(taskId, result.message || 'No ECG data found for the given criteria.');
  // alertSystem.warning('No Results', 'No ECG data found for the given criteria.');
          }
        } catch (error) {
          // Search failed
          taskManager.errorTask(taskId, error.message);
  // alertSystem.error('Search Failed', 'An error occurred while searching. Please try again.');
        }
      }, 100));
    }
  
    const loadPage = async (page) => {
      const pageLoader = document.getElementById('page-loader');
      if (pageLoader) pageLoader.style.display = 'flex';
  
      try {
        const dataSource = sessionStorage.getItem('dataSource') || '';
        let url;
        if (dataSource === 'random') {
          const arr = sessionStorage.getItem('selectedArrhythmia') || '';
          url = new URL(`/ommecgdata/fetch_random_ecg_data/${arr}/`, window.location.origin);
        } else {
          url = new URL('/ommecgdata/fetch_ecg_data/', window.location.origin);
          if (searchContext) {
            Object.entries(searchContext).forEach(([key, value]) => {
              if (value) url.searchParams.append(key, value);
            });
          }
        }
        url.searchParams.append('page', page);
  
        const result = await safeFetch(url.toString(), {
          method: 'GET',
          headers: { 'X-Requested-With': 'XMLHttpRequest' }
        });
  
        if (result.data && Array.isArray(result.data)) {
          currentPage = page;
          totalPages = result.total_pages || totalPages;
          sessionStorage.setItem('totalPages', totalPages);
          sessionStorage.setItem(dataSource === 'random' ? 'randomData' : 'searchResults', JSON.stringify(result.data));
          updateTableWithData(result.data);
          updatePaginationDisplay();
        } else {
          alertSystem.info('info','No data available for this page.');
        }
      } catch (error) {
        alertSystem.error('Error','loading page data.');
      } finally {
        if (pageLoader) pageLoader.style.display = 'none';
      }
    };
  
    if (document.getElementById('nextBtn')) {
      document.getElementById('nextBtn').addEventListener('click', debounce(() => {
        if (currentPage < totalPages) loadPage(currentPage + 1);
      }, 100));
    }
  
    if (document.getElementById('prevBtn')) {
      document.getElementById('prevBtn').addEventListener('click', debounce(() => {
        if (currentPage > 1) loadPage(currentPage - 1);
      }, 100));
    }
  
  const deleteData = (objectId, collection) => {
    // Show confirmation alert
    const alertId = alertSystem.show(
      'warning',
      'Confirm Deletion',
      'Are you sure you want to delete this record?',
      {
        actions: [
          {
            text: 'Delete',
            type: 'primary',
            handler: async () => {
              //Manually close the alert
              document.getElementById(alertId)?.remove();
  
              const pageLoader = document.getElementById('page-loader');
              if (pageLoader) pageLoader.style.display = 'flex';
  
              const formData = new FormData();
              formData.append('object_id', objectId);
              formData.append('collection_name', collection);
  
              try {
                const data = await safeFetch('/ommecgdata/delete_data/', {
                  method: 'POST',
                  headers: { 'X-CSRFToken': getCSRFToken() },
                  body: formData
                });
  
                if (data?.status === 'success') {
                  alertSystem.success('Success', 'Data deleted successfully!');
  
                  // Remove rows
                  document.querySelector(`tr[data-object-id="${objectId}"]`)?.remove();
                  document.querySelector(`tr[data-plot-id="${objectId}"]`)?.remove();
  
                  const remainingRows = document.querySelectorAll('table tbody tr:not(.plot-row)').length;
  
                  if (remainingRows > 0) {
                    reorderSerialNumbers();
                    return;
                  }
  
                  // No rows left Fetch next or redirect
                  const dataSource = sessionStorage.getItem('dataSource') || '';
                  let url;
                  if (dataSource === 'random') {
                    const arr = sessionStorage.getItem('selectedArrhythmia') || '';
                    url = new URL(`/ommecgdata/fetch_random_ecg_data/${arr}/`, window.location.origin);
                  } else {
                    url = new URL(`/ommecgdata/fetch_ecg_data/?page=${currentPage}`, window.location.origin);
                    if (searchContext) {
                      Object.entries(searchContext).forEach(([key, value]) => {
                        if (value) url.searchParams.append(key, value);
                      });
                    }
                  }
  
                  try {
                    const updatedData = await safeFetch(url.toString(), {
                      method: 'GET',
                      headers: { 'X-Requested-With': 'XMLHttpRequest' }
                    });
  
                    if (updatedData?.total_records !== undefined) {
                      const totalRecords = updatedData.total_records;
                      totalPages = Math.ceil(totalRecords / 10);
                      sessionStorage.setItem('totalPages', totalPages);
  
                      if (totalRecords === 0) {
                        window.location.href = "/ommecgdata/";
                        return;
                      }
  
                      if (currentPage > totalPages) currentPage = totalPages || 1;
  
                      let currentPageData = updatedData.data || [];
                      updateTableWithData(currentPageData);
                      reorderSerialNumbers();
                    } else {
                      window.location.href = "/ommecgdata/";
                    }
                  } catch {
                    window.location.href = "/ommecgdata/";
                  }
                } else {
                  alertSystem.error('Error', 'Could not delete data.');
                }
              } catch (error) {
                console.error('Delete error:', error);
                alertSystem.error('Error', 'An error occurred while deleting the record.');
              } finally {
                if (pageLoader) pageLoader.style.display = 'none';
              }
            }
          },
          {
            text: 'Cancel',
            type: 'secondary',
            handler: () => {
              //Manually close the alert
              document.getElementById(alertId)?.remove();
              alertSystem.info('Cancelled', 'Delete action cancelled.');
            }
          }
        ],
        persistent: true
      }
    );
  };
  
    const reorderSerialNumbers = () => {
      const rows = document.querySelectorAll('table tbody tr:not(.plot-row)');
      rows.forEach((row, index) => {
        const srNoCell = row.querySelector('td');
        if (srNoCell) srNoCell.textContent = (currentPage - 1) * 10 + (index + 1);
      });
    };
  
  const fetchAndPlotECG = async (ecgData, leadType, patientId, objectId, leadConfig) => {
      const pageLoader = document.getElementById('page-loader');
      if (pageLoader) pageLoader.style.display = 'flex';
  
      const leadNumber = parseInt(leadType);
      const plotElement = document.getElementById(`plot-${objectId}`);
      if (!plotElement) {
        alertSystem.warning('Warning', 'Plot element not found.');
        if (pageLoader) pageLoader.style.display = 'none';
        return;
      }
  
      plotElement.style.width = '100%';
      plotElement.style.height = '650px';
      Plotly.purge(plotElement);
      plotElement.innerHTML = '';
  
      // Min-max scaling function
      const minMaxScale = (data, minVal = 0, maxVal = 4) => {
        const min = Math.min(...data);
        const max = Math.max(...data);
        if (max === min) return data.map(() => (maxVal + minVal) / 2);
        return data.map(value => ((value - min) / (max - min)) * (maxVal - minVal) + minVal);
      };
  
      // Enhanced peak detection (used as fallback)
      const detectPeaks = (data) => {
        const peaks = { p: [], q: [], r: [], s: [], t: [] };
        const windowSize = 200;
        for (let i = windowSize; i < data.length - windowSize; i++) {
          const segment = data.slice(i - windowSize, i + windowSize + 1);
          const maxIdx = segment.indexOf(Math.max(...segment)) + (i - windowSize);
          const minIdx = segment.indexOf(Math.min(...segment)) + (i - windowSize);
          if (data[maxIdx] > 2.0 && !peaks.r.includes(maxIdx)) peaks.r.push(maxIdx);
          if (data[minIdx] < 1.0 && !peaks.q.includes(minIdx) && minIdx < peaks.r[peaks.r.length - 1]) peaks.q.push(minIdx);
          if (data[minIdx] < 1.0 && !peaks.s.includes(minIdx) && minIdx > peaks.r[peaks.r.length - 1]) peaks.s.push(minIdx);
          if (peaks.r.length > 1) {
            const prevR = peaks.r[peaks.r.length - 2];
            const nextR = peaks.r[peaks.r.length - 1];
            const pIdx = Math.floor((prevR + nextR) / 4);
            const tIdx = Math.floor((3 * prevR + nextR) / 4);
            if (data[pIdx] > 1.0 && !peaks.p.includes(pIdx)) peaks.p.push(pIdx);
            if (data[tIdx] > 1.0 && !peaks.t.includes(tIdx)) peaks.t.push(tIdx);
          }
        }
        return peaks;
      };
  
      // Fetch PQRST data with lead_config
      let pqrst = null;
      try {
        const pqrstResponse = await safeFetch('/ommecgdata/get_pqrst_data/', {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
            'X-CSRFToken': getCSRFToken()
          },
          body: JSON.stringify({
            object_id: objectId,
            arrhythmia: sessionStorage.getItem('selectedArrhythmia') || '',
            lead_config: leadConfig
          })
        });
        if (pqrstResponse.status === 'success') {
          pqrst = {
            p_points: pqrstResponse.p_points || [],
            q_points: pqrstResponse.q_points || [],
            r_peaks: pqrstResponse.r_peaks || [],
            s_points: pqrstResponse.s_points || [],
            t_points: pqrstResponse.t_points || [],
            lead_peaks: pqrstResponse.lead_peaks || null
          };
          // Store PQRST data in window.ecgData
          window.ecgData[objectId].pqrst = JSON.parse(JSON.stringify(pqrst));
        }
      } catch (error) {
        console.error('Error fetching PQRST data:', error);
        alertSystem.error('Error', 'Failed to fetch PQRST data. Using fallback peak detection.');
      }
  
      function enableSynchronizedCursor(objectId) {
        const plotContainer = document.getElementById(`plot-${objectId}`);
        const plotDivs = plotContainer.querySelectorAll('.js-plotly-plot');
  
        plotDivs.forEach((plotDiv, index) => {
          // Add a new trace (red cursor circle) to each plot
          Plotly.addTraces(plotDiv, {
            x: [],
            y: [],
            mode: 'markers',
            marker: {
              size: 10,
              color: 'red',
              symbol: 'circle'
            },
            name: 'cursor',
            hoverinfo: 'skip',
            showlegend: false
          });
        });
  
        plotDivs.forEach((plotDiv, index) => {
          plotDiv.on('plotly_hover', function(eventData) {
            const x = eventData.points[0].x;
  
            // Loop through each plot and update the cursor trace
            plotDivs.forEach((targetPlot, i) => {
              const traceData = targetPlot.data[0];  // assuming main signal is trace 0
              const yData = traceData.y;
              const xData = traceData.x;
  
              // Find closest point to x
              let closestIndex = 0;
              let minDiff = Infinity;
              for (let j = 0; j < xData.length; j++) {
                const diff = Math.abs(xData[j] - x);
                if (diff < minDiff) {
                  closestIndex = j;
                  minDiff = diff;
                }
              }
  
              Plotly.restyle(targetPlot, {
                x: [[xData[closestIndex]]],
                y: [[yData[closestIndex]]]
              }, targetPlot.data.length - 1); // last trace = cursor
            });
          });
  
          plotDiv.on('plotly_unhover', function() {
            // Hide cursor when mouse leaves
            plotDivs.forEach(p => {
              Plotly.restyle(p, {
                x: [[]],
                y: [[]]
              }, p.data.length - 1);
            });
          });
        });
      }
  
      if (leadNumber === 7 || leadNumber === 12) {
         plotElement.style.height = 'auto';
        const leadData = ecgData.leadDict;
        const leadNames = Object.keys(leadData);
        enableSynchronizedCursor(objectId);
        const traces = [];
        const gridRows = leadNames.length;
  
        // Map PQRST indices to each lead
        const allPeaks = {};
        if (pqrst && pqrst.lead_peaks) {
          // Use per-lead PQRST data if available
          leadNames.forEach(lead => {
            allPeaks[lead] = {
              p: pqrst.lead_peaks[lead]?.p_points || [],
              q: pqrst.lead_peaks[lead]?.q_points || [],
              r: pqrst.lead_peaks[lead]?.r_peaks || [],
              s: pqrst.lead_peaks[lead]?.s_points || [],
              t: pqrst.lead_peaks[lead]?.t_points || []
            };
            Object.keys(allPeaks[lead]).forEach(type => {
              allPeaks[lead][type] = allPeaks[lead][type].filter(i => i >= 0 && i < leadData[lead].length);
            });
          });
        } else if (pqrst) {
          // Fallback: Use global PQRST indices and scale them to each lead
          const refLength = leadData[leadNames[0]].length;
          const refMax = Math.max(...Object.values(leadData).map(arr => arr.length - 1));
          leadNames.forEach(lead => {
            const leadLength = leadData[lead].length;
            allPeaks[lead] = {
              p: pqrst.p_points ? pqrst.p_points.map(i => Math.round((i / refMax) * (leadLength - 1))) : [],
              q: pqrst.q_points ? pqrst.q_points.map(i => Math.round((i / refMax) * (leadLength - 1))) : [],
              r: pqrst.r_peaks ? pqrst.r_peaks.map(i => Math.round((i / refMax) * (leadLength - 1))) : [],
              s: pqrst.s_points ? pqrst.s_points.map(i => Math.round((i / refMax) * (leadLength - 1))) : [],
              t: pqrst.t_points ? pqrst.t_points.map(i => Math.round((i / refMax) * (leadLength - 1))) : []
            };
            Object.keys(allPeaks[lead]).forEach(type => {
              allPeaks[lead][type] = allPeaks[lead][type].filter(i => i >= 0 && i < leadLength);
            });
          });
        } else {
          // Fallback to client-side peak detection
          leadNames.forEach(lead => {
            allPeaks[lead] = detectPeaks(leadData[lead].map(y => minMaxScale([y])[0]));
          });
        }
  
        // Store allPeaks in window.ecgData for use in pqrst_csv case
        window.ecgData[objectId].allPeaks = JSON.parse(JSON.stringify(allPeaks));
  
        leadNames.forEach((lead, idx) => {
          const y = leadData[lead];
          const x = Array.from({ length: y.length }, (_, i) => i);
          const scaledY = minMaxScale(y);
          const xAxisName = `x${idx + 1}`;
          const yAxisName = `y${idx + 1}`;
  
          traces.push({
            x,
            y: scaledY,
            name: 'ECG',
            type: 'scatter',
            mode: 'lines',
            line: { width: 1.3, color: 'black' },
            xaxis: xAxisName,
            yaxis: yAxisName,
            showlegend: idx === 0
          });
  
          const peaks = allPeaks[lead];
          ['p', 'q', 'r', 's', 't'].forEach(type => {
            if (peaks[type].length > 0) {
              traces.push({
                x: peaks[type].map(i => x[i]),
                y: peaks[type].map(i => scaledY[i]),
                mode: 'markers',
                name: type.toUpperCase(),
                type: 'scatter',
                marker: { 
                  color: type === 'p' ? 'orange' : type === 'q' ? 'blue' : type === 'r' ? 'red' : type === 's' ? 'green' : 'purple',
                  size: 7,
                  symbol: 'circle'
                },
                xaxis: xAxisName,
                yaxis: yAxisName,
                showlegend: idx === 0,
                customdata: { lead, type, indices: peaks[type] },
                ids: peaks[type].map((i, idx) => `${type}-${idx}`)
              });
            }
          });
        });
        const firstLead = leadNames[0];
        const dataLength = leadData[firstLead].length;
  
        const windowSize = 2000;  // for future scroll/pan if needed
  
        const rawDtick = (dataLength < windowSize)
          ? Math.round(dataLength / 25)   // e.g., 500/25 = 20
          : 100;
  
        const xDtick = Math.ceil(rawDtick / 5) * 5;  // round to nearest 5
        const xMinorDtick = xDtick / 5;
        const layout = {
          grid: { rows: gridRows, columns: 1, pattern: 'independent' },
          height: 500 * gridRows,
          showlegend: true,
          legend: { x: 1, xanchor: 'right', y: 1, bgcolor: 'rgba(255, 255, 255, 0.5)' },
          margin: { t: 40, b: 40, l: 50, r: 40 },
          plot_bgcolor: document.body.dataset.theme === 'dark' ? '#1e1e2f' : 'white',
          paper_bgcolor: document.body.dataset.theme === 'dark' ? '#1e1e2f' : 'white',
          font: { color: document.body.dataset.theme === 'dark' ? '#ffffff' : '#000000' }
        };
  
        leadNames.forEach((lead, idx) => {
          const axisX = `xaxis${idx + 1}`;
          const axisY = `yaxis${idx + 1}`;
          layout[axisX] = {
            range: [0, leadData[lead].length<= windowSize ?leadData[lead].length:windowSize],
            title: { text: 'Time Index', standoff: 20, font: { size: 12 } },
            showgrid: true,
            gridcolor: 'rgba(245, 129, 129, 1)',
            zeroline: false,
            dtick: xDtick,
            tickfont: { size: 12 },
            minor: { showgrid: true, gridcolor: 'rgba(244, 210, 216, 1)',dtick: xMinorDtick,tick0: 0,
            },
            fixedrange: false 
          };
          layout[axisY] = {
            range: [0, 4],
            title: { text: lead, standoff: 15, font: { size: 14 } },
            showgrid: true,
            gridcolor: 'rgba(245, 129, 129, 1)',
            zeroline: false,
            dtick: 0.5,
            tickfont: { size: 12 },
            minor: { showgrid: true, gridcolor: 'rgba(244, 210, 216, 1)'},fixedrange: false 
          };
        });
  
        const config = {
          responsive: true,
          displaylogo: false,
          modeBarButtonsToAdd: [ 'pan2d', 'zoom2d', 'select2d', 'autoScale2d', 'resetScale2d', 'zoomIn2d', 'zoomOut2d', 'toImage']
        };
  
        Plotly.newPlot(plotElement, traces, layout, config).then(() => {
          let selectedPoint = null;
          let lastUpdateTime = 0;
          const debounceDelay = 50; // ms
  
          plotElement.on('plotly_click', function (eventData) {
            const pt = eventData.points[0];
            const traceName = pt.data.name;
            const key = traceName.toLowerCase();
            if (!['p', 'q', 'r', 's', 't'].includes(key)) return;
  
            selectedPoint = {
              key: key,
              pointIndex: pt.pointIndex,
              traceIndices: traces
                .map((t, i) => ({ trace: t, index: i }))
                .filter(t => t.trace.customdata?.type === key)
                .map(t => t.index),
              offsetY: pt.y
            };
          });
  
          plotElement.addEventListener('mousemove', function (e) {
            if (!selectedPoint) return;
  
            const now = Date.now();
            if (now - lastUpdateTime < debounceDelay) return;
            lastUpdateTime = now;
  
            const boundingRect = plotElement.getBoundingClientRect();
            const xFrac = (e.clientX - boundingRect.left) / boundingRect.width;
  
            // Use the first subplot's xaxis for consistency
            const xaxis = plotElement._fullLayout.xaxis || plotElement._fullLayout.xaxis1;
            const newTime = xaxis.p2l(xFrac * boundingRect.width);
  
            // Find closest index using first lead's x values
            const x = Array.from({ length: leadData[leadNames[0]].length }, (_, i) => i);
            let closestIndex = 0;
            let minDiff = Infinity;
            for (let i = 0; i < x.length; i++) {
              const diff = Math.abs(x[i] - newTime);
              if (diff < minDiff) {
                closestIndex = i;
                minDiff = diff;
              }
            }
            if (closestIndex < 0 || closestIndex >= x.length) return;
  
            // Update peak index across all leads
            const updateX = [];
            const updateY = [];
            leadNames.forEach(lead => {
              const scaledY = minMaxScale(leadData[lead]);
              const peaks = allPeaks[lead];
              if (peaks[selectedPoint.key].length > selectedPoint.pointIndex) {
                peaks[selectedPoint.key][selectedPoint.pointIndex] = closestIndex;
              }
              updateX.push(peaks[selectedPoint.key].map(i => x[i]));
              updateY.push(peaks[selectedPoint.key].map(i => scaledY[i]));
            });
  
            Plotly.restyle(plotElement, {
              x: updateX,
              y: updateY
            }, selectedPoint.traceIndices);
          });
  
          plotElement.addEventListener('mouseup', function () {
            if (selectedPoint) {
              const finalIndex = allPeaks[leadNames[0]][selectedPoint.key][selectedPoint.pointIndex];
              window.ecgData[objectId].allPeaks = JSON.parse(JSON.stringify(allPeaks));
              selectedPoint = null;
            }
          });
  
          plotElement.on('plotly_selected', (eventData) => {
            if (eventData && eventData.points && eventData.points.length > 0) {
              const xValues = eventData.points.map(pt => pt.x);
              const minX = Math.min(...xValues);
              const maxX = Math.max(...xValues);
  
              const rawEcgData = window.ecgData[objectId];
              if (!rawEcgData || !rawEcgData.leadDict) {
                window.selectedData = null;
                alertSystem.warning('Warning', 'No raw ECG data available for selection.');
                return;
              }
  
              const selectedData = {};
              Object.keys(rawEcgData.leadDict).forEach(lead => {
                const leadData = rawEcgData.leadDict[lead];
                const x = Array.from({ length: leadData.length }, (_, i) => i);
                const selectedIndices = x
                  .map((xVal, idx) => ({ x: xVal, idx }))
                  .filter(item => item.x >= minX && item.x <= maxX)
                  .map(item => item.idx);
  
                selectedData[lead] = {
                  x: selectedIndices.map(idx => x[idx]),
                  y: selectedIndices.map(idx => leadData[idx])
                };
              });
  
              window.selectedData = selectedData;
            } else {
              window.selectedData = null;
            }
          });
        }).finally(() => {
          if (pageLoader) pageLoader.style.display = 'none';
        });
  
  
      } else if (leadNumber === 2) {
        const processData = async (data) => {
          try {
            const correctedData = await safeFetch('/ommecgdata/process_ecg/', {
              method: 'POST',
              headers: {
                'Content-Type': 'application/json',
                'X-CSRFToken': getCSRFToken()
              },
              body: JSON.stringify({ ...data, objectId, arrhythmia: sessionStorage.getItem('selectedArrhythmia') || '', lead_config: leadConfig })
            });
  
            if (!correctedData?.x || !correctedData?.y || correctedData.x.length === 0 || correctedData.y.length === 0) {
              throw new Error('Invalid processed ECG data');
            }
  
            const scaledY = minMaxScale(correctedData.y);
            const updatedData = { ...correctedData, y: scaledY };
  
            // Ensure PQRST data is stored
            if (!window.ecgData[objectId].pqrst && pqrst) {
              window.ecgData[objectId].pqrst = JSON.parse(JSON.stringify(pqrst));
            }
            plotECG(updatedData , pqrst, objectId);
          } catch (error) {
            plotElement.innerHTML = `<p style="text-align:center; color: red;">Error processing ECG data: ${error.message}</p>`;
            console.error('Error processing ECG data:', error);
          } finally {
            if (pageLoader) pageLoader.style.display = 'none';
          }
        };
  
        if (ecgData?.x && ecgData?.y && ecgData.x.length > 0 && ecgData.y.length > 0) {
          await processData(ecgData);
        } else {
          plotElement.innerHTML = '<p style="text-align:center; color: red;">Invalid ECG data format received.</p>';
          console.error('Invalid ecgData format:', ecgData);
          if (pageLoader) pageLoader.style.display = 'none';
        }
      } else {
        plotElement.innerHTML = '<p style="text-align:center; color: red;">Invalid lead configuration.</p>';
        console.error('Invalid leadNumber:', leadNumber);
        if (pageLoader) pageLoader.style.display = 'none';
      }
    };
  
   const plotECG = (data, pqrst = null, objectId) => {
    const pageLoader = document.getElementById('page-loader');
    if (pageLoader) pageLoader.style.display = 'flex';
  
    const plotElement = document.getElementById(`plot-${objectId}`);
    if (!plotElement) {
      alertSystem.error('Error', 'Plot element not found.');
      if (pageLoader) pageLoader.style.display = 'none';
      return;
    }
  
    Plotly.purge(plotElement);
    plotElement.innerHTML = '';
  
    if (!data.x?.length || !data.y?.length) {
      plotElement.innerHTML = '<p style="text-align:center; color: red;">No valid data to plot.</p>';
      if (pageLoader) pageLoader.style.display = 'none';
      return;
    }
  
    // --- Scale Y between 0 and 4 ---
    const minMaxScale = (arr, minVal = 0, maxVal = 4) => {
      const min = Math.min(...arr);
      const max = Math.max(...arr);
      if (max === min) return arr.map(() => (maxVal + minVal) / 2);
      return arr.map(v => ((v - min) / (max - min)) * (maxVal - minVal) + minVal);
    };
    const scaledY = minMaxScale(data.y);
  
    // --- Main ECG waveform trace ---
    const mainTrace = {
      x: data.x,
      y: scaledY,
      mode: 'lines',
      name: 'ECG',
      line: { color: 'black', width: 1.5 },
      hoverinfo: 'x+y',
      xaxis: 'x2', // main waveform (range slider enabled)
      yaxis: 'y'
    };
  
    // --- Generate marker traces (PQRST) ---
    const markerTraces = [];
    if (pqrst) {
      const markerSize = 8;
      const points = [
        ['p_points', 'P', 'orange'],
        ['q_points', 'Q', 'blue'],
        ['r_peaks', 'R', 'red'],
        ['s_points', 'S', 'green'],
        ['t_points', 'T', 'purple']
      ];
  
      points.forEach(([key, label, color]) => {
        if (pqrst[key]?.length) {
          markerTraces.push({
            x: pqrst[key].map(i => data.x[i]),
            y: pqrst[key].map(i => scaledY[i]),
            mode: 'markers',
            name: label,
            type: 'scatter',
            marker: { color, size: markerSize, symbol: 'circle' },
            hoverinfo: 'x+y',
            xaxis: 'x', // Overlay axis (no range slider)
            yaxis: 'y'
          });
        }
      });
    }
  
    // --- Grid calculations ---
    const dataLength = data.x.length;
    const windowSize = 2000;
    const rawDtick = dataLength < windowSize ? Math.round(dataLength / 25) : 100;
    const xDtick = Math.ceil(rawDtick / 5) * 5;
    const xMinorDtick = xDtick / 5;
  
    // --- Layout ---
    const layout = {
      xaxis2: {
        domain: [0, 1],
        range: [0, Math.min(windowSize, dataLength)],
        autorange: false,
        title: { text: 'Time Index', standoff: 20 },
        showgrid: true,
        gridcolor: 'rgba(245, 121, 121, 0.93)',
        zeroline: false,
        dtick: xDtick,
        minor: {
          showgrid: true,
          gridcolor: 'rgba(244, 210, 216, 0.92)',
          dtick: xMinorDtick,
          tick0: 0
        },
        rangeslider: {
          visible: true,
          thickness: 0.07,
          bgcolor: 'rgba(245,121,121,0.1)',
          bordercolor: 'rgba(245,121,121,0.4)',
        }
      },
  
      xaxis: {
        domain: [0, 1],
        overlaying: 'x2',
        matches: 'x2',
        showgrid: false,
        zeroline: false,
        visible: false // hidden but active for markers
      },
  
      yaxis: {
        range: [0, 4.1],
        title: { text: 'ECG (mV)' },
        showgrid: true,
        gridcolor: 'rgba(245, 121, 121, 0.93)',
        zeroline: false,
        dtick: 0.5,
        minor: { showgrid: true, gridcolor: 'rgba(244, 210, 216, 0.92)' }
      },
  
      margin: { t: 60, b: 70, l: 40, r: 40 },
      showlegend: true,
      legend: { x: 1, xanchor: 'right', y: 1, bgcolor: 'rgba(255,255,255,0.5)' },
      plot_bgcolor: document.body.dataset.theme === 'dark' ? '#1e1e2f' : 'white',
      paper_bgcolor: document.body.dataset.theme === 'dark' ? '#1e1e2f' : 'white',
      font: { color: document.body.dataset.theme === 'dark' ? '#fff' : '#000' }
    };
  
    const config = {
      responsive: true,
      displaylogo: false,
      modeBarButtonsToAdd: ['pan2d', 'zoom2d', 'select2d', 'autoScale2d', 'resetScale2d']
    };
  
    // --- Draw main waveform first, then add markers ---
    Plotly.newPlot(plotElement, [mainTrace], layout, config).then(() => {
      if (markerTraces.length) {
        Plotly.addTraces(plotElement, markerTraces);
      }
  
      enablePQRSTEditing(plotElement, data, pqrst);
    }).finally(() => {
      if (pageLoader) pageLoader.style.display = 'none';
    });
  
    // --- Editable marker movement logic ---
    function enablePQRSTEditing(plotElement, data, pqrst) {
      if (!pqrst) return;
      let selectedPoint = null;
      const updated = JSON.parse(JSON.stringify(pqrst));
  
      plotElement.on('plotly_click', event => {
        const pt = event.points[0];
        const traceName = pt.data.name;
        const key = traceName.toLowerCase();
        if (!['p', 'q', 'r', 's', 't'].includes(key)) return;
        selectedPoint = {
          key: key === 'r' ? 'r_peaks' : `${key}_points`,
          pointIndex: pt.pointIndex,
          traceIndex: pt.curveNumber
        };
      });
  
      plotElement.addEventListener('mousemove', e => {
        if (!selectedPoint) return;
        const rect = plotElement.getBoundingClientRect();
        const xFrac = (e.clientX - rect.left) / rect.width;
        const xaxis = plotElement._fullLayout.xaxis2; // FIXED: correct axis for visible waveform
        const newTime = xaxis.range[0] + xFrac * (xaxis.range[1] - xaxis.range[0]);
        const closestIndex = data.x.findIndex(v => Math.round(v) === Math.round(newTime));
        if (closestIndex === -1) return;
        updated[selectedPoint.key][selectedPoint.pointIndex] = closestIndex;
        Plotly.restyle(plotElement, {
          x: [updated[selectedPoint.key].map(i => data.x[i])],
          y: [updated[selectedPoint.key].map(i => scaledY[i])]
        }, [selectedPoint.traceIndex]);
      });
  
      plotElement.addEventListener('mouseup', () => {
        if (selectedPoint) {
          window.ecgData[objectId].pqrst = JSON.parse(JSON.stringify(updated));
          selectedPoint = null;
        }
      });
    }
  };
  
    window.clearForms = () => {
      ['insertForm', 'fetchEcgForm'].forEach(id => {
        const form = document.getElementById(id);
        if (form) form.reset();
      });
    };
  
  window.previewECG = async function () {
      const csvFileInput = document.getElementById('csv_file');
      const leadTypeSelect = document.getElementById('leadTypeMI');
      const pageLoader = document.getElementById('page-loader');
      const previewModal = document.getElementById('previewLeadModal');
  
      if (!csvFileInput.files.length) {
          alertSystem.warning('Warning', 'Please upload a CSV file.');
          return;
      }
      if (!leadTypeSelect.value) {
          alertSystem.warning('Warning', 'Please select a lead type.');
          return;
      }
      if (!previewModal) {
          alertSystem.error('Error', 'Preview modal not found.');
          return;
      }
  
      const leadType = leadTypeSelect.value;
      const file = csvFileInput.files[0];
  
      if (pageLoader) pageLoader.style.display = 'flex';
  
      // Clear previous preview content
      const plotContainer = document.getElementById('preview-lead-container');
      if (plotContainer) {
          plotContainer.innerHTML = '<p style="text-align:center;font-weight:bold;">Loading ECG data...</p>';
          Plotly.purge(plotContainer);
      }
  
      try {
          // Read CSV file
          const text = await new Promise((resolve, reject) => {
              const reader = new FileReader();
              reader.onload = (event) => resolve(event.target.result);
              reader.onerror = () => reject(new Error('Failed to read CSV file.'));
              reader.readAsText(file);
          });
          const rows = text.trim().split('\n').map(row => row.split(',').map(cell => cell.trim()));
  
          let ecgData;
          let leadNames;
          
          // Find the first non-empty row as the header
          let headerIndex = 0;
          while (headerIndex < rows.length && (rows[headerIndex].length === 0 || rows[headerIndex].every(cell => cell === ''))) {
              headerIndex++;
          }
          if (headerIndex >= rows.length) {
              throw new Error('No valid header or data found in CSV.');
          }
          const header = rows[headerIndex];
  
          // Validate CSV lead count
          const detectedLeadCount = header[0].toLowerCase() === 'datetime' ? header.length - 1 : header.length;
  
          if (leadType === '1' && detectedLeadCount !== 2) {
              throw new Error(`Expected 2-lead CSV.`);
          } else if (leadType === '7' && detectedLeadCount !== 7) {
              if (detectedLeadCount === 12) {
                  throw new Error('12-lead CSV detected. Please select 12-lead option instead of 7-lead.');
              } else {
                  throw new Error(`Expected 7-lead CSV.`);
              }
          } else if (leadType === '12' && detectedLeadCount !== 12) {
              throw new Error(`Expected 12-lead CSV.`);
          }
        if (leadType === '2') {
              // Handle 2-lead CSV 
              const header = rows[headerIndex].length > 1 ? rows[headerIndex] : ['Index', 'II']; 
              let x = [], y = [];
  
              for (let i = headerIndex + 1; i < rows.length; i++) {
                  const row = rows[i];
                  if (row.length < 2) continue;
                  const xRaw = parseFloat(row[0]);
                  const yRaw = parseFloat(row[1]);
  
                  if (isNaN(yRaw)) continue;
                  y.push(yRaw);
  
                  if (isNaN(xRaw) || xRaw === 2024 || (x.length > 0 && xRaw === x[x.length - 1])) {
                      x.push(i - headerIndex); 
                  } else {
                      x.push(xRaw);
                  } 
              }
  
              if (x.length === 0 || y.length === 0) {
                  throw new Error('Invalid or empty CSV data for 2-lead ECG. Ensure the second column contains numeric data.');
              }
  
              const maxPoints = 2000;
              if (x.length > maxPoints) {
                  const step = Math.floor(x.length / maxPoints);
                  ecgData = {
                      x: x.filter((_, i) => i % step === 0),
                      y: y.filter((_, i) => i % step === 0)
                  };
              } else {
                  ecgData = { x, y };
              }
  
              if (new Set(ecgData.x).size === 1 || ecgData.x.length !== ecgData.y.length) {
                  ecgData.x = Array.from({ length: ecgData.y.length }, (_, i) => i);
              }
  
              leadNames = ['II'];
          } else if (leadType === '7' || leadType === '12') {
              // Handle 7-lead or 12-lead CSV
              let adjustedHeader = header;
              let startIdx = 0;
              const expectedLeadCount = leadType === '7' ? 7 : 12;
  
              if (header[0].toLowerCase() === 'datetime') {
                  adjustedHeader = header.slice(1);
                  startIdx = 1;
                  if (adjustedHeader.length !== expectedLeadCount) {
                      throw new Error(`After skipping DateTime column, found ${adjustedHeader.length} leads, but expected ${expectedLeadCount} leads.`);
                  }
              }
  
              leadNames = adjustedHeader;
              const leadDict = {};
              leadNames.forEach(lead => leadDict[lead] = []);
  
              for (let i = headerIndex + 1; i < rows.length; i++) {
                  const row = rows[i];
                  if (row.length >= leadNames.length + startIdx) {
                      leadNames.forEach((lead, idx) => {
                          const value = parseFloat(row[idx + startIdx]);
                          if (!isNaN(value)) {
                              leadDict[lead].push(value);
                          } else {
                              console.warn(`Row ${i}, Column ${lead} has invalid data: ${row[idx + startIdx]}`);
                          }
                      });
                  }
              }
  
              if (Object.values(leadDict).some(arr => arr.length === 0)) {
                  throw new Error('Invalid or empty CSV data for multi-lead ECG.');
              }
  
              // Sample data to 2000 points
              const maxPoints = 2000;
              const sampledLeadDict = {};
              for (const lead in leadDict) {
                  if (leadDict[lead].length > maxPoints) {
                      const step = Math.floor(leadDict[lead].length / maxPoints);
                      sampledLeadDict[lead] = leadDict[lead].filter((_, i) => i % step === 0);
                  } else {
                      sampledLeadDict[lead] = leadDict[lead];
                  }
              }
              if (Object.values(sampledLeadDict).some(arr => arr.length === 0)) {
                  throw new Error('Sampling resulted in empty data.');
              }
              ecgData = { leadDict: sampledLeadDict };
          } else {
              throw new Error('Invalid lead type selected.');
          }
  
          // Show the preview modal
          const modalInstance = bootstrap.Modal.getInstance(previewModal) || new bootstrap.Modal(previewModal);
          modalInstance.show();
  
          // Plot the ECG data
          await window.plotECGPreview(ecgData, leadType, leadNames, plotContainer);
      } catch (error) {
          plotContainer.innerHTML = `<p style="text-align:center; color: red;">Error loading ECG data: ${error.message}</p>`;
          console.error('Error in previewECG:', error);
          alertSystem.error('Error', `Failed to load ECG data: ${error.message}`);
      } finally {
          if (pageLoader) pageLoader.style.display = 'none';
      }
  };
  
  // Modified plotECGPreview with enhanced resize handling
  window.plotECGPreview = async function (ecgData, leadType, leadNames, plotContainer) {
      const pageLoader = document.getElementById('page-loader');
      if (pageLoader) pageLoader.style.display = 'flex';
  
      if (!plotContainer) {
          alertSystem.error('Error', 'Plot container not found.');
          if (pageLoader) pageLoader.style.display = 'none';
          return;
      }
      plotContainer.innerHTML = ''; // Clear old plots
  
      // Detect current theme mode
      const isDarkMode = document.documentElement.getAttribute('data-theme') === 'dark';
      const bgColor = isDarkMode ? '#1e2a44' : '#ffffff';
      const fontColor = isDarkMode ? '#fff' : '#000';
  
      if (typeof Plotly === 'undefined') {
          plotContainer.innerHTML = '<p style="text-align:center; color: red;">Plotly library not loaded.</p>';
          console.error('Plotly is undefined');
          if (pageLoader) pageLoader.style.display = 'none';
          return;
      }
  
      // Min-max scaling function
      const minMaxScale = (data, minVal = 0, maxVal = 4) => {
          const validData = data.filter(v => !isNaN(v));
          if (validData.length === 0) {
              console.error("No valid numeric values in Y array.");
              return data.map(() => (maxVal + minVal) / 2);
          }
          const min = Math.min(...validData);
          const max = Math.max(...validData);
          if (max === min) {
              console.warn('Flat data detected. All Y values are the same.');
              return data.map(() => (maxVal + minVal) / 2);
          }
          return data.map(value => ((value - min) / (max - min)) * (maxVal - minVal) + minVal);
      };
  
      // Clear previous content
      plotContainer.innerHTML = '';
      plotContainer.style.padding = '10px';
  
      if (leadType === '2') {
          if (!ecgData.x || !ecgData.y || ecgData.x.length === 0 || ecgData.y.length === 0) {
              plotContainer.innerHTML = '<p style="text-align:center; color: red;">No valid data to plot for 2-lead (II).</p>';
              console.error('Invalid data for 2-lead plotting:', ecgData);
              if (pageLoader) pageLoader.style.display = 'none';
              return;
          }
  
          const plotDiv = document.createElement('div');
          plotDiv.id = 'preview-plot-lead-II';
          plotDiv.style.width = '100%';
          plotDiv.style.height = '150px';
          plotDiv.style.marginBottom = '10px';
          plotContainer.appendChild(plotDiv);
  
          const isFlat = new Set(ecgData.y).size === 1;
          if (isFlat) {
              plotContainer.innerHTML += '<p style="color:orange;text-align:center;">Flat ECG signal detected. Please verify the CSV content.</p>';
          }
  
          const scaledY = minMaxScale(ecgData.y);
          const traces = [{
              x: Array.from({ length: scaledY.length }, (_, i) => i), // Use indices based on sampled length
              y: scaledY,
              mode: 'lines',
              line: { color: 'green', width: 1 },
              name: 'II'
          }];
  
          const layout = {
              title: { text: 'Lead II', font: { size: 12, color: fontColor } },
              plot_bgcolor: bgColor,
              paper_bgcolor: bgColor,
              font: { color: fontColor },
              xaxis: { visible: false }, // Hide x-axis to avoid confusion with original values
              yaxis: { visible: false },
              margin: { t: 20, b: 0, l: 0, r: 0 },
              showlegend: false
          };
          
              const config = {
                  responsive: true,
                  displayModeBar: false
              };
          
             setTimeout(() => {
              Plotly.newPlot(plotDiv, traces, layout, config).then(() => {
                          Plotly.Plots.resize(plotDiv);
              }).catch(err => {
                  plotDiv.innerHTML = `<p style="text-align:center; color: red;">Error rendering 2-lead plot: ${err.message}</p>`;
                  console.error('Plotly rendering error for 2-lead:', err);
              });
          }, 50);
  
      } else if (leadType === '7' || leadType === '12') {
          const leadData = ecgData.leadDict;
          if (leadNames.length === 0 || Object.values(leadData).some(arr => arr.length === 0)) {
              plotContainer.innerHTML = '<p style="text-align:center; color: red;">No valid data to plot.</p>';
              console.error('Invalid data for multi-lead plotting:', ecgData);
              if (pageLoader) pageLoader.style.display = 'none';
              return;
          }
  
          leadNames.forEach((lead, idx) => {
              const plotDiv = document.createElement('div');
              plotDiv.id = `preview-plot-lead-${idx}`;
              plotDiv.style.width = '100%';
              plotDiv.style.height = '150px';
              plotDiv.style.marginBottom = '10px'; 
              plotContainer.appendChild(plotDiv);
  
              const y = leadData[lead];
              const x = Array.from({ length: y.length }, (_, i) => i);
              if (y.length === 0) {
                  console.error(`No data for lead ${lead} at index ${idx}`);
                  plotDiv.innerHTML = `<p style="text-align:center; color: red;">No data for ${lead}</p>`;
                  return;
              }
              const scaledY = minMaxScale(y);
              const traces = [{
                  x,
                  y: scaledY,
                  type: 'scatter',
                  mode: 'lines',
                  line: { width: 1, color: 'green' },
                  name: lead
              }];
  
          const layout = {
              title: { text: `Lead ${lead}`, font: { size: 12, color: fontColor } },
              plot_bgcolor: bgColor,
              paper_bgcolor: bgColor,
              font: { color: fontColor },
              xaxis: { visible: false },  
              yaxis: { visible: false },
              margin: { t: 20, b: 0, l: 0, r: 0 },
              showlegend: false,
              editable: false,
              modebar: { orientation: 'h', bgcolor: 'transparent', color: 'transparent', activecolor: 'transparent' },
              autosize: true
          };
  
          const config = {
              responsive: true,
              displayModeBar: false
          };
              Plotly.newPlot(plotDiv, traces, layout, config).then(() => {
                  Plotly.Plots.resize(plotDiv);
              }).catch(err => {
                  plotDiv.innerHTML = `<p style="text-align:center; color: red;">Error rendering ${lead} plot: ${err.message}</p>`;
                  console.error(`Plotly rendering error for ${lead}:`, err);
              });
          });
      } else {
          plotContainer.innerHTML = '<p style="text-align:center; color: red;">Invalid lead configuration.</p>';
          console.error('Invalid leadType:', leadType);
      }
  
      // Function to handle resize events
      function handleResize() {
          const plotDivs = plotContainer.querySelectorAll('div[id^="preview-plot-lead-"]');
          plotDivs.forEach(plotDiv => {
              Plotly.Plots.resize(plotDiv);
          });
          plotContainer.style.padding = '10px';
          plotContainer.style.height = '100%';
     }
  
      handleResize();
      window.addEventListener('resize', handleResize);
  
      if (pageLoader) pageLoader.style.display = 'none';
  };
  window.submitForm = debounce(async () => {
      const form = document.getElementById('insertForm');
      if (!form) return;
  
      const insertModal = document.getElementById('insertModal');
      const previewModal = document.getElementById('previewLeadModal');
      if (insertModal) {
          const modalInstance = bootstrap.Modal.getInstance(insertModal) || new bootstrap.Modal(insertModal);
          modalInstance.hide();
      }
      if (previewModal) {
          const modalInstance = bootstrap.Modal.getInstance(previewModal) || new bootstrap.Modal(previewModal);
          modalInstance.hide();
      }
  
      const formData = new FormData(form);
      const pageLoader = document.getElementById('page-loader');
      const saveButton = form.querySelector('button[onclick="submitForm()"]');
      const plotContainer = document.getElementById('preview-lead-container');
  
      if (pageLoader) pageLoader.style.display = 'flex';
      if (saveButton) saveButton.disabled = true;
      if (plotContainer) {
          plotContainer.innerHTML = '';
          Plotly.purge(plotContainer);
      }
  
      try {
          const data = await safeFetch('/ommecgdata/new_insert_data/', {
              method: 'POST',
              body: formData,
              headers: { 'X-CSRFToken': getCSRFToken() }
          });
  
          clearForms();
  
          setTimeout(() => {
              if (data.status === 'success') {
                  alertSystem.success('Success', 'Data added successfully');
              } else {
                  alertSystem.warning('Warning', `Failed to insert data: ${data.message}`);
              }
              setTimeout(() => {
                  location.reload();
              }, 2100);
          }, 100);
      } catch (error) {
          setTimeout(() => {
              alertSystem.error('Error', 'An unexpected error occurred while inserting data.');
              setTimeout(() => {
                  location.reload();
              }, 2100);
          }, 100);
      } finally {
          if (pageLoader) pageLoader.style.display = 'none';
          if (saveButton) saveButton.disabled = false;
      }
  }, 100);
  
  document.addEventListener('contextmenu', function (e) {
    if (e.target.closest('.plot')) {
      e.preventDefault();
  
    }
  });
  
  window.insert = debounce(() => {
      const form = document.getElementById("dbinsertForm");
      if (!form) return;
  
      const saveButton = form.querySelector('button[onclick="insert()"]');
      if (saveButton) saveButton.disabled = true;
  
      // Close modal immediately
      const modalElement = document.getElementById('insertDataModal');
      if (modalElement) {
          const modalInstance = bootstrap.Modal.getInstance(modalElement) || new bootstrap.Modal(modalElement);
          modalInstance.hide();
      }
  
      // Show info alert instantly
      alertSystem.info('Info', 'Data will store in the short time');
      setTimeout(() => {
                  location.reload();
              }, 2100);
  
      // Prepare form data
      const formData = new FormData(form);
  
      // Send request in background fire-and-forget
      (async () => {
          try {
              await fetch('/ommecgdata/insert_db_Data/', {
                  method: "POST",
                  headers: {
                      'X-CSRFToken': document.querySelector('[name=csrfmiddlewaretoken]').value
                  },
                  body: formData
              });
              // No blocking ? not waiting for JSON or reload
          } catch (error) {
              console.error("Error sending data:", error);
          } finally {
              if (saveButton) saveButton.disabled = false;
          }
      })();
  }, 100);
  
  ['openInsertModalBtn', 'openUnifiedSearchBtn', 'insertData'].forEach(buttonId => {
    const button = document.getElementById(buttonId);
    if (!button) return;
  
    button.addEventListener('click', () => {
      let modalId;
  
      if (buttonId === 'openInsertModalBtn') {
        modalId = 'insertModal';
      } else if (buttonId === 'openUnifiedSearchBtn') {
        modalId = 'unifiedSearchModal';
      } else if (buttonId === 'insertData') {
        modalId = 'insertDataModal';
      }
  
      const modal = document.getElementById(modalId);
      if (!modal) return;
      cleanupModalBackdrops();
      // Reset all form fields inside the modal
      modal.querySelectorAll('input, select, textarea').forEach(field => {
        if (field.type === 'checkbox' || field.type === 'radio') {
          field.checked = false;
        } else {
          field.value = '';
          if (field.tagName.toLowerCase() === 'select') {
            field.selectedIndex = 0;
          }
        }
      });
  
      // Optional: Clear ECG plot if exists
      const plotContainer = document.getElementById('localPlot');
      if (plotContainer) {
        Plotly.purge(plotContainer);
        plotContainer.innerHTML = '';
        plotContainer.style.display = 'none';
      }
  
      // Optional: Reset plot status
      const plotStatus = document.getElementById('plotStatus');
      if (plotStatus) {
        plotStatus.style.display = 'none';
      }
  
      // Optional: Reset global ECG array
      window._ecgArray = null;
    });
  });
  
  const scrollToTopBtn = document.getElementById('scrollToTop');
  
  if (scrollToTopBtn) {
    // Show/hide scroll-to-top button based on scroll position (20% of document height)
    window.addEventListener('scroll', function() {
      const scrollHeight = document.documentElement.scrollHeight - window.innerHeight;
      const scrollPosition = window.scrollY;
      const showThreshold = scrollHeight * 0.2; // 20% of scrollable height
      
      if (scrollPosition > showThreshold) {
        scrollToTopBtn.classList.add('visible');
      } else {
        scrollToTopBtn.classList.remove('visible');
      }
    });
  
    // Scroll to top and close all plots when button is clicked
    scrollToTopBtn.addEventListener('click', function() {
      // Add clicked class for rotation animation
      scrollToTopBtn.classList.add('clicked');
      
      // Close all open plots
      document.querySelectorAll('.plot-row').forEach(plotRow => {
        plotRow.style.display = 'none';
      });
      
      // Smooth scroll to top
      window.scrollTo({
        top: 0,
        behavior: 'smooth'
      });
  
      // Remove clicked class after animation
      setTimeout(() => {
        scrollToTopBtn.classList.remove('clicked');
      }, 500); // Matches --transition-slow duration
    });
  }
  window.addArrhythmiaGroup = function () {
    const container = document.getElementById('arrhythmiaContainer');
    const firstGroup = container?.querySelector('.arrhythmia-group');
  
    if (!firstGroup) return;
  
    const newGroup = firstGroup.cloneNode(true);
  
    const arrhythmiaSelect = newGroup.querySelector('.arrhythmia-select');
    const subArrhythmiaSelect = newGroup.querySelector('.sub-arrhythmia-select');
  
    if (arrhythmiaSelect) arrhythmiaSelect.value = '';
    if (subArrhythmiaSelect) subArrhythmiaSelect.value = '';
    // Replace buttons
    const buttonArea = newGroup.querySelector('.col-md-2.d-flex');
    buttonArea.innerHTML = `
        <button type="button" class="btn btn-outline-danger btn-sm remove-btn" onclick="removeArrhythmiaGroup(this)">-</button>
    `;
    container.appendChild(newGroup);
  };
  window.removeArrhythmiaGroup = function (button) {
    const group = button.closest('.arrhythmia-group');
    group.remove();
  };
  // Handle showing correct sub-arrhythmias
  document.addEventListener('change', function (e) {
      if (e.target.classList.contains('arrhythmia-select')) {
          const arrhythmia = e.target.value;
          const subSelect = e.target.closest('.arrhythmia-group').querySelector('.sub-arrhythmia-select');
  
          // Hide all options
          Array.from(subSelect.options).forEach(opt => {
              opt.style.display = 'none';
              if (!opt.value) opt.style.display = 'block'; // keep default
          });
  
          // Show only matching arrhythmia sub-types
          Array.from(subSelect.options).forEach(opt => {
              if (opt.dataset.arrhythmia === arrhythmia) {
                  opt.style.display = 'block';
              }
          });
  
          // Reset sub-selection
          subSelect.value = '';
      }
  });
  function handleArrhythmiaChange(select) {
      const group = select.closest('.arrhythmia-group');
      const subSelect = group.querySelector('.sub-arrhythmia-select');
      const selectedArrhythmia = select.value;
  
      // Reset sub-select
      subSelect.value = "";
      [...subSelect.options].forEach(opt => {
          if (!opt.value) {
              opt.style.display = "block"; // keep "Select Sub-Arrhythmia"
          } else if (opt.dataset.arrhythmia === selectedArrhythmia) {
              opt.style.display = "block";
          } else {
              opt.style.display = "none";
          }
      });
  }
  
  // Attach change event on page load
  document.addEventListener("DOMContentLoaded", () => {
      document.querySelectorAll('.arrhythmia-group select[name="arrhythmia[]"]').forEach(sel => {
          sel.addEventListener("change", () => handleArrhythmiaChange(sel));
      });
  });
  window.addArrhythmiaDurationGroup = function () {
    const container = document.getElementById('arrhythmiaDurationContainer');
    const firstGroup = container.querySelector('.arrhythmia-group');
  
    if (!firstGroup) return;
  
    const newGroup = firstGroup.cloneNode(true);
  
   // Reset values
      const arrhythmiaSelect = newGroup.querySelector('select[name="arrhythmia[]"]');
      const subArrhythmiaSelect = newGroup.querySelector('.sub-arrhythmia-select');
      const input = newGroup.querySelector('input');
  
      if (arrhythmiaSelect) arrhythmiaSelect.value = '';
      if (subArrhythmiaSelect) {
          subArrhythmiaSelect.value = '';
          [...subArrhythmiaSelect.options].forEach(opt => {
              if (opt.value) opt.style.display = "none";
          });
      }
      if (input) input.value = '';
     // Replace buttons
    const buttonArea = newGroup.querySelector('.col-md-2.d-flex');
    buttonArea.innerHTML = `
        <button type="button" class="btn btn-outline-danger btn-sm remove-btn" onclick="removeArrhythmiaGroup(this)">-</button>
    `;
        // Attach new change event
      arrhythmiaSelect.addEventListener("change", () => handleArrhythmiaChange(arrhythmiaSelect));
    container.appendChild(newGroup);
  };
  window.removeArrhythmiaGroup = function (button) {
    const group = button.closest('.arrhythmia-group');
    group.remove();
  };
  // Enhanced multiple search with task management
  const attachGetPageFormListener = () => {
    document.addEventListener('submit', async (e) => {
      if (e.target.id !== 'getPageForm') return;
      e.preventDefault();
     
      // --- Cleanup modal if open ---
      document.querySelectorAll(".modal-backdrop").forEach(backdrop => backdrop.remove());
      document.body.classList.remove("modal-open");
      document.body.style.removeProperty("overflow");
      const modalEl = document.getElementById("getPageModal");
      if (modalEl) {
        const modalInstance = bootstrap.Modal.getInstance(modalEl);
        if (modalInstance) {
          modalInstance.hide();
          modalInstance.dispose();
        }
      }
     
      const selectors = {
        arrhythmiaGroups: '#arrhythmiaDurationContainer .arrhythmia-group',
        leadInput: '[name="leadTypess"]',
        frequencyInput: '[name="frequencyss"]',
        csrfInput: '[name="csrfmiddlewaretoken"]'
      };
  
      const elements = {
        arrhythmiaGroups: document.querySelectorAll(selectors.arrhythmiaGroups),
        leadInput: document.querySelector(selectors.leadInput),
        frequencyInput: document.querySelector(selectors.frequencyInput),
        csrfInput: document.querySelector(selectors.csrfInput)
      };
  
      //Basic validation
      if (!elements.arrhythmiaGroups.length || !elements.leadInput || !elements.frequencyInput || !elements.csrfInput) {
        return;
      }
  
      const arrhythmiaData = Array.from(elements.arrhythmiaGroups)
        .map(group => ({
          arrhythmia: group.querySelector('select')?.value,
          duration: parseInt(group.querySelector('input')?.value) || 0
        }))
        .filter(data => data.arrhythmia && data.duration);
  
      if (!arrhythmiaData.length) {
        return;
      }
  
      const formData = {
        lead: elements.leadInput.value,
        frequency: parseInt(elements.frequencyInput.value),
        arrhythmiaData
      };
      // Close modal immediately
      const modal = document.getElementById('unifiedSearchModal');
      if (modal) {
        const modalInstance = bootstrap.Modal.getInstance(modal) || new bootstrap.Modal(modal);
        modalInstance.hide();
      }
      cleanupModalBackdrops();
      // Create a task for this search
      const arrhythmiaNames = arrhythmiaData.map(a => a.arrhythmia).join(', ');
      const taskId = taskManager.createTask(
        'multiple_search',
        'Multiple ECG Search',
        `Searching for: ${arrhythmiaNames} (Lead: ${formData.lead}, Freq: ${formData.frequency} Hz)`,
        formData
      );
  
      // Execute search in background
      try {
        taskManager.updateTask(taskId, { progress: 20 });
  
        // First request - get multiple segments
        const response = await backgroundFetch('/ommecgdata/get_multiple_segments/', {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
            'X-Requested-With': 'XMLHttpRequest',
            'X-CSRFToken': elements.csrfInput.value,
          },
          body: JSON.stringify(formData)
        }, taskId);
  
          // Deduplicate by patient_id
          if (response.status === 'success' && response.data?.length > 0) {
            taskManager.updateTask(taskId, { progress: 60 });
  
            // Get unique arrhythmias from backend response
            const arrhythmias = [...new Set(response.data.map(d => d.arrhythmia?.toLowerCase()))];
            let allDetails = [];
  
            // Fetch ECG details for each arrhythmia
            for (const arr of arrhythmias) {
                if (!arr) continue;
                try {
                    const detailsResponse = await fetch(`/ommecgdata/ecg_details/${arr}/`, {
                        headers: { 'X-Requested-With': 'XMLHttpRequest' }
                    });
  
                    const contentType = detailsResponse.headers.get('content-type');
                    if (contentType && contentType.includes('application/json')) {
                        const json = await detailsResponse.json();
                        allDetails.push(...(json.data || []));
                    }
                } catch (err) {
                    console.error(`Failed to fetch details for ${arr}:`, err);
                }
            }
  
            // Remove deduplication: use allDetails directly
            if (allDetails.length > 0) {
                taskManager.completeTask(taskId, {
                    data: allDetails,   // show all duplicates
                    total_pages: 1,
                    arrhythmias: arrhythmias
                });
            } else {
                taskManager.errorTask(taskId, 'No ECG data found for selected segments.');
            }
        } else {
            taskManager.errorTask(taskId, response.message || 'No valid data found.');
        }
      } catch (err) {
        console.error('Error submitting multiple search:', err);
        taskManager.errorTask(taskId, err.message);
      }
    });
  };
  
  attachGetPageFormListener();
  attachRowEventListeners();
  loadSessionData();
  });
  
  document.querySelectorAll('.share-icon').forEach(icon => {
    icon.addEventListener('click', function () {
      const menu = this.nextElementSibling;
      menu.classList.toggle('active');
    });
  });
   const openSearchBtn = document.getElementById('openUnifiedSearchBtn');
  
    function cleanupModalBackdrops() {
      document.querySelectorAll(".modal-backdrop").forEach(el => el.remove());
      document.body.classList.remove("modal-open");
      document.body.style.removeProperty("overflow");
      document.body.style.removeProperty("paddingRight");
    }
  
    if (openSearchBtn) {
      openSearchBtn.addEventListener('click', () => {
        cleanupModalBackdrops(); // Clean old backdrops
        const modal = document.getElementById('unifiedSearchModal');
        if (modal) {
          const bsModal = new bootstrap.Modal(modal, { backdrop: 'static', keyboard: false });
          bsModal.show();
        }
      });
    }
  document.addEventListener('DOMContentLoaded', function () {
    // Select all share dropdowns on the page
    const shareDropdowns = document.querySelectorAll('.share-dropdown');
  
    shareDropdowns.forEach(dropdown => {
      const shareIcon = dropdown.querySelector('.share-icon');
      const shareMenu = dropdown.querySelector('.share-menu');
  
      if (!shareIcon || !shareMenu) return;
  
      // Toggle only this menu
      shareIcon.addEventListener('click', (e) => {
        e.stopPropagation();
  
        // Close any other open menus
        document.querySelectorAll('.share-menu.active').forEach(menu => {
          if (menu !== shareMenu) menu.classList.remove('active');
        });
  
        shareMenu.classList.toggle('active');
      });
  
      // Close menu on click outside this dropdown
      document.addEventListener('click', (e) => {
        if (!dropdown.contains(e.target)) {
          shareMenu.classList.remove('active');
        }
      });
  
      // Auto-close on mouse leave
      shareMenu.addEventListener('mouseleave', () => {
        shareMenu.classList.remove('active');
      });
  
      shareIcon.addEventListener('mouseleave', () => {
        setTimeout(() => {
          if (!shareMenu.matches(':hover') && !shareIcon.matches(':hover')) {
            shareMenu.classList.remove('active');
          }
        }, 150);
      });
    });
  });
