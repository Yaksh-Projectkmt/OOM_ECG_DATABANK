  const debounce = (func, wait) => {
    let timeout;
    return (...args) => {
      clearTimeout(timeout);
      timeout = setTimeout(() => func(...args), wait);
    };
  };

// ====== CARD CLICK HANDLER ======
const handleCardClick = (card, selectorType) => {
  card.style.cursor = 'pointer';

  card.addEventListener('click', async () => {
    // Use debounce wrapper here — not around the whole listener
    debounce(async () => {
      const arr = card.dataset.arrhythmia;
      if (!arr) return console.error(`No data-arrhythmia attribute found on ${selectorType}!`);

      // Clear session
      sessionStorage.removeItem('randomData');
      sessionStorage.removeItem('searchResults');
      sessionStorage.removeItem('totalPages');
      sessionStorage.removeItem('selectedArrhythmia');

      // Set new data
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

          // Redirect
          window.location.href = card.dataset.redirect || `/ommecgdata/ecg_details/${arr}/`;
        } else {
          alertSystem.info('Info', json.message || `No random ECG data found for ${arr}`);
        }
      } catch (error) {
        alertSystem.error('Error', 'Fetching random ECG data failed. Please try again.');
      }
    }, 100)();
  });
};

// ====== CARD DATA ======
const cardData = [
  { id: '1', title: 'Premature Ventricular Contraction', delay: 0 },
  { id: '2', title: 'Premature Atrial Contraction', delay: 100 },
  { id: '3', title: 'Atrial Fibrillation & Atrial Flutter', delay: 200 },
  { id: '4', title: 'Heart Block', delay: 300 },
  { id: '5', title: 'Junctional', delay: 400 },
  { id: '6', title: 'Ventricular Fibrillation and Asystole', delay: 500 },
  { id: '7', title: 'Myocardial Infarction', delay: 600 },
  { id: '8', title: 'Noise', delay: 700 },
  { id: '9', title: 'LBBB & RBBB', delay: 800 }
];

// ====== INITIALIZE ======
document.addEventListener('DOMContentLoaded', () => {
  // Apply theme
  const savedTheme = localStorage.getItem('theme') || 'dark';
  document.body.className = `${savedTheme}-mode`;

  // Render cards
  renderCards();

  // Attach click handlers (AFTER cards exist)
  document.querySelectorAll('.condition-card').forEach(card => {
    handleCardClick(card, 'condition-card');
  });
});

// ====== RENDER CARDS ======
function renderCards() {
  const cardsGrid = document.getElementById('cards-grid');
  cardsGrid.innerHTML = ''; // prevent duplicates
  cardData.forEach(card => {
    const cardElement = createCard(card);
    cardsGrid.appendChild(cardElement);
  });
}

// ====== CREATE CARD ELEMENT ======
function createCard(card) {
  const cardDiv = document.createElement('div');
  cardDiv.className = 'condition-card';
  cardDiv.style.animationDelay = `${card.delay}ms`;

  // Add dataset attributes
  cardDiv.dataset.arrhythmia = card.title;
  cardDiv.dataset.redirect = `/ommecgdata/ecg_details/${encodeURIComponent(card.title)}/`;

  cardDiv.innerHTML = `
    <div class="card-gradient"></div>
    <div class="ecg-pattern">
      <svg viewBox="0 0 100 40" preserveAspectRatio="none">
        <polyline class="ecg-line" points="0,20 15,20 18,8 21,32 24,12 27,20 35,20 38,15 41,25 44,18 47,20 100,20"></polyline>
      </svg>
    </div>
    <div class="card-content">
      <div class="card-header">
        <h3 class="card-title">${card.title}</h3>
        <svg class="card-icon" width="20" height="20" viewBox="0 0 24 24"
          fill="none" stroke="currentColor" stroke-width="2"
          stroke-linecap="round" stroke-linejoin="round">
          <polyline points="22 12 18 12 15 21 9 3 6 12 2 12"></polyline>
        </svg>
      </div>
    </div>
     <div class="card-indicator"></div>
    <div class="card-corner-glow">
    </div>
  `;

  return cardDiv;
}
