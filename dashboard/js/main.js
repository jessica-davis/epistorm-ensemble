/* main.js — App initialization, state management, data loading */

const AppState = {
  activeTab: 'overview',
  selectedLocation: 'US',
  selectedLocationName: 'United States',
  selectedDate: null,
  selectedModel: 'Median Epistorm Ensemble',
  dateRange: '3months',
  // Evaluation filters
  evalLocation: 'all',
  evalModels: [],
  evalHorizon: 'all',
  evalRefDateRange: [null, null],
  // Overview-specific
  overviewHorizon: 3,
  overviewRefDate: null,
  // Forecast map
  forecastHorizon: 0,
  forecastPerCapita: false,
};

async function init() {
  try {
    const [observed, ensemble, ensembleLop, categorical,
           activityLevels, thresholds, wisRatio, coverage, locations, usStates] =
      await Promise.all([
        d3.json('data/observed.json'),
        d3.json('data/ensemble.json'),
        d3.json('data/ensemble_lop.json'),
        d3.json('data/categorical.json'),
        d3.json('data/activity_levels.json'),
        d3.json('data/thresholds.json'),
        d3.json('data/wis_ratio.json'),
        d3.json('data/coverage.json'),
        d3.json('data/locations.json'),
        d3.json('data/us-states.json'),
      ]);

    // Parse dates on observed data
    observed.forEach(d => {
      d.date = parseDate(d.date);
    });

    // Parse dates on pivoted ensemble data
    ensemble.forEach(d => {
      d.reference_date = parseDate(d.reference_date);
      d.target_end_date = parseDate(d.target_end_date);
    });
    ensembleLop.forEach(d => {
      d.reference_date = parseDate(d.reference_date);
      d.target_end_date = parseDate(d.target_end_date);
    });

    // Parse dates on categorical
    categorical.forEach(d => {
      d.reference_date = parseDate(d.reference_date);
    });

    // Parse dates on activity levels
    activityLevels.forEach(d => {
      d.reference_date = parseDate(d.reference_date);
    });

    // Parse dates on wis/coverage
    wisRatio.forEach(d => {
      d.target_end_date = parseDate(d.target_end_date);
      d.reference_date = parseDate(d.reference_date);
    });
    coverage.forEach(d => {
      d.target_end_date = parseDate(d.target_end_date);
      d.reference_date = parseDate(d.reference_date);
    });

    // Store globally
    window.DATA = {
      observed, ensemble, ensembleLop, categorical,
      activityLevels, thresholds, wisRatio, coverage, locations, usStates,
    };

    // Derive available reference dates from ensemble data
    const dateSet = new Set(ensemble.map(d => d.reference_date.toISOString().slice(0, 10)));
    const dates = [...dateSet].sort();
    window.DATA.availableDates = dates.map(d => parseDate(d));

    // Set initial state
    AppState.selectedDate = window.DATA.availableDates[window.DATA.availableDates.length - 1];
    AppState.overviewRefDate = AppState.selectedDate;
    AppState.evalModels = [...ALL_MODELS];

    // Parse URL hash
    parseHash();

    // Initialize all components
    initControls();
    initOverview();
    initForecasts();
    initForecastMap();
    initEvaluation();

    updateAll();

    // Hide loading
    document.getElementById('loading').style.display = 'none';
  } catch (err) {
    document.querySelector('.loading-content').innerHTML =
      '<p style="color:#c00;">Error loading data. Make sure to serve via a local web server:</p>' +
      '<code style="display:block;margin-top:8px;font-size:13px;">cd dashboard && python -m http.server 8000</code>';
    console.error('Init error:', err);
  }
}

function updateAll() {
  switch (AppState.activeTab) {
    case 'overview':
      updateOverview();
      break;
    case 'forecasts':
      updateForecasts();
      updateForecastMap();
      break;
    case 'evaluation':
      updateEvaluation();
      break;
  }
}

// --- Tab switching ---
function switchTab(tabName) {
  AppState.activeTab = tabName;

  document.querySelectorAll('.tab').forEach(t => {
    t.classList.toggle('active', t.dataset.tab === tabName);
  });
  document.querySelectorAll('.tab-content').forEach(s => {
    s.classList.toggle('active', s.id === 'tab-' + tabName);
  });

  updateHash();
  updateAll();
}

document.querySelectorAll('.tab').forEach(tab => {
  tab.addEventListener('click', () => switchTab(tab.dataset.tab));
});

// --- URL hash ---
function updateHash() {
  const loc = AppState.selectedLocation;
  window.location.hash = `${AppState.activeTab}/${loc}`;
}

function parseHash() {
  const hash = window.location.hash.slice(1);
  if (!hash) return;
  const [tab, loc] = hash.split('/');
  if (['overview', 'forecasts', 'evaluation'].includes(tab)) {
    AppState.activeTab = tab;
    document.querySelectorAll('.tab').forEach(t => {
      t.classList.toggle('active', t.dataset.tab === tab);
    });
    document.querySelectorAll('.tab-content').forEach(s => {
      s.classList.toggle('active', s.id === 'tab-' + tab);
    });
  }
  if (loc) {
    const found = DATA.locations.find(l => l.location === loc);
    if (found) {
      AppState.selectedLocation = loc;
      AppState.selectedLocationName = getLocationName(loc, DATA.locations);
    }
  }
}

// --- Resize handler ---
let resizeTimer;
window.addEventListener('resize', () => {
  clearTimeout(resizeTimer);
  resizeTimer = setTimeout(() => updateAll(), 200);
});

document.addEventListener('DOMContentLoaded', init);
