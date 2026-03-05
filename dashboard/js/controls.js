/* controls.js — UI controls: location selectors, date slider, model toggles, filters */

function initControls() {
  const locations = DATA.locations;
  const sortedLocs = locations
    .filter(l => l.location !== 'US')
    .sort((a, b) => a.location_name.localeCompare(b.location_name));
  const usLoc = locations.find(l => l.location === 'US');
  const allLocs = usLoc ? [usLoc, ...sortedLocs] : sortedLocs;

  // --- Overview location dropdown ---
  const overviewSelect = document.getElementById('overview-location-select');
  allLocs.forEach(l => {
    const opt = document.createElement('option');
    opt.value = l.location;
    opt.textContent = l.location_name;
    if (l.location === AppState.selectedLocation) opt.selected = true;
    overviewSelect.appendChild(opt);
  });
  overviewSelect.addEventListener('change', () => {
    setLocation(overviewSelect.value);
  });

  // --- Forecasts tab location radio list ---
  const forecastList = document.getElementById('forecast-location-list');
  allLocs.forEach(l => {
    const div = document.createElement('div');
    div.className = 'location-radio-item' + (l.location === AppState.selectedLocation ? ' active' : '');
    div.dataset.location = l.location;
    div.textContent = l.location_name;
    div.addEventListener('click', () => {
      setLocation(l.location);
      forecastList.querySelectorAll('.location-radio-item').forEach(el => {
        el.classList.toggle('active', el.dataset.location === l.location);
      });
    });
    forecastList.appendChild(div);
  });

  // --- Overview date slider ---
  const dates = DATA.availableDates;
  const ovSlider = document.getElementById('overview-date-slider');
  const ovDateLabel = document.getElementById('overview-date-label');
  const ovLatestBtn = document.getElementById('overview-latest-btn');

  ovSlider.max = dates.length - 1;
  ovSlider.value = dates.length - 1;
  ovDateLabel.textContent = dates[dates.length - 1].toISOString().slice(0, 10);
  ovLatestBtn.classList.add('is-latest');

  ovSlider.addEventListener('input', () => {
    const idx = +ovSlider.value;
    AppState.overviewRefDate = dates[idx];
    ovDateLabel.textContent = dates[idx].toISOString().slice(0, 10);
    ovLatestBtn.classList.toggle('is-latest', idx === dates.length - 1);
    updateAll();
  });

  ovLatestBtn.addEventListener('click', () => {
    const lastIdx = dates.length - 1;
    ovSlider.value = lastIdx;
    AppState.overviewRefDate = dates[lastIdx];
    ovDateLabel.textContent = dates[lastIdx].toISOString().slice(0, 10);
    ovLatestBtn.classList.add('is-latest');
    updateAll();
  });

  // --- Overview horizon toggle ---
  document.querySelectorAll('#overview-horizon-toggle .toggle-btn').forEach(btn => {
    btn.addEventListener('click', () => {
      document.querySelectorAll('#overview-horizon-toggle .toggle-btn').forEach(b => b.classList.remove('active'));
      btn.classList.add('active');
      AppState.overviewHorizon = +btn.dataset.horizon;
      updateAll();
    });
  });

  // --- Overview range buttons ---
  initRangeButtons('overview-range-buttons');

  // --- Forecast date slider ---
  const slider = document.getElementById('forecast-date-slider');
  const fcDateLabel = document.getElementById('forecast-date-label');
  const fcLatestBtn = document.getElementById('forecast-latest-btn');

  slider.max = dates.length - 1;
  slider.value = dates.length - 1;
  fcDateLabel.textContent = dates[dates.length - 1].toISOString().slice(0, 10);
  fcLatestBtn.classList.add('is-latest');

  slider.addEventListener('input', () => {
    const idx = +slider.value;
    AppState.selectedDate = dates[idx];
    fcDateLabel.textContent = dates[idx].toISOString().slice(0, 10);
    fcLatestBtn.classList.toggle('is-latest', idx === dates.length - 1);
    updateAll();
  });

  fcLatestBtn.addEventListener('click', () => {
    const lastIdx = dates.length - 1;
    slider.value = lastIdx;
    AppState.selectedDate = dates[lastIdx];
    fcDateLabel.textContent = dates[lastIdx].toISOString().slice(0, 10);
    fcLatestBtn.classList.add('is-latest');
    updateAll();
  });

  // --- Build timeline ticks for both sliders ---
  buildSliderTicks('overview-slider-ticks', dates);
  buildSliderTicks('forecast-slider-ticks', dates);

  // --- Model toggle ---
  document.querySelectorAll('#model-toggle .toggle-btn').forEach(btn => {
    btn.addEventListener('click', () => {
      document.querySelectorAll('#model-toggle .toggle-btn').forEach(b => b.classList.remove('active'));
      btn.classList.add('active');
      AppState.selectedModel = btn.dataset.model;
      updateAll();
    });
  });

  // --- Forecast range buttons ---
  initRangeButtons('forecast-range-buttons');

  // --- Evaluation filters ---
  initEvalFilters(allLocs);
}

function setLocation(code) {
  AppState.selectedLocation = code;
  AppState.selectedLocationName = getLocationName(code, DATA.locations);

  // Sync all location selectors
  const overviewSelect = document.getElementById('overview-location-select');
  if (overviewSelect.value !== code) overviewSelect.value = code;

  document.querySelectorAll('#forecast-location-list .location-radio-item').forEach(el => {
    el.classList.toggle('active', el.dataset.location === code);
  });

  updateHash();
  updateAll();
}

function initRangeButtons(containerId) {
  const container = document.getElementById(containerId);
  container.querySelectorAll('.range-btn').forEach(btn => {
    btn.addEventListener('click', () => {
      container.querySelectorAll('.range-btn').forEach(b => b.classList.remove('active'));
      btn.classList.add('active');
      AppState.dateRange = btn.dataset.range;
      updateAll();
    });
  });
}

function initEvalFilters(allLocs) {
  // Location dropdown
  const evalLocSelect = document.getElementById('eval-location-select');
  allLocs.forEach(l => {
    const opt = document.createElement('option');
    opt.value = l.location;
    opt.textContent = l.location_name;
    evalLocSelect.appendChild(opt);
  });
  evalLocSelect.addEventListener('change', () => {
    AppState.evalLocation = evalLocSelect.value;
    updateAll();
  });

  // Horizon filter
  document.getElementById('eval-horizon-select').addEventListener('change', (e) => {
    AppState.evalHorizon = e.target.value;
    updateAll();
  });

  // Forecast date filters
  ['eval-ref-start', 'eval-ref-end'].forEach(id => {
    document.getElementById(id).addEventListener('change', () => {
      const rs = document.getElementById('eval-ref-start').value;
      const re = document.getElementById('eval-ref-end').value;
      AppState.evalRefDateRange = [rs ? parseDate(rs) : null, re ? parseDate(re) : null];
      updateAll();
    });
  });

  // Model checkboxes
  const checkContainer = document.getElementById('eval-model-checkboxes');

  // Ensemble models group
  const ensLabel = document.createElement('div');
  ensLabel.className = 'model-group-label';
  ensLabel.textContent = 'Ensemble Models';
  checkContainer.appendChild(ensLabel);
  ENSEMBLE_MODELS.forEach(m => addModelCheckbox(checkContainer, m));

  // Individual models group
  const indLabel = document.createElement('div');
  indLabel.className = 'model-group-label';
  indLabel.textContent = 'Individual Models';
  checkContainer.appendChild(indLabel);
  MODELS.forEach(m => addModelCheckbox(checkContainer, m));

  updateModelCount();
}

function addModelCheckbox(container, model) {
  const item = document.createElement('label');
  item.className = 'model-checkbox-item';

  const cb = document.createElement('input');
  cb.type = 'checkbox';
  cb.checked = true;
  cb.value = model;
  cb.addEventListener('change', () => {
    if (cb.checked) {
      if (!AppState.evalModels.includes(model)) AppState.evalModels.push(model);
    } else {
      AppState.evalModels = AppState.evalModels.filter(m => m !== model);
    }
    updateModelCount();
    updateAll();
  });

  const swatch = document.createElement('span');
  swatch.className = 'model-color-swatch';
  swatch.style.background = COLOR_MAP[model] || '#999';

  const label = document.createElement('span');
  label.textContent = model;

  item.appendChild(cb);
  item.appendChild(swatch);
  item.appendChild(label);
  container.appendChild(item);
}

function updateModelCount() {
  const el = document.getElementById('eval-model-count');
  el.textContent = `${AppState.evalModels.length} of ${ALL_MODELS.length} models selected`;
}

function buildSliderTicks(containerId, dates) {
  const container = document.getElementById(containerId);
  if (!container || dates.length === 0) return;

  // Find the first date of each month represented in the dates array
  const monthTicks = [];
  let lastMonth = -1;
  dates.forEach((d, i) => {
    const m = d.getMonth();
    if (m !== lastMonth) {
      monthTicks.push({ index: i, date: d });
      lastMonth = m;
    }
  });

  const total = dates.length - 1;
  if (total <= 0) return;

  monthTicks.forEach(tick => {
    const pct = (tick.index / total) * 100;
    const label = tick.date.toLocaleDateString('en-US', { month: 'short', year: '2-digit' });
    const el = document.createElement('span');
    el.className = 'slider-tick';
    el.style.left = pct + '%';
    el.textContent = label;
    container.appendChild(el);
  });
}
