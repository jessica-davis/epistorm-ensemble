/* forecasts.js — Forecasts tab: quantile chart with prediction intervals */

function initForecasts() {
  // Nothing to pre-build; updateForecasts handles everything
}

function updateForecasts() {
  const loc = AppState.selectedLocation;
  const refDate = AppState.selectedDate;
  const modelName = AppState.selectedModel;
  const modelColor = COLOR_MAP[modelName] || '#2B7A8F';
  const refStr = refDate ? refDate.toISOString().slice(0, 10) : null;

  // Get ensemble data for selected model
  const ensData = modelName === 'LOP Epistorm Ensemble' ? DATA.ensembleLop : DATA.ensemble;

  // Filter forecast data for this location and reference date (exclude backcasts with horizon < 0)
  const forecastData = ensData
    .filter(d => d.location === loc && d.reference_date.toISOString().slice(0, 10) === refStr && d.horizon >= 0)
    .sort((a, b) => a.target_end_date - b.target_end_date);

  // Get ALL observed data for this location (always show future data beyond ref date)
  const locObs = DATA.observed
    .filter(d => d.location === loc && d.value != null)
    .sort((a, b) => a.date - b.date);

  // Update headline
  const locName = getLocationName(loc, DATA.locations);
  document.getElementById('forecast-chart-headline').textContent =
    `Flu Hospitalization Forecasts \u2014 ${locName}`;

  // Update summary text
  updateForecastText(forecastData, locObs, loc, refDate, modelName);

  // Draw chart
  drawForecastChart(forecastData, locObs, refDate, modelColor);
}

function updateForecastText(forecastData, locObs, loc, refDate, modelName) {
  const summaryEl = document.getElementById('forecast-summary-text');
  const piEl = document.getElementById('forecast-pi-text');
  const locName = getLocationName(loc, DATA.locations);
  const population = getPopulation(loc, DATA.locations);

  if (forecastData.length === 0) {
    summaryEl.textContent = 'No forecast data available for the selected date and location.';
    piEl.textContent = '';
    return;
  }

  // Use the furthest horizon forecast
  const lastFc = forecastData[forecastData.length - 1];
  const firstFc = forecastData[0];
  const median = lastFc['0.5'];
  const lower95 = lastFc['0.025'];
  const upper95 = lastFc['0.975'];
  const targetDate = fmtDate(lastFc.target_end_date);
  const refDateStr = fmtDate(refDate);
  const weeksAhead = lastFc.horizon + 1;

  const perCapMedian = population ? perCapita(median, population) : null;

  // Build summary with horizon context
  let summaryHtml =
    `As of ${refDateStr}, the ${modelName} projects <strong>${fmtNum(median)}</strong> influenza hospital admissions` +
    (perCapMedian != null ? ` (<strong>${perCapMedian.toFixed(1)}</strong> per 100,000)` : '') +
    ` in ${locName} by ${targetDate} (${weeksAhead} week${weeksAhead > 1 ? 's' : ''} ahead).`;

  // Compare to most recent observed value
  const obsBeforeRef = locObs.filter(d => d.date <= refDate);
  if (obsBeforeRef.length > 0) {
    const latestObs = obsBeforeRef[obsBeforeRef.length - 1];
    const diff = median - latestObs.value;
    const pctChange = latestObs.value > 0 ? Math.abs(diff / latestObs.value * 100) : 0;
    if (Math.abs(diff) > 1) {
      const direction = diff > 0 ? 'increase' : 'decrease';
      summaryHtml += ` This represents a projected <strong>${pctChange.toFixed(0)}% ${direction}</strong> from the most recent observed value of ${fmtNum(latestObs.value)} (${fmtDate(latestObs.date)}).`;
    } else {
      summaryHtml += ` This is roughly in line with the most recent observed value of ${fmtNum(latestObs.value)} (${fmtDate(latestObs.date)}).`;
    }
  }

  summaryEl.innerHTML = summaryHtml;

  if (lower95 != null && upper95 != null) {
    const perCapLo = population ? perCapita(lower95, population) : null;
    const perCapHi = population ? perCapita(upper95, population) : null;
    piEl.innerHTML =
      `95% prediction interval: <strong>${fmtNum(lower95)}\u2013${fmtNum(upper95)}</strong> hospital admissions` +
      (perCapLo != null && perCapHi != null ?
        ` (<strong>${perCapLo.toFixed(1)}\u2013${perCapHi.toFixed(1)}</strong> per 100,000)` : '') +
      '. The forecast covers horizons of 1 to ' + weeksAhead + ' weeks ahead.';
  } else {
    piEl.textContent = '';
  }
}

function drawForecastChart(forecastData, locObs, refDate, modelColor) {
  const container = document.getElementById('forecast-chart-container');
  container.innerHTML = '';

  if (locObs.length === 0 && forecastData.length === 0) {
    container.innerHTML = '<div class="empty-state">No data available.</div>';
    return;
  }

  // Determine the latest date we need to show (max of observed and forecast)
  const latestObsDate = locObs.length > 0 ? locObs[locObs.length - 1].date : null;
  const latestFcDate = forecastData.length > 0
    ? forecastData[forecastData.length - 1].target_end_date : null;
  const chartMaxDate = d3.max([latestObsDate, latestFcDate].filter(Boolean));

  // Date range cutoff applies to the left bound only (how far back to show historical)
  const cutoff = getDateRangeCutoff(chartMaxDate, AppState.dateRange);
  const filteredObs = cutoff ? locObs.filter(d => d.date >= cutoff) : locObs;

  // Chart dimensions
  const containerWidth = container.clientWidth || 800;
  const width = containerWidth;
  const height = 420;
  const margin = { top: 20, right: 30, bottom: 40, left: 65 };
  const innerW = width - margin.left - margin.right;
  const innerH = height - margin.top - margin.bottom;

  const svg = d3.select(container)
    .append('svg')
    .attr('viewBox', `0 0 ${width} ${height}`)
    .attr('preserveAspectRatio', 'xMidYMid meet');

  const g = svg.append('g').attr('transform', `translate(${margin.left},${margin.top})`);
  const bgLayer = g.append('g').attr('class', 'bg-layer');
  const dataLayer = g.append('g').attr('class', 'data-layer');
  const axisLayer = g.append('g').attr('class', 'axis-layer');
  const annotLayer = g.append('g').attr('class', 'annot-layer');

  // Compute combined domain
  const allVisibleDates = [
    ...filteredObs.map(d => d.date),
    ...forecastData.map(d => d.target_end_date),
  ];
  const xMin = d3.min(allVisibleDates);
  const xMax = d3.max(allVisibleDates);

  const obsMax = d3.max(filteredObs, d => d.value) || 0;
  const fcMax = forecastData.length > 0
    ? d3.max(forecastData, d => d['0.975'] || d['0.5'] || 0)
    : 0;
  const yMaxVal = Math.max(obsMax, fcMax) * 1.1;

  const x = d3.scaleTime().domain([xMin, xMax]).range([0, innerW]);
  const y = d3.scaleLinear().domain([0, yMaxVal]).range([innerH, 0]);

  // Axes
  axisLayer.append('g')
    .attr('class', 'axis')
    .attr('transform', `translate(0,${innerH})`)
    .call(d3.axisBottom(x).ticks(5).tickFormat(d3.timeFormat('%b %Y')));

  axisLayer.append('g')
    .attr('class', 'axis')
    .call(d3.axisLeft(y).ticks(6).tickFormat(d => fmtNum(d)));

  axisLayer.append('text')
    .attr('class', 'axis-label')
    .attr('transform', 'rotate(-90)')
    .attr('x', -innerH / 2).attr('y', -50)
    .text('Weekly Incident Flu Hospitalizations');

  // --- Forecast bands ---
  if (forecastData.length > 0) {
    // 95% PI band
    const area95 = d3.area()
      .x(d => x(d.target_end_date))
      .y0(d => y(d['0.025'] || 0))
      .y1(d => y(d['0.975'] || 0))
      .defined(d => d['0.025'] != null && d['0.975'] != null);

    dataLayer.append('path')
      .datum(forecastData)
      .attr('d', area95)
      .attr('fill', hexToRgba(modelColor, 0.2))
      .attr('stroke', 'none');

    // 50% PI band
    const area50 = d3.area()
      .x(d => x(d.target_end_date))
      .y0(d => y(d['0.25'] || 0))
      .y1(d => y(d['0.75'] || 0))
      .defined(d => d['0.25'] != null && d['0.75'] != null);

    dataLayer.append('path')
      .datum(forecastData)
      .attr('d', area50)
      .attr('fill', hexToRgba(modelColor, 0.4))
      .attr('stroke', 'none');

    // Median line
    const medianLine = d3.line()
      .x(d => x(d.target_end_date))
      .y(d => y(d['0.5']))
      .defined(d => d['0.5'] != null);

    dataLayer.append('path')
      .datum(forecastData)
      .attr('d', medianLine)
      .attr('fill', 'none')
      .attr('stroke', modelColor)
      .attr('stroke-width', 2);

    // Median markers
    dataLayer.selectAll('.fc-dot')
      .data(forecastData.filter(d => d['0.5'] != null))
      .join('circle')
      .attr('class', 'fc-dot')
      .attr('cx', d => x(d.target_end_date))
      .attr('cy', d => y(d['0.5']))
      .attr('r', 3.5)
      .attr('fill', modelColor)
      .attr('stroke', '#fff')
      .attr('stroke-width', 1);
  }

  // --- Observed line (always show all available data in range, including after ref date) ---
  if (filteredObs.length > 0) {
    const obsLine = d3.line()
      .x(d => x(d.date))
      .y(d => y(d.value))
      .defined(d => d.value != null);

    dataLayer.append('path')
      .datum(filteredObs)
      .attr('d', obsLine)
      .attr('fill', 'none')
      .attr('stroke', '#1a1a1a')
      .attr('stroke-width', 2);

    // Markers: solid before ref date, open after
    filteredObs.forEach(d => {
      const isAfterRef = refDate && d.date > refDate;
      dataLayer.append('circle')
        .attr('cx', x(d.date))
        .attr('cy', y(d.value))
        .attr('r', 2.5)
        .attr('fill', isAfterRef ? '#fff' : '#1a1a1a')
        .attr('stroke', '#1a1a1a')
        .attr('stroke-width', isAfterRef ? 1.5 : 0);
    });
  }

  // --- Reference date vertical line ---
  if (refDate && x(refDate) >= 0 && x(refDate) <= innerW) {
    annotLayer.append('line')
      .attr('x1', x(refDate)).attr('x2', x(refDate))
      .attr('y1', 0).attr('y2', innerH)
      .attr('stroke', '#c00')
      .attr('stroke-width', 1.5)
      .attr('stroke-dasharray', '5,4');

    annotLayer.append('text')
      .attr('x', x(refDate) + 5)
      .attr('y', 12)
      .attr('fill', '#c00')
      .attr('font-size', '10px')
      .text(`Forecast Date: ${refDate.toISOString().slice(0, 10)}`);
  }

  // --- Tooltip ---
  const tooltip = d3.select('#tooltip');
  const obsBisect = d3.bisector(d => d.date).left;
  const fcBisect = d3.bisector(d => d.target_end_date).left;

  // Build a quick lookup from date string → observed value
  const obsByDate = new Map();
  filteredObs.forEach(d => obsByDate.set(d.date.toISOString().slice(0, 10), d));

  svg.append('rect')
    .attr('transform', `translate(${margin.left},${margin.top})`)
    .attr('width', innerW).attr('height', innerH)
    .attr('fill', 'none').attr('pointer-events', 'all')
    .on('mousemove', function(event) {
      const [mx] = d3.pointer(event, this);
      const hoverDate = x.invert(mx);

      // Find closest observed point
      let obsPoint = null;
      let obsDist = Infinity;
      if (filteredObs.length > 0) {
        const idx = obsBisect(filteredObs, hoverDate, 1);
        const d0 = filteredObs[Math.max(0, idx - 1)];
        const d1 = filteredObs[Math.min(filteredObs.length - 1, idx)];
        obsPoint = d1 && Math.abs(hoverDate - d0.date) > Math.abs(hoverDate - d1.date) ? d1 : d0;
        obsDist = Math.abs(hoverDate - obsPoint.date);
      }

      // Find closest forecast point
      let fcPoint = null;
      let fcDist = Infinity;
      if (forecastData.length > 0) {
        const fi = fcBisect(forecastData, hoverDate, 1);
        const f0 = forecastData[Math.max(0, fi - 1)];
        const f1 = forecastData[Math.min(forecastData.length - 1, fi)];
        fcPoint = f1 && Math.abs(hoverDate - f0.target_end_date) > Math.abs(hoverDate - f1.target_end_date) ? f1 : f0;
        fcDist = Math.abs(hoverDate - fcPoint.target_end_date);
      }

      // Snap to whichever data point is closest
      let html = '';
      const snapThreshold = 5 * 86400000; // 5 days in ms

      if (fcPoint && fcDist <= obsDist && fcDist < snapThreshold) {
        // Snapped to a forecast point — show forecast + observed if available
        const fcDateStr = fcPoint.target_end_date.toISOString().slice(0, 10);
        const matchObs = obsByDate.get(fcDateStr);
        const weeksAhead = fcPoint.horizon + 1;
        const horizonLabel = `${weeksAhead} Week${weeksAhead > 1 ? 's' : ''} Ahead`;

        html += `<div class="tooltip-title">${fmtDate(fcPoint.target_end_date)}</div>`;
        html += `<div style="font-size:11px;color:#888;margin-bottom:4px;">Horizon: ${horizonLabel}</div>`;
        if (matchObs) {
          html += `<div class="tooltip-row"><span class="tooltip-label">Observed:</span> <span class="tooltip-value">${fmtNum(matchObs.value)}</span></div>`;
        }
        html += `<div class="tooltip-row"><span class="tooltip-label">Forecast Median:</span> <span class="tooltip-value">${fmtNum(fcPoint['0.5'])}</span></div>`;
        html += `<div class="tooltip-row"><span class="tooltip-label">50% PI:</span> <span class="tooltip-value">${fmtNum(fcPoint['0.25'])}\u2013${fmtNum(fcPoint['0.75'])}</span></div>`;
        html += `<div class="tooltip-row"><span class="tooltip-label">95% PI:</span> <span class="tooltip-value">${fmtNum(fcPoint['0.025'])}\u2013${fmtNum(fcPoint['0.975'])}</span></div>`;
        if (matchObs) {
          const error = matchObs.value - fcPoint['0.5'];
          html += `<div style="font-size:11px;color:${Math.abs(error) < 1 ? '#059669' : '#888'};margin-top:4px;">Error: ${error > 0 ? '+' : ''}${fmtNum(error)}</div>`;
        }
      } else if (obsPoint && obsDist < snapThreshold) {
        // Snapped to an observed-only point
        html += `<div class="tooltip-title">${fmtDate(obsPoint.date)}</div>`;
        html += `<div class="tooltip-row"><span class="tooltip-label">Observed:</span> <span class="tooltip-value">${fmtNum(obsPoint.value)}</span></div>`;
        if (obsPoint.weekly_rate != null) {
          html += `<div class="tooltip-row"><span class="tooltip-label">Rate per 100k:</span> <span class="tooltip-value">${obsPoint.weekly_rate.toFixed(2)}</span></div>`;
        }
      }

      if (html) {
        tooltip.classed('visible', true)
          .style('left', (event.clientX + 12) + 'px')
          .style('top', (event.clientY - 10) + 'px')
          .html(html);
      } else {
        tooltip.classed('visible', false);
      }
    })
    .on('mouseleave', () => tooltip.classed('visible', false));
}
