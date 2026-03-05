/* forecast-map.js — Choropleth map on Forecast tab colored by median forecast values */

let fmapSvg = null;
let fmapPath = null;
let fmapStates = null;
let fmapFipsToLoc = {};
let fmapColorScale = null;

function initForecastMap() {
  const container = document.getElementById('forecast-map-container');
  if (!container || !DATA.usStates) return;

  const width = 700;
  const height = 440;

  // Build FIPS-to-location mapping
  DATA.locations.forEach(l => {
    if (l.location !== 'US') {
      fmapFipsToLoc[String(+l.location)] = l.location;
      fmapFipsToLoc[l.location] = l.location;
    }
  });

  fmapSvg = d3.select(container)
    .append('svg')
    .attr('viewBox', `0 0 ${width} ${height}`)
    .attr('preserveAspectRatio', 'xMidYMid meet');

  const projection = d3.geoAlbersUsa()
    .fitSize([width - 40, height - 40],
      topojson.feature(DATA.usStates, DATA.usStates.objects.states));

  fmapPath = d3.geoPath().projection(projection);

  const states = topojson.feature(DATA.usStates, DATA.usStates.objects.states).features;

  fmapStates = fmapSvg.selectAll('.fmap-state')
    .data(states)
    .join('path')
    .attr('class', 'fmap-state')
    .attr('d', fmapPath)
    .attr('fill', '#ddd')
    .attr('stroke', '#fff')
    .attr('stroke-width', 0.8)
    .style('cursor', 'pointer')
    .on('click', function(event, d) {
      const locCode = fmapFipsToLoc[d.id];
      if (locCode) setLocation(locCode);
    })
    .on('mouseenter', function(event, d) {
      d3.select(this).attr('stroke', '#333').attr('stroke-width', 2);
      showForecastMapTooltip(event, d);
    })
    .on('mousemove', function(event) {
      d3.select('#tooltip')
        .style('left', (event.clientX + 14) + 'px')
        .style('top', (event.clientY - 10) + 'px');
    })
    .on('mouseleave', function() {
      updateForecastMapHighlight();
      d3.select('#tooltip').classed('visible', false);
    });

  // State borders overlay
  fmapSvg.append('path')
    .datum(topojson.mesh(DATA.usStates, DATA.usStates.objects.states, (a, b) => a !== b))
    .attr('fill', 'none')
    .attr('stroke', '#fff')
    .attr('stroke-width', 0.8)
    .attr('d', fmapPath)
    .style('pointer-events', 'none');

  initForecastMapControls();
}

function initForecastMapControls() {
  document.getElementById('forecast-horizon-select').addEventListener('change', function(e) {
    AppState.forecastHorizon = +e.target.value;
    updateForecastMap();
  });

  document.querySelectorAll('#percapita-toggle .toggle-btn').forEach(btn => {
    btn.addEventListener('click', function() {
      document.querySelectorAll('#percapita-toggle .toggle-btn').forEach(b => b.classList.remove('active'));
      btn.classList.add('active');
      AppState.forecastPerCapita = (btn.dataset.mode === 'percapita');
      updateForecastMap();
    });
  });
}

function updateForecastMap() {
  if (!fmapStates) return;

  const refDate = AppState.selectedDate;
  const horizon = AppState.forecastHorizon;
  const modelName = AppState.selectedModel;
  const usePerCapita = AppState.forecastPerCapita;
  const refStr = refDate ? refDate.toISOString().slice(0, 10) : null;

  const ensData = modelName === 'LOP Epistorm Ensemble' ? DATA.ensembleLop : DATA.ensemble;

  // Build location → {median, lo95, hi95} lookup
  const dataByLoc = {};
  ensData.forEach(d => {
    if (d.reference_date.toISOString().slice(0, 10) === refStr &&
        d.horizon === horizon && d.location !== 'US') {
      const median = d['0.5'];
      if (median != null) {
        const pop = getPopulation(d.location, DATA.locations);
        dataByLoc[d.location] = {
          median: median,
          lo95: d['0.025'],
          hi95: d['0.975'],
          displayVal: usePerCapita && pop ? perCapita(median, pop) : median,
        };
      }
    }
  });

  // Auto-scale color domain
  const values = Object.values(dataByLoc).map(d => d.displayVal);
  const maxVal = values.length > 0 ? d3.max(values) : 1;

  const modelColor = COLOR_MAP[modelName] || '#4A9E8E';
  fmapColorScale = d3.scaleSequential(t => d3.interpolateRgb('#f0f4f8', modelColor)(t))
    .domain([0, maxVal]);

  // Update state fills
  fmapStates
    .transition().duration(300)
    .attr('fill', function(d) {
      const locCode = fmapFipsToLoc[d.id];
      if (!locCode) return '#eee';
      const data = dataByLoc[locCode];
      return data ? fmapColorScale(data.displayVal) : '#eee';
    });

  updateForecastMapHighlight();
  drawForecastMapLegend(maxVal, usePerCapita);
}

function updateForecastMapHighlight() {
  if (!fmapStates) return;
  const selectedLoc = AppState.selectedLocation;

  fmapStates
    .attr('stroke-width', function(d) {
      const locCode = fmapFipsToLoc[d.id];
      return (locCode === selectedLoc && selectedLoc !== 'US') ? 2.5 : 0.8;
    })
    .attr('stroke', function(d) {
      const locCode = fmapFipsToLoc[d.id];
      return (locCode === selectedLoc && selectedLoc !== 'US') ? '#1a1a1a' : '#fff';
    });
}

function showForecastMapTooltip(event, d) {
  const locCode = fmapFipsToLoc[d.id];
  if (!locCode) return;

  const locName = getLocationName(locCode, DATA.locations);
  const refDate = AppState.selectedDate;
  const horizon = AppState.forecastHorizon;
  const modelName = AppState.selectedModel;
  const usePerCapita = AppState.forecastPerCapita;
  const refStr = refDate ? refDate.toISOString().slice(0, 10) : null;
  const ensData = modelName === 'LOP Epistorm Ensemble' ? DATA.ensembleLop : DATA.ensemble;
  const population = getPopulation(locCode, DATA.locations);

  // Find forecast for this state/date/horizon
  const fc = ensData.find(r =>
    r.location === locCode &&
    r.reference_date.toISOString().slice(0, 10) === refStr &&
    r.horizon === horizon
  );

  const weeksAhead = horizon + 1;
  let html = `<div class="tooltip-title">${locName}</div>`;
  html += `<div style="font-size:11px;color:#888;margin-bottom:4px;">${weeksAhead} Week${weeksAhead > 1 ? 's' : ''} Ahead</div>`;

  if (fc && fc['0.5'] != null) {
    if (usePerCapita && population) {
      html += `<div class="tooltip-row"><span class="tooltip-label">Median:</span> <span class="tooltip-value">${perCapita(fc['0.5'], population).toFixed(1)} per 100k</span></div>`;
      html += `<div class="tooltip-row"><span class="tooltip-label">95% PI:</span> <span class="tooltip-value">${perCapita(fc['0.025'], population).toFixed(1)}\u2013${perCapita(fc['0.975'], population).toFixed(1)}</span></div>`;
    } else {
      html += `<div class="tooltip-row"><span class="tooltip-label">Median:</span> <span class="tooltip-value">${fmtNum(fc['0.5'])}</span></div>`;
      html += `<div class="tooltip-row"><span class="tooltip-label">95% PI:</span> <span class="tooltip-value">${fmtNum(fc['0.025'])}\u2013${fmtNum(fc['0.975'])}</span></div>`;
    }
  } else {
    html += `<div style="font-size:11px;color:#aaa;">No forecast data</div>`;
  }

  // Add sparkline
  html += buildSparkline(locCode, refDate, ensData);

  d3.select('#tooltip')
    .classed('visible', true)
    .style('left', (event.clientX + 14) + 'px')
    .style('top', (event.clientY - 10) + 'px')
    .html(html);
}

function buildSparkline(locCode, refDate, ensData) {
  // 3 months of observed data
  const threeMonthsAgo = new Date(refDate);
  threeMonthsAgo.setMonth(threeMonthsAgo.getMonth() - 3);

  const locObs = DATA.observed
    .filter(d => d.location === locCode && d.value != null && d.date >= threeMonthsAgo)
    .sort((a, b) => a.date - b.date);

  const refStr = refDate.toISOString().slice(0, 10);
  const fcData = ensData
    .filter(d => d.location === locCode &&
            d.reference_date.toISOString().slice(0, 10) === refStr &&
            d.horizon >= 0)
    .sort((a, b) => a.target_end_date - b.target_end_date);

  if (locObs.length === 0 && fcData.length === 0) return '';

  const sparkW = 200;
  const sparkH = 70;
  const m = { top: 6, right: 6, bottom: 6, left: 6 };
  const iW = sparkW - m.left - m.right;
  const iH = sparkH - m.top - m.bottom;

  // Combined domain
  const allDates = [
    ...locObs.map(d => d.date.getTime()),
    ...fcData.map(d => d.target_end_date.getTime())
  ];
  const allValues = [
    ...locObs.map(d => d.value),
    ...fcData.map(d => d['0.975'] || d['0.5'] || 0)
  ];

  if (allDates.length === 0) return '';

  const xMin = d3.min(allDates);
  const xMax = d3.max(allDates);
  const xRange = xMax - xMin || 1;
  const yMax = d3.max(allValues) * 1.05 || 1;

  const sx = (t) => m.left + ((t - xMin) / xRange) * iW;
  const sy = (v) => m.top + iH - (v / yMax) * iH;

  const modelColor = COLOR_MAP[AppState.selectedModel] || '#2B7A8F';

  let svg = `<svg width="${sparkW}" height="${sparkH}" style="margin-top:6px;display:block;">`;

  // Forecast 95% PI area
  if (fcData.length > 0) {
    const upper = fcData.map(d => `${sx(d.target_end_date.getTime())},${sy(d['0.975'] || 0)}`);
    const lower = [...fcData].reverse().map(d => `${sx(d.target_end_date.getTime())},${sy(d['0.025'] || 0)}`);
    svg += `<polygon points="${upper.join(' ')} ${lower.join(' ')}" fill="${hexToRgba(modelColor, 0.25)}" stroke="none"/>`;

    // Forecast median line
    const fcPath = fcData.map((d, i) =>
      `${i === 0 ? 'M' : 'L'}${sx(d.target_end_date.getTime())},${sy(d['0.5'])}`
    ).join('');
    svg += `<path d="${fcPath}" fill="none" stroke="${modelColor}" stroke-width="1.5"/>`;
  }

  // Observed line
  if (locObs.length > 0) {
    const obsPath = locObs.map((d, i) =>
      `${i === 0 ? 'M' : 'L'}${sx(d.date.getTime())},${sy(d.value)}`
    ).join('');
    svg += `<path d="${obsPath}" fill="none" stroke="#1a1a1a" stroke-width="1.5"/>`;
  }

  // Reference date vertical line
  const refTime = refDate.getTime();
  if (refTime >= xMin && refTime <= xMax) {
    const rx = sx(refTime);
    svg += `<line x1="${rx}" x2="${rx}" y1="${m.top}" y2="${m.top + iH}" stroke="#c00" stroke-width="1" stroke-dasharray="3,2"/>`;
  }

  svg += '</svg>';
  return svg;
}

function drawForecastMapLegend(maxVal, usePerCapita) {
  const container = document.getElementById('forecast-map-legend');
  container.innerHTML = '';

  if (maxVal <= 0) return;

  const marginL = 30;
  const marginR = 30;
  const barW = 240;
  const totalW = marginL + barW + marginR;
  const legendH = 40;
  const barH = 12;
  const barY = 4;

  const svg = d3.select(container)
    .append('svg')
    .attr('width', totalW)
    .attr('height', legendH);

  // Gradient matching the model color scale
  const modelColor = COLOR_MAP[AppState.selectedModel] || '#4A9E8E';
  const interpolator = t => d3.interpolateRgb('#f0f4f8', modelColor)(t);

  const defs = svg.append('defs');
  const gradient = defs.append('linearGradient')
    .attr('id', 'fmap-legend-grad');

  const numStops = 10;
  for (let i = 0; i <= numStops; i++) {
    const t = i / numStops;
    gradient.append('stop')
      .attr('offset', `${t * 100}%`)
      .attr('stop-color', interpolator(t));
  }

  svg.append('rect')
    .attr('x', marginL).attr('y', barY)
    .attr('width', barW).attr('height', barH)
    .attr('fill', 'url(#fmap-legend-grad)')
    .attr('rx', 2);

  // Axis
  const scale = d3.scaleLinear().domain([0, maxVal]).range([marginL, marginL + barW]);
  const axis = d3.axisBottom(scale).ticks(5).tickSize(4);

  if (usePerCapita) {
    axis.tickFormat(d => d.toFixed(1));
  } else {
    axis.tickFormat(d => fmtNum(d));
  }

  svg.append('g')
    .attr('transform', `translate(0, ${barY + barH})`)
    .call(axis)
    .selectAll('text')
    .style('font-size', '10px')
    .style('fill', '#718096');

  // Remove axis line
  svg.select('.domain').remove();

  // Unit label
  const unitText = usePerCapita ? 'Median forecast (per 100k)' : 'Median forecast (admissions)';
  container.insertAdjacentHTML('beforeend',
    `<div style="font-size:11px;color:#a0aec0;margin-top:2px;">${unitText}</div>`
  );
}
