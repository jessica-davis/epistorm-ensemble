/* forecast-map.js — Choropleth map on Forecast tab colored by median forecast values */

let fmapSvg = null;
let fmapPath = null;
let fmapStates = null;
let fmapFipsToLoc = {};
let fmapColorScale = null;
let fmapDcRect = null;
let fmapPrEl = null;
let fmapUsRect = null;

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

  // Hatch pattern for no-data states
  const defs = fmapSvg.append('defs');
  const hatch = defs.append('pattern')
    .attr('id', 'fmap-hatch')
    .attr('patternUnits', 'userSpaceOnUse')
    .attr('width', 6).attr('height', 6);
  hatch.append('rect')
    .attr('width', 6).attr('height', 6)
    .attr('fill', '#f0f0f0');
  hatch.append('path')
    .attr('d', 'M0,6 L6,0')
    .attr('stroke', '#d0d0d0')
    .attr('stroke-width', 1);

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
      const locCode = fmapFipsToLoc[d.id];
      if (locCode) showMapTooltipForLoc(event, locCode);
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

  // --- DC Inset (small square near DC's actual position) ---
  const dcFeature = states.find(f => String(f.id) === '11');
  if (dcFeature) {
    const dcCentroid = fmapPath.centroid(dcFeature);
    if (dcCentroid && !isNaN(dcCentroid[0])) {
      const dcSize = 16;
      const dcX = dcCentroid[0] + 16;
      const dcY = dcCentroid[1] + 8;
      const dcG = fmapSvg.append('g').attr('class', 'fmap-dc-inset');

      // Leader line from actual DC to inset square
      dcG.append('line')
        .attr('x1', dcCentroid[0]).attr('y1', dcCentroid[1])
        .attr('x2', dcX + dcSize / 2).attr('y2', dcY + dcSize / 2)
        .attr('stroke', '#a0aec0').attr('stroke-width', 0.7)
        .attr('stroke-dasharray', '2,1')
        .style('pointer-events', 'none');

      fmapDcRect = dcG.append('rect')
        .attr('x', dcX).attr('y', dcY)
        .attr('width', dcSize).attr('height', dcSize)
        .attr('rx', 2)
        .attr('fill', '#ddd')
        .attr('stroke', '#fff').attr('stroke-width', 1);
      attachInsetEvents(fmapDcRect, '11');
    }
  }

  // --- PR Inset (bottom-right area, near Florida) ---
  const prFeature = states.find(f => String(f.id) === '72');
  const prInsetX = 570;
  const prInsetY = 365;

  if (prFeature) {
    // Render actual PR shape with a separate projection
    const prGeo = { type: 'FeatureCollection', features: [prFeature] };
    const prProj = d3.geoMercator().fitExtent([[0, 0], [50, 28]], prGeo);
    const prPathGen = d3.geoPath().projection(prProj);

    const prG = fmapSvg.append('g')
      .attr('class', 'fmap-pr-inset')
      .attr('transform', `translate(${prInsetX}, ${prInsetY})`);

    const prBounds = prPathGen.bounds(prFeature);
    const prCx = (prBounds[0][0] + prBounds[1][0]) / 2;

    fmapPrEl = prG.append('path')
      .datum(prFeature)
      .attr('d', prPathGen)
      .attr('fill', '#ddd')
      .attr('stroke', '#fff').attr('stroke-width', 0.8);
    attachInsetEvents(fmapPrEl, '72');
  } else {
    // Fallback: rectangle for PR if not in TopoJSON
    const prG = fmapSvg.append('g')
      .attr('class', 'fmap-pr-inset')
      .attr('transform', `translate(${prInsetX}, ${prInsetY})`);

    fmapPrEl = prG.append('rect')
      .attr('x', 0).attr('y', 0)
      .attr('width', 40).attr('height', 18)
      .attr('rx', 3)
      .attr('fill', '#ddd')
      .attr('stroke', '#fff').attr('stroke-width', 1);
    attachInsetEvents(fmapPrEl, '72');
  }

  // --- US National box (top center, above MI/NY area) ---
  const usBoxW = 54;
  const usBoxH = 28;
  const usBoxX = 490;
  const usBoxY = 6;
  const usG = fmapSvg.append('g').attr('class', 'fmap-us-inset');

  usG.append('text')
    .attr('x', usBoxX + usBoxW / 2).attr('y', usBoxY + usBoxH + 12)
    .attr('text-anchor', 'middle')
    .attr('font-size', '8px').attr('fill', '#718096').attr('font-weight', '500')
    .text('National-Level Forecast')
    .style('pointer-events', 'none');

  fmapUsRect = usG.append('rect')
    .attr('x', usBoxX).attr('y', usBoxY)
    .attr('width', usBoxW).attr('height', usBoxH)
    .attr('rx', 4)
    .attr('fill', '#ddd')
    .attr('stroke', '#fff').attr('stroke-width', 1);
  attachInsetEvents(fmapUsRect, 'US');

  initForecastMapControls();
}

/* Shared event wiring for inset elements (DC, PR, US) */
function attachInsetEvents(el, locCode) {
  el.style('cursor', 'pointer')
    .on('click', () => setLocation(locCode))
    .on('mouseenter', function(event) {
      d3.select(this).attr('stroke', '#333').attr('stroke-width', 2);
      showMapTooltipForLoc(event, locCode);
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
}

function initForecastMapControls() {
  document.querySelectorAll('#horizon-toggle .toggle-btn').forEach(btn => {
    btn.addEventListener('click', function() {
      document.querySelectorAll('#horizon-toggle .toggle-btn').forEach(b => b.classList.remove('active'));
      btn.classList.add('active');
      AppState.forecastHorizon = +btn.dataset.horizon;
      updateForecastMap();
    });
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

  // Build location → {median, lo95, hi95, displayVal} lookup (exclude US for color scale)
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

  // Auto-scale color domain from state values (excludes US to avoid dominating the scale)
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
      if (!locCode) return 'url(#fmap-hatch)';
      const data = dataByLoc[locCode];
      return data ? fmapColorScale(data.displayVal) : 'url(#fmap-hatch)';
    });

  // Color DC inset
  if (fmapDcRect) {
    const dcData = dataByLoc['11'];
    fmapDcRect.transition().duration(300)
      .attr('fill', dcData ? fmapColorScale(dcData.displayVal) : 'url(#fmap-hatch)');
  }

  // Color PR inset
  if (fmapPrEl) {
    const prData = dataByLoc['72'];
    fmapPrEl.transition().duration(300)
      .attr('fill', prData ? fmapColorScale(prData.displayVal) : 'url(#fmap-hatch)');
  }

  // Color US box (US value may exceed state max — clamp to scale)
  if (fmapUsRect) {
    const usFc = ensData.find(d =>
      d.location === 'US' &&
      d.reference_date.toISOString().slice(0, 10) === refStr &&
      d.horizon === horizon
    );
    if (usFc && usFc['0.5'] != null) {
      const usPop = getPopulation('US', DATA.locations);
      const usVal = usePerCapita && usPop ? perCapita(usFc['0.5'], usPop) : usFc['0.5'];
      fmapUsRect.transition().duration(300)
        .attr('fill', fmapColorScale(Math.min(usVal, maxVal)));
    } else {
      fmapUsRect.transition().duration(300)
        .attr('fill', 'url(#fmap-hatch)');
    }
  }

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

  // DC inset
  if (fmapDcRect) {
    fmapDcRect
      .attr('stroke', selectedLoc === '11' ? '#1a1a1a' : '#fff')
      .attr('stroke-width', selectedLoc === '11' ? 2.5 : 1);
  }
  // PR inset
  if (fmapPrEl) {
    fmapPrEl
      .attr('stroke', selectedLoc === '72' ? '#1a1a1a' : '#fff')
      .attr('stroke-width', selectedLoc === '72' ? 2.5 : 0.8);
  }
  // US inset
  if (fmapUsRect) {
    fmapUsRect
      .attr('stroke', selectedLoc === 'US' ? '#1a1a1a' : '#fff')
      .attr('stroke-width', selectedLoc === 'US' ? 2.5 : 1);
  }
}

/* Unified tooltip for any location (states, DC, PR, US) */
function showMapTooltipForLoc(event, locCode) {
  const locName = getLocationName(locCode, DATA.locations);
  const refDate = AppState.selectedDate;
  const horizon = AppState.forecastHorizon;
  const modelName = AppState.selectedModel;
  const usePerCapita = AppState.forecastPerCapita;
  const refStr = refDate ? refDate.toISOString().slice(0, 10) : null;
  const ensData = modelName === 'LOP Epistorm Ensemble' ? DATA.ensembleLop : DATA.ensemble;
  const population = getPopulation(locCode, DATA.locations);

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

  const marginL = 40;
  const marginR = 40;
  const barW = 360;
  const totalW = marginL + barW + marginR;
  const legendH = 46;
  const barH = 16;
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
    .style('font-size', '11px')
    .style('fill', '#718096');

  // Remove axis line
  svg.select('.domain').remove();

  // Unit label + no-data indicator
  const unitText = usePerCapita ? 'Median forecast (per 100k)' : 'Median forecast (admissions)';
  container.insertAdjacentHTML('beforeend',
    `<div style="display:flex;align-items:center;gap:16px;margin-top:4px;">
      <span style="font-size:12px;color:#718096;">${unitText}</span>
      <span style="display:flex;align-items:center;gap:5px;font-size:11px;color:#a0aec0;">
        <svg width="14" height="14"><rect width="14" height="14" fill="url(#fmap-hatch)" rx="2" stroke="#d0d0d0" stroke-width="0.5"/></svg>
        No data
      </span>
    </div>`
  );
}
