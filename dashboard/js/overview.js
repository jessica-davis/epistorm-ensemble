/* overview.js — Overview tab: activity headline, observed chart, forecast summary */

let overviewChart = null;

function initOverview() {
  // Build activity legend
  const legendEl = document.getElementById('activity-legend');
  ACTIVITY_ORDER.forEach(level => {
    const item = document.createElement('div');
    item.className = 'legend-item';
    const swatch = document.createElement('span');
    swatch.className = 'legend-swatch';
    swatch.style.background = ACTIVITY_COLORS[level];
    const label = document.createElement('span');
    label.textContent = level;
    item.appendChild(swatch);
    item.appendChild(label);
    legendEl.appendChild(item);
  });
  // Add observed line legend
  const lineItem = document.createElement('div');
  lineItem.className = 'legend-item';
  lineItem.innerHTML = '<span style="width:14px;height:2px;background:#1a1a1a;display:inline-block;vertical-align:middle;border-radius:1px;"></span><span>Observed</span>';
  legendEl.appendChild(lineItem);
}

function updateOverview() {
  const loc = AppState.selectedLocation;
  const locThresh = getThresholdsForLocation(loc, DATA.thresholds);

  // --- Activity headline ---
  updateHeadline(loc, locThresh);

  // --- Observed chart ---
  updateActivityChart(loc, locThresh);

  // --- Forecast summary ---
  updateForecastSummary(loc);
}

function updateHeadline(loc, locThresh) {
  const locObs = DATA.observed.filter(d => d.location === loc).sort((a, b) => a.date - b.date);
  const headlineEl = document.getElementById('activity-headline');
  const compEl = document.getElementById('activity-comparison');

  if (locObs.length === 0) {
    headlineEl.textContent = 'No data available for this location.';
    compEl.textContent = '';
    return;
  }

  const latest = locObs[locObs.length - 1];
  const level = locThresh ? getActivityLevel(latest.value, locThresh) : 'Unknown';
  const levelColor = ACTIVITY_COLORS[level] || '#666';
  const locName = getLocationName(loc, DATA.locations);
  const dateStr = fmtDate(latest.date);

  headlineEl.innerHTML =
    `The flu activity level in ${locName} is currently <strong style="color:${levelColor}">${level}</strong> as of <strong>${dateStr}</strong>.`;

  // Year-over-year comparison
  const oneYearAgo = new Date(latest.date);
  oneYearAgo.setDate(oneYearAgo.getDate() - 364);
  const closest = locObs.reduce((best, d) => {
    const diff = Math.abs(d.date - oneYearAgo);
    return diff < Math.abs(best.date - oneYearAgo) ? d : best;
  });
  const daysDiff = Math.abs(closest.date - oneYearAgo) / 86400000;

  if (daysDiff < 14 && closest.value != null && latest.value != null) {
    const lastYearLevel = locThresh ? getActivityLevel(closest.value, locThresh) : null;
    const lastYearColor = lastYearLevel ? (ACTIVITY_COLORS[lastYearLevel] || '#666') : '#666';
    if (lastYearLevel) {
      compEl.innerHTML =
        `This compares to <strong style="color:${lastYearColor}">${lastYearLevel}</strong> activity during the same week last year (${fmtNum(closest.value)} admissions on ${fmtDate(closest.date)}).`;
    } else {
      compEl.textContent =
        `Last year at this time there were ${fmtNum(closest.value)} admissions (${fmtDate(closest.date)}).`;
    }
  } else {
    compEl.textContent = '';
  }
}

function updateActivityChart(loc, locThresh) {
  const container = document.getElementById('activity-chart-container');
  const locObs = DATA.observed
    .filter(d => d.location === loc && d.value != null)
    .sort((a, b) => a.date - b.date);

  if (locObs.length === 0) {
    container.innerHTML = '<div class="empty-state">No observed data for this location.</div>';
    return;
  }

  // Date range filter
  const maxDate = locObs[locObs.length - 1].date;
  const cutoff = getDateRangeCutoff(maxDate, AppState.dateRange);
  const filtered = cutoff ? locObs.filter(d => d.date >= cutoff) : locObs;

  if (filtered.length === 0) {
    container.innerHTML = '<div class="empty-state">No data in selected range.</div>';
    return;
  }

  // Chart dimensions
  const containerWidth = container.clientWidth || 800;
  const width = containerWidth;
  const height = 340;
  const margin = { top: 20, right: 20, bottom: 40, left: 75 };
  const innerW = width - margin.left - margin.right;
  const innerH = height - margin.top - margin.bottom;

  // Clear and redraw
  container.innerHTML = '';
  const svg = d3.select(container)
    .append('svg')
    .attr('viewBox', `0 0 ${width} ${height}`)
    .attr('preserveAspectRatio', 'xMidYMid meet');

  const g = svg.append('g').attr('transform', `translate(${margin.left},${margin.top})`);
  const bgLayer = g.append('g').attr('class', 'bg-layer');
  const dataLayer = g.append('g').attr('class', 'data-layer');
  const axisLayer = g.append('g').attr('class', 'axis-layer');

  // Scales
  const x = d3.scaleTime()
    .domain(d3.extent(filtered, d => d.date))
    .range([0, innerW]);

  const yMax = d3.max(filtered, d => d.value) * 1.1;
  const y = d3.scaleLinear()
    .domain([0, yMax])
    .range([innerH, 0]);

  // Threshold bands
  if (locThresh) {
    const bands = [
      { label: 'Low', lo: 0, hi: locThresh['Medium'], color: ACTIVITY_COLORS['Low'] },
      { label: 'Moderate', lo: locThresh['Medium'], hi: locThresh['High'], color: ACTIVITY_COLORS['Moderate'] },
      { label: 'High', lo: locThresh['High'], hi: locThresh['Very High'], color: ACTIVITY_COLORS['High'] },
      { label: 'Very High', lo: locThresh['Very High'], hi: yMax * 2, color: ACTIVITY_COLORS['Very High'] },
    ];
    bands.forEach(b => {
      const yTop = Math.max(y(Math.min(b.hi, yMax * 1.5)), 0);
      const yBot = Math.min(y(b.lo), innerH);
      if (yBot > yTop) {
        bgLayer.append('rect')
          .attr('x', 0).attr('y', yTop)
          .attr('width', innerW).attr('height', yBot - yTop)
          .attr('fill', b.color).attr('opacity', 0.45);
        // Direct label on the band (left-aligned, with background pill)
        const bandMidY = (yTop + yBot) / 2;
        if (yBot - yTop > 14) {
          // Measure text width to size the pill
          const tempText = bgLayer.append('text')
            .attr('font-size', '9px')
            .attr('font-weight', '600')
            .text(b.label);
          const textW = tempText.node().getComputedTextLength();
          tempText.remove();
          // Background pill
          const pillPad = 4;
          bgLayer.append('rect')
            .attr('x', 3)
            .attr('y', bandMidY - 7)
            .attr('width', textW + pillPad * 2)
            .attr('height', 14)
            .attr('rx', 3)
            .attr('fill', 'rgba(255,255,255,0.7)');
          // Label text
          bgLayer.append('text')
            .attr('x', 3 + pillPad)
            .attr('y', bandMidY)
            .attr('dy', '0.35em')
            .attr('text-anchor', 'start')
            .attr('font-size', '9px')
            .attr('font-weight', '600')
            .attr('fill', '#4a5568')
            .text(b.label);
        }
      }
    });
  }

  // Axes
  axisLayer.append('g')
    .attr('class', 'axis')
    .attr('transform', `translate(0,${innerH})`)
    .call(d3.axisBottom(x).ticks(6).tickFormat(d3.timeFormat('%b %Y')));

  axisLayer.append('g')
    .attr('class', 'axis')
    .call(d3.axisLeft(y).ticks(5).tickFormat(d => fmtNum(d)));

  // Y-axis label
  axisLayer.append('text')
    .attr('class', 'axis-label')
    .attr('transform', 'rotate(-90)')
    .attr('x', -innerH / 2).attr('y', -58)
    .text('Weekly Flu Hospitalizations');

  // Line
  const line = d3.line()
    .x(d => x(d.date))
    .y(d => y(d.value))
    .defined(d => d.value != null);

  dataLayer.append('path')
    .datum(filtered)
    .attr('d', line)
    .attr('fill', 'none')
    .attr('stroke', '#1a1a1a')
    .attr('stroke-width', 2);

  // Circle markers
  dataLayer.selectAll('.obs-dot')
    .data(filtered)
    .join('circle')
    .attr('class', 'obs-dot')
    .attr('cx', d => x(d.date))
    .attr('cy', d => y(d.value))
    .attr('r', 3)
    .attr('fill', '#fff')
    .attr('stroke', '#1a1a1a')
    .attr('stroke-width', 1.5);

  // Tooltip interaction
  const tooltip = d3.select('#tooltip');
  const bisect = d3.bisector(d => d.date).left;

  svg.append('rect')
    .attr('transform', `translate(${margin.left},${margin.top})`)
    .attr('width', innerW).attr('height', innerH)
    .attr('fill', 'none')
    .attr('pointer-events', 'all')
    .on('mousemove', function(event) {
      const [mx] = d3.pointer(event, this);
      const date = x.invert(mx);
      const idx = bisect(filtered, date, 1);
      const d0 = filtered[idx - 1];
      const d1 = filtered[idx];
      const d = d1 && (date - d0.date > d1.date - date) ? d1 : d0;
      if (!d) return;

      tooltip.classed('visible', true)
        .style('left', (event.clientX + 12) + 'px')
        .style('top', (event.clientY - 10) + 'px')
        .html(`<div class="tooltip-title">${fmtDate(d.date)}</div>
               <div class="tooltip-row"><span class="tooltip-label">Admissions:</span> <span class="tooltip-value">${fmtNum(d.value)}</span></div>
               ${d.weekly_rate != null ? `<div class="tooltip-row"><span class="tooltip-label">Rate per 100k:</span> <span class="tooltip-value">${d.weekly_rate.toFixed(2)}</span></div>` : ''}`);
    })
    .on('mouseleave', () => tooltip.classed('visible', false));
}

function updateForecastSummary(loc) {
  const refDate = AppState.overviewRefDate;
  const horizon = AppState.overviewHorizon;
  const refStr = refDate ? refDate.toISOString().slice(0, 10) : null;

  // Get activity level predictions (horizon in data is 0-indexed: 0–3)
  const actData = DATA.activityLevels.filter(d =>
    d.location === loc &&
    d.reference_date.toISOString().slice(0, 10) === refStr &&
    d.horizon === horizon
  );

  // Get trend predictions
  const catData = DATA.categorical.filter(d =>
    d.location === loc &&
    d.reference_date.toISOString().slice(0, 10) === refStr &&
    d.horizon === horizon
  );

  // Headline
  const headlineEl = document.getElementById('forecast-summary-headline');
  const narrativeEl = document.getElementById('forecast-summary-narrative');

  // Find most likely activity level
  let topActivity = null;
  let topActivityProb = 0;
  ACTIVITY_ORDER.forEach(level => {
    const match = actData.find(d => d.output_type_id === level);
    if (match && match.value > topActivityProb) {
      topActivity = level;
      topActivityProb = match.value;
    }
  });

  // Find most likely trend
  let topTrend = null;
  let topTrendProb = 0;
  CATEGORY_ORDER.forEach(cat => {
    const match = catData.find(d => formatCategory(d.output_type_id) === cat);
    if (match && match.value > topTrendProb) {
      topTrend = cat;
      topTrendProb = match.value;
    }
  });

  if (topActivity && topTrend) {
    const actColor = ACTIVITY_COLORS[topActivity] || '#666';
    const trendColor = CATEGORY_COLORS[topTrend] || '#666';
    headlineEl.innerHTML =
      `Forecast summary &middot; <strong style="color:${actColor}">${topActivity}</strong> activity, trending <strong style="color:${trendColor}">${topTrend.toLowerCase()}</strong>`;

    const locName = getLocationName(loc, DATA.locations);
    const weeksAhead = horizon + 1;
    const horizonLabel = `the next ${weeksAhead} week${weeksAhead > 1 ? 's' : ''}`;
    // Compute the target forecast date
    const targetDate = new Date(refDate);
    targetDate.setDate(targetDate.getDate() + weeksAhead * 7);
    const targetDateStr = fmtDate(targetDate);
    const trendVerb = topTrend.includes('Increase') ? 'increase' : topTrend.includes('Decrease') ? 'decrease' : 'remain stable';

    // Get median forecast value from ensemble data
    const ensFc = DATA.ensemble.find(d =>
      d.location === loc &&
      d.reference_date.toISOString().slice(0, 10) === refStr &&
      d.horizon === horizon
    );
    const medianStr = ensFc && ensFc['0.5'] != null ? ` with a median forecast of <strong>${fmtNum(ensFc['0.5'])}</strong> admissions` : '';

    narrativeEl.innerHTML =
      `Over ${horizonLabel} (${targetDateStr}), flu hospitalizations in ${locName} are projected to ${trendVerb}${medianStr} — activity levels are forecast to stay <strong style="color:${actColor}">${topActivity.toLowerCase()}</strong>.`;
  } else {
    headlineEl.textContent = 'Forecast summary';
    narrativeEl.textContent = 'No forecast data available for the selected date and location.';
  }

  // Draw probability bars
  drawProbBars('activity-bars-container', actData, ACTIVITY_ORDER, ACTIVITY_COLORS, 'output_type_id', false);
  drawProbBars('trend-bars-container', catData, CATEGORY_ORDER, CATEGORY_COLORS, 'output_type_id', true);
}

function drawProbBars(containerId, data, categories, colors, idField, formatCat) {
  const container = document.getElementById(containerId);
  container.innerHTML = '';

  if (data.length === 0) {
    container.innerHTML = '<div class="empty-state" style="padding:12px;font-size:12px;">No data</div>';
    return;
  }

  // Find winner
  let maxProb = 0;
  let winner = null;
  const probs = {};
  categories.forEach(cat => {
    const match = data.find(d => {
      const id = formatCat ? formatCategory(d[idField]) : d[idField];
      return id === cat;
    });
    probs[cat] = match ? match.value : 0;
    if (probs[cat] > maxProb) {
      maxProb = probs[cat];
      winner = cat;
    }
  });

  const barHeight = 24;
  const labelWidth = 100;
  const pctWidth = 45;
  const maxBarWidth = 150;

  categories.forEach(cat => {
    const prob = probs[cat] || 0;
    const isWinner = cat === winner;
    const opacity = isWinner ? 1 : 0.35;

    const row = document.createElement('div');
    row.style.cssText = `display:flex;align-items:center;margin-bottom:4px;opacity:${opacity};`;

    const labelSpan = document.createElement('span');
    labelSpan.style.cssText = `width:${labelWidth}px;font-size:12px;color:#333;flex-shrink:0;`;
    labelSpan.textContent = cat;

    const barWrap = document.createElement('div');
    barWrap.style.cssText = `flex:1;height:${barHeight}px;background:#f0f0f0;border-radius:3px;overflow:hidden;max-width:${maxBarWidth}px;`;

    const bar = document.createElement('div');
    bar.style.cssText = `height:100%;width:${Math.max(prob * 100, 1)}%;background:${colors[cat] || '#999'};border-radius:3px;transition:width 0.3s ease;`;

    barWrap.appendChild(bar);

    const pctSpan = document.createElement('span');
    pctSpan.style.cssText = `width:${pctWidth}px;text-align:right;font-size:12px;font-weight:600;color:#333;flex-shrink:0;margin-left:8px;`;
    pctSpan.textContent = fmtPct(prob);

    row.appendChild(labelSpan);
    row.appendChild(barWrap);
    row.appendChild(pctSpan);
    container.appendChild(row);
  });
}
