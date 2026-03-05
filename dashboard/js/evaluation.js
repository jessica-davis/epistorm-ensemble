/* evaluation.js — Evaluation tab: WIS boxplots, coverage calibration */

function initEvaluation() {
  // Nothing to pre-build; updateEvaluation handles everything
}

function updateEvaluation() {
  const filtered = applyEvalFilters();
  drawWISBoxplot(filtered.wis);
  drawCoverageChart(filtered.cov);
}

function applyEvalFilters() {
  let wis = DATA.wisRatio;
  let cov = DATA.coverage;

  // Location filter
  if (AppState.evalLocation !== 'all') {
    wis = wis.filter(d => d.location === AppState.evalLocation);
    cov = cov.filter(d => d.location === AppState.evalLocation);
  }

  // Horizon filter
  if (AppState.evalHorizon && AppState.evalHorizon !== 'all') {
    const h = +AppState.evalHorizon;
    if (!isNaN(h)) {
      wis = wis.filter(d => d.horizon === h);
      cov = cov.filter(d => d.horizon === h);
    }
  }

  // Forecast date range (reference_date)
  const [refStart, refEnd] = AppState.evalRefDateRange;
  if (refStart) {
    wis = wis.filter(d => d.reference_date >= refStart);
    cov = cov.filter(d => d.reference_date >= refStart);
  }
  if (refEnd) {
    wis = wis.filter(d => d.reference_date <= refEnd);
    cov = cov.filter(d => d.reference_date <= refEnd);
  }

  // Model filter
  const models = AppState.evalModels;
  wis = wis.filter(d => models.includes(d.model));
  cov = cov.filter(d => models.includes(d.model));

  return { wis, cov };
}

function drawWISBoxplot(wisData) {
  const container = document.getElementById('wis-chart-container');
  const emptyEl = document.getElementById('wis-empty');
  container.innerHTML = '';

  if (wisData.length === 0) {
    emptyEl.style.display = 'block';
    return;
  }
  emptyEl.style.display = 'none';

  // Group by model, compute box stats
  const byModel = d3.group(wisData, d => d.model);
  const stats = [];
  byModel.forEach((values, model) => {
    const ratios = values.map(d => d.wis_ratio).filter(v => v != null && isFinite(v)).sort(d3.ascending);
    if (ratios.length === 0) return;
    const q1 = d3.quantile(ratios, 0.25);
    const median = d3.quantile(ratios, 0.5);
    const q3 = d3.quantile(ratios, 0.75);
    const iqr = q3 - q1;
    const whiskerLo = Math.max(d3.min(ratios), q1 - 1.5 * iqr);
    const whiskerHi = Math.min(d3.max(ratios), q3 + 1.5 * iqr);
    stats.push({ model, q1, median, q3, whiskerLo, whiskerHi, n: ratios.length });
  });

  // Sort by median ascending
  stats.sort((a, b) => a.median - b.median);

  // Chart dimensions
  const containerWidth = container.clientWidth || 700;
  const boxH = 28;
  const boxGap = 6;
  const margin = { top: 20, right: 30, bottom: 40, left: 200 };
  const innerW = containerWidth - margin.left - margin.right;
  const innerH = stats.length * (boxH + boxGap);
  const height = innerH + margin.top + margin.bottom;

  const svg = d3.select(container)
    .append('svg')
    .attr('viewBox', `0 0 ${containerWidth} ${height}`)
    .attr('preserveAspectRatio', 'xMidYMid meet');

  const g = svg.append('g').attr('transform', `translate(${margin.left},${margin.top})`);

  // X scale
  const xMin = d3.min(stats, d => d.whiskerLo);
  const xMax = d3.max(stats, d => d.whiskerHi);
  const xPad = (xMax - xMin) * 0.1;
  const x = d3.scaleLinear()
    .domain([Math.min(xMin - xPad, 0), xMax + xPad])
    .range([0, innerW]);

  // Y scale (band)
  const y = d3.scaleBand()
    .domain(stats.map(d => d.model))
    .range([0, innerH])
    .padding(0.15);

  // X axis
  g.append('g')
    .attr('class', 'axis')
    .attr('transform', `translate(0,${innerH})`)
    .call(d3.axisBottom(x).ticks(6).tickFormat(d3.format('.1f')));

  g.append('text')
    .attr('class', 'axis-label')
    .attr('x', innerW / 2).attr('y', innerH + 35)
    .text('WIS Ratio');

  // Y axis (model names) — bold ensemble models
  g.append('g')
    .attr('class', 'axis')
    .call(d3.axisLeft(y).tickSize(0))
    .selectAll('text')
    .style('font-size', '11px')
    .style('font-weight', d => ENSEMBLE_MODELS.includes(d) ? '700' : '400')
    .style('fill', d => ENSEMBLE_MODELS.includes(d) ? '#1a202c' : '#718096');

  // Red dashed line at x=1
  if (x(1) >= 0 && x(1) <= innerW) {
    g.append('line')
      .attr('x1', x(1)).attr('x2', x(1))
      .attr('y1', 0).attr('y2', innerH)
      .attr('stroke', '#c00')
      .attr('stroke-width', 1.5)
      .attr('stroke-dasharray', '5,4');

    g.append('text')
      .attr('x', x(1) + 4).attr('y', -6)
      .attr('fill', '#c00')
      .attr('font-size', '10px')
      .text('WIS ratio = 1');
  }

  // Tooltip ref
  const tooltip = d3.select('#tooltip');

  // Draw boxes
  stats.forEach(s => {
    const color = COLOR_MAP[s.model] || '#999';
    const yPos = y(s.model);
    const bh = y.bandwidth();

    // Whisker line
    g.append('line')
      .attr('x1', x(s.whiskerLo)).attr('x2', x(s.whiskerHi))
      .attr('y1', yPos + bh / 2).attr('y2', yPos + bh / 2)
      .attr('stroke', color).attr('stroke-width', 1.5);

    // Whisker caps
    [s.whiskerLo, s.whiskerHi].forEach(w => {
      g.append('line')
        .attr('x1', x(w)).attr('x2', x(w))
        .attr('y1', yPos + bh * 0.2).attr('y2', yPos + bh * 0.8)
        .attr('stroke', color).attr('stroke-width', 1.5);
    });

    // Box (Q1 to Q3)
    g.append('rect')
      .attr('x', x(s.q1))
      .attr('y', yPos)
      .attr('width', Math.max(x(s.q3) - x(s.q1), 1))
      .attr('height', bh)
      .attr('fill', hexToRgba(color, 0.4))
      .attr('stroke', color)
      .attr('stroke-width', 1.5)
      .attr('rx', 2);

    // Median line
    g.append('line')
      .attr('x1', x(s.median)).attr('x2', x(s.median))
      .attr('y1', yPos).attr('y2', yPos + bh)
      .attr('stroke', color).attr('stroke-width', 2.5);

    // Invisible hover rect for tooltip
    g.append('rect')
      .attr('x', x(s.whiskerLo) - 4)
      .attr('y', yPos)
      .attr('width', Math.max(x(s.whiskerHi) - x(s.whiskerLo) + 8, bh))
      .attr('height', bh)
      .attr('fill', 'transparent')
      .attr('cursor', 'pointer')
      .on('mousemove', function(event) {
        const isEns = ENSEMBLE_MODELS.includes(s.model);
        tooltip.classed('visible', true)
          .style('left', (event.clientX + 14) + 'px')
          .style('top', (event.clientY - 10) + 'px')
          .html(
            `<div class="tooltip-title" style="color:${color}">${s.model}${isEns ? ' ★' : ''}</div>` +
            `<div class="tooltip-row"><span class="tooltip-label">Median WIS Ratio:</span> <span class="tooltip-value">${s.median.toFixed(3)}</span></div>` +
            `<div class="tooltip-row"><span class="tooltip-label">Q1 – Q3:</span> <span class="tooltip-value">${s.q1.toFixed(3)} – ${s.q3.toFixed(3)}</span></div>` +
            `<div class="tooltip-row"><span class="tooltip-label">Whiskers:</span> <span class="tooltip-value">${s.whiskerLo.toFixed(3)} – ${s.whiskerHi.toFixed(3)}</span></div>` +
            `<div class="tooltip-row"><span class="tooltip-label">Observations:</span> <span class="tooltip-value">${s.n}</span></div>` +
            (s.median < 1 ? `<div style="font-size:11px;color:#059669;margin-top:4px;">▼ Better than baseline</div>` :
             s.median > 1 ? `<div style="font-size:11px;color:#c00;margin-top:4px;">▲ Worse than baseline</div>` : '')
          );
      })
      .on('mouseleave', () => tooltip.classed('visible', false));
  });
}

function drawCoverageChart(covData) {
  const container = document.getElementById('coverage-chart-container');
  const emptyEl = document.getElementById('coverage-empty');
  container.innerHTML = '';

  if (covData.length === 0) {
    emptyEl.style.display = 'block';
    return;
  }
  emptyEl.style.display = 'none';

  // Compute mean coverage per model per interval
  const byModel = d3.group(covData, d => d.model);
  const modelLines = [];
  byModel.forEach((values, model) => {
    if (!AppState.evalModels.includes(model)) return;
    const points = INTERVAL_RANGES.map(iv => {
      const colName = iv + '_cov';
      const vals = values.map(d => d[colName]).filter(v => v != null && isFinite(v));
      const mean = vals.length > 0 ? d3.mean(vals) : null;
      return { interval: iv, coverage: mean };
    }).filter(p => p.coverage != null);
    if (points.length > 0) {
      modelLines.push({ model, points });
    }
  });

  if (modelLines.length === 0) {
    emptyEl.style.display = 'block';
    return;
  }

  // Chart dimensions
  const containerWidth = container.clientWidth || 700;
  const legendWidth = 180;
  const width = containerWidth;
  const height = 420;
  const margin = { top: 20, right: legendWidth + 20, bottom: 50, left: 60 };
  const innerW = width - margin.left - margin.right;
  const innerH = height - margin.top - margin.bottom;

  const svg = d3.select(container)
    .append('svg')
    .attr('viewBox', `0 0 ${width} ${height}`)
    .attr('preserveAspectRatio', 'xMidYMid meet');

  const g = svg.append('g').attr('transform', `translate(${margin.left},${margin.top})`);

  const x = d3.scaleLinear().domain([0, 100]).range([0, innerW]);
  const y = d3.scaleLinear().domain([0, 100]).range([innerH, 0]);

  // Axes
  g.append('g')
    .attr('class', 'axis')
    .attr('transform', `translate(0,${innerH})`)
    .call(d3.axisBottom(x).tickValues(INTERVAL_RANGES.filter(v => v !== 95)).tickFormat(d => d + '%'));

  g.append('g')
    .attr('class', 'axis')
    .call(d3.axisLeft(y).ticks(5).tickFormat(d => d + '%'));

  g.append('text')
    .attr('class', 'axis-label')
    .attr('x', innerW / 2).attr('y', innerH + 40)
    .text('Prediction Interval (%)');

  g.append('text')
    .attr('class', 'axis-label')
    .attr('transform', 'rotate(-90)')
    .attr('x', -innerH / 2).attr('y', -45)
    .text('Coverage (%)');

  // Perfect calibration diagonal
  g.append('line')
    .attr('x1', x(0)).attr('x2', x(100))
    .attr('y1', y(0)).attr('y2', y(100))
    .attr('stroke', '#1a1a1a')
    .attr('stroke-width', 1.5)
    .attr('stroke-dasharray', '6,4')
    .attr('opacity', 0.5);

  // Separate ensemble vs individual models — draw individual first (faded), ensemble on top
  const isEnsemble = m => ENSEMBLE_MODELS.includes(m);
  const individualLines = modelLines.filter(ml => !isEnsemble(ml.model));
  const ensembleLines = modelLines.filter(ml => isEnsemble(ml.model));

  const line = d3.line()
    .x(d => x(d.interval))
    .y(d => y(d.coverage * 100));

  // Tooltip ref
  const tooltip = d3.select('#tooltip');

  // Helper to draw a model's line + dots with tooltip
  function drawModelLine(ml, isEns) {
    const color = COLOR_MAP[ml.model] || '#999';
    const opacity = isEns ? 1 : 0.3;
    const strokeW = isEns ? 2.5 : 1.5;
    const dotR = isEns ? 4 : 2.5;

    g.append('path')
      .datum(ml.points)
      .attr('d', line)
      .attr('fill', 'none')
      .attr('stroke', color)
      .attr('stroke-width', strokeW)
      .attr('opacity', opacity);

    g.selectAll(null)
      .data(ml.points)
      .join('circle')
      .attr('cx', d => x(d.interval))
      .attr('cy', d => y(d.coverage * 100))
      .attr('r', dotR)
      .attr('fill', color)
      .attr('stroke', '#fff')
      .attr('stroke-width', isEns ? 1.5 : 0.5)
      .attr('opacity', opacity);

    // Larger invisible hit-area circles for tooltips
    g.selectAll(null)
      .data(ml.points)
      .join('circle')
      .attr('cx', d => x(d.interval))
      .attr('cy', d => y(d.coverage * 100))
      .attr('r', 10)
      .attr('fill', 'transparent')
      .attr('cursor', 'pointer')
      .on('mousemove', function(event, d) {
        const covPct = (d.coverage * 100).toFixed(1);
        const diff = d.coverage * 100 - d.interval;
        const calibLabel = Math.abs(diff) < 2 ? 'Well calibrated' :
          diff > 0 ? 'Over-covered (conservative)' : 'Under-covered (overconfident)';
        const calibColor = Math.abs(diff) < 2 ? '#059669' : diff > 0 ? '#5B89B5' : '#c00';
        tooltip.classed('visible', true)
          .style('left', (event.clientX + 14) + 'px')
          .style('top', (event.clientY - 10) + 'px')
          .html(
            `<div class="tooltip-title" style="color:${color}">${ml.model}${isEns ? ' ★' : ''}</div>` +
            `<div class="tooltip-row"><span class="tooltip-label">Prediction Interval:</span> <span class="tooltip-value">${d.interval}%</span></div>` +
            `<div class="tooltip-row"><span class="tooltip-label">Actual Coverage:</span> <span class="tooltip-value">${covPct}%</span></div>` +
            `<div style="font-size:11px;color:${calibColor};margin-top:4px;">${calibLabel} (${diff > 0 ? '+' : ''}${diff.toFixed(1)}pp)</div>`
          );
      })
      .on('mouseleave', () => tooltip.classed('visible', false));
  }

  // Individual models — faded (drawn first, behind)
  individualLines.forEach(ml => drawModelLine(ml, false));

  // Ensemble models — bold and prominent (drawn on top)
  ensembleLines.forEach(ml => drawModelLine(ml, true));

  // Legend (right side) — ensemble first, then individual
  const legend = svg.append('g')
    .attr('transform', `translate(${width - legendWidth - 10},${margin.top})`);

  // Diagonal legend item
  legend.append('line')
    .attr('x1', 0).attr('x2', 16)
    .attr('y1', 7).attr('y2', 7)
    .attr('stroke', '#1a1a1a')
    .attr('stroke-width', 1.5)
    .attr('stroke-dasharray', '4,3');
  legend.append('text')
    .attr('x', 22).attr('y', 10)
    .attr('font-size', '11px').attr('fill', '#666')
    .text('Perfect Calibration');

  const orderedLines = [...ensembleLines, ...individualLines];
  orderedLines.forEach((ml, i) => {
    const yOff = 24 + i * 18;
    const color = COLOR_MAP[ml.model] || '#999';
    const ens = isEnsemble(ml.model);
    const opacity = ens ? 1 : 0.4;
    legend.append('line')
      .attr('x1', 0).attr('x2', 16)
      .attr('y1', yOff).attr('y2', yOff)
      .attr('stroke', color).attr('stroke-width', ens ? 2.5 : 1.5)
      .attr('opacity', opacity);
    legend.append('circle')
      .attr('cx', 8).attr('cy', yOff)
      .attr('r', ens ? 3.5 : 2.5).attr('fill', color)
      .attr('opacity', opacity);
    legend.append('text')
      .attr('x', 22).attr('y', yOff + 3)
      .attr('font-size', ens ? '11px' : '10px')
      .attr('font-weight', ens ? '600' : '400')
      .attr('fill', ens ? '#1a1a1a' : '#888')
      .text(ml.model.length > 22 ? ml.model.slice(0, 20) + '...' : ml.model);
  });
}
