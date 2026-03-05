/* map.js — US choropleth map (TopoJSON + D3 geo) */

let mapSvg = null;
let mapPath = null;
let mapStates = null;

function initMap() {
  const container = document.getElementById('overview-map-container');
  if (!container) return;

  const width = 500;
  const height = 320;

  mapSvg = d3.select(container)
    .append('svg')
    .attr('viewBox', `0 0 ${width} ${height}`)
    .attr('preserveAspectRatio', 'xMidYMid meet');

  const projection = d3.geoAlbersUsa()
    .fitSize([width - 20, height - 20], topojson.feature(DATA.usStates, DATA.usStates.objects.states));

  mapPath = d3.geoPath().projection(projection);

  const states = topojson.feature(DATA.usStates, DATA.usStates.objects.states).features;

  // Build FIPS → location mapping
  const fipsToLoc = {};
  DATA.locations.forEach(l => {
    // TopoJSON uses numeric string IDs; locations.csv uses zero-padded 2-digit
    fipsToLoc[String(+l.location)] = l.location;
    fipsToLoc[l.location] = l.location;
  });

  mapStates = mapSvg.selectAll('.state')
    .data(states)
    .join('path')
    .attr('class', 'state')
    .attr('d', mapPath)
    .attr('fill', '#ddd')
    .attr('stroke', '#fff')
    .attr('stroke-width', 0.8)
    .style('cursor', 'pointer')
    .on('click', function(event, d) {
      const locCode = fipsToLoc[d.id];
      if (locCode) {
        setLocation(locCode);
      }
    })
    .on('mouseenter', function(event, d) {
      d3.select(this).attr('stroke', '#333').attr('stroke-width', 1.5);
      const locCode = fipsToLoc[d.id];
      const locName = locCode ? getLocationName(locCode, DATA.locations) : d.properties.name;
      const tooltip = d3.select('#tooltip');
      tooltip.classed('visible', true)
        .style('left', (event.clientX + 12) + 'px')
        .style('top', (event.clientY - 10) + 'px')
        .html(`<div class="tooltip-title">${locName}</div>`);
    })
    .on('mousemove', function(event) {
      d3.select('#tooltip')
        .style('left', (event.clientX + 12) + 'px')
        .style('top', (event.clientY - 10) + 'px');
    })
    .on('mouseleave', function() {
      d3.select(this).attr('stroke', '#fff').attr('stroke-width', 0.8);
      d3.select('#tooltip').classed('visible', false);
    });

  // State borders
  mapSvg.append('path')
    .datum(topojson.mesh(DATA.usStates, DATA.usStates.objects.states, (a, b) => a !== b))
    .attr('fill', 'none')
    .attr('stroke', '#fff')
    .attr('stroke-width', 0.8)
    .attr('d', mapPath)
    .style('pointer-events', 'none');
}

function updateMap() {
  if (!mapStates) return;

  // Build FIPS → location mapping
  const fipsToLoc = {};
  DATA.locations.forEach(l => {
    fipsToLoc[String(+l.location)] = l.location;
    fipsToLoc[l.location] = l.location;
  });

  // Get latest observed value per location
  const latestByLoc = {};
  DATA.observed.forEach(d => {
    if (!latestByLoc[d.location] || d.date > latestByLoc[d.location].date) {
      latestByLoc[d.location] = d;
    }
  });

  mapStates
    .transition().duration(300)
    .attr('fill', function(d) {
      const locCode = fipsToLoc[d.id];
      if (!locCode) return '#eee';
      const obs = latestByLoc[locCode];
      if (!obs || obs.value == null) return '#eee';
      const thresh = getThresholdsForLocation(locCode, DATA.thresholds);
      if (!thresh) return '#eee';
      const level = getActivityLevel(obs.value, thresh);
      return ACTIVITY_COLORS[level] || '#eee';
    });

  // Highlight selected state
  mapStates
    .attr('stroke-width', function(d) {
      const locCode = fipsToLoc[d.id];
      return locCode === AppState.selectedLocation ? 2.5 : 0.8;
    })
    .attr('stroke', function(d) {
      const locCode = fipsToLoc[d.id];
      return locCode === AppState.selectedLocation ? '#1a1a1a' : '#fff';
    });

  // Update tooltip with value info on hover
  mapStates
    .on('mouseenter', function(event, d) {
      d3.select(this).attr('stroke', '#333').attr('stroke-width', 2);
      const locCode = fipsToLoc[d.id];
      const locName = locCode ? getLocationName(locCode, DATA.locations) : d.properties.name;
      const obs = latestByLoc[locCode];
      const thresh = locCode ? getThresholdsForLocation(locCode, DATA.thresholds) : null;
      const level = obs && thresh ? getActivityLevel(obs.value, thresh) : null;

      let html = `<div class="tooltip-title">${locName}</div>`;
      if (obs && obs.value != null) {
        html += `<div class="tooltip-row"><span class="tooltip-label">Admissions:</span> <span class="tooltip-value">${fmtNum(obs.value)}</span></div>`;
      }
      if (level) {
        html += `<div class="tooltip-row"><span class="tooltip-label">Activity:</span> <span class="tooltip-value" style="color:${ACTIVITY_COLORS[level]}">${level}</span></div>`;
      }

      d3.select('#tooltip')
        .classed('visible', true)
        .style('left', (event.clientX + 12) + 'px')
        .style('top', (event.clientY - 10) + 'px')
        .html(html);
    });
}
