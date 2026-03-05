/* utils.js — Shared helper functions */

function hexToRgba(hex, alpha) {
  const h = hex.replace('#', '');
  const r = parseInt(h.substring(0, 2), 16);
  const g = parseInt(h.substring(2, 4), 16);
  const b = parseInt(h.substring(4, 6), 16);
  return `rgba(${r},${g},${b},${alpha})`;
}

function formatCategory(cat) {
  const map = {
    'large_decrease': 'Large Decrease',
    'decrease': 'Decrease',
    'stable': 'Stable',
    'increase': 'Increase',
    'large_increase': 'Large Increase',
  };
  return map[cat.toLowerCase().trim()] || cat;
}

function getLocationName(code, locations) {
  if (code === 'US') return 'United States';
  const loc = locations.find(l => l.location === code);
  return loc ? loc.location_name : code;
}

function getActivityLevel(value, thresholds) {
  if (value >= thresholds['Very High']) return 'Very High';
  if (value >= thresholds['High']) return 'High';
  if (value >= thresholds['Medium']) return 'Moderate';
  return 'Low';
}

function fmtNum(n) {
  if (n == null || isNaN(n)) return '—';
  return new Intl.NumberFormat().format(Math.round(n));
}

function fmtPct(n) {
  if (n == null || isNaN(n)) return '—';
  return (n * 100).toFixed(0) + '%';
}

function fmtDate(d) {
  if (!d) return '—';
  const date = d instanceof Date ? d : new Date(d);
  return date.toLocaleDateString('en-US', { month: 'long', day: 'numeric', year: 'numeric' });
}

function parseDate(str) {
  return new Date(str + 'T00:00:00');
}

/** Filter dates array to those within the range string relative to maxDate */
function getDateRangeCutoff(maxDate, range) {
  const d = new Date(maxDate);
  switch (range) {
    case '3months':  d.setMonth(d.getMonth() - 3); break;
    case '6months':  d.setMonth(d.getMonth() - 6); break;
    case '1year':    d.setFullYear(d.getFullYear() - 1); break;
    case '2years':   d.setFullYear(d.getFullYear() - 2); break;
    case 'all':      return null; // no cutoff — caller should use data min
    default:         d.setMonth(d.getMonth() - 3);
  }
  return d;
}

/** Get population for a location code */
function getPopulation(code, locations) {
  const loc = locations.find(l => l.location === code);
  return loc ? loc.population : null;
}

/** Compute per-capita rate per 100k */
function perCapita(value, population) {
  if (!population || !value) return null;
  return (value / population) * 100000;
}

/** Get thresholds for a specific location */
function getThresholdsForLocation(locationCode, thresholds) {
  return thresholds.find(t => t.location === locationCode) || null;
}
