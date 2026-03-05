# Epistorm Ensemble Dashboard — Build Instructions

## Overview

Build a **commercial-grade influenza hospitalization forecast dashboard** styled after New York Times data visualizations. This is a complete rewrite of the existing Streamlit dashboard (`epistorm-ensemble-dashboard.py`) as a static vanilla HTML/CSS/JS application using D3.js.

**Reference implementation:** [jessica-davis/flu-dashboard-traj](https://github.com/jessica-davis/flu-dashboard-traj) — follow its architecture patterns (centralized app state, D3 v7 charting, modular JS files, responsive SVGs).

---

## Tech Stack

- **No frameworks, no build step.** Vanilla HTML, CSS, JavaScript only.
- **D3.js v7** (CDN: `https://d3js.org/d3.v7.min.js`) — all charts, maps, interactions
- **TopoJSON v3** (CDN: `https://d3js.org/topojson.v3.min.js`) — US state map
- **Pre-processed JSON files** for all data (converted from existing parquet/CSV)
- Serve locally with `python -m http.server 8000` from the `dashboard/` directory

---

## File Structure

```
dashboard/
├── index.html              # Single-page app: Overview, Forecasts, Evaluation (tab navigation)
├── about.html              # About/methodology page
├── css/
│   └── style.css           # Complete NYT-inspired design system
├── js/
│   ├── main.js             # App init, data loading (Promise.all), state management, updateAll()
│   ├── colors.js           # All color constants (models, activity levels, trends, categories)
│   ├── overview.js         # Overview tab: activity chart, forecast summary bars, headline
│   ├── forecasts.js        # Forecasts tab: quantile chart with prediction intervals
│   ├── evaluation.js       # Evaluation tab: WIS boxplots, coverage calibration
│   ├── map.js              # US choropleth map (TopoJSON + D3 geo)
│   ├── controls.js         # Location selector, date slider, model toggles, range buttons
│   └── utils.js            # Shared: formatters, tooltip builder, responsive helpers
├── data/                   # Pre-processed JSON (see Data Preparation section)
│   ├── observed.json
│   ├── forecasts.json
│   ├── ensemble.json
│   ├── ensemble_lop.json
│   ├── thresholds.json
│   ├── activity_levels.json
│   ├── categorical.json
│   ├── wis_ratio.json
│   ├── coverage.json
│   ├── locations.json
│   └── us-states.json
├── assets/
│   ├── epistorm-ensemble-logo.png
│   ├── epistorm-logo.png
│   └── northeastern-logo.png
└── scripts/
    └── prepare_data.py     # Python script to convert source data → JSON
```

---

## Design System (NYT-Inspired)

### Typography
```css
/* Headlines, titles, narrative text */
font-family: Georgia, 'Times New Roman', serif;

/* UI controls, labels, axis text, tooltips */
font-family: -apple-system, BlinkMacSystemFont, 'Helvetica Neue', Arial, sans-serif;
```

- Page title: 26px, bold, #1a1a1a
- Section headlines: 20–22px, semibold, #333
- Body/narrative text: 16–18px, regular, #333
- Captions/labels: 12–13px, #666
- Micro-labels: 10–11px, #999

### Color Palette

**Neutrals (primary UI):**
- Text: `#1a1a1a` (near-black), `#333`, `#555`, `#666`, `#999`
- Backgrounds: `#ffffff` (cards), `#f7f7f7` (page background)
- Borders: `#ddd`, `#e0e0e0`, `#eee`

**Model colors** (preserve exactly from existing dashboard):
```javascript
const COLOR_MAP = {
  'Median Epistorm Ensemble': '#2B7A8F',
  'LOP Epistorm Ensemble':    '#3CAAA0',
  'FluSight-ensemble':        '#3D5A80',
  'CEPH-Rtrend_fluH':         '#E07B54',
  'MOBS-EpyStrain_Flu':       '#C94F7C',
  'MOBS-GLEAM_RL_FLUH':       '#7B61A0',
  'NU-PGF_FLUH':              '#4A9E6F',
  'NEU_ISI-FluBcast':         '#D4A843',
  'NEU_ISI-AdaptiveEnsemble': '#5B8DB8',
  'Gatech-ensemble_prob':     '#A0522D',
  'Gatech-ensemble_stat':     '#7A9E7E',
  'MIGHTE-Nsemble':           '#C06B3A',
  'MIGHTE-Joint':             '#6B8E9F',
};
```

**Activity level colors:**
```javascript
const ACTIVITY_COLORS = {
  'Low':       '#7DD4C8',
  'Moderate':  '#3CAAA0',
  'High':      '#2B7A8F',
  'Very High': '#3D5A80',
};
```

**Trend direction colors:**
```javascript
const CATEGORY_COLORS = {
  'Large Decrease': '#006d77',
  'Decrease':       '#83c5be',
  'Stable':         '#aaaaaa',
  'Increase':       '#e29578',
  'Large Increase': '#bc4749',
};
```

### Layout
- Max-width: `1200px`, centered with `margin: 0 auto`
- Cards: white background, `border: 1px solid #ddd`, `border-radius: 4px`, `box-shadow: 0 1px 4px rgba(0,0,0,0.06)`
- Spacing: 16px–24px padding inside cards, 16px–20px gaps between cards
- Responsive: flex-wrap at narrow viewports, single-column below 768px

### Interactive Elements
- Tabs: underline-style active indicator (2px bottom border, #1a1a1a)
- Buttons/toggles: bordered style, `#1a1a1a` active, `#f5f5f5` hover
- Segmented controls: vertical button groups with active state highlight (reference repo pattern)
- Transitions: 300ms ease on all data updates, color changes, layout shifts

---

## App State Management

Follow the reference repo pattern — centralized state object:

```javascript
const AppState = {
  activeTab: 'overview',        // 'overview' | 'forecasts' | 'evaluation'
  selectedLocation: 'US',       // location code
  selectedLocationName: 'United States',
  selectedDate: null,           // reference date (latest by default)
  selectedModel: 'Median Epistorm Ensemble',
  dateRange: '3months',         // '3months' | '6months' | '1year' | '2years' | 'all'
  // Evaluation filters
  evalLocation: 'all',
  evalHorizon: 'all',           // 'all' | 0 | 1 | 2 | 3
  evalModels: [],               // all selected by default
  evalRefDateRange: [null, null],
  evalTargetDateRange: [null, null],
  // Overview-specific
  overviewHorizon: 3,
  overviewRefDate: null,        // latest by default
};
```

State changes trigger `updateAll()` which re-renders affected components.

---

## Data Preparation

### Source Files (relative to project root `epistorm-ensemble/`)

| Source | Format | Dashboard JSON |
|--------|--------|---------------|
| `data/observed_data.csv` | CSV: date, location, value | `observed.json` |
| `data/all_forecasts.parquet` | Parquet: reference_date, location, horizon, target, target_end_date, output_type, output_type_id, value, model | `forecasts.json` |
| `data/quantile_ensemble.pq` | Parquet: same schema, model='Median Epistorm Ensemble' | `ensemble.json` |
| `data/quantile_ensemble_LOP.pq` | Parquet: same schema, model='LOP Epistorm Ensemble' | `ensemble_lop.json` |
| `data/categorical_ensemble.pq` | Parquet: pmf output_type, trend categories | `categorical.json` |
| `data/activity_level_ensemble.pq` | Parquet: activity level probabilities | `activity_levels.json` |
| `data/threshold_levels.csv` | CSV: location, Medium, High, Very High thresholds | `thresholds.json` |
| `data/wis_ratio_epistorm_models_2526.pq` | Parquet: Model, location, horizon, target_end_date, wis_ratio | `wis_ratio.json` |
| `data/coverage_epistorm_models_2526.pq` | Parquet: Model, location, horizon, target_end_date, {interval}_cov columns | `coverage.json` |
| `locations.csv` | CSV: location, location_name, population | `locations.json` |

### prepare_data.py Script

```python
"""Convert source data files to dashboard-ready JSON.

Run from project root:
    python dashboard/scripts/prepare_data.py
"""
import pandas as pd
import json
from pathlib import Path

OUTPUT_DIR = Path("dashboard/data")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def to_json(df, filename, orient='records', date_format='iso'):
    path = OUTPUT_DIR / filename
    df.to_json(path, orient=orient, date_format=date_format, indent=None)
    print(f"  → {path} ({len(df)} rows)")

# Observed data
obs = pd.read_csv("data/observed_data.csv")
obs['date'] = pd.to_datetime(obs['date']).dt.strftime('%Y-%m-%d')
to_json(obs, "observed.json")

# Forecasts (filter to horizon >= 0, quantile + pmf types only)
fc = pd.read_parquet("data/all_forecasts.parquet")
fc['reference_date'] = pd.to_datetime(fc['reference_date']).dt.strftime('%Y-%m-%d')
fc['target_end_date'] = pd.to_datetime(fc['target_end_date']).dt.strftime('%Y-%m-%d')
fc = fc[fc['horizon'] >= 0]
to_json(fc, "forecasts.json")

# Ensemble quantile
ens = pd.read_parquet("data/quantile_ensemble.pq")
ens['reference_date'] = pd.to_datetime(ens['reference_date']).dt.strftime('%Y-%m-%d')
ens['target_end_date'] = pd.to_datetime(ens['target_end_date']).dt.strftime('%Y-%m-%d')
ens['model'] = 'Median Epistorm Ensemble'
to_json(ens, "ensemble.json")

# LOP Ensemble
lop = pd.read_parquet("data/quantile_ensemble_LOP.pq")
lop['reference_date'] = pd.to_datetime(lop['reference_date']).dt.strftime('%Y-%m-%d')
lop['target_end_date'] = pd.to_datetime(lop['target_end_date']).dt.strftime('%Y-%m-%d')
lop['model'] = 'LOP Epistorm Ensemble'
to_json(lop, "ensemble_lop.json")

# Categorical ensemble
cat = pd.read_parquet("data/categorical_ensemble.pq")
cat['reference_date'] = pd.to_datetime(cat['reference_date']).dt.strftime('%Y-%m-%d')
to_json(cat, "categorical.json")

# Activity level ensemble
act = pd.read_parquet("data/activity_level_ensemble.pq")
act['reference_date'] = pd.to_datetime(act['reference_date']).dt.strftime('%Y-%m-%d')
to_json(act, "activity_levels.json")

# Thresholds
thresh = pd.read_csv("data/threshold_levels.csv")
to_json(thresh, "thresholds.json")

# WIS ratio
wis = pd.read_parquet("data/wis_ratio_epistorm_models_2526.pq")
wis['target_end_date'] = pd.to_datetime(wis['target_end_date']).dt.strftime('%Y-%m-%d')
wis['location'] = wis['location'].astype(str).str.zfill(2)
to_json(wis, "wis_ratio.json")

# Coverage
cov = pd.read_parquet("data/coverage_epistorm_models_2526.pq")
cov['target_end_date'] = pd.to_datetime(cov['target_end_date']).dt.strftime('%Y-%m-%d')
cov['location'] = cov['location'].astype(str).str.zfill(2)
to_json(cov, "coverage.json")

# Locations
loc = pd.read_csv("locations.csv")
to_json(loc, "locations.json")

print("Done.")
```

For the US states TopoJSON, download from: `https://cdn.jsdelivr.net/npm/us-atlas@3/states-10m.json` and save as `dashboard/data/us-states.json`.

---

## Page Structure (index.html)

### HTML Skeleton

```
<body>
  <!-- HEADER -->
  <header>
    <div class="container">
      <img src="assets/epistorm-ensemble-logo.png" alt="Epistorm Ensemble" width="250">
      <nav><!-- About link --></nav>
    </div>
  </header>

  <!-- TAB BAR -->
  <div class="tab-bar container">
    <button class="tab active" data-tab="overview">Overview</button>
    <button class="tab" data-tab="forecasts">Forecasts</button>
    <button class="tab" data-tab="evaluation">Evaluation</button>
  </div>

  <!-- TAB CONTENT -->
  <main class="container">
    <section id="tab-overview" class="tab-content active">...</section>
    <section id="tab-forecasts" class="tab-content">...</section>
    <section id="tab-evaluation" class="tab-content">...</section>
  </main>

  <!-- FOOTER -->
  <footer class="container">
    <img src="assets/northeastern-logo.png" width="150">
    <img src="assets/epistorm-logo.png" width="150">
  </footer>

  <!-- Scripts -->
  <script src="https://d3js.org/d3.v7.min.js"></script>
  <script src="https://d3js.org/topojson.v3.min.js"></script>
  <script src="js/colors.js"></script>
  <script src="js/utils.js"></script>
  <script src="js/controls.js"></script>
  <script src="js/map.js"></script>
  <script src="js/overview.js"></script>
  <script src="js/forecasts.js"></script>
  <script src="js/evaluation.js"></script>
  <script src="js/main.js"></script>
</body>
```

---

## Feature Specifications

### Tab 1: Overview

**Layout:** Full-width headline card → two-column (forecast summary + explainer)

#### 1.1 Activity Level Headline
- Large narrative text: "The flu activity level in {location} is currently **{level}** as of {date}."
- Year-over-year comparison sentence below (same epiweek last year)
- Use serif font, 22px for headline, 16px for comparison

#### 1.2 Observed Hospitalizations Chart
- D3 line chart with activity level threshold bands (horizontal colored rectangles behind the data)
- X-axis: dates, Y-axis: weekly flu hospitalizations
- Line: black, 2px stroke, circle-open markers
- Threshold bands use `ACTIVITY_COLORS` with 0.7 opacity, layered below the line
- Date range buttons below: "Last 3 months", "Last 6 months", "Last year", "Last 2 years", "All data"
- Custom legend row for activity level colors

#### 1.3 Forecast Summary Card (left column)
- Headline: "Forecast summary · **{activity level}** activity, trending **{trend direction}**"
- Narrative sentence: "Over the next {N} weeks, flu hospitalizations in {location} are projected to {trend} — activity levels are forecast to stay {level} through {date}."
- Controls: Forecast Date dropdown, Forecast Horizon dropdown (1–4 weeks ahead)
- Two horizontal bar charts side by side:
  - **Activity Level Predictions:** horizontal bars for Low/Moderate/High/Very High with probabilities
  - **Trend Direction:** horizontal bars for Large Decrease/Decrease/Stable/Increase/Large Increase
- Winner category at full opacity, others at 35% opacity
- Percentage labels outside bars

#### 1.4 Location Selector
- Dropdown at top: "United States" + all state names alphabetically
- Also consider adding a US choropleth map that's clickable to select state

#### 1.5 US Map (from reference repo)
- D3 choropleth using TopoJSON
- Color states by current activity level or most-likely forecast activity level
- Click state → updates `AppState.selectedLocation`, re-renders all components
- Tooltip on hover showing state name + value

### Tab 2: Forecasts

**Layout:** Left sidebar (controls, ~25% width) + main chart area (~75%)

#### 2.1 Controls Sidebar
- **Location selector:** scrollable radio-button list (US + all states), inside a bordered card with scroll
- Highlight currently selected location

#### 2.2 Main Forecast Chart
- **Headline:** "Flu Hospitalization Forecasts — {location name}" (serif, colored #518fb0)
- **Forecast summary text:** "As of {date}, the {model} forecasts **{median}** influenza hospital admissions (**{per_capita}** per 100,000 people) by {target date} for {location}."
- "95% prediction interval: **{lower}–{upper}** hospital admissions (**{per_capita_lower}–{per_capita_upper}** per 100,000 people)."

#### 2.3 D3 Quantile Forecast Line Chart
- **Observed data:** black line (2px), solid circle markers for in-sample, open circle markers for out-of-sample (after reference date)
- **Forecast bands:**
  - 95% PI: `fill` between 0.025 and 0.975 quantiles, model color at 20% opacity
  - 50% PI: `fill` between 0.25 and 0.75 quantiles, model color at 40% opacity
  - Median: solid line + small circle markers, model color, 2px
- **Reference date line:** red dashed vertical line with annotation "Forecast Date: YYYY-MM-DD"
- **Axes:** x = date, y = "Weekly Incident Flu Hospitalizations"
- **Hover:** unified x-hover showing all values at that date

#### 2.4 Below-Chart Controls
- **Date slider:** select forecast reference date from available dates, styled with model color (#2B7A8F)
- **Model toggle:** two buttons ("Median Epistorm Ensemble" / "LOP Epistorm Ensemble"), primary/secondary styling
- **Historical Data Range:** button group (Last 3 Months, Last 6 Months, Last year, Last 2 Years, All data)

#### 2.5 Info Cards (below controls)
- "What are Prediction Intervals?" — explanatory text in bordered card
- "How are the ensemble models created?" — explanatory text in bordered card

### Tab 3: Evaluation

**Layout:** Left sidebar (filters, ~25% width) + main chart area (~75%)

#### 3.1 Filter Sidebar
- Location dropdown (All Locations + states)
- Horizon dropdown (All Horizons, 0, 1, 2, 3)
- Date filters (reference date range, target end date range) — collapsible
- Model checkboxes grouped: Ensemble models first, then Individual models — collapsible
- Model count caption at bottom

#### 3.2 WIS Ratio Boxplot
- **Title:** "Weighted Interval Score (WIS) Ratio"
- **Description:** "The WIS ratio compares each model's Weighted Interval Score to the FluSight-baseline model. A ratio less than 1 indicates better performance than baseline."
- D3 horizontal boxplot (one box per model)
- Models sorted by median WIS ratio (ascending — best first)
- Each box colored by `COLOR_MAP`
- Red dashed vertical line at x=1 with annotation "WIS ratio = 1"
- X-axis limited to whisker range (1.5×IQR), hide outlier markers
- Y-axis: model names (reversed order for top-down reading)

#### 3.3 Coverage Calibration Chart
- **Title:** "Prediction Interval Coverage"
- **Description:** "Coverage measures the proportion of observed values that fall within each prediction interval. A well-calibrated model should have coverage close to the diagonal."
- D3 line chart
- Black dashed diagonal = perfect calibration
- One line per model (colored by `COLOR_MAP`), with circle markers
- X-axis: Prediction Interval % (10, 20, 30, 40, 50, 60, 70, 80, 90, 95, 98)
- Y-axis: Coverage (0% – 100%)
- Legend to the right of chart

---

## Data Constants

```javascript
const MODELS = [
  'MIGHTE-Nsemble', 'MIGHTE-Joint', 'CEPH-Rtrend_fluH',
  'MOBS-EpyStrain_Flu', 'MOBS-GLEAM_RL_FLUH', 'NU-PGF_FLUH',
  'NEU_ISI-FluBcast', 'NEU_ISI-AdaptiveEnsemble',
  'Gatech-ensemble_prob', 'Gatech-ensemble_stat'
];

const QUANTILES = [
  '0.01', '0.025', '0.05', '0.1', '0.15', '0.2', '0.25', '0.3', '0.35',
  '0.4', '0.45', '0.5', '0.55', '0.6', '0.65', '0.7', '0.75', '0.8',
  '0.85', '0.9', '0.95', '0.975', '0.99'
];

const CATEGORY_ORDER = ['Large Decrease', 'Decrease', 'Stable', 'Increase', 'Large Increase'];
const ACTIVITY_ORDER = ['Low', 'Moderate', 'High', 'Very High'];
const INTERVAL_RANGES = [10, 20, 30, 40, 50, 60, 70, 80, 90, 95, 98];
```

---

## D3 Patterns to Follow

### Responsive SVGs
```javascript
const svg = d3.select("#chart-container")
  .append("svg")
  .attr("viewBox", `0 0 ${width} ${height}`)
  .attr("preserveAspectRatio", "xMidYMid meet");
```

### Layered Groups (z-ordering)
```javascript
const g = svg.append("g").attr("transform", `translate(${margin.left},${margin.top})`);
const bgLayer = g.append("g").attr("class", "bg-layer");      // threshold bands
const dataLayer = g.append("g").attr("class", "data-layer");   // lines, areas
const axisLayer = g.append("g").attr("class", "axis-layer");   // axes
const annotLayer = g.append("g").attr("class", "annot-layer"); // annotations
```

### Scales
```javascript
const x = d3.scaleTime().domain(d3.extent(data, d => d.date)).range([0, innerWidth]);
const y = d3.scaleLinear().domain([0, yMax * 1.05]).range([innerHeight, 0]);
```

### Area fills for prediction intervals
```javascript
const area = d3.area()
  .x(d => x(d.date))
  .y0(d => y(d.lower))
  .y1(d => y(d.upper))
  .defined(d => d.lower != null && d.upper != null);

dataLayer.append("path")
  .datum(bandData)
  .attr("d", area)
  .attr("fill", hexToRgba(color, 0.2))
  .attr("stroke", "none");
```

### Transitions
```javascript
selection.transition().duration(300).ease(d3.easeCubicOut)
  .attr("d", updatedPath);
```

### Tooltips
Build a shared tooltip `<div>` positioned absolutely, shown on mouseover:
```javascript
const tooltip = d3.select("body").append("div")
  .attr("class", "tooltip")
  .style("opacity", 0);
```

---

## UX Requirements

1. **Loading state:** Show a subtle spinner or "Loading data..." message while JSON files load
2. **Smooth transitions:** All chart updates animate over 300ms with `d3.easeCubicOut`
3. **Responsive:** Charts resize on window resize (use `viewBox` + `preserveAspectRatio`)
4. **Mobile-friendly:** Stack columns vertically below 768px, enlarge touch targets to 44px minimum
5. **Accessible:** All interactive elements keyboard-focusable, ARIA labels on controls, sufficient color contrast
6. **No-data states:** Show friendly messages when filters return empty data ("No data available for selected filters")
7. **Number formatting:** Use `Intl.NumberFormat` for comma-separated numbers, `.1%` for percentages
8. **Date formatting:** Display dates as "Month DD, YYYY" in narrative text, "YYYY-MM-DD" in controls
9. **URL hash state:** Persist tab + location in URL hash for shareable links (e.g., `#forecasts/06` for California forecasts)
10. **Print-friendly:** Hide interactive controls in `@media print`, show charts at full width

---

## Utility Functions (utils.js)

```javascript
// Convert hex color to rgba string
function hexToRgba(hex, alpha) {
  const h = hex.replace('#', '');
  const r = parseInt(h.substring(0, 2), 16);
  const g = parseInt(h.substring(2, 4), 16);
  const b = parseInt(h.substring(4, 6), 16);
  return `rgba(${r},${g},${b},${alpha})`;
}

// Format category string
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

// Get location name from code
function getLocationName(code, locations) {
  if (code === 'US') return 'United States';
  const loc = locations.find(l => l.location === code);
  return loc ? loc.location_name : code;
}

// Get activity level from value + thresholds
function getActivityLevel(value, thresholds) {
  if (value >= thresholds['Very High']) return 'Very High';
  if (value >= thresholds['High']) return 'High';
  if (value >= thresholds['Medium']) return 'Moderate';
  return 'Low';
}

// Format number with commas
function fmtNum(n) { return new Intl.NumberFormat().format(Math.round(n)); }

// Format percentage
function fmtPct(n) { return (n * 100).toFixed(0) + '%'; }
```

---

## Data Loading Pattern (main.js)

```javascript
async function init() {
  try {
    const [observed, forecasts, ensemble, ensembleLop, categorical,
           activityLevels, thresholds, wisRatio, coverage, locations, usStates] =
      await Promise.all([
        d3.json('data/observed.json'),
        d3.json('data/forecasts.json'),
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

    // Parse dates
    observed.forEach(d => d.date = new Date(d.date));
    // ... parse other date fields ...

    // Combine ensemble data into forecasts
    const allForecasts = [...forecasts, ...ensemble, ...ensembleLop];

    // Store in global data object
    window.DATA = { observed, forecasts: allForecasts, categorical, activityLevels,
                    thresholds, wisRatio, coverage, locations, usStates };

    // Set initial state
    const dates = [...new Set(allForecasts.map(d => d.reference_date))].sort();
    AppState.selectedDate = dates[dates.length - 1];
    AppState.overviewRefDate = dates[dates.length - 1];

    // Initialize components
    initControls();
    initMap();
    initOverview();
    initForecasts();
    initEvaluation();

    updateAll();
  } catch (err) {
    document.getElementById('loading').textContent =
      'Error loading data. Make sure to serve with a local web server.';
    console.error(err);
  }
}

function updateAll() {
  updateOverview();
  updateForecasts();
  updateEvaluation();
  updateMap();
}

document.addEventListener('DOMContentLoaded', init);
```

---

## Build Order

When implementing, build in this order:

1. **`prepare_data.py`** — run it, verify JSON files are created and valid
2. **`index.html`** — page skeleton with all div containers, tab structure
3. **`css/style.css`** — complete design system (can iterate, but get foundations in place)
4. **`js/colors.js`** — all color constants
5. **`js/utils.js`** — helper functions
6. **`js/main.js`** — data loading, state management, tab switching
7. **`js/controls.js`** — location selector, date controls
8. **`js/overview.js`** — overview tab (start here for visual validation)
9. **`js/forecasts.js`** — forecast chart (most complex visualization)
10. **`js/evaluation.js`** — WIS + coverage charts
11. **`js/map.js`** — US choropleth (enhancement)
12. **`about.html`** — static about page

---

## Testing / Verification

1. Run `python dashboard/scripts/prepare_data.py` from project root
2. `cd dashboard && python -m http.server 8000`
3. Open `http://localhost:8000` in browser
4. Verify:
   - All three tabs render and switch correctly
   - Overview: activity headline, hospitalizations chart with colored bands, forecast summary bars
   - Forecasts: observed + forecast lines with shaded PI bands, date slider works, model toggle works
   - Evaluation: WIS boxplots render, coverage chart renders, filters update charts
   - Location changes propagate across all tabs
   - Charts are responsive (resize browser window)
   - No console errors
