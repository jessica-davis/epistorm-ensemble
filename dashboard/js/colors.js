/* colors.js — All color constants and data arrays */

const COLOR_MAP = {
  // Ensemble models — muted blue/green/teal family
  'Median Epistorm Ensemble': '#4A9E8E',
  'LOP Epistorm Ensemble':    '#5B89B5',
  'FluSight-ensemble':        '#8D88BF',
  // Individual models — muted, soft-toned palette
  'CEPH-Rtrend_fluH':         '#C4946B',
  'MOBS-EpyStrain_Flu':       '#C47A8E',
  'MOBS-GLEAM_RL_FLUH':       '#7A9B6B',
  'NU-PGF_FLUH':              '#6BA88D',
  'NEU_ISI-FluBcast':         '#B5A45A',
  'NEU_ISI-AdaptiveEnsemble': '#CB7D7D',
  'Gatech-ensemble_prob':     '#A68B6B',
  'Gatech-ensemble_stat':     '#6AA3AD',
  'MIGHTE-Nsemble':           '#C49060',
  'MIGHTE-Joint':             '#8B9EAA',
};

const ACTIVITY_COLORS = {
  'Low':       '#7DD4C8',
  'Moderate':  '#3CAAA0',
  'High':      '#2B7A8F',
  'Very High': '#3D5A80',
};

const CATEGORY_COLORS = {
  'Large Decrease': '#006d77',
  'Decrease':       '#83c5be',
  'Stable':         '#aaaaaa',
  'Increase':       '#e29578',
  'Large Increase': '#bc4749',
};

const MODELS = [
  'MIGHTE-Nsemble', 'MIGHTE-Joint', 'CEPH-Rtrend_fluH',
  'MOBS-EpyStrain_Flu', 'MOBS-GLEAM_RL_FLUH', 'NU-PGF_FLUH',
  'NEU_ISI-FluBcast', 'NEU_ISI-AdaptiveEnsemble',
  'Gatech-ensemble_prob', 'Gatech-ensemble_stat',
];

const ENSEMBLE_MODELS = [
  'Median Epistorm Ensemble', 'LOP Epistorm Ensemble', 'FluSight-ensemble',
];

const ALL_MODELS = [...ENSEMBLE_MODELS, ...MODELS];

const QUANTILES = [
  '0.01', '0.025', '0.05', '0.1', '0.15', '0.2', '0.25', '0.3', '0.35',
  '0.4', '0.45', '0.5', '0.55', '0.6', '0.65', '0.7', '0.75', '0.8',
  '0.85', '0.9', '0.95', '0.975', '0.99',
];

const CATEGORY_ORDER = ['Large Decrease', 'Decrease', 'Stable', 'Increase', 'Large Increase'];
const ACTIVITY_ORDER = ['Low', 'Moderate', 'High', 'Very High'];
const INTERVAL_RANGES = [10, 20, 30, 40, 50, 60, 70, 80, 90, 95, 98];
