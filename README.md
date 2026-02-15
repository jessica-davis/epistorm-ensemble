# Epistorm Ensemble

Interactive dashboard for creating and visualizing influenza forecast ensembles from the [Epistorm](https://www.epistorm.org/) consortium's contributing models, built with Streamlit.

**Live dashboard:** [epistorm-ensemble-forecasts.streamlit.app](https://epistorm-ensemble-forecasts.streamlit.app/)

## Overview

This project combines forecasts from multiple influenza models participating in the CDC's [FluSight](https://www.cdc.gov/fluview/php/about/forecasting.html) initiative into a unified ensemble forecast. The dashboard provides three views:

- **Overview** - National and state-level summary of current flu activity and ensemble predictions
- **Forecasts** - Detailed quantile and categorical forecasts by location, with individual model comparisons
- **Evaluation** - Model performance metrics including WIS (Weighted Interval Score) ratios and prediction interval coverage

### Contributing Models

MIGHTE-Nsemble, MIGHTE-Joint, CEPH-Rtrend_fluH, MOBS-EpyStrain_Flu, MOBS-GLEAM_RL_FLUH, NU-PGF_FLUH, NEU_ISI-FluBcast, NEU_ISI-AdaptiveEnsemble, Gatech-ensemble_prob, Gatech-ensemble_stat

## Repository Structure

```
epistorm-ensemble/
├── epistorm-ensemble-dashboard.py   # Main Streamlit app
├── src/                             # Pipeline & library code
│   ├── ensemble.py                  #   Ensemble creation methods
│   ├── fetch_data.py                #   Fetches forecasts from FluSight hub
│   ├── create_ensemble_forecasts.py #   Generates ensemble outputs
│   └── calculate_scores.py          #   Computes WIS scores & coverage
├── data/                            # Processed data files (updated weekly)
├── assets/                          # Logo images
├── notebooks/                       # Development/analysis notebooks
├── .github/workflows/               # Automated weekly data pipeline
├── .streamlit/config.toml           # Streamlit theme config
└── requirements.txt                 # Python dependencies
```

## Data Pipeline

A GitHub Actions workflow runs weekly (Thursdays) to update the forecast data:

1. **Fetch** (`src/fetch_data.py`) - Downloads the latest model forecasts and observed data from the [FluSight-forecast-hub](https://github.com/cdcepi/FluSight-forecast-hub)
2. **Ensemble** (`src/create_ensemble_forecasts.py`) - Creates quantile, categorical, and activity-level ensemble forecasts using methods in `src/ensemble.py`
3. **Score** (`src/calculate_scores.py`) - Calculates WIS ratios and prediction interval coverage for all models
4. **Deploy** - Commits updated data files to the repo; Streamlit Cloud auto-deploys

## Local Development

### Setup

```bash
pip install -r requirements.txt
```

### Run the dashboard

```bash
streamlit run epistorm-ensemble-dashboard.py
```

### Run the data pipeline manually

```bash
python src/fetch_data.py
python src/create_ensemble_forecasts.py
python src/calculate_scores.py
```

Note: `create_ensemble_forecasts.py` requires a `COVIDCAST_API_KEY` environment variable for accessing the Delphi Epidata API.
