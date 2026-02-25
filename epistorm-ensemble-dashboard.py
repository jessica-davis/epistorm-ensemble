import streamlit as st
import pandas as pd
from epiweeks import Week
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import requests
from io import StringIO
import numpy as np
import os
import base64
from pathlib import Path
from ensemble import create_ensemble_method1, create_ensemble_method2, create_categorical_ensemble, create_categorical_ensemble_quantile

# Page config
#st.set_page_config(page_title="Epistorm Influenza Forecasts", layout="wide")


import plotly.io as pio

def img_to_html(path, width=150):
    with open(path, "rb") as f:
        data = base64.b64encode(f.read()).decode()
    return f'<img src="data:image/png;base64,{data}" width="{width}">'


# Set Plotly to always use light theme
pio.templates.default = "plotly_white"

st.set_page_config(
    page_title="Epistorm Ensemble Dashboard",
    layout="wide",
    initial_sidebar_state="collapsed",
    menu_items={
        'Get Help': None,
        'Report a bug': None,
        'About': None
    }
)


st.markdown(
    """
    <style>
        [data-testid="stSidebar"] { display: none; }
        [data-testid="collapsedControl"] { display: none; }
        .stTabs [data-baseweb="tab-list"] button [data-testid="stMarkdownContainer"] p {
            font-size: 1.2rem;
            font-weight: 600;}
        .block-container {
    padding-top: 3rem;
}
    [data-testid="stContainer"] .stMarkdown, 
    [data-testid="stContainer"] .stPlotlyChart {
    margin-bottom: -15px;
}
    </style>
    """,
    unsafe_allow_html=True,
)


# Constants
MODELS = [
    'MIGHTE-Nsemble', 
    'MIGHTE-Joint', 
    'CEPH-Rtrend_fluH', 
    'MOBS-EpyStrain_Flu', 
    'MOBS-GLEAM_RL_FLUH', 
    'NU-PGF_FLUH', 
    'NEU_ISI-FluBcast', 
    'NEU_ISI-AdaptiveEnsemble',
    'Gatech-ensemble_prob',
    'Gatech-ensemble_stat'
]

QUANTILES = ['0.01', '0.025', '0.05', '0.1', '0.15', '0.2', '0.25', '0.3', '0.35', 
             '0.4', '0.45', '0.5', '0.55', '0.6', '0.65', '0.7', '0.75', '0.8', 
             '0.85', '0.9', '0.95', '0.975', '0.99']

COLOR_MAP = {
    'Median Epistorm Ensemble': '#2B7A8F',   # teal blue — your primary ensemble
    'LOP Epistorm Ensemble':    '#3CAAA0',   # lighter teal — secondary ensemble
    'FluSight-ensemble':        '#3D5A80',   # navy
    'CEPH-Rtrend_fluH':         '#E07B54',   # warm orange
    'MOBS-EpyStrain_Flu':       '#C94F7C',   # raspberry
    'MOBS-GLEAM_RL_FLUH':       '#7B61A0',   # purple
    'NU-PGF_FLUH':              '#4A9E6F',   # green
    'NEU_ISI-FluBcast':         '#D4A843',   # amber
    'NEU_ISI-AdaptiveEnsemble': '#5B8DB8',   # steel blue
    'Gatech-ensemble_prob':     '#A0522D',   # sienna
    'Gatech-ensemble_stat':     '#7A9E7E',   # sage green
    'MIGHTE-Nsemble':           '#C06B3A',   # burnt orange
    'MIGHTE-Joint':             '#6B8E9F',   # slate
}

CATEGORY_COLORS = {
    'Large Decrease': '#006d77',
    'Decrease': '#83c5be',
    'Stable': '#e5e5e5',
    'Increase': '#e29578',
    'Large Increase': '#bc4749'
}

CATEGORY_ORDER = ['Large Decrease', 'Decrease', 'Stable', 'Increase', 'Large Increase']

INTERVAL_RANGES = [10, 20, 30, 40, 50, 60, 70, 80, 90, 95, 98]

DATA_DIR = Path("data")
DATA_DIR.mkdir(exist_ok=True)


@st.cache_data(ttl=1)
def load_locations():
    """Load locations from local file or fallback to original source"""
    try:
        locations_df = pd.read_csv('locations.csv')
        return locations_df
    except Exception as e:
        st.error(f"Error loading locations: {e}")
        return None

@st.cache_data(ttl=1)
def load_observed_data():
    """Load observed data from local cache or fetch if not available"""
    local_file = DATA_DIR / "observed_data.csv"
    
    # Try to load from local file first
    if local_file.exists():
        try:
            data = pd.read_csv(local_file)
            data['date'] = pd.to_datetime(data['date'])
            return data
        except Exception as e:
            st.warning(f"Error loading cached data: {e}. Fetching fresh data...")
    
    # Fallback to fetching from URL
    url = "https://raw.githubusercontent.com/cdcepi/FluSight-forecast-hub/refs/heads/main/target-data/target-hospital-admissions.csv"
    try:
        response = requests.get(url)
        data = pd.read_csv(StringIO(response.text))
        data['date'] = pd.to_datetime(data['date'])
        return data
    except Exception as e:
        st.error(f"Error loading observed data: {e}")
        return None


@st.cache_data(ttl=1)
def load_thresholds():
    """Load observed data from local cache or fetch if not available"""
    local_file = DATA_DIR / "threshold_levels.csv"
    
    # Try to load from local file first
    if local_file.exists():
        try:
            data = pd.read_csv(local_file)
            return data
        except Exception as e:
            st.warning(f"Error loading cached data: {e}. Fetching fresh data...")
    


@st.cache_data(ttl=1)
def load_all_forecasts():
    """Load all forecasts from local cache or fetch if not available"""
    local_file = DATA_DIR / "all_forecasts.parquet"
    
    # Try to load from local file first
    if local_file.exists():
        try:
            combined = pd.read_parquet(local_file)
            combined['reference_date'] = pd.to_datetime(combined['reference_date'])
            combined['target_end_date'] = pd.to_datetime(combined['target_end_date'])
            #st.info(f"Loaded data (last updated: {pd.Timestamp(local_file.stat().st_mtime, unit='s').strftime('%Y-%m-%d')})")
            return combined
        except Exception as e:
            st.warning(f"Error loading cached forecasts: {e}. Fetching fresh data...")
    
    # Fallback to fetching from URLs
    all_data = []
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    for idx, model in enumerate(MODELS):
        status_text.text(f"Loading {model}...")
        df = load_model_forecasts(model)
        if not df.empty:
            all_data.append(df)
        progress_bar.progress((idx + 1) / len(MODELS))
    
    status_text.empty()
    progress_bar.empty()
    
    if all_data:
        combined = pd.concat(all_data, ignore_index=True)
        combined['reference_date'] = pd.to_datetime(combined['reference_date'])
        combined['target_end_date'] = pd.to_datetime(combined['target_end_date'])
        return combined
    return pd.DataFrame()

@st.cache_data(ttl=1)
def load_baseline_forecasts():
    """Load baseline forecasts from local cache or fetch if not available"""
    local_file = DATA_DIR / "baseline_forecasts.parquet"
    
    # Try to load from local file first
    if local_file.exists():
        try:
            df = pd.read_parquet(local_file)
            df['reference_date'] = pd.to_datetime(df['reference_date'])
            df['target_end_date'] = pd.to_datetime(df['target_end_date'])
            return df
        except Exception as e:
            st.warning(f"Error loading cached baseline: {e}. Fetching fresh data...")
    
    # Fallback to fetching
    baseline_model = 'FluSight-baseline'
    df = load_model_forecasts(baseline_model)
    if not df.empty:
        df['reference_date'] = pd.to_datetime(df['reference_date'])
        df['target_end_date'] = pd.to_datetime(df['target_end_date'])
    return df



@st.cache_data(ttl=1)
def load_model_forecasts(model_name):
    base_url = f"https://raw.githubusercontent.com/cdcepi/FluSight-forecast-hub/main/model-output/{model_name}"
    all_forecasts = []
    start_date = datetime(2025, 11, 1)
    #end_date = datetime.now()
    end_date = pd.to_datetime(Week.fromdate(datetime.now()).enddate())
    current_date = start_date
    while current_date <= end_date:
        date_str = current_date.strftime('%Y-%m-%d')
        file_url = f"{base_url}/{date_str}-{model_name}.csv"
        try:
            response = requests.get(file_url, timeout=5)
            if response.status_code == 200:
                df = pd.read_csv(StringIO(response.text))
                df['model'] = model_name
                if model_name in ['Gatech-ensemble_prob', 'Gatech-ensemble_stat']:
                    if df['location'].dtype in ['int64', 'int32']:
                        df['location'] = df['location'].astype(int).astype(str).str.zfill(2)
                all_forecasts.append(df)
        except Exception as e:
            pass
        current_date += pd.Timedelta(days=7)
    if all_forecasts:
        return pd.concat(all_forecasts, ignore_index=True)
    return pd.DataFrame()


@st.cache_data(ttl=1)
def load_wis_data():
    try:
        df = pd.read_parquet('./data/wis_ratio_epistorm_models_2526.pq')
        df['target_end_date'] = pd.to_datetime(df['target_end_date'])
        df['reference_date'] = df['target_end_date'] - pd.to_timedelta(df['horizon'] * 7, unit='D')
        df['location'] = df['location'].astype(str).str.zfill(2)
        return df
    except Exception as e:
        st.error(f"Error loading WIS data: {e}")
        return None

@st.cache_data(ttl=1)
def load_coverage_data():
    try:
        df = pd.read_parquet('./data/coverage_epistorm_models_2526.pq')
        df['target_end_date'] = pd.to_datetime(df['target_end_date'])
        df['reference_date'] = df['target_end_date'] - pd.to_timedelta(df['horizon'] * 7, unit='D')
        df['location'] = df['location'].astype(str).str.zfill(2)
        return df
    except Exception as e:
        st.error(f"Error loading coverage data: {e}")
        return None

def create_ensemble_forecasts(forecast_data):
   # ensemble1 = create_ensemble_method1(forecast_data)
   # ensemble1['model'] = 'Median Epistorm Ensemble'
   # categorical_ensemble = create_categorical_ensemble_quantile(ensemble1[ensemble1.horizon>=0])
   # categorical_ensemble['model'] = 'Median Epistorm Ensemble'

    ensemble1 = pd.read_parquet('./data/quantile_ensemble.pq')
    categorical_ensemble = pd.read_parquet('./data/categorical_ensemble.pq')
    ensemble2 = pd.read_parquet('./data/quantile_ensemble_LOP.pq')

    ensemble1['model'] = 'Median Epistorm Ensemble'
    categorical_ensemble['model'] = 'Median Epistorm Ensemble'
    ensemble2['model'] = 'LOP Epistorm Ensemble'
    #ensemble2['target'] = 'wk inc flu hosp'  # Ensure target is set correctly

    return pd.concat([ensemble1, categorical_ensemble, ensemble2], ignore_index=True)

def hex_to_rgba(hex_color, alpha):
    hex_color = hex_color.lstrip('#')
    r, g, b = tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))
    return f'rgba({r},{g},{b},{alpha})'

def format_category(category):
    category_str = str(category).strip().lower()
    word_map = {
        'large_decrease': 'Large Decrease',
        'large decrease': 'Large Decrease',
        'decrease': 'Decrease',
        'stable': 'Stable',
        'increase': 'Increase',
        'large_increase': 'Large Increase',
        'large increase': 'Large Increase'
    }
    return word_map.get(category_str, category_str.title())

def get_location_name(location_code, locations_df):
    if location_code == 'US':
        return 'United States'
    if locations_df is not None:
        loc_info = locations_df[locations_df['location'] == location_code]
        if not loc_info.empty:
            return loc_info.iloc[0]['location_name']
    return location_code

def plot_wis_boxplot_evaluation(wis_df, selected_models, locations_df):
    if wis_df is None or wis_df.empty:
        return None
    fig = go.Figure()
    models_with_data = [m for m in selected_models if m in wis_df['Model'].unique()]
    
    # Calculate median WIS ratio for each model and sort
    model_medians = {}
    for model in models_with_data:
       # if model=='FluSight-ensemble':
        #    continue
        model_data = wis_df[wis_df['Model'] == model]
        if not model_data.empty:
            model_medians[model] = model_data['wis_ratio'].median()
    
    # Sort models by median WIS ratio (ascending - best performers first)
    sorted_models = sorted(model_medians.keys(), key=lambda m: model_medians[m])
    
    for model in sorted_models:
        model_data = wis_df[wis_df['Model'] == model]
        if not model_data.empty:
            fig.add_trace(go.Box(
                x=model_data['wis_ratio'],
                y=[model] * len(model_data),
                name=model,
                marker_color=COLOR_MAP.get(model, '#808080'),
                boxmean=False,
                orientation='h',
                hoverinfo='skip',
                boxpoints='outliers',
                marker=dict(opacity=0)  
            ))
    fig.add_vline(
        x=1,
        line_dash="dash",
        line_color="red",
        line_width=2,
        annotation_text="WIS ratio = 1",
        annotation_position="top"
    )

    # Calculate x-axis limits based on whiskers (1.5*IQR)
    all_whisker_mins = []
    all_whisker_maxs = []
    for model in sorted_models:
        model_data = wis_df[wis_df['Model'] == model]['wis_ratio']
        q1 = model_data.quantile(0.25)
        q3 = model_data.quantile(0.75)
        iqr = q3 - q1
        whisker_min = max(model_data.min(), q1 - 1.5 * iqr)
        whisker_max = min(model_data.max(), q3 + 1.5 * iqr)
        all_whisker_mins.append(whisker_min)
        all_whisker_maxs.append(whisker_max)

    x_min = min(all_whisker_mins) - 0.1
    x_max = max(all_whisker_maxs) + 0.1

    fig.update_layout(
        title="",
        xaxis_title="WIS Ratio (Model / FluSight Baseline)",
        yaxis_title="",
        height=max(50 * len(sorted_models) + 150, 400),
        width=800,
        showlegend=False,
        margin=dict(l=200, r=50, t=80, b=50),
        xaxis=dict(range=[x_min, x_max]),
        yaxis=dict(
            categoryorder='array',
            categoryarray=list(reversed(sorted_models))
        )
    )
    return fig

def plot_coverage_evaluation(coverage_df, selected_models):
    if coverage_df is None or coverage_df.empty:
        return None
    fig = go.Figure()
    intervals = [10, 20, 30, 40, 50, 60, 70, 80, 90, 95, 98]
    fig.add_trace(go.Scatter(
        x=intervals,
        y=[i/100 for i in intervals],
        mode='lines',
        name='Perfect Calibration',
        line=dict(color='black', dash='dash', width=2),
        showlegend=True
    ))
    models_with_data = [m for m in selected_models if m in coverage_df['Model'].unique()]
    for model in models_with_data:
        #if model=='FluSight-ensemble':
         #   continue
        model_data = coverage_df[coverage_df['Model'] == model]
        if not model_data.empty:
            avg_coverage = []
            for interval in intervals:
                col_name = f'{interval}_cov'
                if col_name in model_data.columns:
                    avg_coverage.append(model_data[col_name].mean())
                else:
                    avg_coverage.append(np.nan)
            fig.add_trace(go.Scatter(
                x=intervals,
                y=avg_coverage,
                mode='lines+markers',
                name=model,
                line=dict(color=COLOR_MAP.get(model, '#808080'), width=2),
                marker=dict(size=8),
                hovertemplate=f'{model}<br>Interval: %{{x}}%<br>Coverage: %{{y:.1%}}<extra></extra>'
            ))
    fig.update_layout(
        title="",
        xaxis_title="Prediction Interval (%)",
        yaxis_title="Coverage",
        height=600,
        width=900,
        yaxis=dict(tickformat='.0%', range=[0, 1.05]),
        xaxis=dict(range=[0, 105], tickvals=intervals),
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=1.02
        ),
        margin=dict(l=60, r=10, t=50, b=60)
    )
    return fig

def plot_categorical_forecasts(forecast_data, selected_location, selected_date, selected_models):
    cat_data = forecast_data[
        (forecast_data['output_type'] == 'pmf') &
        (forecast_data['target'] == 'wk flu hosp rate change') &
        (forecast_data['location'] == selected_location) &
        (forecast_data['reference_date'] == selected_date) &
        (forecast_data['model'].isin(selected_models))
    ].copy()
    if cat_data.empty:
        return None, []
    cat_data['category'] = cat_data['output_type_id'].apply(format_category)
    cat_data = cat_data[cat_data['category'].isin(CATEGORY_ORDER)]
    available_models = []
    models_without_data = []
    for model in selected_models:
        model_data = cat_data[cat_data['model'] == model]
        if not model_data.empty:
            available_models.append(model)
        else:
            models_without_data.append(model)
    if not available_models:
        return None, models_without_data
    n_models = len(available_models)
    fig = make_subplots(
        rows=n_models,
        cols=1,
        subplot_titles=available_models,
        vertical_spacing=0.2 / n_models if n_models > 1 else 0.1,
        row_heights=[1] * n_models
    )
    legend_added = set()
    for idx, model in enumerate(available_models):
        model_data = cat_data[cat_data['model'] == model].copy()
        horizons = [0, 1, 2, 3]
        for horizon in horizons:
            horizon_data = model_data[model_data['horizon'] == horizon]
            cumulative = 0
            for category in CATEGORY_ORDER:
                cat_rows = horizon_data[horizon_data['category'] == category]
                if not cat_rows.empty:
                    prob = cat_rows['value'].values[0]
                else:
                    prob = 0
                show_in_legend = category not in legend_added
                if show_in_legend:
                    legend_added.add(category)
                fig.add_trace(
                    go.Bar(
                        x=[horizon],
                        y=[prob],
                        name=category,
                        marker_color=CATEGORY_COLORS[category],
                        legendgroup=category,
                        showlegend=show_in_legend,
                        hovertemplate=f'{category}: {prob:.1%}<extra></extra>' if prob > 0 else None,
                        base=cumulative,
                        width=0.6
                    ),
                    row=idx + 1,
                    col=1
                )
                cumulative += prob
        fig.update_xaxes(
            title_text="Horizon" if idx == n_models - 1 else "",
            tickmode='array',
            tickvals=[0, 1, 2, 3],
            ticktext=['0', '1', '2', '3'],
            range=[-0.5, 3.5],
            row=idx + 1,
            col=1
        )
        fig.update_yaxes(
            title_text="Probability",
            tickformat='.0%',
            range=[0, 1.0],
            row=idx + 1,
            col=1
        )
    fig.update_layout(
        barmode='stack',
        height=max(300 * n_models, 400),
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.1,
            xanchor="center",
            x=0.5,
            traceorder="normal"
        ),
        margin=dict(t=120, b=50, l=80, r=50)
    )
    return fig, models_without_data

def plot_forecasts(observed_data, forecast_data, selected_location, selected_date, selected_models, available_dates, start_date, end_date):
    
    fig = go.Figure()
    obs_filtered = observed_data[
        (observed_data['location'] == selected_location) &
        (observed_data['date'] >= start_date) &
        (observed_data['date'] <= end_date)
    ].copy()
    obs_filtered = obs_filtered.sort_values('date')

    if not obs_filtered.empty:
        reference_date = pd.Timestamp(selected_date)

        obs_past = obs_filtered[obs_filtered['date'] < reference_date]
        obs_future = obs_filtered[obs_filtered['date'] >= reference_date]

        # Continuous line across all observed data
        fig.add_trace(go.Scatter(
            x=obs_filtered['date'],
            y=obs_filtered['value'],
            mode='lines',
            showlegend=False,
            line=dict(color='black', width=2),
            hovertemplate='Value: %{y:,.0f}<extra></extra>'
        ))

        # Solid markers for past
        if not obs_past.empty:
            fig.add_trace(go.Scatter(
                x=obs_past['date'],
                y=obs_past['value'],
                mode='lines+markers',
                name='Observed (in sample)',
                showlegend=True,
                marker=dict(size=8, symbol='circle', color='black'),
                hovertemplate='Value: %{y:,.0f}<extra></extra>'
            ))

        # Open markers for preliminary
        if not obs_future.empty:
            fig.add_trace(go.Scatter(
                x=obs_future['date'],
                y=obs_future['value'],
                mode='lines+markers',
                name='Observed (out of sample)',
                showlegend=True,
                marker=dict(size=8, symbol='circle-open', color='black'),
                hovertemplate='Value: %{y:,.0f}<extra></extra>'
            ))


    forecast_filtered = forecast_data[
        (forecast_data['location'] == selected_location) &
        (forecast_data['reference_date'] == selected_date) &
        (forecast_data['target'] == 'wk inc flu hosp') &
        (forecast_data['model'].isin(selected_models))
    ].copy()

    max_forecast_date = end_date
    for model in selected_models:
        #st.write(f"model: '{model}', rows: {len(forecast_filtered[forecast_filtered['model'] == model])}")
    
        model_data = forecast_filtered[forecast_filtered['model'] == model]
        if model_data.empty:
            continue
        color = COLOR_MAP.get(model, '#808080')
        quantiles_df = model_data[model_data['output_type'] == 'quantile'].copy()
        quantiles_df['output_type_id'] = quantiles_df['output_type_id'].astype(float)
        pivot_df = quantiles_df.pivot_table(
            index='target_end_date',
            columns='output_type_id',
            values='value'
        ).reset_index()
        pivot_df = pivot_df.sort_values('target_end_date')
        if not pivot_df.empty:
            max_forecast_date = max(max_forecast_date, pivot_df['target_end_date'].max())
        if 0.025 in pivot_df.columns and 0.975 in pivot_df.columns:
            fig.add_trace(go.Scatter(
                x=pivot_df['target_end_date'],
                y=pivot_df[0.975],
                mode='lines',
                line=dict(width=0),
                showlegend=False,
                hoverinfo='skip',
                legendgroup=model
            ))
            fig.add_trace(go.Scatter(
                x=pivot_df['target_end_date'],
                y=pivot_df[0.025],
                mode='lines',
                line=dict(width=0),
                fillcolor=hex_to_rgba(color, 0.2),
                fill='tonexty',
                name=f'{model} (95% PI)',
                showlegend=True,
                customdata=pivot_df[[0.025, 0.975]],
                hovertemplate='95% PI: (%{customdata[0]:,.0f} - %{customdata[1]:,.0f}<extra></extra>)',
                legendgroup=model
            ))
        if 0.25 in pivot_df.columns and 0.75 in pivot_df.columns:
            fig.add_trace(go.Scatter(
                x=pivot_df['target_end_date'],
                y=pivot_df[0.75],
                mode='lines',
                line=dict(width=0),
                showlegend=False,
                hoverinfo='skip',
                legendgroup=model
            ))
            fig.add_trace(go.Scatter(
                x=pivot_df['target_end_date'],
                y=pivot_df[0.25],
                mode='lines',
                line=dict(width=0),
                fillcolor=hex_to_rgba(color, 0.4),
                fill='tonexty',
                name=f'{model} (50% PI)',
                showlegend=True,
                customdata=pivot_df[[0.25, 0.75]],
                hovertemplate='50% PI: (%{customdata[0]:,.0f} - %{customdata[1]:,.0f}<extra></extra>)',
                legendgroup=model
            ))
        if 0.5 in pivot_df.columns:
            fig.add_trace(go.Scatter(
                x=pivot_df['target_end_date'],
                y=pivot_df[0.5],
                mode='lines+markers',
                name=f'{model} (median)',
                line=dict(color=color, width=2),
                marker=dict(size=6, color=color),
                customdata=pivot_df[[0.5]],
                hovertemplate='Median: %{customdata[0]:,.0f}<extra></extra>',
                legendgroup=model
            ))
    #for ref_date in available_dates:
     #   fig.add_shape(
      #      type="line",
       #     x0=ref_date, x1=ref_date,
        #    y0=0, y1=1,
         #   yref="paper",
          #  line=dict(color='gray', width=1, dash='dot'),
           # opacity=0.3
       # )
    fig.add_shape(
        type="line",
        x0=selected_date, x1=selected_date,
        y0=0, y1=1,
        yref="paper",
        line=dict(color='red', width=2, dash='dash')
    )
    fig.add_annotation(
        x=selected_date,
        y=1,
        yref="paper",
        text=f"Forecast Date: {selected_date.strftime('%Y-%m-%d')}",
        showarrow=False,
        yshift=10
    )
    location_name = "US" if selected_location == "US" else selected_location
    if selected_location != "US":
        locations_df = load_locations()
        if locations_df is not None:
            loc_info = locations_df[locations_df['location'] == selected_location]
            if not loc_info.empty:
                location_name = loc_info.iloc[0]['location_name']
    fig.update_layout(
        xaxis_title="Date",
        yaxis_title="Weekly Incident Flu Hospitalizations",
        hovermode='x unified',
        height=600,
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.01
        ),
        xaxis=dict(
            range=[start_date, max_forecast_date]
        )
    )
    return fig, location_name

st.markdown(
    f"""
    <div style="display: flex; justify-content: flex-start;margin-bottom: 20px;">
        {img_to_html("assets/epistorm-ensemble-logo.png", width=250)}
    </div>
    """,
    unsafe_allow_html=True,
)



tab_overview, tab_forecasts, tab_evaluation = st.tabs([ "Overview", "Forecasts", "Evaluation"])


# Main app
#st.markdown("<h1 style='color: #415584;'>Epistorm Influenza Forecasts 2025-26</h1>", unsafe_allow_html=True)
#st.markdown('<p style="font-size: 20px; color: black;">Interactive visualization of influenza hospitalization forecasts from models in the Epistorm consortium for the 2025-26 season.</p>',
#    unsafe_allow_html=True)

# View selection in sidebar
#tab_selection = st.sidebar.selectbox(   "Select Tab", ["Forecasts", "Evaluation"],  key="tab_selection")


st.sidebar.markdown("---")

# Load common data
with st.spinner("Loading data..."):
    locations_df = load_locations()
    observed_data = load_observed_data()
    forecast_data = load_all_forecasts()
    baseline_data = load_baseline_forecasts()
    thresholds = load_thresholds()

if forecast_data.empty:
    st.error("No forecast data could be loaded. Please check your internet connection and try again.")
    st.stop()

if observed_data is None or observed_data.empty:
    st.warning("Could not load observed data. Continuing with forecasts only.")
    observed_data = pd.DataFrame(columns=['date', 'location', 'value'])

with st.spinner("Creating ensemble forecasts..."):
    try:
        if 'Median Epistorm Ensemble' not in forecast_data['model'].values or 'LOP Epistorm Ensemble' not in forecast_data['model'].values:
            ensemble_data = create_ensemble_forecasts(forecast_data)
            forecast_data = pd.concat([forecast_data, ensemble_data], ignore_index=True)
        forecast_data = forecast_data[forecast_data['horizon'] >= 0]
    except Exception as e:
        st.warning(f"Could not create ensemble forecasts: {e}")
        import traceback
        st.error(traceback.format_exc())

# ============== SIDEBAR CONTROLS ==============
#if tab_selection == "Forecasts":
with tab_forecasts:
    controls_col, chart_col = st.columns([1, 3], gap="large")
    
    with controls_col:
        #st.markdown("### Forecast Controls")
        
        # Date selector

        # Model selection (ensemble only)
       # selected_models = ['Median Epistorm Ensemble']
        #st.session_state.selected_models = selected_models

       
       # Location selector
        if locations_df is not None:
            state_locations_df = locations_df[locations_df['location'] != 'US']
            location_options = ['US'] + state_locations_df['location'].tolist()
            location_names = ['United States'] + state_locations_df['location_name'].tolist()
            location_dict = dict(zip(location_names, location_options))

            if 'selected_location_name' not in st.session_state: 
                st.session_state.selected_location_name = 'United States'

            st.markdown("""
                <style>
                div[data-testid="stExpander"] div[data-testid="stVerticalBlock"] {
                    max-height: 650px;
                    overflow-y: auto;
                }
                </style>
            """, unsafe_allow_html=True)

            with st.expander("Select Location", expanded=True):
                selected = st.radio(
                    label="",
                    options=location_names,
                    index=location_names.index(st.session_state.selected_location_name),
                    key="selected_location_name"
                )
            selected_location_name = st.session_state.selected_location_name
            selected_location = location_dict[selected_location_name]
        else:
            selected_location = st.text_input("Enter Location Code", value="US", key="forecast_location_text")



    with chart_col:

        available_dates = sorted(
            [pd.Timestamp(d).to_pydatetime() for d in forecast_data['reference_date'].unique()]
        )

        if available_dates:
            if 'selected_date' not in st.session_state:
                st.session_state.selected_date = available_dates[-1]
            selected_date = st.session_state.get('forecast_date', st.session_state.selected_date)
        else:
            st.error("No forecast dates available")
            st.stop()


        model_options = {
            "Median Epistorm Ensemble": "Median Epistorm Ensemble",
            "LOP Epistorm Ensemble": "LOP Epistorm Ensemble",
        }

        if 'selected_model' not in st.session_state:
            st.session_state.selected_model = "Median Epistorm Ensemble"

        selected_models = [st.session_state.selected_model]

        if selected_models:
            
            if not observed_data.empty:
                min_obs_date = observed_data['date'].min().date()
                max_obs_date = observed_data['date'].max().date()
            else:
                min_obs_date = datetime(2024, 1, 1).date()
                max_obs_date = datetime.now().date()

            latest_forecast_date = None
            if not forecast_data.empty and 'target_end_date' in forecast_data.columns:
                selected_forecast_data = forecast_data[forecast_data['reference_date'] == selected_date]
                if not selected_forecast_data.empty:
                    latest_forecast_date = pd.Timestamp(selected_forecast_data['target_end_date'].max()).date()

            max_end_date = max(max_obs_date, latest_forecast_date) if latest_forecast_date else max_obs_date

            range_options = {
                "Last 3 Months": 3,
                "Last 6 Months": 6,
                "Last year": 12,
                "Last 2 Years": 24,
                "All data": None,
            }

            if 'selected_range' not in st.session_state:
                st.session_state.selected_range = "Last 3 Months"

            months = range_options[st.session_state.selected_range]
            if months is None:
                start_date = min_obs_date
            else:
                start_date = max_end_date - pd.DateOffset(months=months)
                start_date = max(start_date.date(), min_obs_date)

            end_date = max_end_date
            start_date_ts = pd.Timestamp(start_date)
            end_date_ts = pd.Timestamp(end_date)

            #st.write("calling plot_forecasts with models:", selected_models)

            fig, location_name = plot_forecasts(observed_data, forecast_data, selected_location, selected_date, selected_models, available_dates, start_date_ts, end_date_ts)
            
            st.markdown(f"<h2 style='color: #518fb0;'>Flu Hospitalization Forecasts - {location_name}</h2>", unsafe_allow_html=True)

            ensemble_forecast_data = forecast_data[
                (forecast_data['model'] == 'Median Epistorm Ensemble') &
                (forecast_data['location'] == selected_location) &
                (forecast_data['reference_date'] == selected_date) &
                (forecast_data['horizon'] == 3) &
                (forecast_data['output_type'] == 'quantile')
            ].copy()
            
            if not ensemble_forecast_data.empty:
                ensemble_forecast_data['output_type_id'] = ensemble_forecast_data['output_type_id'].astype(float)
                median_val = ensemble_forecast_data[ensemble_forecast_data['output_type_id'] == 0.5]['value'].values
                lower_val = ensemble_forecast_data[ensemble_forecast_data['output_type_id'] == 0.025]['value'].values
                upper_val = ensemble_forecast_data[ensemble_forecast_data['output_type_id'] == 0.975]['value'].values
                
                if len(median_val) > 0 and len(lower_val) > 0 and len(upper_val) > 0:
                    median = int(round(median_val[0]))
                    lower = int(round(lower_val[0]))
                    upper = int(round(upper_val[0]))
                    
                    population = None
                    if locations_df is not None:
                        if selected_location == 'US':
                            pop_row = locations_df[locations_df['location'] == 'US']
                        else:
                            pop_row = locations_df[locations_df['location'] == selected_location]
                        if not pop_row.empty and 'population' in pop_row.columns:
                            population = pop_row.iloc[0]['population']
                    
                    end_date_4_weeks = ensemble_forecast_data['target_end_date'].iloc[0].strftime('%B %d, %Y')
                    heading = f"As of {selected_date.strftime('%B %d, %Y')}, the Epistorm Ensemble forecasts <b>{median:,}</b> influenza hospital admissions"
                    
                    if population and population > 0:
                        per_capita_median = int(round((median / population) * 100000))
                        per_capita_lower = int(round((lower / population) * 100000))
                        per_capita_upper = int(round((upper / population) * 100000))
                        heading += f" (<b>{per_capita_median:,}</b> per 100,000 people)"
                    
                    heading += f" by {end_date_4_weeks} for {location_name}."
                    pi_text = f"95% prediction interval: <b>{lower:,}–{upper:,}</b> hospital admissions"
                    
                    if population and population > 0:
                        pi_text += f" (<b>{per_capita_lower:,}–{per_capita_upper:,}</b> per 100,000 people)"
                    pi_text += "."
                    
                    st.markdown(f"<p style='font-size: 22px;'>{heading}</p>", unsafe_allow_html=True)
                    st.markdown(f"<p style='font-size: 18px;'>{pi_text}</p>", unsafe_allow_html=True)

            # Date range
            
           # with st.expander("Historical Data Range", expanded=True):
            #    cols = st.columns(len(range_options))
             #   for col, label in zip(cols, range_options):
              #      with col:
               #         if st.button(label, key=f"range_{label}", use_container_width=True):
                #            st.session_state.selected_range = label
                 #           st.rerun()

            #st.markdown("<style>.stPlotlyChart { margin-top: -5rem; }</style>", unsafe_allow_html=True)

            event = st.plotly_chart(fig, key="forecast_plot", on_select="rerun", use_container_width=True, config={'displayModeBar': False})
            
            if event and hasattr(event, 'selection') and event.selection:
                try:
                    if 'points' in event.selection and len(event.selection['points']) > 0:
                        clicked_x = event.selection['points'][0]['x']
                        clicked_date = pd.to_datetime(clicked_x)
                        closest_date = min(available_dates, key=lambda d: abs((d - clicked_date).total_seconds()))
                        if closest_date != st.session_state.selected_date:
                            st.session_state.selected_date = closest_date
                            st.rerun()
                except:
                    pass
        
            st.markdown("""
                <style>
                .stSlider [data-baseweb="slider"] [role="slider"] {
                    background-color: #2B7A8F;
                }
                .stSlider [data-baseweb="slider"] [data-testid="stThumbValue"] {
                    color: #2B7A8F;
                }
                .stSlider [data-baseweb="slider"] div[data-testid="stSliderTrackFill"] {
                    background-color: #2B7A8F;
                }
                .stSlider [data-baseweb="slider"] [data-testid="stSliderTrack"] {
                    background-color: #a8d4db;
                }
                </style>
            """, unsafe_allow_html=True)

            # Slider lives below the plot
            selected_date = st.select_slider(
                "**Select Forecast Date**",
                options=available_dates,
                value=st.session_state.selected_date if st.session_state.selected_date in available_dates else available_dates[-1],
                format_func=lambda x: x.strftime('%Y-%m-%d'),
                key="forecast_date"
            )
           # st.session_state.selected_date = selected_date



            st.markdown("""
                <style>
                div[data-testid="stHorizontalBlock"] button[kind="primary"] {
                    background-color: #2B7A8F;
                    border-color: #2B7A8F;
                    color: white;
                }
                div[data-testid="stHorizontalBlock"] button[kind="secondary"] {
                    background-color: white;
                    border-color: #2B7A8F;
                    color: #2B7A8F;
                }
                </style>
            """, unsafe_allow_html=True)

            st.markdown("**Select Model**")

            cols = st.columns(2)
            for col, label in zip(cols, list(model_options.keys())):
                with col:
                    is_selected = st.session_state.selected_model == label
                    if st.button(
                        f"✓ {label}" if is_selected else label,
                        key=f"model_{label}",
                        use_container_width=True,
                        type="primary" if is_selected else "secondary"
                    ):
                        st.session_state.selected_model = label
                        st.rerun()

            with st.expander("Historical Data Range", expanded=True):
                cols = st.columns(len(range_options))
                for col, label in zip(cols, range_options):
                    with col:
                        if st.button(label, key=f"range_{label}", use_container_width=True):
                            st.session_state.selected_range = label
                            st.rerun()


            #st.markdown("**Tip:** Change the location, forecast date, or historical date range in the sidebar.")
           
        else:
            st.warning("Please select at least one model to display")

        st.markdown("<div style='margin-top: -50rem;'>", unsafe_allow_html=True)
        with st.container(border=True):
            st.markdown("**What are Prediction Intervals?**")
            st.markdown(
                "The shaded regions around the forecast line show the range of values the model expects future observations to fall within. "
                "The darker shaded region represents the **50% interval** — there is a 50% chance the true value will fall within this range. "
                "The lighter shaded region represents the **90% interval** — there is a 90% chance the true value will fall within this range."
            )


        
        



#if tab_selection == "Evaluation":
with tab_evaluation:
    #st.sidebar.header("Evaluation Controls")

    controls_col, chart_col = st.columns([1, 3], gap="large")
    
    with controls_col:
        #st.markdown("### Evaluation Controls")

        wis_data = load_wis_data()
        coverage_data = load_coverage_data()

        if wis_data is None and coverage_data is None:
            st.error("Could not load evaluation data.")
            st.stop()

        # Get available models
        eval_models = []
        if wis_data is not None:
            eval_models = sorted(wis_data['Model'].unique().tolist())
        elif coverage_data is not None:
            eval_models = sorted(coverage_data['Model'].unique().tolist())

        # Location selector
        eval_location_map = {'All Locations': 'all'}
        if locations_df is not None:
            eval_location_codes = []
            if wis_data is not None:
                eval_location_codes = sorted(wis_data['location'].unique().tolist())
            elif coverage_data is not None:
                eval_location_codes = sorted(coverage_data['location'].unique().tolist())

            eval_location_names = ['All Locations']
            for loc_code in eval_location_codes:
                loc_name = get_location_name(loc_code, locations_df)
                eval_location_names.append(loc_name)
                eval_location_map[loc_name] = loc_code

            selected_eval_location_name = st.selectbox(
                "Select Location",
                eval_location_names,
                index=0,
                key="eval_location"
            )
            selected_eval_location = eval_location_map[selected_eval_location_name]
        else:
            selected_eval_location = st.text_input(
                "Enter Location Code (or 'all')", value="all", key="eval_location_text"
            )

        # Horizon selector
        selected_horizon = st.selectbox(
            "Select Horizon",
            ['All Horizons', 0, 1, 2, 3],
            index=0,
            key="eval_horizon"
        )

        # Date filters
        with st.expander("Date Filters"):
            if wis_data is not None:
                min_ref_date = wis_data['reference_date'].min().date()
                max_ref_date = wis_data['reference_date'].max().date()
                min_target_date = wis_data['target_end_date'].min().date()
                max_target_date = wis_data['target_end_date'].max().date()
            else:
                min_ref_date = datetime(2024, 1, 1).date()
                max_ref_date = datetime.now().date()
                min_target_date = min_ref_date
                max_target_date = max_ref_date

            ref_date_range = st.date_input(
                "Reference Date Range",
                value=(min_ref_date, max_ref_date),
                min_value=min_ref_date,
                max_value=max_ref_date,
                key="eval_ref_date_range"
            )
            target_date_range = st.date_input(
                "Target End Date Range",
                value=(min_target_date, max_target_date),
                min_value=min_target_date,
                max_value=max_target_date,
                key="eval_target_date_range"
            )

        if isinstance(ref_date_range, tuple) and len(ref_date_range) == 2:
            ref_start, ref_end = ref_date_range
        else:
            ref_start = ref_end = ref_date_range

        if isinstance(target_date_range, tuple) and len(target_date_range) == 2:
            target_start, target_end = target_date_range
        else:
            target_start = target_end = target_date_range

        # Model selection
        with st.expander("Select Models", expanded=True):
            eval_ensemble_models = [m for m in eval_models if m == 'Median Epistorm Ensemble']
            eval_individual_models = [m for m in eval_models if m != 'Median Epistorm Ensemble' ]

            if 'selected_eval_models' not in st.session_state:
                st.session_state.selected_eval_models = eval_models.copy()

            selected_eval_models = []
            if eval_ensemble_models:
                st.markdown("**Ensemble:**")
                for model in eval_ensemble_models:
                    if st.checkbox(model, value=model in st.session_state.selected_eval_models, key=f"eval_model_{model}"):
                        selected_eval_models.append(model)
            if eval_individual_models:
                st.markdown("**Individual:**")
                for model in eval_individual_models:
                    if st.checkbox(model, value=model in st.session_state.selected_eval_models, key=f"eval_model_{model}"):
                        selected_eval_models.append(model)
            st.session_state.selected_eval_models = selected_eval_models

        # Data summary
        st.markdown("---")
        st.caption(f"Models selected: {len(selected_eval_models)}")

    # ---- Filter data ----
    filtered_wis = wis_data.copy() if wis_data is not None else None
    filtered_coverage = coverage_data.copy() if coverage_data is not None else None

    if filtered_wis is not None:
        if selected_eval_models:
            filtered_wis = filtered_wis[filtered_wis['Model'].isin(selected_eval_models)]
        if selected_eval_location != 'all':
            filtered_wis = filtered_wis[filtered_wis['location'] == selected_eval_location]
        if selected_horizon != 'All Horizons':
            filtered_wis = filtered_wis[filtered_wis['horizon'] == selected_horizon]
        filtered_wis = filtered_wis[
            (filtered_wis['reference_date'].dt.date >= ref_start) &
            (filtered_wis['reference_date'].dt.date <= ref_end) &
            (filtered_wis['target_end_date'].dt.date >= target_start) &
            (filtered_wis['target_end_date'].dt.date <= target_end)
        ]

    if filtered_coverage is not None:
        if selected_eval_models:
            filtered_coverage = filtered_coverage[filtered_coverage['Model'].isin(selected_eval_models)]
        if selected_eval_location != 'all':
            filtered_coverage = filtered_coverage[filtered_coverage['location'] == selected_eval_location]
        if selected_horizon != 'All Horizons':
            filtered_coverage = filtered_coverage[filtered_coverage['horizon'] == selected_horizon]
        filtered_coverage = filtered_coverage[
            (filtered_coverage['reference_date'].dt.date >= ref_start) &
            (filtered_coverage['reference_date'].dt.date <= ref_end) &
            (filtered_coverage['target_end_date'].dt.date >= target_start) &
            (filtered_coverage['target_end_date'].dt.date <= target_end)
        ]




    with chart_col:    

        # Check if evaluation data was loaded
        wis_data = load_wis_data()
        coverage_data = load_coverage_data()
        
        if wis_data is None and coverage_data is None:
            st.error("Could not load evaluation data. Please ensure the parquet files exist.")
        else:
            # Get filter values from sidebar (already set above)
            selected_eval_models = st.session_state.get('selected_eval_models', [])
            selected_eval_location = st.session_state.get('eval_location', 'All Locations')

            st.markdown(f"<h2 style='color: #518fb0;'>Forecast Evaluation - {selected_eval_location}</h2>", unsafe_allow_html=True)
            st.markdown("Evaluate forecast performance using the Weighted Interval Score (WIS) and prediction interval coverage metrics.")
        

            if selected_eval_location != 'All Locations':
                # Convert location name back to code if needed
                if locations_df is not None:
                    eval_location_codes = sorted(wis_data['location'].unique().tolist()) if wis_data is not None else []
                    eval_location_map = {'All Locations': 'all'}
                    for loc_code in eval_location_codes:
                        loc_name = get_location_name(loc_code, locations_df)
                        eval_location_map[loc_name] = loc_code
                    selected_eval_location = eval_location_map.get(selected_eval_location, 'all')
            else:
                selected_eval_location = 'all'
            
            selected_horizon = st.session_state.get('eval_horizon', 'All Horizons')
            
            ref_date_range = st.session_state.get('eval_ref_date_range', None)
            target_date_range = st.session_state.get('eval_target_date_range', None)
            
            if wis_data is not None:
                min_ref_date = wis_data['reference_date'].min().date()
                max_ref_date = wis_data['reference_date'].max().date()
                min_target_date = wis_data['target_end_date'].min().date()
                max_target_date = wis_data['target_end_date'].max().date()
            else:
                min_ref_date = datetime(2024, 1, 1).date()
                max_ref_date = datetime.now().date()
                min_target_date = min_ref_date
                max_target_date = max_ref_date
            
            if ref_date_range is not None:
                if isinstance(ref_date_range, tuple) and len(ref_date_range) == 2:
                    ref_start, ref_end = ref_date_range
                else:
                    ref_start = ref_end = ref_date_range
            else:
                ref_start, ref_end = min_ref_date, max_ref_date
            
            if target_date_range is not None:
                if isinstance(target_date_range, tuple) and len(target_date_range) == 2:
                    target_start, target_end = target_date_range
                else:
                    target_start = target_end = target_date_range
            else:
                target_start, target_end = min_target_date, max_target_date
            
            # Filter data
            filtered_wis = wis_data.copy() if wis_data is not None else None
            filtered_coverage = coverage_data.copy() if coverage_data is not None else None
            
            if filtered_wis is not None:
                if selected_eval_models:
                    filtered_wis = filtered_wis[filtered_wis['Model'].isin(selected_eval_models)]
                if selected_eval_location != 'all':
                    filtered_wis = filtered_wis[filtered_wis['location'] == selected_eval_location]
                if selected_horizon != 'All Horizons':
                    filtered_wis = filtered_wis[filtered_wis['horizon'] == selected_horizon]
                filtered_wis = filtered_wis[
                    (filtered_wis['reference_date'].dt.date >= ref_start) &
                    (filtered_wis['reference_date'].dt.date <= ref_end) &
                    (filtered_wis['target_end_date'].dt.date >= target_start) &
                    (filtered_wis['target_end_date'].dt.date <= target_end)
                ]
            
            if filtered_coverage is not None:
                if selected_eval_models:
                    filtered_coverage = filtered_coverage[filtered_coverage['Model'].isin(selected_eval_models)]
                if selected_eval_location != 'all':
                    filtered_coverage = filtered_coverage[filtered_coverage['location'] == selected_eval_location]
                if selected_horizon != 'All Horizons':
                    filtered_coverage = filtered_coverage[filtered_coverage['horizon'] == selected_horizon]
                filtered_coverage = filtered_coverage[
                    (filtered_coverage['reference_date'].dt.date >= ref_start) &
                    (filtered_coverage['reference_date'].dt.date <= ref_end) &
                    (filtered_coverage['target_end_date'].dt.date >= target_start) &
                    (filtered_coverage['target_end_date'].dt.date <= target_end)
                ]
            
            # Display WIS boxplot
            st.markdown("### Weighted Interval Score (WIS) Ratio")
            st.markdown("""
            The WIS ratio compares each model's Weighted Interval Score to the FluSight-baseline model. 
            A ratio less than 1 indicates better performance than baseline, while a ratio greater than 1 indicates worse performance.
            """)
            
            if filtered_wis is not None and not filtered_wis.empty:
                wis_fig = plot_wis_boxplot_evaluation(filtered_wis, selected_eval_models, locations_df)
                if wis_fig:
                    st.plotly_chart(wis_fig, use_container_width=False, config={'displayModeBar': False})
                else:
                    st.warning("No WIS data available for the selected filters.")
            else:
                st.warning("No WIS data available for the selected filters.")
            
            st.markdown("---")
            
            # Display coverage plot
            st.markdown("### Prediction Interval Coverage")
            st.markdown("""
            Coverage measures the proportion of observed values that fall within each prediction interval. 
            A well-calibrated model should have coverage close to the diagonal line (e.g., 50% of observations within the 50% prediction interval).
            """)
            
            if filtered_coverage is not None and not filtered_coverage.empty:
                cov_fig = plot_coverage_evaluation(filtered_coverage, selected_eval_models)
                if cov_fig:
                    st.plotly_chart(cov_fig, use_container_width=False, config={'displayModeBar': False})
                else:
                    st.warning("No coverage data available for the selected filters.")
            else:
                st.warning("No coverage data available for the selected filters.")


with tab_overview:
    if 'overview_ref_date' not in st.session_state:
        cat_dates = sorted(forecast_data['reference_date'].unique(), reverse=True)
        st.session_state.overview_ref_date = cat_dates[0]
    if 'overview_horizon' not in st.session_state:
        st.session_state.overview_horizon = 3
    
    sel_col, _ = st.columns([1, 2])  # 1/3 width for selector
    with sel_col:
        if locations_df is not None:
            state_locations_df = locations_df[locations_df['location'] != 'US'].sort_values('location_name')
            overview_location_names = ['United States'] + state_locations_df['location_name'].tolist()
            overview_location_dict = dict(
                zip(overview_location_names, ['US'] + state_locations_df['location'].tolist())
            )
            overview_location_name = st.selectbox(
                "Location",
                overview_location_names,
                index=0,
                key="overview_location"
            )
            overview_location = overview_location_dict[overview_location_name]
        else:
            overview_location = "US"
            overview_location_name = "United States"

        

    row1_col1, row1_col2 = st.columns([4,2], gap="large")
    row2_col1, row2_col2 = st.columns(2, gap="large")

    
    with row1_col1:
        with st.container(border=True, height=700):
            
           # st.markdown("### Observed Hospitalizations")
           # Filter and plot
            obs_filtered = observed_data[ (observed_data['location'] == overview_location)
            ].sort_values('date')


            loc_thresholds = thresholds[ (thresholds['location'] == overview_location)]


            if 'overview_date_range' not in st.session_state:
                st.session_state.overview_date_range = "Last 3 months"


            if not obs_filtered.empty:
                thresh = loc_thresholds.iloc[0]
                activity_levels = [
                        ('Low', 0, thresh['Medium']),
                        ('Moderate', thresh['Medium'], thresh['High']),
                        ('High', thresh['High'], thresh['Very High']),
                        ('Very High', thresh['Very High'], thresh['Very High'] * 5),
                    ]

                ACTIVITY_COLORS = {
                        'Low': '#7DD4C8',
                        'Moderate': '#3CAAA0',
                        'High': '#2B7A8F',
                        'Very High': '#3D5A80'
                    }

                max_date = obs_filtered['date'].max()
                date_range_option = st.session_state.overview_date_range
                
                if date_range_option == "Last 3 months":
                    obs_filtered = obs_filtered[obs_filtered['date'] >= max_date - pd.DateOffset(months=3)]
                elif date_range_option == "Last 6 months":
                    obs_filtered = obs_filtered[obs_filtered['date'] >= max_date - pd.DateOffset(months=6)]
                elif date_range_option == "Last year":
                    obs_filtered = obs_filtered[obs_filtered['date'] >= max_date - pd.DateOffset(years=1)]
                elif date_range_option == "Last 2 years":
                    obs_filtered = obs_filtered[obs_filtered['date'] >= max_date - pd.DateOffset(years=2)]


                recent_date = obs_filtered[obs_filtered['date'] == obs_filtered['date'].max()] if not obs_filtered.empty else None
                value = recent_date['value'].iloc[0] if recent_date is not None else None
                threshold_dat = thresholds[ (thresholds['location'] == overview_location)].iloc[0]

                if value >= threshold_dat['Very High']:
                    current_threshold = 'Very High'
                elif value >= threshold_dat['High']:
                    current_threshold = 'High'
                elif value >= threshold_dat['Medium']:
                    current_threshold = 'Moderate'
                else:
                    current_threshold = 'Low'

                loc_text = 'the United States' if overview_location=='US' else overview_location_name
                threshold_color = ACTIVITY_COLORS.get(current_threshold, 'black')
                heading = f"The flu activity level in {loc_text} is currently " f"<b style='color: {threshold_color};'>{current_threshold}</b> " + \
                    f"as of {obs_filtered['date'].max().strftime('%B %d, %Y')}."
                

                current_date    = obs_filtered['date'].max()
                current_epiweek = Week.fromdate(current_date)

                # Same epiweek, prior year
                prior_epiweek = Week(current_epiweek.year - 1, current_epiweek.week)
                prior_start   = prior_epiweek.startdate()
                prior_end     = prior_epiweek.enddate()

                obs_all = observed_data[ observed_data['location'] == overview_location].sort_values('date')
                obs_loc = obs_all.copy()

                prior_row = obs_loc[
                    (obs_loc['date'] >= pd.Timestamp(prior_start)) &
                    (obs_loc['date'] <= pd.Timestamp(prior_end))
                ]

                # Calculate current threshold
                recent_date       = obs_loc[obs_loc['date'] == current_date]
                current_value     = recent_date['value'].iloc[0] if not recent_date.empty else None
                threshold_dat     = thresholds[thresholds['location'] == overview_location].iloc[0]

                def get_threshold(val):
                    if val is None:
                        return None
                    if val >= threshold_dat['Very High']:
                        return 'Very High'
                    elif val >= threshold_dat['High']:
                        return 'High'
                    elif val >= threshold_dat['Medium']:
                        return 'Moderate'
                    else:
                        return 'Low'

                current_threshold = get_threshold(current_value)

                st.markdown(f"<p style='font-size: 22px;'>{heading}</p>", unsafe_allow_html=True)
                

                if not prior_row.empty and current_threshold is not None:
                    prior_value     = prior_row['value'].iloc[0]
                    prior_threshold = get_threshold(prior_value)
                    prior_date_fmt  = prior_row['date'].iloc[0].strftime('%B %d, %Y')
                    prior_color     = ACTIVITY_COLORS.get(prior_threshold, 'black')
                    cur_color       = ACTIVITY_COLORS.get(current_threshold, 'black')

                    if prior_threshold == current_threshold:
                        comparison = (
                            f"This is <b>consistent</b> with the same epiweek last year, "
                            f"when activity was also "
                            f"<b style='color:{prior_color};'>{prior_threshold}</b> "
                            f"({prior_date_fmt})."
                        )
                    else:
                        comparison = (
                            f"This compares to "
                            f"<b style='color:{prior_color};'>{prior_threshold}</b> "
                            f"activity during the same week last year ({prior_date_fmt})."
                        )
                    st.markdown(
                        f"<div style='padding: 2px 4px 8px 4px; color:#666; font-size:0.95rem;'>"
                        f"{comparison}</div>",
                        unsafe_allow_html=True
                    )

                  #  st.markdown(
                  #      f"<div style='padding: 2px 4px 8px 4px; color:#666; font-size:0.95rem;'>"
                  #      f"{comparison}</div>",
                  #      unsafe_allow_html=True
                  #  )


                   

                    



                fig = go.Figure()

                if not loc_thresholds.empty:
                    thresh = loc_thresholds.iloc[0]
 
                    y_max = obs_filtered['value'].max() * 1.1

                    for level, lower, upper in activity_levels:
                        fig.add_hrect(
                            y0=lower, y1=upper,
                            fillcolor=ACTIVITY_COLORS[level],
                            line_width=0,
                            opacity=0.7,
                            layer="below"
                           ) 

                    fig.update_layout(yaxis=dict(range=[0, y_max]))


                fig.add_trace(go.Scatter(
                    x=obs_filtered['date'],
                    y=obs_filtered['value'],
                    mode='lines+markers',
                    line=dict(color="#000000", width=2),
                    marker=dict(size=5, symbol='circle-open', color='black'),
                    hovertemplate='%{x|%b %d, %Y}<br>Value: %{y:,.0f}<extra></extra>'
                ))
                fig.update_layout(
                    xaxis_title="Date",
                    yaxis_title="Weekly Flu Hospitalizations",
                    height=450,
                    margin=dict(l=50, r=20, t=20, b=2),
                    showlegend=False,
                    yaxis=dict(range=[0, y_max], showgrid=False)
                )
                st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})


                legend_html = """
                <div style="display: flex; gap: 12px; justify-content: flex-end; margin-top: -10px; align-items: center;">
                    <span style="font-size: 12px; color: gray; font-weight: 600;">Activity Level:</span>
                    <span style="display: flex; align-items: center; gap: 4px;">
                        <span style="width: 12px; height: 12px; background: #7DD4C8; display: inline-block;"></span>
                        <span style="font-size: 12px; color: gray;">Low</span>
                    </span>
                    <span style="display: flex; align-items: center; gap: 4px;">
                        <span style="width: 12px; height: 12px; background: #3CAAA0; display: inline-block;"></span>
                        <span style="font-size: 12px; color: gray;">Moderate</span>
                    </span>
                    <span style="display: flex; align-items: center; gap: 4px;">
                        <span style="width: 12px; height: 12px; background: #2B7A8F; display: inline-block;"></span>
                        <span style="font-size: 12px; color: gray;">High</span>
                    </span>
                    <span style="display: flex; align-items: center; gap: 4px;">
                        <span style="width: 12px; height: 12px; background: #3D5A80; display: inline-block;"></span>
                        <span style="font-size: 12px; color: gray;">Very High</span>
                    </span>
                </div>
                """
                st.markdown(legend_html, unsafe_allow_html=True)

            else:
                st.warning("No observed data available for this location.")


            st.markdown("<p style='font-size: 15px; color: gray; margin-bottom: 2px;'>Date Range:</p>", unsafe_allow_html=True)
            range_options = ["Last 3 months", "Last 6 months", "Last year", "Last 2 years", "All data"]
            cols = st.columns(len(range_options))
            for col, label in zip(cols, range_options):
                with col:
                    if st.button(label, key=f"overview_range_{label}", use_container_width=True):
                        st.session_state.overview_date_range = label
                        st.rerun()
            
  


    with row1_col2:
        with st.container(border=True, height=700):
            st.markdown("<p style='font-size: 22px;'><b>What are Flu Activity Levels?</b></p>", unsafe_allow_html=True)
            st.markdown(
                """
                Activity levels describe the intensity of flu hospitalization activity in a given location, 
                based on thresholds derived from historical data.

                - <b style='color: #7DD4C8;'>Low</b> — Flu hospitalizations are below typical seasonal levels. Little to no widespread activity.
                - <b style='color: #3CAAA0;'>Moderate</b> — Flu activity is picking up. Hospitalizations are above baseline but within expected seasonal range.
                - <b style='color: #2B7A8F;'>High</b> — Elevated flu activity. Hospitalizations are significantly above typical levels.
                - <b style='color: #3D5A80;'>Very High</b> — Exceptional flu activity. Hospitalizations are at or near peak seasonal levels.

                Thresholds are location-specific and are calculated based on historical flu hospitalization patterns for each state and the United States overall.
                """,
                unsafe_allow_html=True
            )      

    # ── Replace your row2_col1 and row2_col2 blocks with this ──────────────────────

    # ── Outer layout: forecast card (60) + new container (40) ─────────────────────
    outer_left, outer_right = st.columns([4,2], gap="large")

    with outer_left:
        with st.container(border=True, height=600):

            # ── Load & filter activity level data ──────────────────────────────────
            act_df = pd.read_parquet('./data/activity_level_ensemble.pq')
            act_df['reference_date'] = pd.to_datetime(act_df['reference_date'])
            act_df = act_df[act_df['location'] == overview_location]
            act_plot = act_df[
                (act_df['reference_date'] == st.session_state.overview_ref_date) &
                (act_df['horizon'] == st.session_state.overview_horizon)
            ].copy()

            # ── Load & filter trend data ────────────────────────────────────────────
            cat_df = pd.read_parquet('./data/categorical_ensemble.pq')
            cat_df['reference_date'] = pd.to_datetime(cat_df['reference_date'])
            cat_df = cat_df[cat_df['location'] == overview_location]
            cat_plot = cat_df[
                (cat_df['reference_date'] == st.session_state.overview_ref_date) &
                (cat_df['horizon'] == st.session_state.overview_horizon)
            ].copy()

            # ── Derive headline values ──────────────────────────────────────────────
            act_max_idx = act_plot['value'].dropna().idxmax()
            act_level   = act_plot.loc[act_max_idx, 'output_type_id']
            act_prob    = act_plot.loc[act_max_idx, 'value']
            act_color   = ACTIVITY_COLORS.get(act_level, 'black')

            cat_max_idx = cat_plot['value'].dropna().idxmax()
            cat_label   = format_category(cat_plot.loc[cat_max_idx, 'output_type_id'])
            cat_prob    = cat_plot.loc[cat_max_idx, 'value']
            cat_color   = CATEGORY_COLORS.get(cat_label, 'black')
            if cat_label == 'Stable':
                cat_color = 'dimgray'

            # ── Headline ───────────────────────────────────────────────────────────
            st.markdown(
                f"""
                <div style="padding: 8px 4px 4px 4px;">
                    <span style="font-size:1.25rem; font-weight:600; color:#333;">
                        Forecast summary &nbsp;·&nbsp;
                    </span>
                    <span style="font-size:1.25rem; font-weight:700; color:{act_color};">
                        {act_level}
                    </span>
                    <span style="font-size:1.25rem; font-weight:600; color:#333;">
                        &nbsp;activity, trending&nbsp;
                    </span>
                    <span style="font-size:1.25rem; font-weight:700; color:{cat_color};">
                        {cat_label.lower()}
                    </span>
                </div>
                """,
                unsafe_allow_html=True
            )

            # ── Dynamic summary sentence ───────────────────────────────────────────
            horizon_weeks = st.session_state.overview_horizon + 1
            target_date   = (pd.Timestamp(st.session_state.overview_ref_date) +
                            pd.Timedelta(weeks=horizon_weeks)).strftime('%B %d, %Y')

            st.markdown(
                f"""
                <div style="padding: 2px 4px 8px 4px; color:#666; font-size:0.95rem;">
                    Over the next <b>{horizon_weeks} week{'s' if horizon_weeks > 1 else ''}</b>, 
                    flu hospitalizations in {'the ' if overview_location == 'US' else ''} <b>{overview_location}</b> are projected to 
                    <b style="color:{cat_color};">{cat_label.lower()}</b> — 
                    activity levels are forecast to stay <b style="color:{act_color};">{act_level}</b> 
                    through <b>{target_date}</b>.
                </div>
                """,
                unsafe_allow_html=True
            )


            # ── Controls row ───────────────────────────────────────────────────────
            ctrl1, ctrl2, _ = st.columns([1, 1, 2])
            with ctrl1:
                cat_dates = sorted(forecast_data['reference_date'].unique(), reverse=True)
                st.selectbox(
                    "Forecast Date",
                    cat_dates,
                    format_func=lambda x: pd.Timestamp(x).strftime('%Y-%m-%d'),
                    index=0,
                    key="overview_ref_date"
                )
            with ctrl2:
                horizon_labels = {0: "1 week ahead", 1: "2 weeks ahead",
                                2: "3 weeks ahead", 3: "4 weeks ahead"}
                st.selectbox(
                    "Forecast Horizon",
                    [0, 1, 2, 3],
                    index=3,
                    format_func=lambda x: horizon_labels[x],
                    key="overview_horizon"
                )

            st.divider()

            # ── Chart columns ──────────────────────────────────────────────────────
            left_col, mid_col, right_col = st.columns([10, 1, 10])

            # ── LEFT: Activity level donut ─────────────────────────────────────────
            with left_col:
                act_order = ['Low', 'Moderate', 'High', 'Very High']
                act_colors_map = {
                    'Low':       '#7DD4C8',
                    'Moderate':  '#3CAAA0',
                    'High':      '#2B7A8F',
                    'Very High': '#3D5A80',
                }

                act_plot = act_plot.set_index('output_type_id').reindex(act_order).reset_index()
                act_plot['value'] = act_plot['value'].fillna(0)

                slice_colors = [act_colors_map[r] for r in act_order]

                fig_act = go.Figure()
                fig_act.add_trace(go.Pie(
                    labels=act_order,
                    values=act_plot['value'],
                    hole=0.6,
                    marker=dict(colors=slice_colors, line=dict(color='white', width=2)),
                    textinfo='none',
                    hovertemplate='%{label}: %{percent}<extra></extra>',
                    sort=False,
                    direction='clockwise',
                ))

                fig_act.update_layout(
                    title=dict(text="Activity Level", font=dict(size=13, color='#555'), x=0),
                    annotations=[
                        dict(
                            text=f"<b>{act_level}</b><br>{act_prob:.0%}",
                            x=0.5, y=0.5,
                            font=dict(size=14, color=act_color),
                            showarrow=False
                        )
                    ],
                    showlegend=True,
                    legend=dict(
                        orientation='v',
                        x=1.0, y=0.5,
                        font=dict(size=11),
                        itemsizing='constant'
                    ),
                    height=260,
                    margin=dict(l=20, r=80, t=40, b=20),
                    plot_bgcolor='rgba(0,0,0,0)',
                    paper_bgcolor='rgba(0,0,0,0)',
                )

                st.plotly_chart(fig_act, use_container_width=True, config={'displayModeBar': False})

            # ── MIDDLE: Vertical divider ───────────────────────────────────────────
            with mid_col:
                st.markdown(
                    """
                    <div style="display:flex; justify-content:center; height:260px;
                                margin-top:40px;">
                        <div style="width:1px; background-color:#e0e0e0; height:100%;"></div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

            # ── RIGHT: Trend horizontal bars ───────────────────────────────────────
            with right_col:
                cat_order  = ['large_decrease', 'decrease', 'stable', 'increase', 'large_increase']
                cat_labels = ['Large Decrease', 'Decrease', 'Stable', 'Increase', 'Large Increase']
                cat_colors_map = {
                    'Large Decrease': '#006d77',
                    'Decrease':       '#83c5be',
                    'Stable':         '#aaaaaa',
                    'Increase':       '#e29578',
                    'Large Increase': '#bc4749',
                }

                cat_plot = cat_plot.set_index('output_type_id').reindex(cat_order).reset_index()
                cat_plot['value'] = cat_plot['value'].fillna(0)

                # Winner full opacity, others muted but still their own color
                bar_colors = []
                for lbl in cat_labels:
                    hex_col = cat_colors_map[lbl]
                    if lbl == cat_label:
                        bar_colors.append(hex_col)
                    else:
                        # Convert hex to rgba with reduced opacity
                        h = hex_col.lstrip('#')
                        r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
                        bar_colors.append(f'rgba({r},{g},{b},0.35)')

                fig_cat = go.Figure()
                fig_cat.add_trace(go.Bar(
                    y=cat_labels,
                    x=cat_plot['value'],
                    orientation='h',
                    marker_color=bar_colors,
                    marker_line_width=0,
                    text=[f"{v:.0%}" if v > 0 else "" for v in cat_plot['value']],
                    textposition='outside',
                    textfont=dict(size=11, color='#555'),
                    hovertemplate='%{y}: %{x:.1%}<extra></extra>',
                    showlegend=False
                ))

                fig_cat.update_layout(
                    title=dict(text="Trend Direction", font=dict(size=13, color='#555'), x=0),
                    xaxis=dict(
                        tickformat='.0%', range=[0, 1.15],
                        showgrid=False, zeroline=False, showticklabels=False
                    ),
                    yaxis=dict(
                        categoryorder='array', categoryarray=list(reversed(cat_labels)),
                        showgrid=False, tickfont=dict(size=12)
                    ),
                    height=260,
                    margin=dict(l=110, r=60, t=40, b=20),
                    plot_bgcolor='rgba(0,0,0,0)',
                    paper_bgcolor='rgba(0,0,0,0)',
                )

                st.plotly_chart(fig_cat, use_container_width=True, config={'displayModeBar': False})

    with outer_right:
        with st.container(border=True, height=600):
            st.markdown("#### Trend forecasts")
            st.markdown(
                """
                **Trend direction** describes how hospitalization levels are expected to change 
                over time, relative to the current period. Forecasts are expressed as probabilities across five categories:

                <ul style="line-height:2;">
                    <li><span style="color:#006d77; font-weight:700;">Large Decrease</span> — a substantial decline in activity</li>
                    <li><span style="color:#83c5be; font-weight:700;">Decrease</span> — a moderate decline in activity</li>
                    <li><span style="color:#aaaaaa; font-weight:700;">Stable</span> — little to no change expected</li>
                    <li><span style="color:#e29578; font-weight:700;">Increase</span> — a moderate rise in activity</li>
                    <li><span style="color:#bc4749; font-weight:700;">Large Increase</span> — a substantial rise in activity</li>
                </ul>
                """,
                unsafe_allow_html=True
            )

            st.markdown(
                """
                Probabilities collective uncertainty in these categories. The highlighted bar represents 
                the most likely outcome, but the full distribution should be considered 
                when interpreting the forecast.
                """
            )

        


st.divider()
st.markdown(
    f"""
    <div style="display: flex; align-items: center; gap: 30px; justify-content: flex-start;">
        {img_to_html("assets/northeastern-logo.png", width=150)}
        {img_to_html("assets/epistorm-logo.png", width=150)}
    </div>
    """,
    unsafe_allow_html=True,
)
