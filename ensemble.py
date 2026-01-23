import pandas as pd
import numpy as np
from typing import List, Tuple

def create_ensemble_method1(forecast_data):
    """
    Ensemble Method 1: Create ensemble forecasts from individual model predictions
    
    Parameters:
    -----------
    forecast_data : pd.DataFrame
        DataFrame containing all model forecasts with columns:
        - reference_date
        - location
        - horizon
        - target
        - target_end_date
        - output_type
        - output_type_id
        - value
        - model
    
    Returns:
    --------
    pd.DataFrame
        Ensemble forecasts in the same format as input (without 'model' column,
        as it will be added by the calling function)
    
    uses median across models for each quantile.
    """
    
    # Filter for quantile forecasts only
    quantile_data = forecast_data[(forecast_data['output_type'] == 'quantile') & (forecast_data['target']=='wk inc flu hosp')].copy() 
    
    # Group by all relevant columns except model and value
    grouping_cols = ['reference_date', 'location', 'horizon', 'target',  'target_end_date', 'output_type', 'output_type_id']
    
    # Calculate median across models for each quantile
    ensemble = quantile_data.groupby(grouping_cols, as_index=False)['value'].median()
    
    return ensemble


def create_categorical_ensemble(forecast_data):
    """
    Create ensemble categorical forecasts by averaging probabilities across models
    
    Parameters:
    -----------
    forecast_data : pd.DataFrame
        DataFrame containing all model forecasts with columns:
        - reference_date, location, horizon, target, target_end_date
        - output_type, output_type_id, value, model
    
    Returns:
    --------
    pd.DataFrame
        Ensemble categorical forecasts with normalized probabilities
    """
    
    # Filter for categorical forecasts
    df = forecast_data[
        (forecast_data['output_type'] == 'pmf') & 
        (forecast_data['target'] == 'wk flu hosp rate change')
    ].copy()
    
    if df.empty:
        return pd.DataFrame()
    
    # Convert date columns
    df['reference_date'] = pd.to_datetime(df['reference_date'])
    df['target_end_date'] = pd.to_datetime(df['target_end_date'])
    
    # Group by all dimensions except model and value
    group_cols = ['reference_date', 'location', 'horizon', 'target', 
                  'target_end_date', 'output_type', 'output_type_id']
    
    # Calculate mean probability across models
    ensemble = df.groupby(group_cols)['value'].mean().reset_index()
    
    # Normalize probabilities to sum to 1 for each (reference_date, location, horizon) group
    normalize_cols = ['reference_date', 'location', 'horizon']
    
    # Calculate sum of probabilities for each group
    prob_sums = ensemble.groupby(normalize_cols)['value'].transform('sum')
    
    # Normalize (handle division by zero)
    ensemble['value'] = np.where(
        prob_sums > 0,
        ensemble['value'] / prob_sums,
        0
    )
    
    # Add model identifier
    ensemble['model'] = 'Median Epistorm Ensemble'
    
    return ensemble


def create_ensemble_method2(forecast_data: pd.DataFrame) -> pd.DataFrame:
    """
    Optimized Ensemble Method 2: Create ensemble forecasts from individual model predictions
    
    Performance improvements:
    - Vectorized operations instead of loops where possible
    - Eliminated repeated DataFrame concatenations
    - Pre-allocated arrays for results
    - Cached repeated computations
    - Used groupby operations more efficiently
    
    Parameters:
    -----------
    forecast_data : pd.DataFrame
        DataFrame containing all model forecasts with columns:
        - reference_date, location, horizon, target, target_end_date
        - output_type, output_type_id, value, model
    
    Returns:
    --------
    pd.DataFrame
        Ensemble forecasts in the same format as input
    """
    
    # Filter once at the beginning
    df = forecast_data[
        (forecast_data['output_type'] == 'quantile') & 
        (forecast_data['target'] == 'wk inc flu hosp')
    ].copy()
    
    # Convert types once
    df['output_type_id'] = df['output_type_id'].astype(float)
    df['reference_date'] = pd.to_datetime(df['reference_date'])
    df['target_end_date'] = pd.to_datetime(df['target_end_date'])
    
    # Exclude specific locations upfront
    df = df[~df['location'].isin(['66', '72'])]
    
    # Group by date and location to process in batches
    grouped = df.groupby(['reference_date', 'location'])
    
    # Collect results in a list for efficient concatenation
    results_list = []
    
    for (date, location), group_data in grouped:
        try:
            ensemble_result = process_location_date(group_data, date, location)
            if ensemble_result is not None:
                results_list.append(ensemble_result)
        except Exception as e:
            print(f'Error processing {date}, location {location}: {e}')
            continue
    
    if not results_list:
        return pd.DataFrame()
    
    # Single concatenation at the end
    return pd.concat(results_list, ignore_index=True)


def process_location_date(group_data: pd.DataFrame, date, location) -> pd.DataFrame:
    """Process ensemble for a single location and date"""
    
    # Get unique models and horizons
    models = group_data['model'].unique()
    horizons = sorted(group_data['horizon'].unique())
    
    if len(models) == 0:
        return None
    
    # Pre-compute interpolation values for all horizons
    interp_data = {}
    
    for horizon in horizons:
        horizon_data = group_data[group_data['horizon'] == horizon]
        
        # Collect all values from all models for this horizon
        all_values = horizon_data['value'].values
        
        if len(all_values) == 0:
            continue
            
        interp_data[horizon] = all_values
    
    if not interp_data:
        return None
    
    # Process each model's quantiles
    quantile_results = []
    
    for model in models:
        model_data = group_data[group_data['model'] == model]
        
        for horizon in horizons:
            if horizon not in interp_data:
                continue
                
            horizon_model_data = model_data[model_data['horizon'] == horizon]
            
            if len(horizon_model_data) == 0:
                continue
            
            # Sort by quantile for interpolation
            horizon_model_data = horizon_model_data.sort_values('output_type_id')
            quantiles = horizon_model_data['output_type_id'].values
            values = horizon_model_data['value'].values
            
            # Interpolate
            interp_vals = interp_data[horizon]
            interp_quantiles = np.interp(interp_vals, values, quantiles)
            
            # Store results
            for val, quant in zip(interp_vals, interp_quantiles):
                quantile_results.append({
                    'horizon': horizon,
                    'xvalue': val,
                    'quantile_val': quant
                })
    
    if not quantile_results:
        return None
    
    # Convert to DataFrame and compute means
    quant_df = pd.DataFrame(quantile_results)
    avg_quantiles = quant_df.groupby(['horizon', 'xvalue'])['quantile_val'].mean().reset_index()
    
    # Get target quantiles from first model
    first_model = group_data[group_data['model'] == models[0]]
    
    # Build final results
    final_results = []
    
    for horizon in horizons:
        horizon_avg = avg_quantiles[avg_quantiles['horizon'] == horizon]
        
        if len(horizon_avg) == 0:
            continue
        
        # Get target quantiles and end date for this horizon
        horizon_first = first_model[first_model['horizon'] == horizon]
        
        if len(horizon_first) == 0:
            continue
            
        target_quantiles = sorted(horizon_first['output_type_id'].unique())
        target_end_date = horizon_first['target_end_date'].iloc[0]
        
        # Sort for interpolation
        horizon_avg = horizon_avg.sort_values('quantile_val')
        source_quantiles = horizon_avg['quantile_val'].values
        source_values = horizon_avg['xvalue'].values
        
        # Interpolate back to target quantiles
        final_values = np.interp(target_quantiles, source_quantiles, source_values)
        
        # Create result rows
        for quant, val in zip(target_quantiles, final_values):
            final_results.append({
                'output_type_id': quant,
                'value': val,
                'horizon': horizon,
                'target_end_date': target_end_date,
                'location': location,
                'reference_date': date
            })
    
    if not final_results:
        return None
    
    return pd.DataFrame(final_results)


# Additional optimization: Parallel processing option
def create_ensemble_method2_parallel(forecast_data: pd.DataFrame, n_jobs: int = -1) -> pd.DataFrame:
    """
    Parallel version using joblib for even faster processing
    Install with: pip install joblib
    
    Parameters:
    -----------
    forecast_data : pd.DataFrame
        Input forecast data
    n_jobs : int
        Number of parallel jobs (-1 for all CPUs)
    """
    from joblib import Parallel, delayed
    
    # Same preprocessing
    df = forecast_data[
        (forecast_data['output_type'] == 'quantile') & 
        (forecast_data['target'] == 'wk inc flu hosp')
    ].copy()
    
    df['output_type_id'] = df['output_type_id'].astype(float)
    df['reference_date'] = pd.to_datetime(df['reference_date'])
    df['target_end_date'] = pd.to_datetime(df['target_end_date'])
    df = df[~df['location'].isin(['66', '72'])]
    
    grouped = df.groupby(['reference_date', 'location'])
    
    # Process in parallel
    results = Parallel(n_jobs=n_jobs)(
        delayed(process_location_date)(group_data, date, location)
        for (date, location), group_data in grouped
    )
    
    # Filter out None results and concatenate
    results = [r for r in results if r is not None]
    
    if not results:
        return pd.DataFrame()
    
    return pd.concat(results, ignore_index=True)