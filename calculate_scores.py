"""
Script to calculate WIS and coverage scores for forecast models
Run by GitHub Actions weekly after data fetching
"""

import pandas as pd
import numpy as np
from datetime import datetime
from pathlib import Path

# ============================================================================
# ENSEMBLE CREATION FUNCTION
# ============================================================================

def create_ensemble_method1(forecast_data):
    """
    Ensemble Method 1: Create ensemble forecasts from individual model predictions
    
    Parameters:
    -----------
    forecast_data : pd.DataFrame
        DataFrame containing all model forecasts
    
    Returns:
    --------
    pd.DataFrame
        Ensemble forecasts using median across models for each quantile
    """
    
    # Filter for quantile forecasts only
    quantile_data = forecast_data[(forecast_data['output_type'] == 'quantile') & 
                                  (forecast_data['target'] == 'wk inc flu hosp')].copy() 
    
    # Group by all relevant columns except model and value
    grouping_cols = ['reference_date', 'location', 'horizon', 'target', 
                     'target_end_date', 'output_type', 'output_type_id']
    
    # Calculate median across models for each quantile
    ensemble = quantile_data.groupby(grouping_cols, as_index=False)['value'].median()
    
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
        (forecast_data['target'] == 'wk inc flu hosp') &
        (forecast_data['model'] != 'FluSight-ensemble')
    ].copy()
    
    # Convert types once
    df['output_type_id'] = df['output_type_id'].astype(float)
    df['reference_date'] = pd.to_datetime(df['reference_date'])
    df['target_end_date'] = pd.to_datetime(df['target_end_date'])
    
    # Exclude specific locations upfront
    df = df[~df['location'].isin(['66',])]
    
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
    

    # Add model identifier
    ensemble = pd.concat(results_list, ignore_index=True)
    ensemble['model'] = 'LOP Epistorm Ensemble'
    ensemble['output_type_id'] = ensemble['output_type_id'].astype(str)
    ensemble['output_type'] = 'quantile'
    ensemble['target'] = 'wk inc flu hosp'
    
    # Single concatenation at the end
    return ensemble


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




# ============================================================================
# SCORING FUNCTIONS
# ============================================================================

class scoring_functions:
    """
    Compute scores for forecasts
    
    Methods:
    --------
    timestamp_wis : Computes the WIS at each time point
    interval_score : Computes the interval score (part of WIS)
    coverage : Calculates coverage for a prediction interval
    get_all_coverages : Calculates coverage for all prediction intervals
    get_wis_scores : Calculate WIS for all models, horizons, locations, dates
    calculate_forecast_coverage : Calculate coverage for all forecasts
    """
      
    def timestamp_wis(self, observations, predsfilt, 
                     interval_ranges=[10,20,30,40,50,60,70,80,90,95,98]):
        """
        Calculate weighted interval score at a timestamp
        
        Parameters:
        -----------
        observations : DataFrame
            Observations across time
        predsfilt : DataFrame
            Predictions (quantile and point) across time
        interval_ranges : list
            Percentage covered by each interval
            
        Returns:
        --------
        DataFrame with WIS scores
        """
        # Get all quantiles
        quantiles = np.array(predsfilt.sort_values(by='output_type_id').output_type_id)

        qs = []
        # Get the values of the quantiles from the forecast 
        for q in quantiles:
            df = predsfilt[predsfilt.output_type_id == q].sort_values(by='target_end_date')
            val = np.array(df.value)
            qs.append(val)

        Q = np.array(qs)  # quantiles of forecast array
        y = np.array(observations.value)  # observations array

        if quantiles[11] != 0.5:
            print(f'Warning: quantiles[11] = {quantiles[11]}, not median!')

        # Calculate WIS
        WIS = np.zeros(len(y))

        # Calculate interval score for each prediction interval
        for i in range(len(quantiles) // 2):
            interval_range = 100 * (quantiles[-i-1] - quantiles[i])
            alpha = 1 - (quantiles[-i-1] - quantiles[i])
            IS = self.interval_score(y, Q[i], Q[-i-1], interval_range)
            WIS += IS['interval_score'] * alpha / 2
        
        WIS += 0.5 * np.abs(Q[11] - y)

        WISlist = np.array(WIS) / (len(interval_ranges) + 0.5)

        # Save scores in a dataframe
        df = pd.DataFrame({
            'Model': predsfilt.Model.unique(),
            'location': predsfilt.location.unique(),
            'horizon': predsfilt.horizon.unique(),
            'reference_date': predsfilt.reference_date.unique(),
            'target_end_date': predsfilt.target_end_date.unique(),
            'wis': WISlist[0]
        }, index=[0])

        return df

    def interval_score(self, observation, lower, upper, interval_range):
        """
        Calculate interval score
        
        Parameters:
        -----------
        observation : array_like
            Vector of observations
        lower : array_like
            Prediction for the lower quantile
        upper : array_like
            Prediction for the upper quantile
        interval_range : int
            Percentage covered by the interval
            
        Returns:
        --------
        dict with interval scores and components
        """
        if len(lower) != len(upper) or len(lower) != len(observation):
            raise ValueError("vector shape mismatch")
        if interval_range > 100 or interval_range < 0:
            raise ValueError("interval range should be between 0 and 100")

        obs, l, u = np.array(observation), np.array(lower), np.array(upper)
        alpha = 1 - interval_range / 100

        # Get interval score components
        dispersion = u - l
        underprediction = (2 / alpha) * (l - obs) * (obs < l)
        overprediction = (2 / alpha) * (obs - u) * (obs > u)
        score = dispersion + underprediction + overprediction

        out = {
            'interval_score': score,
            'dispersion': dispersion,
            'underprediction': underprediction,
            'overprediction': overprediction
        }

        return out

    def coverage(self, observation, lower, upper):
        """
        Calculate fraction of observations within lower and upper bounds
        
        Parameters:
        -----------
        observation : array_like
            Vector of observations
        lower : array_like
            Prediction for the lower quantile
        upper : array_like
            Prediction for the upper quantile
            
        Returns:
        --------
        float: Fraction of observations within bounds
        """
        if len(lower) != len(upper) or len(lower) != len(observation):
            raise ValueError("vector shape mismatch")

        obs, l, u = np.array(observation), np.array(lower), np.array(upper)
        return np.mean(np.logical_and(obs >= l, obs <= u))

    def get_all_coverages(self, observations, predictions, 
                         interval_ranges=[10,20,30,40,50,60,70,80,90,95,98]):
        """
        Get coverages for all prediction intervals
        
        Parameters:
        -----------
        observations : DataFrame
            Surveillance data
        predictions : DataFrame
            Forecasts to score
        interval_ranges : list
            Prediction intervals to calculate coverage for
            
        Returns:
        --------
        dict with coverage for each interval
        """
        out = dict()
        for interval_range in interval_ranges:
            q_low = 0.5 - interval_range / 200
            q_upp = 0.5 + interval_range / 200
            cov = self.coverage(
                observations.value,
                predictions[predictions.output_type_id == round(q_low, 3)].value,
                predictions[predictions.output_type_id == round(q_upp, 3)].value
            )
            out[f'{interval_range}_cov'] = cov

        return out
    
    def get_wis_scores(self, predsall, surv, models, dates, save_location=False):
        """
        Calculate WIS for each model, horizon, location, and date
        
        Parameters:
        -----------
        predsall : DataFrame
            All forecasts
        surv : DataFrame
            Surveillance data
        models : list
            Models to evaluate
        dates : list
            Dates to evaluate
        save_location : Path or False
            If not False, saves WIS scores to given Path
            
        Returns:
        --------
        DataFrame with WIS scores
        """
        dfwis = pd.DataFrame()
        results = []
        
        surv= surv.copy()
        surv['date'] = pd.to_datetime(surv['date'])
        for horizon in [0, 1, 2, 3]:
            for model in models:
                for date in dates: 
                    for location in predsall.location.unique():
                        # Filter by horizon, model and submission date
                        predsfilt = predsall[
                            (predsall.horizon == horizon) & 
                            (predsall.Model == model) & 
                            (predsall.reference_date == date) & 
                            (predsall.target_end_date <= surv.date.max()) &
                            (predsall.location == location)
                        ]

                        if len(predsfilt) == 0:
                            continue

                        # Filter date and location
                        observations = surv[
                            (surv['date'] == predsfilt.target_end_date.unique()[0]) &
                            (surv['location'] == location)
                        ]

                        if len(observations) == 0:
                            continue

                        out = self.timestamp_wis(observations, predsfilt)
                        results.append(out)
                        #dfwis = pd.concat([dfwis, out])
        dfwis = pd.concat(results, ignore_index=True)

        if save_location:
            dfwis.to_pickle(f'{save_location}fluforecast_timestamp_wis_{datetime.today().date()}.pkl')

        return dfwis
    
    def calculate_forecast_coverage(self, predsall, surv, models, dates, 
                                    save_location=False):
        """
        Calculate coverage for each model, horizon, location, and date
        
        Parameters:
        -----------
        predsall : DataFrame
            All forecasts
        surv : DataFrame
            Surveillance data
        models : list
            Models to evaluate
        dates : list
            Dates to evaluate
        save_location : Path or False
            If not False, saves coverage scores to given Path
            
        Returns:
        --------
        DataFrame with coverage values
        """
        #dfcoverage = pd.DataFrame()
        results = []

        surv = surv.copy()
        surv['date'] = pd.to_datetime(surv['date'])

        for date in dates:
            for model in models:
                for location in predsall.location.unique():
                    for horizon in [0, 1, 2, 3]:
                        # Filter by model and submission date
                        pred = predsall[
                            (predsall.Model == model) & 
                            (predsall.reference_date == date) &
                            (predsall.horizon == horizon) & 
                            (predsall.target_end_date <= surv.date.max()) &
                            (predsall.location == location)
                        ]
                         
                        if len(pred) == 0:
                            continue

                        # Filter date and location
                        observations = surv[
                            surv['date'] == pred.target_end_date.unique()[0]
                        ]
                        observations = observations[
                            observations['location'] == location
                        ]

                        if len(observations) == 0:
                            continue

                        # Calculate coverage
                        out = self.get_all_coverages(observations, pred)
                        out = pd.DataFrame(out, index=[0])
                        out['Model'] = model
                        out['reference_date'] = date
                        out['target_end_date'] = pred.target_end_date.unique()[0]
                        out['horizon'] = horizon
                        out['location'] = location

                        #dfcoverage = pd.concat([dfcoverage, out])
                        results.append(out)
        dfcoverage = pd.concat(results, ignore_index=True)

        #dfcoverage = dfcoverage.reset_index().drop(columns=['index'])

        if save_location:
            dfcoverage.to_pickle(f'{save_location}fluforecast_coverage_{datetime.today().date()}.pkl')

        return dfcoverage


# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("Starting score calculation process...")
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)
    
    DATA_DIR = Path("data")
    
    # Load all the forecasts
    print("\n1. Loading forecast data...")
    all_forecasts = pd.read_parquet(DATA_DIR / "all_forecasts.parquet")
    baseline_forecasts = pd.read_parquet(DATA_DIR / "baseline_forecasts.parquet")
    observed_data = pd.read_csv(DATA_DIR / "observed_data.csv")
    print(f"   ✓ Loaded {len(all_forecasts)} forecast rows")
    print(f"   ✓ Loaded {len(baseline_forecasts)} baseline rows")
    print(f"   ✓ Loaded {len(observed_data)} observation rows")
    
    # Create ensemble
    print("\n2. Creating ensemble forecasts...")
    ensemble1 = create_ensemble_method1(all_forecasts)
    ensemble1['model'] = 'Median Epistorm Ensemble'
    print(f"   ✓ Created ensemble with {len(ensemble1)} rows")

    # Create ensemble
    ensemble2 = create_ensemble_method2(all_forecasts)
    ensemble2['model'] = 'LOP Epistorm Ensemble'
    print(f"   ✓ Created LOP ensemble with {len(ensemble2)} rows")
    
    # Combine all forecasts
    print("\n3. Combining all forecasts...")
    all_forecasts = pd.concat([all_forecasts, baseline_forecasts, ensemble1, ensemble2], 
                              ignore_index=True)
    print(f"   ✓ Combined total: {len(all_forecasts)} rows")
    
    # Prepare data for scoring
    print("\n4. Preparing data for scoring...")
    predsall = all_forecasts[all_forecasts.output_type == 'quantile']
    predsall['target_end_date'] = pd.to_datetime(predsall['target_end_date'])
    predsall['output_type_id'] = predsall["output_type_id"].astype(float)
    predsall = predsall[predsall.target == 'wk inc flu hosp']
    predsall = predsall.rename(columns={'model': 'Model'})
    print(f"   ✓ Prepared {len(predsall)} quantile predictions")
    
    # Initialize scoring
    scoring = scoring_functions()
    
    # Calculate WIS
    print("\n5. Calculating WIS scores...")
    print("   This may take several minutes...")
    dfwis = scoring.get_wis_scores(
        predsall, 
        observed_data, 
        models=predsall.Model.unique(), 
        dates=predsall.reference_date.unique(),
        save_location=False
    )
    print(f"   ✓ Calculated WIS for {len(dfwis)} forecast-observation pairs")
    
    # Compute WIS ratio
    print("\n6. Computing WIS ratios...")
    baseline = dfwis[dfwis.Model == 'FluSight-baseline'] 
    baseline = baseline.rename(columns={'wis': 'wis_baseline', 'Model': 'baseline'})
    dfwis_test = dfwis[dfwis.Model != 'FluSight-baseline']
    
    dfwis_ratio = pd.merge(
        dfwis_test, 
        baseline, 
        how='inner', 
        on=['location', 'target_end_date', 'horizon', 'reference_date']
    )
    dfwis_ratio['wis_ratio'] = dfwis_ratio['wis'] / dfwis_ratio['wis_baseline']
    
    # Save WIS ratio
    output_file = './data/wis_ratio_epistorm_models_2526.pq'
    dfwis_ratio.to_parquet(output_file)
    print(f"   ✓ Saved WIS ratios to {output_file}")
    print(f"   ✓ Total WIS ratio records: {len(dfwis_ratio)}")
    
    # Calculate coverage
    print("\n7. Calculating coverage scores...")
    print("   This may take several minutes...")
    dfcoverage = scoring.calculate_forecast_coverage(
        predsall, 
        observed_data, 
        models=predsall.Model.unique(), 
        dates=predsall.reference_date.unique(),
        save_location=False
    )
    
    # Save coverage
    output_file = './data/coverage_epistorm_models_2526.pq'
    dfcoverage.to_parquet(output_file)
    print(f"   ✓ Saved coverage scores to {output_file}")
    print(f"   ✓ Total coverage records: {len(dfcoverage)}")
    
    print("\n" + "=" * 60)
    print("Score calculation completed successfully!")
    print("=" * 60)