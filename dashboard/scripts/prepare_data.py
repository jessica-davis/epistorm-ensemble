"""Convert source data files to dashboard-ready JSON.

Run from project root:
    python dashboard/scripts/prepare_data.py

Ensemble data is pivoted so each row contains all quantile values as columns,
reducing JSON size from ~32 MB to ~3 MB total.
"""
import pandas as pd
import urllib.request
from pathlib import Path

OUTPUT_DIR = Path("dashboard/data")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def to_json(df, filename, orient="records", date_format="iso"):
    path = OUTPUT_DIR / filename
    df.to_json(path, orient=orient, date_format=date_format, indent=None)
    size_mb = path.stat().st_size / (1024 * 1024)
    print(f"  -> {path} ({len(df)} rows, {size_mb:.1f} MB)")


def pivot_quantile(df):
    """Pivot quantile data: one row per (ref_date, target_end_date, location, horizon)
    with quantile values as columns (0.01, 0.025, ..., 0.99)."""
    quantile_df = df[df["output_type"] == "quantile"].copy()
    pivoted = quantile_df.pivot_table(
        index=["reference_date", "target_end_date", "location", "horizon"],
        columns="output_type_id",
        values="value",
        aggfunc="first",
    ).reset_index()
    pivoted.columns.name = None
    # Round numeric columns to 1 decimal
    num_cols = [c for c in pivoted.columns if c not in
                ["reference_date", "target_end_date", "location", "horizon"]]
    pivoted[num_cols] = pivoted[num_cols].round(1)
    return pivoted


# --- Observed data ---
print("Processing observed data...")
obs = pd.read_csv("data/observed_data.csv")
obs["date"] = pd.to_datetime(obs["date"]).dt.strftime("%Y-%m-%d")
obs["value"] = obs["value"].round(1)
obs["weekly_rate"] = obs["weekly_rate"].round(4)
to_json(obs, "observed.json")

# --- Ensemble quantile (Median) — pivoted ---
print("Processing median ensemble (pivoted)...")
ens = pd.read_parquet("data/quantile_ensemble.pq")
ens["reference_date"] = pd.to_datetime(ens["reference_date"]).dt.strftime("%Y-%m-%d")
ens["target_end_date"] = pd.to_datetime(ens["target_end_date"]).dt.strftime("%Y-%m-%d")
ens_pivot = pivot_quantile(ens)
to_json(ens_pivot, "ensemble.json")

# --- LOP Ensemble — pivoted ---
print("Processing LOP ensemble (pivoted)...")
lop = pd.read_parquet("data/quantile_ensemble_LOP.pq")
lop["reference_date"] = pd.to_datetime(lop["reference_date"]).dt.strftime("%Y-%m-%d")
lop["target_end_date"] = pd.to_datetime(lop["target_end_date"]).dt.strftime("%Y-%m-%d")
lop_pivot = pivot_quantile(lop)
to_json(lop_pivot, "ensemble_lop.json")

# --- Categorical ensemble ---
print("Processing categorical ensemble...")
cat = pd.read_parquet("data/categorical_ensemble.pq")
cat["reference_date"] = pd.to_datetime(cat["reference_date"]).dt.strftime("%Y-%m-%d")
if "Model" in cat.columns and "model" not in cat.columns:
    cat = cat.rename(columns={"Model": "model"})
to_json(cat, "categorical.json")

# --- Activity level ensemble ---
print("Processing activity levels...")
act = pd.read_parquet("data/activity_level_ensemble.pq")
act["reference_date"] = pd.to_datetime(act["reference_date"]).dt.strftime("%Y-%m-%d")
to_json(act, "activity_levels.json")

# --- Thresholds ---
print("Processing thresholds...")
thresh = pd.read_csv("data/threshold_levels.csv")
to_json(thresh, "thresholds.json")

# --- WIS ratio ---
print("Processing WIS ratio...")
wis = pd.read_parquet("data/wis_ratio_epistorm_models_2526.pq")
wis["target_end_date"] = pd.to_datetime(wis["target_end_date"]).dt.strftime("%Y-%m-%d")
wis["reference_date"] = pd.to_datetime(wis["reference_date"]).dt.strftime("%Y-%m-%d")
wis["location"] = wis["location"].astype(str).str.zfill(2)
if "Model" in wis.columns and "model" not in wis.columns:
    wis = wis.rename(columns={"Model": "model"})
to_json(wis, "wis_ratio.json")

# --- Coverage ---
print("Processing coverage...")
cov = pd.read_parquet("data/coverage_epistorm_models_2526.pq")
cov["target_end_date"] = pd.to_datetime(cov["target_end_date"]).dt.strftime("%Y-%m-%d")
cov["reference_date"] = pd.to_datetime(cov["reference_date"]).dt.strftime("%Y-%m-%d")
cov["location"] = cov["location"].astype(str).str.zfill(2)
if "Model" in cov.columns and "model" not in cov.columns:
    cov = cov.rename(columns={"Model": "model"})
to_json(cov, "coverage.json")

# --- Locations ---
print("Processing locations...")
loc = pd.read_csv("locations.csv")
to_json(loc, "locations.json")

# --- US States TopoJSON ---
topo_path = OUTPUT_DIR / "us-states.json"
if not topo_path.exists():
    print("Downloading US states TopoJSON...")
    url = "https://cdn.jsdelivr.net/npm/us-atlas@3/states-10m.json"
    urllib.request.urlretrieve(url, topo_path)
    size_mb = topo_path.stat().st_size / (1024 * 1024)
    print(f"  -> {topo_path} ({size_mb:.1f} MB)")
else:
    print(f"  -> {topo_path} (already exists)")

print("\nDone.")
