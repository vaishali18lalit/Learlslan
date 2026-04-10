"""Spatial Harmonizer - joins all datasets on county or ED as primary key."""
import pandas as pd
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from ingestion.cso_api import load_cso_data, get_latest_cso, load_cso_ed_data, get_latest_cso_ed
from ingestion.tii_scraper import load_tii_data, get_latest_tii, load_tii_ed_data, get_latest_tii_ed
from ingestion.seai_loader import load_seai_data, load_seai_ed_data
from ingestion.housing_loader import load_housing_data, get_latest_housing, load_housing_ed_data, get_latest_housing_ed


# ══════════════════════════════════════════════════════════════
# COUNTY-LEVEL HARMONIZATION (backward compatible)
# ══════════════════════════════════════════════════════════════

def harmonize_data() -> pd.DataFrame:
    """
    Join all datasets on county, producing one row per county
    with the latest snapshot of all metrics.
    """
    cso_df = load_cso_data()
    tii_df = load_tii_data()
    seai_df = load_seai_data()
    rtb_df = load_housing_data()

    cso_latest = get_latest_cso(cso_df)
    tii_latest = get_latest_tii(tii_df)
    rtb_latest = get_latest_housing(rtb_df)

    merged = cso_latest.merge(tii_latest, on="county", how="outer", suffixes=("", "_tii"))
    merged = merged.merge(seai_df, on="county", how="outer")
    merged = merged.merge(rtb_latest, on="county", how="outer", suffixes=("", "_rtb"))

    month_cols = [c for c in merged.columns if c.startswith("month")]
    if len(month_cols) > 1:
        merged["month"] = merged[month_cols[0]]
        merged.drop(columns=[c for c in month_cols if c != "month"], inplace=True, errors="ignore")

    numeric_cols = merged.select_dtypes(include="number").columns
    for col in numeric_cols:
        merged[col].fillna(merged[col].median(), inplace=True)

    return merged.reset_index(drop=True)


def get_time_series_data() -> dict:
    """
    Get full time series data for each county.
    Returns dict: {county: DataFrame with monthly metrics}
    """
    cso_df = load_cso_data()
    tii_df = load_tii_data()
    rtb_df = load_housing_data()

    ts = cso_df.merge(tii_df, on=["county", "month"], how="outer")
    ts = ts.merge(rtb_df, on=["county", "month"], how="outer")

    county_ts = {}
    for county in ts["county"].unique():
        county_data = ts[ts["county"] == county].sort_values("month").reset_index(drop=True)
        county_ts[county] = county_data

    return county_ts


# ══════════════════════════════════════════════════════════════
# ELECTORAL DIVISION-LEVEL HARMONIZATION (new)
# ══════════════════════════════════════════════════════════════

def harmonize_ed_data() -> pd.DataFrame:
    """
    Join all ED-level datasets on ed_id, producing one row per Electoral Division
    with the latest snapshot of all metrics.
    """
    cso_df = load_cso_ed_data()
    tii_df = load_tii_ed_data()
    seai_df = load_seai_ed_data()
    rtb_df = load_housing_ed_data()

    cso_latest = get_latest_cso_ed(cso_df)
    tii_latest = get_latest_tii_ed(tii_df)
    rtb_latest = get_latest_housing_ed(rtb_df)

    # Join on ed_id (primary key for ED level)
    merged = cso_latest.merge(
        tii_latest, on=["ed_id", "ed_name", "county"], how="outer", suffixes=("", "_tii")
    )
    merged = merged.merge(
        seai_df, on=["ed_id", "ed_name", "county"], how="outer"
    )
    merged = merged.merge(
        rtb_latest, on=["ed_id", "ed_name", "county"], how="outer", suffixes=("", "_rtb")
    )

    # Clean up duplicate month columns
    month_cols = [c for c in merged.columns if c.startswith("month")]
    if len(month_cols) > 1:
        merged["month"] = merged[month_cols[0]]
        merged.drop(columns=[c for c in month_cols if c != "month"], inplace=True, errors="ignore")

    # Fill missing numeric values with median
    numeric_cols = merged.select_dtypes(include="number").columns
    for col in numeric_cols:
        merged[col].fillna(merged[col].median(), inplace=True)

    return merged.reset_index(drop=True)


def get_ed_time_series_data() -> dict:
    """
    Get full time series data for each ED.
    Returns dict: {ed_id: DataFrame with monthly metrics}
    """
    cso_df = load_cso_ed_data()
    tii_df = load_tii_ed_data()
    rtb_df = load_housing_ed_data()

    ts = cso_df.merge(tii_df, on=["ed_id", "ed_name", "county", "month"], how="outer")
    ts = ts.merge(rtb_df, on=["ed_id", "ed_name", "county", "month"], how="outer")

    ed_ts = {}
    for ed_id in ts["ed_id"].unique():
        ed_data = ts[ts["ed_id"] == ed_id].sort_values("month").reset_index(drop=True)
        ed_ts[ed_id] = ed_data

    return ed_ts


def aggregate_to_county(ed_df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate ED-level data to county-level by taking the mean of each numeric column.
    Useful for rolling ED data up to county summaries.
    """
    numeric_cols = ed_df.select_dtypes(include="number").columns.tolist()
    grouped = ed_df.groupby("county")[numeric_cols].mean().reset_index()
    return grouped


def get_county_ed_summary(ed_df: pd.DataFrame, county: str) -> pd.DataFrame:
    """
    Get all EDs within a specific county, sorted by risk_score descending.
    """
    county_eds = ed_df[ed_df["county"] == county].copy()
    if "risk_score" in county_eds.columns:
        county_eds = county_eds.sort_values("risk_score", ascending=False)
    return county_eds.reset_index(drop=True)
