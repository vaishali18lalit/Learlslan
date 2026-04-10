"""Daft.ie Property Data Client — live rental & sale listings via daftlistings."""
import pandas as pd
import numpy as np
import streamlit as st
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import IRISH_COUNTIES


# ── County → daftlistings Location mapping ─────────────────────
def _build_location_map():
    """Build mapping from county name to daftlistings Location enum."""
    from daftlistings import Location

    # County-level Location enum values (e.g. Location.DUBLIN, Location.CORK)
    mapping = {}
    for county in IRISH_COUNTIES:
        enum_name = county.upper()
        if hasattr(Location, enum_name):
            mapping[county] = getattr(Location, enum_name)
    return mapping


COUNTY_LOCATION_MAP = None  # Lazy init


def _get_location_map():
    global COUNTY_LOCATION_MAP
    if COUNTY_LOCATION_MAP is None:
        COUNTY_LOCATION_MAP = _build_location_map()
    return COUNTY_LOCATION_MAP


# ── Core Fetch Functions ───────────────────────────────────────
@st.cache_data(ttl=900, show_spinner="Fetching live Daft.ie listings...")
def fetch_county_listings(county: str, search_type: str = "rent", max_pages: int = 2) -> pd.DataFrame:
    """
    Fetch live property listings from Daft.ie for a given county.
    
    Args:
        county: Irish county name (e.g., "Dublin")
        search_type: "rent" or "sale"
        max_pages: max pages to scrape (each ~20 listings)
    
    Returns:
        DataFrame with columns: title, price, bedrooms, bathrooms, property_type, daft_link, county, search_type
    """
    loc_map = _get_location_map()
    if county not in loc_map:
        return _empty_listings_df()

    try:
        from daftlistings import Daft, SearchType

        daft = Daft()
        daft.set_location(loc_map[county])

        if search_type == "sale":
            daft.set_search_type(SearchType.RESIDENTIAL_SALE)
        else:
            daft.set_search_type(SearchType.RESIDENTIAL_RENT)

        listings = daft.search(max_pages=max_pages)

        rows = []
        for listing in listings:
            try:
                price_raw = listing.price
                price_num = _parse_price(price_raw)
                rows.append({
                    "title": listing.title or "N/A",
                    "price": price_num,
                    "price_display": price_raw or "N/A",
                    "bedrooms": _safe_int(getattr(listing, "bedrooms", None)),
                    "bathrooms": _safe_int(getattr(listing, "bathrooms", None)),
                    "property_type": _extract_property_type(listing.title or ""),
                    "daft_link": listing.daft_link or "",
                    "county": county,
                    "search_type": search_type,
                })
            except Exception:
                continue

        if not rows:
            return _empty_listings_df()

        return pd.DataFrame(rows)

    except Exception as e:
        print(f"Daft.ie fetch error for {county} ({search_type}): {e}")
        return _empty_listings_df()


@st.cache_data(ttl=900, show_spinner="Analysing market data...")
def get_county_market_summary(county: str) -> dict:
    """
    Aggregate live Daft.ie data into a market summary for a county.
    Returns dict with rental and sale metrics.
    """
    rent_df = fetch_county_listings(county, search_type="rent")
    sale_df = fetch_county_listings(county, search_type="sale")

    summary = {
        "county": county,
        # Rental metrics
        "rental_listing_count": 0,
        "rental_median": 0,
        "rental_mean": 0,
        "rental_min": 0,
        "rental_max": 0,
        "rental_price_per_bedroom": 0,
        "rental_prices": [],
        "rental_types": {},
        # Sale metrics
        "sale_listing_count": 0,
        "sale_median": 0,
        "sale_mean": 0,
        "sale_min": 0,
        "sale_max": 0,
        "sale_prices": [],
        "sale_types": {},
        # Derived
        "has_live_data": False,
    }

    # Process rental data
    if len(rent_df) > 0 and rent_df["price"].sum() > 0:
        valid_rent = rent_df[rent_df["price"] > 0]
        if len(valid_rent) > 0:
            summary["rental_listing_count"] = len(valid_rent)
            summary["rental_median"] = valid_rent["price"].median()
            summary["rental_mean"] = valid_rent["price"].mean()
            summary["rental_min"] = valid_rent["price"].min()
            summary["rental_max"] = valid_rent["price"].max()
            summary["rental_prices"] = valid_rent["price"].tolist()
            summary["rental_types"] = valid_rent["property_type"].value_counts().to_dict()
            avg_beds = valid_rent["bedrooms"].replace(0, np.nan).mean()
            if avg_beds and avg_beds > 0:
                summary["rental_price_per_bedroom"] = summary["rental_median"] / avg_beds
            summary["has_live_data"] = True

    # Process sale data
    if len(sale_df) > 0 and sale_df["price"].sum() > 0:
        valid_sale = sale_df[sale_df["price"] > 0]
        if len(valid_sale) > 0:
            summary["sale_listing_count"] = len(valid_sale)
            summary["sale_median"] = valid_sale["price"].median()
            summary["sale_mean"] = valid_sale["price"].mean()
            summary["sale_min"] = valid_sale["price"].min()
            summary["sale_max"] = valid_sale["price"].max()
            summary["sale_prices"] = valid_sale["price"].tolist()
            summary["sale_types"] = valid_sale["property_type"].value_counts().to_dict()
            summary["has_live_data"] = True

    return summary


def get_national_summary(counties: list = None) -> dict:
    """Get aggregated national-level market stats from all counties."""
    if counties is None:
        counties = IRISH_COUNTIES

    all_rental_prices = []
    all_sale_prices = []
    total_rental_listings = 0
    total_sale_listings = 0

    for county in counties:
        summary = get_county_market_summary(county)
        all_rental_prices.extend(summary.get("rental_prices", []))
        all_sale_prices.extend(summary.get("sale_prices", []))
        total_rental_listings += summary.get("rental_listing_count", 0)
        total_sale_listings += summary.get("sale_listing_count", 0)

    return {
        "national_rental_median": np.median(all_rental_prices) if all_rental_prices else 0,
        "national_sale_median": np.median(all_sale_prices) if all_sale_prices else 0,
        "total_rental_listings": total_rental_listings,
        "total_sale_listings": total_sale_listings,
    }


# ── Helpers ────────────────────────────────────────────────────
def _parse_price(price_str) -> float:
    """Parse price string to float."""
    if not price_str:
        return 0.0
    try:
        cleaned = str(price_str).replace("€", "").replace(",", "").replace(" ", "")
        # Handle "per month" / "per week" etc
        cleaned = cleaned.split("per")[0].strip()
        cleaned = cleaned.split("/")[0].strip()
        # Handle price ranges like "1,200 - 1,500"
        if "-" in cleaned:
            parts = cleaned.split("-")
            return float(parts[0].strip())
        return float(cleaned)
    except (ValueError, IndexError):
        return 0.0


def _safe_int(val) -> int:
    """Safely convert to int."""
    try:
        if val is None:
            return 0
        return int(val)
    except (ValueError, TypeError):
        return 0


def _extract_property_type(title: str) -> str:
    """Infer property type from listing title."""
    title_lower = title.lower()
    if "apartment" in title_lower or "apt" in title_lower:
        return "Apartment"
    elif "house" in title_lower or "semi-d" in title_lower or "detached" in title_lower or "terraced" in title_lower:
        return "House"
    elif "studio" in title_lower:
        return "Studio"
    elif "duplex" in title_lower:
        return "Duplex"
    elif "bungalow" in title_lower:
        return "Bungalow"
    elif "penthouse" in title_lower:
        return "Penthouse"
    else:
        return "Other"


def _empty_listings_df() -> pd.DataFrame:
    """Return empty DataFrame with expected columns."""
    return pd.DataFrame(columns=[
        "title", "price", "price_display", "bedrooms", "bathrooms",
        "property_type", "daft_link", "county", "search_type"
    ])
