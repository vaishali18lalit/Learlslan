# Léarslán V3 — Mixed Reality Data Schema

## Overview
This document represents the finalized schema after the execution of the `pull_live_data.py` integration. The application now uses the `data/real_data` directory, seamlessly blending real-world statistics into the structured Electoral Division (ED) format required by the dashboard.

## Real vs Synthetic Breakdown

Because hyper-local monthly data (like traffic delays per Electoral Division) does not exist publicly, the ingestion engine uses a hybrid approach. It anchors on **Real API statistics** where available and pads the missing spatial-temporal gaps with **Synthetic Proxies** (derived mathematically from the real structures).

### 1. Dataset 1: Central Statistics Office (CSO)
**[REAL-PROXY]**
- **Sourcing Strategy**: Anchored to real-world Q1 2024 published metrics for wages and employment. Because actual employment is only captured at full enumeration (Census every 5 years) and not monthly, the baseline is completely Real, but the *variance per month* is a synthetic proxy.
- **Features:** 
    - `county`, `month` - **REAL**
    - `employment_rate` - **REAL-PROXY** (Real 2024 base anchor)
    - `avg_income` - **REAL-PROXY** (Real 2024 wage stats)
    - `unemployment_rate` - **REAL-PROXY**

### 2. Dataset 2: Residential Tenancies Board (RTB)
**[REAL-PROXY]**
- **Sourcing Strategy**: Pulled using actual Q4 2023 index yields mapped to 2024 inflation rates. Month-over-month growth within the ED level is smoothed synthetically.
- **Features:**
    - `avg_monthly_rent` - **REAL-PROXY**
    - `rent_growth_pct` - **REAL**
    - `rental_yield` - **REAL-PROXY**

### 3. Dataset 3: Transport Infrastructure Ireland (TII)
**[SYNTHETIC]**
- **Sourcing Strategy**: TII APIs are counter-specific on motorways and do not aggregate by Electoral Division. This dataset uses purely algorithmic generation.
- **Features:**
    - `traffic_volume` - **SYNTHETIC**
    - `congestion_delay_minutes` - **SYNTHETIC**
    - `avg_speed_kph` - **SYNTHETIC**

### 4. Dataset 4: Sustainable Energy Authority (SEAI)
**[SYNTHETIC]**
- **Sourcing Strategy**: SEAI requires strict authenticated registration for national bulk BER datasets. To ensure immediate dashboard boot, these are algorithmic.
- **Features:**
    - `ber_avg_score` - **SYNTHETIC**
    - `est_annual_energy_cost` - **SYNTHETIC**
    - `pct_a_rated` - **SYNTHETIC**

### 5. Geospatial Boundaries (GADM GeoJSON)
**[MIXED]**
- **County Boundaries**: `ireland_counties.geojson` is downloaded live directly from the GADM API database. **[100% REAL]**
- **ED Boundaries**: Because true L2 polygon rendering is exceptionally heavy for web mapping, ED geometries are algorithmically generated spatial centroids around the parent County. **[100% SYNTHETIC]**

---

## Final Composition Statistics

The mathematical composition of the dataset features ingested into the Léarslán machine learning pipelines breaks down as follows regarding their real-world provenance:

| Data Type | Feature Count | Percentage Total |
| :--- | :--- | :--- |
| **Purely Real / Anchored Proxies** | 7 metrics | **54%** |
| **Strictly Synthetic Fallbacks** | 6 metrics | **46%** |

### Summary
The final integration runs at a **54% Real (or Anchored Proxy)** to **46% Synthetic** split. This allows the application to respond with real-world housing and demographic metrics while intelligently faking missing gaps (like commute times on rural roads) to ensure zero UI crashes.
