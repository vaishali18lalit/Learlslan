# Léarslán Data Sources: Official Repositories

This document outlines the official data sources from which the core datasets for the Léarslán application (County and Electoral Division levels) can be downloaded. 

*Note: While the underlying data pipeline provides ingestion scripts for these sectors, the `data/` folder currently uses synthetically generated data via `generate_synthetic.py` for demonstration and MVP purposes. To replace these with real production data, use the official links provided below.*

## 1. Central Statistics Office (CSO) Data
**Themes:** Employment Rates, Average Income, Population, Demographics
- **CSO Open Data Portal (PxStat):** [https://data.cso.ie/](https://data.cso.ie/)
- **Census Small Area Population Statistics (SAPS)** (Best for ED-level data): [https://www.cso.ie/en/census/census2022/smallareapopulationstatistics/](https://www.cso.ie/en/census/census2022/smallareapopulationstatistics/)

## 2. Residential Tenancies Board (RTB) Data
**Themes:** Average Monthly Rent, Rent Growth, Rental Yields
- **RTB Data & Insights Hub:** [https://www.rtb.ie/data-insights](https://www.rtb.ie/data-insights)
- **RTB Rent Index** (Provides quarterly datasets at County and Local Electoral Area levels): [https://www.rtb.ie/research/rent-index](https://www.rtb.ie/research/rent-index)
- **Open Data Ireland - Rent:** [https://data.gov.ie/dataset/rtb-rent-index](https://data.gov.ie/dataset/rtb-rent-index)

## 3. Sustainable Energy Authority of Ireland (SEAI) Data
**Themes:** Average BER Ratings, Energy Costs, Energy Efficiency
- **National BER Register Public Search:** [https://ndber.seai.ie/](https://ndber.seai.ie/)
- **SEAI Data & Insights:** [https://www.seai.ie/data-and-insights/](https://www.seai.ie/data-and-insights/)
- **National BER Research Dataset** (Full raw dataset available upon request/registration for county/ED analytics): [https://ndber.seai.ie/BERResearchTool/Register/Create](https://ndber.seai.ie/BERResearchTool/Register/Create)

## 4. Transport Infrastructure Ireland (TII) Data
**Themes:** Traffic Volume, Congestion Delays, Commute Times
- **TII Traffic Data Hub** (Live and historical traffic counters): [https://trafficdata.tii.ie/](https://trafficdata.tii.ie/)
- **TII Library & Data Open Data:** [https://www.tii.ie/tii-library/traffic-data/](https://www.tii.ie/tii-library/traffic-data/)

## 5. Live Market Listings & Housing Supply
**Themes:** Active Property Market Data, Prices
- **Daft.ie:** [https://www.daft.ie/](https://www.daft.ie/) (Requires API or web-scraping compliance for live market feeds, referenced in `daft_client.py`)
- **Property Price Register (PPR):** [https://www.propertypriceregister.ie/](https://www.propertypriceregister.ie/)
