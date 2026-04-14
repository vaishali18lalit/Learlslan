# Learslán — Irish Community Intelligence Dashboard

**"Where in Ireland Should You Live?"**

Learslán is a hyper-local cost-of-living intelligence platform for Ireland. It scores and ranks **255 Electoral Divisions** across **26 counties** using machine learning, and provides an AI-powered advisor that answers relocation questions using live model inference.

![Python](https://img.shields.io/badge/Python-3.13-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-1.56-red)
![ML Models](https://img.shields.io/badge/ML_Models-8-green)
![Tests](https://img.shields.io/badge/Tests-45%2F45_passing-brightgreen)

---

## Features

### Dashboard (7 Tabs)
| Tab | Description |
|-----|-------------|
| **Overview** | Choropleth map of Ireland with risk/livability/transport/affordability heatmaps |
| **Property Explorer** | Rent, yield, and affordability metrics per county |
| **Area Duel** | Side-by-side radar chart comparison of any two areas |
| **Area Clusters** | KMeans + UMAP scatter plot with neighbourhood archetypes |
| **Budget Simulator** | Personalized monthly cost breakdown (rent + energy + commute) |
| **Forecast** | ARIMA(1,1,1) 6-month rent projections with confidence intervals |
| **Where to Live?** | TOPSIS multi-criteria ranking with user-controlled priority sliders |

### AI Advisor (Floating Chatbot)
- Floating bottom-right button accessible from **every tab**
- **Live ML inference** per query — runs TOPSIS, SHAP, and ARIMA in real-time
- **Hybrid RAG pipeline** — TF-IDF keyword search + Gemini semantic reranking over 9 policy documents
- **Citations** — every answer references `[Source: document]`, `[Data: metric]`, or `[Model: algorithm]`
- Powered by **LiteLLM** (OpenAI-compatible endpoint) with template fallback

---

## Architecture

```
User Query
    |
    +---> TF-IDF (keyword search) ---> Top 10 candidates
    |                                       |
    +---> Gemini Embeddings (semantic) -----+---> Rerank ---> Top 3 chunks
    |
    +---> Live ML Inference
    |       |-- TOPSIS (salary-based area ranking)
    |       |-- SHAP TreeExplainer (score drivers)
    |       |-- ARIMA (rent forecast)
    |
    +---> Context Assembly (area scores + page context + RAG + ML results)
    |
    +---> LLM (gpt-4o-mini via LiteLLM) ---> Response with citations
```

---

## ML Models (Pre-trained, 45/45 tests passing)

| # | Algorithm | Library | Purpose | Output |
|---|-----------|---------|---------|--------|
| 1 | **GBM x4** | scikit-learn | Score risk, livability, transport, affordability | 4 x [0-100] scores |
| 2 | **KMeans** | scikit-learn | Neighbourhood archetype clustering | 7 clusters (Premium Urban, Hidden Gems, Budget Rural, etc.) |
| 3 | **UMAP** | umap-learn | 2D projection for cluster visualization | (x, y) coordinates |
| 4 | **IsolationForest** | scikit-learn | Anomaly detection (rent spikes, affordability crises) | anomaly flag + severity |
| 5 | **TOPSIS** | numpy | Multi-criteria area ranking by user preferences | match score 0-100 |
| 6 | **ARIMA(1,1,1)** | statsmodels | 6-month rent/metric forecasting | forecast + 80% CI |
| 7 | **SHAP TreeExplainer** | shap | Explain what drives each area's score | per-feature contributions |
| 8 | **TF-IDF + Cosine** | scikit-learn | RAG document retrieval for AI Advisor | top-3 relevant chunks |

Pre-trained models are saved in `ml/models/` — the app loads them directly without retraining.

---

## Data Sources

| Dataset | Source | Granularity |
|---------|--------|-------------|
| Employment & Income | CSO (Central Statistics Office) | 255 EDs x 12 months |
| Rent & Yields | RTB (Residential Tenancies Board) | 255 EDs x 12 months |
| Traffic & Congestion | TII (Transport Infrastructure Ireland) | 255 EDs x 12 months |
| Energy & BER Ratings | SEAI (Sustainable Energy Authority) | 255 EDs (static) |
| County Boundaries | GADM GeoJSON | 26 county polygons |
| ED Boundaries | Generated | 255 ED polygons |

### RAG Corpus (9 documents)
- National Housing Strategy
- RTB Rent Index Report Q4 2024
- CSO Census 2022 Summary
- SEAI Energy & BER Report
- TII Transport Strategy
- Housing for All Plan
- Croi Conaithe Scheme
- Daft.ie Market Report Q4 2024
- Official Data Sources Guide

---

## Quick Start

### 1. Clone
```bash
git clone https://github.com/vaishali18lalit/Learlslan.git
cd Learlslan
```

### 2. Create virtual environment
```bash
python -m venv venv
```

### 3. Install dependencies
```bash
# Using uv (recommended, fast)
uv pip install -r requirements.txt --python venv/Scripts/python.exe

# Or using pip
venv/Scripts/pip install -r requirements.txt
```

### 4. Configure API keys
Copy `.env.example` to `.env` and fill in your keys:
```bash
cp .env.example .env
```

```env
GEMINI_API_KEY=your_gemini_api_key_here
LITELLM_BASE_URL=https://your-litellm-endpoint.com
LITELLM_API_KEY=sk-your-key-here
```

The AI Advisor works with any **OpenAI-compatible endpoint** via LiteLLM. If no keys are configured, it falls back to template-based responses.

### 5. Run
```bash
venv/Scripts/streamlit run app.py
```

Open http://localhost:8501 in your browser.

---

## Project Structure

```
Learlslan/
|-- app.py                    # Main Streamlit dashboard
|-- config.py                 # Configuration, constants, ED registry
|-- requirements.txt          # Python dependencies
|
|-- ml/                       # ML scoring pipeline
|   |-- pipeline.py           # Orchestration: load/save artifacts
|   |-- feature_engineering.py # 6 derived features
|   |-- risk_model.py         # 4x GBM training + labels
|   |-- clustering.py         # KMeans + UMAP
|   |-- anomaly_detector.py   # IsolationForest
|   |-- recommender.py        # TOPSIS ranking
|   |-- forecasting.py        # ARIMA(1,1,1)
|   |-- explainability.py     # SHAP TreeExplainer
|   |-- models/               # Pre-trained model artifacts
|   |   |-- scored_df.parquet
|   |   |-- risk_score_gbm.joblib
|   |   |-- livability_score_gbm.joblib
|   |   |-- transport_score_gbm.joblib
|   |   |-- affordability_score_gbm.joblib
|   |   |-- kmeans_model.joblib
|   |   |-- cluster_scaler.joblib
|   |   +-- feature_columns.json
|   +-- data/                 # Merged CSVs for ML
|
|-- insights/                 # AI Advisor module
|   |-- chat.py               # Chat UI (popover + full page modes)
|   |-- rag_engine.py         # Hybrid TF-IDF + Gemini embedding RAG
|   |-- context.py            # Page context collector
|   +-- ml_tools.py           # Live TOPSIS, SHAP, ARIMA inference
|
|-- ui/                       # Dashboard tab renderers
|   |-- styles.py             # CSS injection + metric cards
|   |-- map_view.py           # Folium choropleth map
|   |-- charts.py             # Plotly charts
|   |-- sidebar.py            # County detail panel
|   |-- tab_property.py       # Property Explorer
|   |-- tab_duel.py           # Area Duel (radar chart)
|   |-- tab_clusters.py       # KMeans cluster scatter
|   |-- tab_budget.py         # Budget Simulator
|   |-- tab_forecast.py       # ARIMA forecast charts
|   |-- tab_recommender.py    # TOPSIS recommender
|   +-- alert_banner.py       # Anomaly alert banner
|
|-- ingestion/                # Data ingestion from external sources
|   |-- spatial_harmonizer.py # Joins all datasets on county/ED
|   |-- cso_api.py            # CSO employment data loader
|   |-- housing_loader.py     # RTB rent data loader
|   |-- tii_scraper.py        # TII traffic data loader
|   |-- seai_loader.py        # SEAI energy data loader
|   +-- daft_client.py        # Daft.ie market data client
|
|-- data/
|   |-- real_data/            # Source CSVs + GeoJSON
|   +-- documents/            # RAG corpus (9 policy documents)
|       +-- doc_metadata.json # Source provenance for citations
|
+-- docs/                     # Architecture & design documents
```

---

## How the AI Advisor Works

When a user asks a question, the advisor:

1. **Detects intent** — parses the query for salary mentions, area names, keywords like "why", "forecast", "where"
2. **Runs live ML models** based on intent:
   - Salary detected → runs **TOPSIS** with personalized budget constraints
   - "Why" questions → runs **SHAP TreeExplainer** on the relevant GBM model
   - "Forecast" questions → runs **ARIMA(1,1,1)** on the area's rent time series
3. **Retrieves RAG context** — hybrid TF-IDF + Gemini embedding search over 9 policy documents
4. **Assembles context** — area scores, SHAP drivers, TOPSIS results, forecasts, RAG chunks, page context
5. **Calls LLM** — sends everything to gpt-4o-mini via LiteLLM with citation instructions
6. **Returns response** with `[Source:]`, `[Data:]`, and `[Model:]` citations

---

## Feature Engineering

| Feature | Formula | Purpose |
|---------|---------|---------|
| `affordability_index` | `(income / 12) / rent` | Can you afford to live here? |
| `rent_to_income_pct` | `(rent x 12) / income x 100` | What % of income goes to rent? |
| `commute_cost_monthly` | `congestion_delay x 2 x 22` | Monthly commute cost proxy |
| `true_cost_index` | `0.5*Norm(rent) + 0.25*Norm(energy) + 0.25*Norm(commute)` | Real cost of living (0-100) |
| `energy_tax` | `rent + (energy_cost / 12)` | True monthly housing bill |
| `commute_to_rent_ratio` | `commute_cost / rent` | Transport burden relative to rent |

---

## Environment Variables

| Variable | Required | Description |
|----------|----------|-------------|
| `GEMINI_API_KEY` | Optional | Google Gemini API key (for RAG embeddings) |
| `LITELLM_BASE_URL` | For AI chat | OpenAI-compatible LLM endpoint URL |
| `LITELLM_API_KEY` | For AI chat | API key for the LLM endpoint |

Without LLM keys, the AI Advisor runs in **template mode** — still shows area data and RAG context, just without conversational AI.

---

## License

Internal project — Hackathon 2025.
