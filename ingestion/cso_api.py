"""CSO PxStat API wrapper - loads employment/income data (county + ED level)."""
import pandas as pd
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import CSO_FILE, CSO_ED_FILE


def load_cso_data() -> pd.DataFrame:
    """Load CSO employment data from local CSV (county level)."""
    if not CSO_FILE.exists():
        raise FileNotFoundError(f"CSO data not found at {CSO_FILE}. Run generate_synthetic.py first.")
    df = pd.read_csv(CSO_FILE, parse_dates=["month"])
    return df


def get_latest_cso(df: pd.DataFrame) -> pd.DataFrame:
    """Get the latest month's CSO data per county."""
    latest_month = df["month"].max()
    return df[df["month"] == latest_month].copy()


# ── Electoral Division Level ──────────────────────────────────

def load_cso_ed_data() -> pd.DataFrame:
    """Load CSO employment data from local CSV (ED level)."""
    if not CSO_ED_FILE.exists():
        raise FileNotFoundError(f"CSO ED data not found at {CSO_ED_FILE}. Run generate_synthetic.py first.")
    df = pd.read_csv(CSO_ED_FILE, parse_dates=["month"])
    return df


def get_latest_cso_ed(df: pd.DataFrame) -> pd.DataFrame:
    """Get the latest month's CSO data per ED."""
    latest_month = df["month"].max()
    return df[df["month"] == latest_month].copy()
