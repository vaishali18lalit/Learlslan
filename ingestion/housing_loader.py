"""RTB Rent / Housing data loader (county + ED level)."""
import pandas as pd
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import RTB_FILE, RTB_ED_FILE


def load_housing_data() -> pd.DataFrame:
    """Load RTB rent data from local CSV (county level)."""
    if not RTB_FILE.exists():
        raise FileNotFoundError(f"RTB data not found at {RTB_FILE}. Run generate_synthetic.py first.")
    df = pd.read_csv(RTB_FILE, parse_dates=["month"])
    return df


def get_latest_housing(df: pd.DataFrame) -> pd.DataFrame:
    """Get the latest month's housing data per county."""
    latest_month = df["month"].max()
    return df[df["month"] == latest_month].copy()


# ── Electoral Division Level ──────────────────────────────────

def load_housing_ed_data() -> pd.DataFrame:
    """Load RTB rent data from local CSV (ED level)."""
    if not RTB_ED_FILE.exists():
        raise FileNotFoundError(f"RTB ED data not found at {RTB_ED_FILE}. Run generate_synthetic.py first.")
    df = pd.read_csv(RTB_ED_FILE, parse_dates=["month"])
    return df


def get_latest_housing_ed(df: pd.DataFrame) -> pd.DataFrame:
    """Get the latest month's housing data per ED."""
    latest_month = df["month"].max()
    return df[df["month"] == latest_month].copy()
