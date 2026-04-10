"""TII Traffic data loader (county + ED level)."""
import pandas as pd
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import TII_FILE, TII_ED_FILE


def load_tii_data() -> pd.DataFrame:
    """Load TII traffic data from local CSV (county level)."""
    if not TII_FILE.exists():
        raise FileNotFoundError(f"TII data not found at {TII_FILE}. Run generate_synthetic.py first.")
    df = pd.read_csv(TII_FILE, parse_dates=["month"])
    return df


def get_latest_tii(df: pd.DataFrame) -> pd.DataFrame:
    """Get the latest month's TII data per county."""
    latest_month = df["month"].max()
    return df[df["month"] == latest_month].copy()


# ── Electoral Division Level ──────────────────────────────────

def load_tii_ed_data() -> pd.DataFrame:
    """Load TII traffic data from local CSV (ED level)."""
    if not TII_ED_FILE.exists():
        raise FileNotFoundError(f"TII ED data not found at {TII_ED_FILE}. Run generate_synthetic.py first.")
    df = pd.read_csv(TII_ED_FILE, parse_dates=["month"])
    return df


def get_latest_tii_ed(df: pd.DataFrame) -> pd.DataFrame:
    """Get the latest month's TII data per ED."""
    latest_month = df["month"].max()
    return df[df["month"] == latest_month].copy()
