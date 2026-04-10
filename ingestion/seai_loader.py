"""SEAI BER energy rating data loader (county + ED level)."""
import pandas as pd
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import SEAI_FILE, SEAI_ED_FILE


def load_seai_data() -> pd.DataFrame:
    """Load SEAI BER data from local CSV (county level)."""
    if not SEAI_FILE.exists():
        raise FileNotFoundError(f"SEAI data not found at {SEAI_FILE}. Run generate_synthetic.py first.")
    df = pd.read_csv(SEAI_FILE)
    return df


# ── Electoral Division Level ──────────────────────────────────

def load_seai_ed_data() -> pd.DataFrame:
    """Load SEAI BER data from local CSV (ED level)."""
    if not SEAI_ED_FILE.exists():
        raise FileNotFoundError(f"SEAI ED data not found at {SEAI_ED_FILE}. Run generate_synthetic.py first.")
    df = pd.read_csv(SEAI_ED_FILE)
    return df
