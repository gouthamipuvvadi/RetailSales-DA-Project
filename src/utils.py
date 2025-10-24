import pandas as pd
from pathlib import Path

RAW_PATH = Path("data/raw/transactions.csv")
PROCESSED_DIR = Path("data/processed")
FIG_DIR = Path("reports/figures")
REPORTS_DIR = Path("reports")

def ensure_dirs():
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

def read_raw() -> pd.DataFrame:
    return pd.read_csv(RAW_PATH)

def write_df(df: pd.DataFrame, name: str):
    ensure_dirs()
    out = PROCESSED_DIR / name
    df.to_csv(out, index=False)
    return out
