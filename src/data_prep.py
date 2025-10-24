import pandas as pd
import numpy as np
from datetime import datetime
from .utils import read_raw, write_df, ensure_dirs

def load_and_clean() -> pd.DataFrame:
    df = read_raw()

    # Standardize dtypes
    df['date'] = pd.to_datetime(df['date'])
    numeric_cols = ['unit_price','quantity','discount_rate','gross_sales','net_sales']
    for c in numeric_cols:
        df[c] = pd.to_numeric(df[c], errors='coerce')

    # Drop impossible values
    df = df[df['unit_price'] >= 0]
    df = df[df['quantity'] >= 1]

    # Remove duplicates by transaction_id
    df = df.drop_duplicates(subset=['transaction_id'])

    # Handle missing: fill discount_rate with 0, others drop if critical
    df['discount_rate'] = df['discount_rate'].fillna(0)
    df = df.dropna(subset=['date','category','unit_price','quantity'])

    # Feature engineering
    df['revenue'] = df['net_sales']
    df['weekday'] = df['date'].dt.day_name()
    df['month'] = df['date'].dt.to_period('M').dt.to_timestamp()
    df['year'] = df['date'].dt.year
    df['is_weekend'] = df['weekday'].isin(['Saturday','Sunday']).astype(int)

    # Save cleaned
    write_df(df, "cleaned_transactions.csv")
    return df
