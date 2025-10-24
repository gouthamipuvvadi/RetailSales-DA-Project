import pandas as pd
import numpy as np
from datetime import datetime
from .utils import write_df

def rfm(df: pd.DataFrame, now: pd.Timestamp=None):
    if now is None:
        now = df['date'].max() + pd.Timedelta(days=1)

    # Compute RFM per customer
    agg = df.groupby('customer_id').agg(
        recency=('date', lambda x: (now - x.max()).days),
        frequency=('transaction_id','nunique'),
        monetary=('revenue','sum')
    ).reset_index()

    # Score 1-5 (higher is better for frequency, monetary; lower is better for recency)
    agg['R'] = pd.qcut(agg['recency'], 5, labels=[5,4,3,2,1]).astype(int)
    agg['F'] = pd.qcut(agg['frequency'].rank(method='first'), 5, labels=[1,2,3,4,5]).astype(int)
    agg['M'] = pd.qcut(agg['monetary'].rank(method='first'), 5, labels=[1,2,3,4,5]).astype(int)
    agg['RFM_Score'] = agg['R']*100 + agg['F']*10 + agg['M']

    # Simple segments
    def segment(row):
        if row['R']>=4 and row['F']>=4 and row['M']>=4:
            return 'Champions'
        if row['R']>=4 and row['F']>=3:
            return 'Loyal'
        if row['R']>=3 and row['M']>=4:
            return 'Big Spenders'
        if row['R']<=2 and row['F']<=2:
            return 'At Risk'
        return 'Regulars'

    agg['Segment'] = agg.apply(segment, axis=1)
    return agg

def run_rfm(df: pd.DataFrame):
    rfm_df = rfm(df)
    write_df(rfm_df, "rfm_segments.csv")
