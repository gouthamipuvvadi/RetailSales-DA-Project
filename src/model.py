import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error
from .utils import write_df

def _calendar_features(df: pd.DataFrame) -> pd.DataFrame:
    monthly = df.groupby('month', as_index=False)['revenue'].sum()
    monthly['year'] = monthly['month'].dt.year
    monthly['month_num'] = monthly['month'].dt.month
    # simple lag features
    monthly = monthly.sort_values('month')
    monthly['lag1'] = monthly['revenue'].shift(1)
    monthly['lag2'] = monthly['revenue'].shift(2)
    monthly['lag3'] = monthly['revenue'].shift(3)
    monthly = monthly.dropna().reset_index(drop=True)
    return monthly

def run_model(df: pd.DataFrame):
    m = _calendar_features(df)
    X = m[['year','month_num','lag1','lag2','lag3']]
    y = m['revenue']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, shuffle=False)

    model = Ridge(alpha=1.0, random_state=0)
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    mae = mean_absolute_error(y_test, preds)

    out = m.copy()
    out['split'] = ['train']*len(X_train) + ['test']*len(X_test)
    out.loc[out['split']=='test','prediction'] = preds
    out.loc[out['split']=='test','abs_error'] = (out.loc[out['split']=='test','prediction'] - y_test).abs()

    write_df(out, "monthly_forecast_baseline.csv")

    # Write a brief summary to reports
    with open('reports/summary.md', 'w') as f:
        f.write(f"""# Results Summary

**Baseline model**: Ridge regression on calendar + lag features.
- Test MAE: {mae:.2f}

Notes:
- This is a simple baseline. Improvements could include holiday effects, promotion flags, hierarchical models per category/city, or dedicated time-series models.
""")
