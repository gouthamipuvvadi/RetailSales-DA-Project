import matplotlib.pyplot as plt
import pandas as pd
from .utils import ensure_dirs, FIG_DIR, write_df

def _save_plot(name: str):
    plt.tight_layout()
    plt.savefig(FIG_DIR / name, dpi=160)
    plt.close()

def run_eda(df: pd.DataFrame):
    ensure_dirs()
    # Sales by month
    monthly = df.groupby('month', as_index=False)['revenue'].sum()
    monthly.plot(x='month', y='revenue', kind='line', title='Monthly Revenue')
    _save_plot('monthly_revenue.png')

    # Sales by category
    by_cat = df.groupby('category', as_index=False)['revenue'].sum().sort_values('revenue', ascending=False)
    by_cat.plot(x='category', y='revenue', kind='bar', title='Revenue by Category')
    _save_plot('revenue_by_category.png')

    # Discount vs revenue per order
    sample = df.sample(min(1000, len(df)), random_state=7)
    sample.plot(x='discount_rate', y='revenue', kind='scatter', title='Discount vs Revenue (Order-Level)')
    _save_plot('discount_vs_revenue.png')

    write_df(monthly, "monthly_revenue.csv")
    write_df(by_cat, "revenue_by_category.csv")
