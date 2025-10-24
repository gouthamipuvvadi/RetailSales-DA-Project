from src.data_prep import load_and_clean

def test_cleaning_basic():
    df = load_and_clean()
    assert (df['unit_price'] >= 0).all()
    assert (df['quantity'] >= 1).all()
    assert df['date'].dtype.kind == 'M'  # datetime
    assert df['revenue'].notna().all()
