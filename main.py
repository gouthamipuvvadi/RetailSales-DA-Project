from src.data_prep import load_and_clean
from src.eda import run_eda
from src.segmentation import run_rfm
from src.model import run_model

def main():
    df = load_and_clean()
    run_eda(df)
    run_rfm(df)
    run_model(df)
    print("Pipeline complete. Check data/processed and reports/figures.")

if __name__ == "__main__":
    main()
