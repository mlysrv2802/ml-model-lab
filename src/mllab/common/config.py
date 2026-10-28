from pathlib import Path

# config.py -> common/ -> mllab/ -> src/ -> project root
PROJECT_ROOT = Path(__file__).resolve().parents[3]
DATA_DIR = PROJECT_ROOT / "data"
CHURN_PATH = DATA_DIR / "telco_churn.csv"
REPORTS_DIR = PROJECT_ROOT / "reports"  # git-ignored: generated charts

RANDOM_STATE = 42  # fixed seed, so every run gives the same numbers
TEST_SIZE = 0.2

TARGET = "Churn"
ID_COLUMN = "customerID"
