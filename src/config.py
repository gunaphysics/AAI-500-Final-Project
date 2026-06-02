from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
INTERIM_DIR = DATA_DIR / "interim"
PROCESSED_DIR = DATA_DIR / "processed"

MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

KAGGLE_DATASET = "nudratabbas/heart-failure-readmission-and-sdoh-dataset"
RAW_FILENAME = "heart_failure_readmission_dataset.csv"

TARGET = "readmitted_30d"
ID_COL = "patient_id"
RANDOM_STATE = 42
